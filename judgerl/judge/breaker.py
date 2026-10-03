"""Circuit breakers (per backend and global).

A breaker trips on any of:
- failure rate over a sliding time window (after ``min_calls`` outcomes in the window);
- ``consecutive_failures`` in a row;
- a fatal status (``auth``, ``quota`` by default);
- cumulative cost reaching ``cost_budget_usd``.

Policy ``pause``: dispatch stops while open; after ``cooldown_s`` the breaker goes half-open and lets
``half_open_probes`` requests through; a success closes it, a failure reopens it. Budget trips never
recover. Policy ``stop``: the trip is permanent and the client raises :class:`JudgeHalted`.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Iterable, Optional, Tuple

from judgerl.judge.clock import Clock
from judgerl.judge.types import CONTROL_STATUSES, FATAL_BACKEND_STATUSES, JudgeStatus

CLOSED, OPEN, HALF_OPEN = "closed", "open", "half_open"


@dataclass
class BreakerSettings:
    window_s: float = 60.0
    min_calls: int = 20
    failure_rate: Optional[float] = 0.5
    consecutive_failures: Optional[int] = 10
    cooldown_s: float = 30.0
    half_open_probes: int = 1
    policy: str = "pause"
    cost_budget_usd: Optional[float] = None
    fatal_statuses: Tuple[JudgeStatus, ...] = tuple(FATAL_BACKEND_STATUSES)
    ignore_statuses: Tuple[JudgeStatus, ...] = tuple(CONTROL_STATUSES)
    enabled: bool = True

    def __post_init__(self):
        if self.policy not in ("pause", "stop"):
            raise ValueError(f"breaker policy must be 'pause' or 'stop', got {self.policy!r}")
        self.fatal_statuses = tuple(JudgeStatus(s) for s in self.fatal_statuses)
        self.ignore_statuses = tuple(JudgeStatus(s) for s in self.ignore_statuses)


@dataclass
class CircuitBreaker:
    name: str
    settings: BreakerSettings = field(default_factory=BreakerSettings)
    clock: Clock = field(default_factory=Clock)

    def __post_init__(self):
        self.state = CLOSED
        self.reason: Optional[str] = None
        self.budget_tripped = False
        self.cost_usd = 0.0
        self.trips = 0
        self._window: Deque[Tuple[float, bool]] = deque()
        self._consecutive = 0
        self._opened_at = 0.0
        self._probes = 0

    # ------------------------------------------------------------------ state
    @property
    def permanent(self) -> bool:
        """Open for good: policy ``stop`` or a budget trip."""
        return self.state == OPEN and (self.settings.policy == "stop" or self.budget_tripped)

    def _maybe_half_open(self) -> None:
        if self.state == OPEN and not self.permanent and self.clock.now() - self._opened_at >= self.settings.cooldown_s:
            self.state = HALF_OPEN
            self._probes = 0

    def allow(self) -> bool:
        """May a request be dispatched now? In half-open state this consumes a probe slot."""
        if not self.settings.enabled:
            return True
        self._maybe_half_open()
        if self.state == CLOSED:
            return True
        if self.state == HALF_OPEN and self._probes < self.settings.half_open_probes:
            self._probes += 1
            return True
        return False

    def release_probe(self) -> None:
        """Give back a probe slot taken by :meth:`allow` when the request was never sent."""
        if self.state == HALF_OPEN and self._probes > 0:
            self._probes -= 1

    def wait_time(self) -> Optional[float]:
        """Seconds until a probe may be sent; 0 if closed; ``None`` if permanently open."""
        if not self.settings.enabled or self.state == CLOSED:
            return 0.0
        if self.permanent:
            return None
        self._maybe_half_open()
        if self.state == HALF_OPEN:
            return 0.0 if self._probes < self.settings.half_open_probes else self.settings.cooldown_s / 10 or 0.01
        return max(0.0, self.settings.cooldown_s - (self.clock.now() - self._opened_at))

    def trip(self, reason: str, budget: bool = False) -> None:
        if self.state != OPEN:
            self.trips += 1
        self.state = OPEN
        self.reason = reason
        self.budget_tripped = self.budget_tripped or budget
        self._opened_at = self.clock.now()
        self._probes = 0

    def reset(self) -> None:
        """Manually close the breaker (e.g. after a budget increase)."""
        self.state, self.reason, self.budget_tripped = CLOSED, None, False
        self._window.clear()
        self._consecutive = 0

    # ------------------------------------------------------------------ recording
    def add_cost(self, usd: float) -> None:
        self.cost_usd += usd
        b = self.settings.cost_budget_usd
        if self.settings.enabled and b is not None and self.cost_usd >= b:
            self.trip(f"{self.name}: cost ${self.cost_usd:.4f} reached budget ${b:.4f}", budget=True)

    def record(self, status: JudgeStatus) -> None:
        if not self.settings.enabled or status in self.settings.ignore_statuses:
            return
        s = self.settings
        now = self.clock.now()
        ok = status == JudgeStatus.OK
        if self.state == HALF_OPEN:
            if ok:
                self.state, self.reason = CLOSED, None
                self._window.clear()
                self._consecutive = 0
            else:
                self.trip(f"{self.name}: half-open probe failed ({status.value})")
            return
        self._window.append((now, ok))
        while self._window and now - self._window[0][0] > s.window_s:
            self._window.popleft()
        self._consecutive = 0 if ok else self._consecutive + 1
        if self.state == OPEN:
            return
        if status in s.fatal_statuses:
            self.trip(f"{self.name}: fatal status {status.value}")
        elif s.consecutive_failures and self._consecutive >= s.consecutive_failures:
            self.trip(f"{self.name}: {self._consecutive} consecutive failures (last {status.value})")
        elif s.failure_rate is not None and len(self._window) >= s.min_calls:
            fails = sum(1 for _, o in self._window if not o)
            rate = fails / len(self._window)
            if rate >= s.failure_rate:
                self.trip(f"{self.name}: failure rate {rate:.2f} >= {s.failure_rate} over {len(self._window)} calls")

    def record_many(self, statuses: Iterable[JudgeStatus]) -> None:
        for st in statuses:
            self.record(st)

    def snapshot(self) -> dict:
        fails = sum(1 for _, o in self._window if not o)
        return {"state": self.state, "reason": self.reason, "trips": self.trips, "cost_usd": self.cost_usd,
                "window_calls": len(self._window), "window_failures": fails, "consecutive": self._consecutive}
