"""Per-backend admission control: requests/min and tokens/min token buckets plus a concurrency cap.

Tokens are reserved up front from an estimate (prompt characters / 4 + the completion cap) and
reconciled with the real usage afterwards. A provider ``retry-after`` pauses the whole backend.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Optional

from judgerl.judge.clock import Clock


class TokenBucket:
    """Classic token bucket: ``rate_per_s`` refill, ``capacity`` burst. Level may go negative on reconcile."""

    def __init__(self, rate_per_s: float, capacity: float, clock: Clock):
        if rate_per_s <= 0 or capacity <= 0:
            raise ValueError("token bucket rate and capacity must be > 0")
        self.rate = float(rate_per_s)
        self.capacity = float(capacity)
        self.clock = clock
        self.level = float(capacity)
        self._t = clock.now()

    def _refill(self) -> None:
        now = self.clock.now()
        if now > self._t:
            self.level = min(self.capacity, self.level + (now - self._t) * self.rate)
        self._t = now

    def wait_time(self, amount: float) -> float:
        """Seconds until ``amount`` can be taken (amounts above capacity wait for a full bucket)."""
        self._refill()
        need = min(amount, self.capacity)
        return 0.0 if self.level >= need else (need - self.level) / self.rate

    def take(self, amount: float) -> None:
        self._refill()
        self.level -= amount

    def give(self, amount: float) -> None:
        self._refill()
        self.level = min(self.capacity, self.level + amount)


@dataclass
class Admission:
    """Handle returned by :meth:`RateLimiter.acquire`; pass it back to :meth:`RateLimiter.release`."""

    tokens_reserved: float = 0.0
    released: bool = False


class RateLimiter:
    """requests/min + tokens/min + max concurrency for one backend. All limits optional."""

    def __init__(self, rpm: Optional[float] = None, tpm: Optional[float] = None,
                 max_concurrency: Optional[int] = None, clock: Optional[Clock] = None,
                 burst_s: float = 1.0):
        self.clock = clock or Clock()
        # burst: at most ``burst_s`` seconds worth of the per-minute budget at once (>= 1 request)
        self.requests = TokenBucket(rpm / 60.0, max(1.0, rpm / 60.0 * burst_s), self.clock) if rpm else None
        self.tokens = TokenBucket(tpm / 60.0, max(1.0, tpm / 60.0 * max(burst_s, 1.0) * 10), self.clock) if tpm else None
        self.max_concurrency = max_concurrency
        self._sem = asyncio.Semaphore(max_concurrency) if max_concurrency else None
        self._lock = asyncio.Lock()
        self._paused_until = 0.0
        self.waited_s = 0.0

    def pause_for(self, seconds: float) -> None:
        """Honor a provider ``retry-after``: no request is admitted before ``now + seconds``."""
        self._paused_until = max(self._paused_until, self.clock.now() + max(0.0, seconds))

    @property
    def inflight(self) -> int:
        return 0 if self._sem is None else self.max_concurrency - self._sem._value  # noqa: SLF001

    async def acquire(self, est_tokens: float, timeout: Optional[float] = None) -> Optional[Admission]:
        """Wait for admission. Returns ``None`` if it cannot be granted within ``timeout`` seconds."""
        start = self.clock.now()
        deadline = None if timeout is None else start + timeout
        if self._sem is not None:
            try:
                if deadline is None:
                    await self._sem.acquire()
                else:
                    await asyncio.wait_for(self._sem.acquire(), max(0.0, deadline - self.clock.now()))
            except asyncio.TimeoutError:
                return None
        try:
            async with self._lock:  # FIFO fairness between waiters of this backend
                while True:
                    wait = max(0.0, self._paused_until - self.clock.now())
                    if self.requests is not None:
                        wait = max(wait, self.requests.wait_time(1.0))
                    if self.tokens is not None and est_tokens > 0:
                        wait = max(wait, self.tokens.wait_time(est_tokens))
                    if wait <= 0:
                        break
                    if deadline is not None and self.clock.now() + wait > deadline:
                        raise asyncio.TimeoutError
                    await self.clock.sleep(wait)
                if self.requests is not None:
                    self.requests.take(1.0)
                if self.tokens is not None and est_tokens > 0:
                    self.tokens.take(est_tokens)
        except BaseException as e:
            if self._sem is not None:
                self._sem.release()
            if isinstance(e, asyncio.TimeoutError):
                return None
            raise
        self.waited_s += self.clock.now() - start
        return Admission(tokens_reserved=est_tokens if self.tokens is not None else 0.0)

    def release(self, adm: Admission, actual_tokens: Optional[float] = None) -> None:
        if adm.released:
            return
        adm.released = True
        if self.tokens is not None and actual_tokens is not None:
            diff = adm.tokens_reserved - actual_tokens
            if diff > 0:
                self.tokens.give(diff)
            elif diff < 0:
                self.tokens.take(-diff)
        if self._sem is not None:
            self._sem.release()


def estimate_prompt_tokens(messages) -> int:
    """Cheap, tokenizer-free estimate (4 characters per token, at least 1 per message)."""
    n = 0
    for m in messages:
        c = m.get("content")
        if isinstance(c, str):
            n += len(c)
        elif isinstance(c, list):
            for part in c:
                if isinstance(part, dict):
                    n += len(str(part.get("text", "")))
    return max(len(messages), n // 4 + 4 * len(messages))
