"""The judge client: queueing, admission, retries, failover, parsing, caching, breakers, budgets.

Life of a request::

    submit ─► priority queue (bounded; submit waits when full)
          ─► worker: global breaker gate ─► cache lookup ─► single-flight (in-process + cross-process)
          ─► attempt loop over the pool:
               pick backend (breaker allows, attempts left) ─► budget reservation ─► rate limiter
               ─► backend.generate (one attempt, per-attempt timeout) ─► classify / parse / validate
               ─► ok: cache + resolve | retryable: backoff (retry-after) or fail over | fatal: next backend
          ─► telemetry (exactly once per request) ─► future resolves with a JudgeResult

Every request resolves exactly once with a :class:`JudgeResult` whose status says what happened; the
only exception is a breaker (or budget) with policy ``stop``, which raises :class:`JudgeHalted` from
the futures of every unresolved request and from later submissions.
"""
from __future__ import annotations

import asyncio
import heapq
import itertools
import logging
import random
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from judgerl.judge.backends.base import JudgeBackend, classify_exception
from judgerl.judge.backends.registry import get_backend_factory, import_object
from judgerl.judge.breaker import CircuitBreaker
from judgerl.judge.cache import JudgeCache, cache_key
from judgerl.judge.clock import Clock
from judgerl.judge.config import BackendConfig, JudgeConfig
from judgerl.judge.ratelimit import RateLimiter, estimate_prompt_tokens
from judgerl.judge.reasoning import reasoning_budget
from judgerl.judge.schema import (correction_message, extract_json, parse_strict, schema_name, schema_to_dict,
                                  split_reasoning, validate, with_schema_instruction)
from judgerl.judge.telemetry import CostModel, Telemetry
from judgerl.judge.types import (BackendCall, BackendError, BackendResponse, Decoding, JudgeHalted, JudgeRequest,
                                 JudgeResult, JudgeStatus, ReasoningSpec, Usage)

log = logging.getLogger("judgerl.judge")

PREFLIGHT_SCHEMA = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}
PREFLIGHT_PROMPT = 'Reply with exactly this JSON object and nothing else: {"ok": true}'


# ----------------------------------------------------------------------------------------- internals
class _Slot:
    """A backend plus its admission control, breaker and prices."""

    def __init__(self, backend: JudgeBackend, cfg: BackendConfig, clock: Clock, use_litellm_costs: bool):
        self.backend = backend
        self.cfg = cfg
        self.name = backend.name
        rl = cfg.rate_limit
        self.limiter = RateLimiter(rl.rpm, rl.tpm, rl.max_concurrency, clock=clock, burst_s=rl.burst_s)
        self.breaker = CircuitBreaker(f"backend:{self.name}", cfg.breaker.settings(), clock)
        prices = {backend.model_id: cfg.prices.prices()} if cfg.prices is not None else {}
        self.costs = CostModel(prices, use_litellm=use_litellm_costs)
        # observed / estimated prompt tokens: raised at once, decays slowly (budget + tpm estimates)
        self.prompt_ratio = 1.0
        # per-backend hard cost cap (breaker.cost_budget_usd): worst-case reservations of in-flight calls
        self.reserved_usd = 0.0
        self.reservations = 0
        self.max_call_cost: Optional[float] = None  # largest observed cost of one call (unpriced models)
        self.cost_cond = asyncio.Condition()

    def calibrate(self, estimated: int, actual: int) -> None:
        if estimated <= 0 or actual <= 0:
            return
        obs = actual / estimated
        self.prompt_ratio = obs if obs > self.prompt_ratio else max(0.25, 0.99 * self.prompt_ratio + 0.01 * obs)

    @property
    def model_id(self) -> str:
        return self.backend.model_id


@dataclass(eq=False)
class _Item:
    request: JudgeRequest
    future: asyncio.Future
    submitted: float
    deadline: Optional[float]
    seq: int
    started: bool = False
    finished: bool = False
    first_dispatch: Optional[float] = None
    task: Optional[asyncio.Task] = None
    cancel_reason: Optional[str] = None
    global_probe: bool = False


@dataclass
class _Outcome:
    status: JudgeStatus
    slot: Optional[_Slot] = None
    response: Optional[BackendResponse] = None
    parsed: Any = None
    lenient: bool = False
    errors: List[str] = field(default_factory=list)
    correction: Optional[str] = None
    error: Optional[str] = None
    retry_after: Optional[float] = None
    retryable: Optional[bool] = None
    cost: float = 0.0
    cost_known: bool = True
    usage: Usage = field(default_factory=Usage)
    latency_s: float = 0.0
    dispatched: bool = False
    failover: bool = False       # not dispatched because this backend cannot serve (e.g. its cost cap)
    answer_text: Optional[str] = None


class _QueueClosed(Exception):
    """Raised to producers waiting for space when the queue is closed (client close or halt)."""


class _PriorityQueue:
    """Bounded heap queue: higher ``priority`` first, then earlier deadline, then FIFO.

    :meth:`close` drains the queue, marks it closed and wakes every waiting producer, which then
    raises :class:`_QueueClosed` instead of enqueueing.
    """

    def __init__(self, maxsize: int):
        self.maxsize = maxsize
        self._heap: List[Tuple] = []
        self._lock = asyncio.Lock()
        self._not_empty = asyncio.Condition(self._lock)
        self._not_full = asyncio.Condition(self._lock)
        self.closed = False
        self.waiting_producers = 0
        self.wake_task: Optional[asyncio.Task] = None

    def __len__(self) -> int:
        return len(self._heap)

    async def put(self, item: _Item) -> None:
        async with self._not_full:
            while not self.closed and len(self._heap) >= self.maxsize:
                self.waiting_producers += 1
                try:
                    await self._not_full.wait()
                finally:
                    self.waiting_producers -= 1
            if self.closed:
                raise _QueueClosed
            dl = item.deadline if item.deadline is not None else float("inf")
            heapq.heappush(self._heap, (-item.request.priority, dl, item.seq, item))
            self._not_empty.notify()

    async def get(self) -> _Item:
        async with self._not_empty:
            while not self._heap:
                await self._not_empty.wait()
            item = heapq.heappop(self._heap)[-1]
            self._not_full.notify()
            return item

    def close(self) -> List[_Item]:
        """Close the queue and return the queued items. Synchronous so it can run from callbacks;
        waiting producers are woken by a task that notifies under the condition's lock."""
        self.closed = True
        items = [e[-1] for e in self._heap]
        self._heap.clear()
        if self.wake_task is None:
            self.wake_task = asyncio.get_running_loop().create_task(self._wake_producers())
        return items

    async def wait_producers_gone(self, timeout: float = 5.0) -> None:
        """After :meth:`close`: wait until every woken producer has left :meth:`put`."""
        if self.wake_task is not None:
            await self.wake_task
        end = asyncio.get_running_loop().time() + timeout
        while self.waiting_producers and asyncio.get_running_loop().time() < end:
            await asyncio.sleep(0)
        await asyncio.sleep(0)

    async def _wake_producers(self) -> None:
        async with self._not_full:
            self._not_full.notify_all()


# ----------------------------------------------------------------------------------------- preflight
@dataclass
class PreflightReport:
    entries: List[Dict[str, Any]]
    problems: List[str]
    warnings: List[str]

    @property
    def ok(self) -> bool:
        return not self.problems

    def __str__(self) -> str:
        lines = ["judge preflight: " + ("OK" if self.ok else f"{len(self.problems)} problem(s)")]
        for e in self.entries:
            lines.append(f"  [{'ok' if e['ok'] else 'FAIL'}] {e['backend']} ({e['model']}): status={e['status']} "
                         f"latency={e['latency_s']:.2f}s cost=${e['cost_usd']:.6f}"
                         + (f" error={e['error']}" if e.get("error") else ""))
        lines += [f"  problem: {p}" for p in self.problems]
        lines += [f"  warning: {w}" for w in self.warnings]
        return "\n".join(lines)

    def raise_if_failed(self) -> "PreflightReport":
        if not self.ok:
            raise PreflightError(str(self))
        return self


class PreflightError(RuntimeError):
    pass


# ----------------------------------------------------------------------------------------- client
class JudgeClient:
    """Async judge client. Create inside (or before) the event loop that will use it.

    ``backends`` may supply ready backend instances (by name, or a sequence using their ``name``);
    config entries with the same name add rate limits / breakers / prices to them.
    """

    def __init__(self, config: Union[JudgeConfig, Mapping[str, Any], str, None] = None, *,
                 backends: Union[Mapping[str, JudgeBackend], Sequence[JudgeBackend], None] = None,
                 clock: Optional[Clock] = None, redact=None, cache: Optional[JudgeCache] = None,
                 telemetry: Optional[Telemetry] = None, seed: Optional[int] = None):
        self.config = JudgeConfig.coerce(config)
        self.clock = clock or Clock()
        cfg = self.config
        tcfg = cfg.telemetry
        self.telemetry = telemetry or Telemetry(tcfg.jsonl_path, tcfg.log_raw, tcfg.latency_window, tcfg.prefix)
        if redact is None and cfg.redact:
            redact = import_object(cfg.redact)
        self.redact = redact
        self.cache = cache
        if self.cache is None and cfg.cache.enabled and cfg.cache.path:
            self.cache = JudgeCache(cfg.cache.path, ttl_s=cfg.cache.ttl_s, lease_ttl_s=cfg.cache.lease_ttl_s,
                                    poll_s=cfg.cache.poll_s)
        self._rng = random.Random(seed)

        instances: Dict[str, JudgeBackend] = {}
        if isinstance(backends, Mapping):
            instances = dict(backends)
        elif backends is not None:
            instances = {b.name: b for b in backends}
        by_name = {b.display_name: b for b in cfg.backends}
        self.slots: Dict[str, _Slot] = {}
        for bc in cfg.backends:
            inst = instances.pop(bc.display_name, None)
            if inst is None:
                inst = self._build_backend(bc)
            self.slots[bc.display_name] = _Slot(inst, bc, self.clock, tcfg.use_litellm_costs)
        for name, inst in instances.items():  # instances without a config entry: default limits
            bc = by_name.get(name) or BackendConfig(type="instance", model=inst.model_id, name=name)
            self.slots[name] = _Slot(inst, bc, self.clock, tcfg.use_litellm_costs)
        if not self.slots:
            raise ValueError("judge config has no backends")
        self.global_breaker = CircuitBreaker("global", cfg.breaker.settings(), self.clock)

        self.halted: Optional[JudgeHalted] = None
        self._queue: Optional[_PriorityQueue] = None
        self._workers: List[asyncio.Task] = []
        self._items: Dict[str, _Item] = {}
        self._seq = itertools.count()
        self._flights: Dict[str, asyncio.Future] = {}
        self._started = False
        self._closed = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._spent = 0.0
        self._reserved = 0.0
        self._budget_cond: Optional[asyncio.Condition] = None
        self._budget_exhausted = False

    # ------------------------------------------------------------------ construction helpers
    @staticmethod
    def _build_backend(bc: BackendConfig) -> JudgeBackend:
        factory = get_backend_factory(bc.type)
        kwargs = bc.constructor_kwargs()
        try:
            return factory(**kwargs)
        except TypeError as e:
            raise ValueError(f"backend {bc.display_name!r} (type {bc.type}): bad options {sorted(kwargs)}: {e}") from e

    @classmethod
    def from_backends(cls, *backends: JudgeBackend, **config: Any) -> "JudgeClient":
        """Programmatic construction: ``JudgeClient.from_backends(b1, b2, retry={"max_attempts": 2})``.
        ``clock``, ``redact``, ``seed`` and :class:`JudgeCache` / :class:`Telemetry` instances go to the
        constructor; everything else is :class:`JudgeConfig` (``backend_configs`` maps a backend name
        to its rate_limit / breaker / prices / reasoning_budget settings)."""
        ctor = {k: config.pop(k) for k in ("clock", "redact", "seed") if k in config}
        for k, cls_ in (("cache", JudgeCache), ("telemetry", Telemetry)):  # instances, not config sections
            if isinstance(config.get(k), cls_):
                ctor[k] = config.pop(k)
        extra = config.pop("backend_configs", {}) or {}
        confs = []
        for b in backends:
            spec = dict(extra.get(b.name, {}))
            spec.update(type="instance", name=b.name, model=b.model_id)
            confs.append(spec)
        config["backends"] = confs
        return cls(JudgeConfig.from_dict(config), backends=list(backends), **ctor)

    def backend(self, name: str) -> JudgeBackend:
        return self.slots[name].backend

    def attach_colocated(self, name: str, engine: Any, awake: Optional[bool] = None) -> None:
        """Bind the trainer-provided engine to a ``colocated`` backend."""
        b = self.slots[name].backend
        if not hasattr(b, "attach"):
            raise TypeError(f"backend {name!r} is not a colocated backend")
        b.attach(engine, awake=awake)

    # ------------------------------------------------------------------ lifecycle
    async def start(self) -> "JudgeClient":
        if self._started:
            return self
        if self._closed:
            raise RuntimeError("judge client is closed")
        self._loop = asyncio.get_running_loop()
        self._queue = _PriorityQueue(self.config.queue.max_queue)
        self._budget_cond = asyncio.Condition()
        self._started = True
        for slot in self.slots.values():
            await slot.backend.start()
        self._workers = [asyncio.create_task(self._worker(), name=f"judge-worker-{i}")
                         for i in range(self.config.queue.max_inflight)]
        return self

    async def close(self) -> None:
        """Cancel outstanding work (resolved as ``cancelled``), stop workers, close backends."""
        if self._closed:
            return
        self._closed = True
        if self._queue is not None:
            for item in self._queue.close():
                self._finish(item, self._result(item, JudgeStatus.CANCELLED, error="client closed"))
            await self._queue.wait_producers_gone()  # woken producers resolve their requests as cancelled
        for item in list(self._items.values()):
            if item.task is not None and not item.task.done():
                item.cancel_reason = "client closed"
                item.task.cancel()
        for item in list(self._items.values()):
            if item.task is not None:
                await asyncio.wait({item.task}, timeout=5)
        for w in self._workers:
            w.cancel()
        for w in self._workers:
            try:
                await w
            except (asyncio.CancelledError, Exception):
                pass
        for item in list(self._items.values()):  # anything left (e.g. never picked up)
            self._finish(item, self._result(item, JudgeStatus.CANCELLED, error="client closed"))
        for slot in self.slots.values():
            try:
                await slot.backend.close()
            except Exception as e:  # pragma: no cover - best effort
                log.warning("closing backend %s failed: %r", slot.name, e)
        self.telemetry.close()
        if self.cache is not None:
            self.cache.close()

    async def __aenter__(self) -> "JudgeClient":
        return await self.start()

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()

    # ------------------------------------------------------------------ submission
    def _prepare(self, request: JudgeRequest) -> JudgeRequest:
        if self.redact is not None:
            request = self.redact(request)
            if not isinstance(request, JudgeRequest):
                raise TypeError("redact hook must return a JudgeRequest")
        return request

    async def submit(self, request: JudgeRequest) -> asyncio.Future:
        """Enqueue a request; waits while the queue is full (backpressure). Returns a future that
        resolves to a :class:`JudgeResult` (or raises :class:`JudgeHalted`)."""
        if not self._started:
            await self.start()
        if self.halted is not None:
            raise self.halted
        if self._closed:
            raise RuntimeError("judge client is closed")
        request = self._prepare(request)
        if request.request_id in self._items:
            raise ValueError(f"duplicate in-flight request_id {request.request_id!r}")
        now = self.clock.now()
        deadline_s = request.deadline_s if request.deadline_s is not None else self.config.defaults.deadline_s
        item = _Item(request=request, future=self._loop.create_future(), submitted=now,
                     deadline=(now + deadline_s) if deadline_s else None, seq=next(self._seq))
        self._items[request.request_id] = item
        item.future.add_done_callback(lambda f, it=item: self._on_future_done(it))
        try:
            await self._queue.put(item)
        except _QueueClosed:  # closed or halted while waiting for space: fail fast, never strand the caller
            if self.halted is not None:
                self._fail(item, self.halted)
                raise self.halted
            self._finish(item, self._result(item, JudgeStatus.CANCELLED, error="client closed"))
            return item.future
        except BaseException:
            self._items.pop(request.request_id, None)
            raise
        return item.future

    async def judge(self, request: JudgeRequest) -> JudgeResult:
        return await (await self.submit(request))

    async def judge_many(self, requests: Iterable[JudgeRequest]) -> List[JudgeResult]:
        futs = [await self.submit(r) for r in requests]
        return list(await asyncio.gather(*futs))

    def cancel(self, request_id: str, reason: str = "cancelled by caller") -> bool:
        """Cancel a queued or running request; it resolves with status ``cancelled``."""
        item = self._items.get(request_id)
        if item is None or item.finished:
            return False
        item.cancel_reason = reason
        if item.task is not None and not item.task.done():
            item.task.cancel()
        elif not item.started:
            self._finish(item, self._result(item, JudgeStatus.CANCELLED, error=reason))
        return True

    def _on_future_done(self, item: _Item) -> None:
        if item.future.cancelled() and not item.finished:
            item.cancel_reason = item.cancel_reason or "future cancelled"
            if item.task is not None and not item.task.done():
                item.task.cancel()
            elif not item.started:
                self._finish(item, self._result(item, JudgeStatus.CANCELLED, error=item.cancel_reason))

    @property
    def queue_depth(self) -> int:
        return len(self._queue) if self._queue is not None else 0

    @property
    def inflight(self) -> int:
        return sum(1 for it in self._items.values() if it.started and not it.finished)

    # ------------------------------------------------------------------ resolution
    def _result(self, item: _Item, status: JudgeStatus, **kw: Any) -> JudgeResult:
        r = item.request
        return JudgeResult(request_id=r.request_id, status=status, tags=dict(r.tags), versions=r.versions, **kw)

    def _finish(self, item: _Item, result: JudgeResult) -> None:
        if item.finished:
            return
        item.finished = True
        if item.future.cancelled() and result.status != JudgeStatus.CANCELLED:
            # the caller cancelled after the work completed but before it was delivered: account for
            # what the caller observed (the cost already spent stays on the record)
            result.error = f"future cancelled by caller; discarded {result.status.value} result"
            result.status = JudgeStatus.CANCELLED
        now = self.clock.now()
        result.latency_s = now - item.submitted
        result.queue_s = (item.first_dispatch or now) - item.submitted
        self._items.pop(item.request.request_id, None)
        self.telemetry.record(item.request, result)
        if not (result.cache_hit or result.coalesced):
            self.global_breaker.record(result.status)
            if self.global_breaker.state == "open" and self.global_breaker.settings.policy == "stop":
                self._halt(JudgeHalted(f"judge halted: {self.global_breaker.reason}"))
        elif item.global_probe:
            self.global_breaker.release_probe()
        if not item.future.done():
            item.future.set_result(result)

    def _fail(self, item: _Item, exc: JudgeHalted) -> None:
        if item.finished:
            return
        item.finished = True
        now = self.clock.now()
        result = self._result(item, exc.status, error=exc.reason, latency_s=now - item.submitted)
        self._items.pop(item.request.request_id, None)
        self.telemetry.record(item.request, result)
        if not item.future.done():
            item.future.set_exception(exc)
            item.future.exception()  # mark retrieved: callers that ignore it do not trigger loop warnings

    def _halt(self, exc: JudgeHalted) -> None:
        if self.halted is not None:
            return
        self.halted = exc
        self.telemetry.event("halted")
        log.error("%s", exc.reason)
        if self._queue is not None:
            for item in self._queue.close():
                self._fail(item, exc)
        for item in list(self._items.values()):
            if item.task is not None and not item.task.done() and item.task is not asyncio.current_task():
                item.task.cancel()

    # ------------------------------------------------------------------ worker
    async def _worker(self) -> None:
        while True:
            item = await self._queue.get()
            if item.finished:
                continue
            if self.halted is not None:
                self._fail(item, self.halted)
                continue
            item.started = True
            item.task = asyncio.create_task(self._process(item))
            try:
                await asyncio.wait({item.task})
            except asyncio.CancelledError:
                item.task.cancel()
                raise
            t = item.task
            if t.cancelled():
                if self.halted is not None:
                    self._fail(item, self.halted)
                else:
                    self._finish(item, self._result(item, JudgeStatus.CANCELLED,
                                                    error=item.cancel_reason or "cancelled"))
            elif t.exception() is not None:
                e = t.exception()
                if isinstance(e, JudgeHalted):
                    self._fail(item, e)
                else:  # a bug in the control plane: never lose the request
                    log.exception("judge request %s failed internally", item.request.request_id, exc_info=e)
                    self._finish(item, self._result(item, JudgeStatus.BACKEND_ERROR, error=f"internal: {e!r}"[:500]))
            else:
                self._finish(item, t.result())

    def _remaining(self, item: _Item) -> Optional[float]:
        return None if item.deadline is None else item.deadline - self.clock.now()

    async def _gate_global(self, item: _Item) -> Optional[JudgeResult]:
        """Wait while the global breaker is open (pause). Returns a result if the request must end."""
        gb = self.global_breaker
        while True:
            if self.halted is not None:
                raise self.halted
            if gb.state == "closed":
                return None
            if gb.allow():
                item.global_probe = gb.state == "half_open"
                return None
            wait = gb.wait_time()
            if wait is None:
                return self._result(item, JudgeStatus.CIRCUIT_OPEN, error=gb.reason)
            rem = self._remaining(item)
            if rem is not None and rem <= 0:
                return self._result(item, JudgeStatus.CIRCUIT_OPEN, error=f"deadline passed while paused: {gb.reason}")
            await self.clock.sleep(max(0.001, min(wait, rem) if rem is not None else wait))

    async def _process(self, item: _Item) -> JudgeResult:
        req = item.request
        rem = self._remaining(item)
        if rem is not None and rem <= 0:
            return self._result(item, JudgeStatus.TIMEOUT, error="deadline passed in queue")
        gated = await self._gate_global(item)
        if gated is not None:
            return gated
        pool = [self.slots[n] for n in self.config.pool_members(req.pool)] if self.config.backends else \
            list(self.slots.values())
        keys = [(slot, self._key(req, slot)) for slot in pool]
        use_cache = self.cache is not None and req.use_cache
        if use_cache:
            hit = await self._cache_lookup(item, keys)
            if hit is not None:
                return hit
        primary = keys[0][1]
        while req.use_cache and primary in self._flights:  # in-process single-flight
            leader = self._flights[primary]
            await asyncio.wait({leader})  # raises only if THIS request is cancelled
            if leader.cancelled():
                continue  # the leader was cancelled: try to lead
            return self._coalesced(item, leader.result())
        flight = self._loop.create_future()
        if req.use_cache:
            self._flights[primary] = flight
        token: Optional[int] = None          # fencing token of our cross-process lease
        lease_state = {"lost": False}
        renew: Optional[asyncio.Task] = None
        result: Optional[JudgeResult] = None
        try:
            if use_cache and self.config.cache.cross_process:
                while True:
                    token = await self.cache.aacquire_lease(primary)
                    if token is not None:
                        break
                    other = await self.cache.wait_for_other(primary, item.deadline, self.clock.now)
                    if other is not None:
                        result = self._from_cache(item, other, keys[0][0])
                        return result
                    rem = self._remaining(item)
                    if rem is not None and rem <= 0:
                        result = self._result(item, JudgeStatus.TIMEOUT, error="deadline passed waiting for "
                                              "another process computing the same request")
                        return result
                hit = await self._cache_lookup(item, keys)
                if hit is not None:
                    result = hit
                    return result
                renew = asyncio.create_task(self._renew_lease(primary, token, lease_state))
            result = await self._execute(item, pool, dict((s.name, k) for s, k in keys))
            if result.ok and use_cache:
                key = dict((s.name, k) for s, k in keys).get(result.backend, primary)
                if token is None:
                    await self.cache.aput(key, self._cacheable(result), raw=result.raw, model=result.model)
                elif not lease_state["lost"]:
                    written = await self.cache.aput(key, self._cacheable(result), raw=result.raw,
                                                    model=result.model, token=token, lease_key=primary)
                    if not written:
                        self._lease_lost(primary, lease_state)
            return result
        finally:
            if renew is not None:
                renew.cancel()
            if token is not None and not lease_state["lost"]:
                try:
                    await asyncio.shield(self.cache.arelease_lease(primary, token))
                except Exception:  # pragma: no cover - lease expires anyway
                    pass
            if self._flights.get(primary) is flight:
                del self._flights[primary]
            if not flight.done():
                if result is not None:
                    flight.set_result(result)
                else:
                    flight.cancel()

    def _lease_lost(self, key: str, state: Dict[str, bool]) -> None:
        if not state["lost"]:
            state["lost"] = True
            self.telemetry.event("cache_lease_lost")
            log.warning("judge cache lease on %s was lost (holder paused past its TTL?); "
                        "the result is returned but not written to the cache", key[:12])

    async def _renew_lease(self, key: str, token: int, state: Dict[str, bool]) -> None:
        while True:
            await asyncio.sleep(self.cache.renew_interval_s)
            if not await self.cache.arenew_lease(key, token):
                self._lease_lost(key, state)
                return

    # ------------------------------------------------------------------ cache helpers
    def _effective(self, req: JudgeRequest) -> Tuple[Decoding, ReasoningSpec]:
        d = self.config.defaults
        dec = req.decoding.merged(Decoding(max_tokens=d.max_tokens, temperature=d.temperature, top_p=d.top_p))
        spec = req.reasoning if req.reasoning.mode != "default" else ReasoningSpec.parse(d.reasoning)
        return dec, spec

    def _structured_mode(self, slot: _Slot) -> str:
        mode = slot.backend.capabilities.structured
        return "none" if mode in (None, "auto") else mode

    def _key(self, req: JudgeRequest, slot: _Slot) -> str:
        dec, spec = self._effective(req)
        return cache_key(messages=req.messages, output_schema=req.output_schema, prompt_version=req.prompt_version,
                         schema_version=req.schema_version, parser_version=req.parser_version, model=slot.model_id,
                         revision=slot.backend.revision, decoding=dec.to_dict(), reasoning=spec.to_dict(),
                         structured=self._structured_mode(slot), sample_index=req.sample_index,
                         extra_params=req.extra_params)

    @staticmethod
    def _cacheable(result: JudgeResult) -> Dict[str, Any]:
        d = result.to_dict(include_raw=False)
        for k in ("attempt_log", "tags", "latency_s", "queue_s", "request_id", "cache_hit", "coalesced"):
            d.pop(k, None)
        return d

    async def _cache_lookup(self, item: _Item, keys) -> Optional[JudgeResult]:
        for slot, key in keys:
            rec = await self.cache.aget(key)
            if rec is not None:
                return self._from_cache(item, rec, slot, key)
        return None

    def _from_cache(self, item: _Item, rec: Dict[str, Any], slot: _Slot, key: Optional[str] = None) -> JudgeResult:
        parsed = rec.get("parsed")
        schema = item.request.output_schema
        if schema is not None:
            value, errs = validate(schema, parsed)
            if not errs:
                parsed = value
        return self._result(item, JudgeStatus.OK, parsed=parsed, text=rec.get("text"),
                            reasoning_text=rec.get("reasoning_text"), finish_reason=rec.get("finish_reason"),
                            backend=rec.get("backend") or slot.name, model=rec.get("model"), cache_hit=True,
                            cache_key=key or rec.get("cache_key"), lenient_parse=bool(rec.get("lenient_parse")),
                            raw=rec.get("raw"))

    def _coalesced(self, item: _Item, led: JudgeResult) -> JudgeResult:
        return self._result(item, led.status, parsed=led.parsed, text=led.text, reasoning_text=led.reasoning_text,
                            finish_reason=led.finish_reason, backend=led.backend, model=led.model,
                            error=led.error, coalesced=True, cache_key=led.cache_key,
                            lenient_parse=led.lenient_parse, raw=led.raw)

    # ------------------------------------------------------------------ attempt loop
    def _pick(self, pool: List[_Slot], tried: Dict[str, int], disabled: set, per_backend: int
              ) -> Tuple[Optional[_Slot], Optional[float], bool]:
        """First usable backend. Otherwise ``(None, wait, blocked)``: ``blocked`` means some backend with
        attempts left is held back by its breaker, ``wait`` is when one may reopen (``None`` = never)."""
        best_wait: Optional[float] = None
        blocked = False
        for slot in pool:
            if slot.name in disabled or tried.get(slot.name, 0) >= per_backend:
                continue
            if slot.breaker.allow():
                return slot, 0.0, False
            blocked = True
            w = slot.breaker.wait_time()
            if w is not None:
                best_wait = w if best_wait is None else min(best_wait, w)
        return None, best_wait, blocked

    def _backoff(self, n: int, retry_after: Optional[float]) -> float:
        rc = self.config.retry
        delay = min(rc.max_delay_s, rc.base_delay_s * (2 ** max(0, n - 1)))
        delay *= 1.0 + rc.jitter * self._rng.random()
        if retry_after:
            delay = max(delay, min(retry_after, rc.max_retry_after_s))
        return delay

    async def _execute(self, item: _Item, pool: List[_Slot], keys: Dict[str, str]) -> JudgeResult:
        req = item.request
        rc = self.config.retry
        retry_on = {JudgeStatus(s) for s in rc.retry_on}
        per_backend = rc.attempts_per_backend or rc.max_attempts
        dec, spec = self._effective(req)
        answer_tokens = int(dec.max_tokens)
        messages = list(req.messages)
        tried: Dict[str, int] = {}
        disabled: set = set()
        log_: List[Dict[str, Any]] = []
        usage = Usage()
        cost, cost_known = 0.0, True
        attempts = reasks = failovers = 0
        last: Optional[_Outcome] = None
        prev_slot: Optional[str] = None

        def finish(status: JudgeStatus, outcome: Optional[_Outcome], error: Optional[str] = None) -> JudgeResult:
            resp = outcome.response if outcome else None
            slot = outcome.slot if outcome else None
            return self._result(
                item, status, parsed=outcome.parsed if outcome and status == JudgeStatus.OK else None,
                text=outcome.answer_text if outcome else None,
                reasoning_text=resp.reasoning_text if resp else None, finish_reason=resp.finish_reason if resp else None,
                usage=usage, cost_usd=cost, cost_known=cost_known, attempts=attempts,
                backend=slot.name if slot else None, model=(resp.model if resp and resp.model else
                                                            slot.model_id if slot else None),
                error=error if error is not None else (outcome.error if outcome else None),
                lenient_parse=bool(outcome and outcome.lenient and status == JudgeStatus.OK), reasks=reasks,
                failovers=failovers, cache_key=keys.get(slot.name) if slot else None, attempt_log=log_,
                raw=resp.raw if resp else None)

        while attempts < rc.max_attempts:
            slot, wait, blocked = self._pick(pool, tried, disabled, per_backend)
            if slot is None:
                if not blocked:
                    break  # every backend exhausted its attempts or was disabled
                if wait is None:
                    reasons = "; ".join(s.breaker.reason for s in pool if s.breaker.reason)
                    return finish(JudgeStatus.CIRCUIT_OPEN, last, error=f"no backend available: {reasons}")
                rem = self._remaining(item)
                if rem is not None and rem <= wait:
                    return finish(JudgeStatus.CIRCUIT_OPEN, last, error="deadline passed while backend breakers open")
                await self.clock.sleep(max(0.001, wait))
                continue
            if prev_slot is not None and slot.name != prev_slot:
                failovers += 1
            prev_slot = slot.name
            outcome = await self._attempt(item, slot, messages, answer_tokens, spec, dec, attempts + 1)
            if outcome.failover and not outcome.dispatched:
                slot.breaker.release_probe()  # its breaker is now permanently open: _pick skips it
                last = last or outcome
                continue
            if outcome.status == JudgeStatus.BUDGET_EXCEEDED or not outcome.dispatched:
                slot.breaker.release_probe()
                last = outcome
                return finish(outcome.status, outcome if outcome.response else last, error=outcome.error)
            attempts += 1
            tried[slot.name] = tried.get(slot.name, 0) + 1
            usage += outcome.usage
            cost += outcome.cost
            cost_known = cost_known and outcome.cost_known
            slot.breaker.record(outcome.status)  # cost was added atomically with the reservation settle
            log_.append({"attempt": attempts, "backend": slot.name, "status": outcome.status.value,
                         "latency_s": round(outcome.latency_s, 4), "error": outcome.error,
                         "finish_reason": outcome.response.finish_reason if outcome.response else None})
            last = outcome
            st = outcome.status
            if st == JudgeStatus.OK:
                return finish(st, outcome)
            if st == JudgeStatus.SEMANTIC_REJECT and self.config.structured.reask and outcome.correction:
                reasks += 1
                prev = (outcome.answer_text or "")[: self.config.structured.reask_max_chars]
                messages = list(req.messages) + [{"role": "assistant", "content": prev},
                                                 {"role": "user", "content": outcome.correction}]
            if st == JudgeStatus.TRUNCATED and rc.truncation_growth > 1.0:
                grown = int(answer_tokens * rc.truncation_growth)
                answer_tokens = min(grown, rc.max_tokens_cap) if rc.max_tokens_cap else grown
            if st in slot.breaker.settings.fatal_statuses or outcome.retryable is False:
                disabled.add(slot.name)  # fail over without retrying this backend
                continue
            if st not in retry_on:
                break
            if st == JudgeStatus.RATE_LIMITED and outcome.retry_after:
                slot.limiter.pause_for(min(outcome.retry_after, rc.max_retry_after_s))
            nxt, _ = self._peek(pool, tried, disabled, per_backend)
            if nxt is not None and nxt.name == slot.name and attempts < rc.max_attempts:
                delay = self._backoff(tried[slot.name], outcome.retry_after)
                rem = self._remaining(item)
                if rem is not None and rem <= delay:
                    break
                await self.clock.sleep(delay)
        if last is None:
            return finish(JudgeStatus.CIRCUIT_OPEN, None, error="no backend could be tried")
        return finish(last.status, last)

    def _peek(self, pool, tried, disabled, per_backend) -> Tuple[Optional[_Slot], None]:
        for slot in pool:
            if slot.name in disabled or tried.get(slot.name, 0) >= per_backend:
                continue
            if slot.breaker.state == "open" and slot.breaker.wait_time() != 0.0:
                continue
            return slot, None
        return None, None

    # ------------------------------------------------------------------ budget
    async def _reserve(self, item: _Item, slot: _Slot, est_prompt: int, max_tokens: int) -> Tuple[Optional[float], str]:
        """Reserve the worst-case cost of one attempt. Returns ``(reservation, error)``; ``None`` = refused."""
        budget = self.config.budget_usd
        if budget is None:
            return 0.0, ""
        est = slot.costs.estimate(slot.model_id, est_prompt, max_tokens) or 0.0
        while True:
            if self._budget_exhausted or self._spent + est > budget or self._spent >= budget:
                self._budget_exhausted = True
                msg = (f"budget ${budget:.4f} reached (spent ${self._spent:.4f}, "
                       f"next attempt up to ${est:.4f})")
                self.telemetry.event("budget_exhausted")
                if self.config.budget_policy == "stop":
                    exc = JudgeHalted(f"judge halted: {msg}", JudgeStatus.BUDGET_EXCEEDED)
                    self._halt(exc)
                    raise exc
                return None, msg
            if self._spent + self._reserved + est <= budget:
                self._reserved += est
                return est, ""
            # in-flight reservations may settle below their worst case: wait for them
            rem = self._remaining(item)
            async with self._budget_cond:
                try:
                    await asyncio.wait_for(self._budget_cond.wait(), None if rem is None else max(0.0, rem))
                except asyncio.TimeoutError:
                    return None, "deadline passed waiting for budget reservations to settle"

    async def _settle(self, reservation: float, actual: float) -> None:
        self._reserved = max(0.0, self._reserved - reservation)
        self._spent += actual
        if self._budget_cond is not None:
            async with self._budget_cond:
                self._budget_cond.notify_all()

    async def _reserve_backend(self, item: _Item, slot: _Slot, est_prompt: int, max_tokens: int
                               ) -> Tuple[Optional[float], str, bool]:
        """Atomically reserve one attempt's worst-case cost against the backend's hard cost cap.

        Returns ``(reservation, reason, exhausted)``; ``reservation is None`` means refused, and
        ``exhausted`` that the cap is reached (the backend's breaker is tripped for good, so the pool
        fails over). The worst case is the priced estimate, else the largest cost observed for one
        call; while neither is known, calls are admitted one at a time so the first observed cost
        bounds the rest.
        """
        br = slot.breaker
        cap = br.settings.cost_budget_usd if br.settings.enabled else None
        if cap is None:
            return 0.0, "", False
        while True:
            priced = slot.costs.estimate(slot.model_id, est_prompt, max_tokens)
            est = priced if priced is not None else slot.max_call_cost
            spent = br.cost_usd
            if br.budget_tripped or spent >= cap or (est is not None and spent + est > cap):
                msg = (f"{slot.name}: cost cap ${cap:.4f} reached (spent ${spent:.4f}, next call up to "
                       f"${(est or 0.0):.4f})")
                if not br.budget_tripped:
                    br.trip(msg, budget=True)
                    self.telemetry.event("backend_budget_exhausted")
                return None, msg, True
            if est is None:
                if slot.reservations == 0:
                    slot.reservations += 1
                    return 0.0, "", False
            elif spent + slot.reserved_usd + est <= cap:
                slot.reserved_usd += est
                slot.reservations += 1
                return est, "", False
            rem = self._remaining(item)  # other calls hold the headroom: wait for them to settle
            async with slot.cost_cond:
                try:
                    await asyncio.wait_for(slot.cost_cond.wait(), None if rem is None else max(0.0, rem))
                except asyncio.TimeoutError:
                    return None, f"{slot.name}: deadline passed waiting for cost reservations to settle", False

    async def _settle_backend(self, slot: _Slot, reservation: Optional[float], actual: float) -> None:
        if reservation is None or slot.breaker.settings.cost_budget_usd is None or not slot.breaker.settings.enabled:
            if actual:
                slot.breaker.add_cost(actual)
            return
        slot.reserved_usd = max(0.0, slot.reserved_usd - reservation)
        slot.reservations = max(0, slot.reservations - 1)
        if actual > 0:
            slot.max_call_cost = max(slot.max_call_cost or 0.0, actual)
        slot.breaker.add_cost(actual)
        async with slot.cost_cond:
            slot.cost_cond.notify_all()

    @property
    def spent_usd(self) -> float:
        return self._spent

    # ------------------------------------------------------------------ one attempt
    async def _attempt(self, item: _Item, slot: _Slot, messages: List[Dict[str, Any]], answer_tokens: int,
                       spec: ReasoningSpec, dec: Decoding, attempt_no: int) -> _Outcome:
        req = item.request
        budget = reasoning_budget(spec, slot.cfg.reasoning_budget)
        max_sent = answer_tokens + budget
        schema_dict = schema_to_dict(req.output_schema)
        mode = self._structured_mode(slot)
        instr = self.config.structured.schema_instruction
        send_msgs = messages
        if schema_dict is not None and (instr == "always" or (instr == "auto" and mode == "none")):
            send_msgs = with_schema_instruction(messages, schema_dict)
        timeout = req.timeout_s or slot.cfg.timeout_s or self.config.defaults.timeout_s
        rem = self._remaining(item)
        if rem is not None:
            if rem <= 0:
                return _Outcome(JudgeStatus.TIMEOUT, slot, error="request deadline passed")
            timeout = min(timeout, rem) if timeout else rem
        call = BackendCall(messages=send_msgs, max_tokens=max_sent, temperature=dec.temperature, top_p=dec.top_p,
                           stop=list(dec.stop) if dec.stop else None, seed=dec.seed, reasoning=spec,
                           reasoning_budget=budget, json_schema=schema_dict if mode != "none" else None,
                           schema_name=schema_name(req.output_schema), timeout_s=timeout,
                           request_id=req.request_id, attempt=attempt_no, extra_params=dict(req.extra_params),
                           tags=dict(req.tags))
        raw_est = estimate_prompt_tokens(send_msgs)
        est_prompt = int(raw_est * slot.prompt_ratio) + 1
        bres, bwhy, exhausted = await self._reserve_backend(item, slot, est_prompt, max_sent)
        if bres is None:
            if exhausted:
                return _Outcome(JudgeStatus.CIRCUIT_OPEN, slot, error=bwhy, failover=True)
            return _Outcome(JudgeStatus.TIMEOUT, slot, error=bwhy)
        try:
            reservation, why = await self._reserve(item, slot, est_prompt, max_sent)
        except BaseException:
            await asyncio.shield(self._settle_backend(slot, bres, 0.0))
            raise
        if reservation is None:
            await self._settle_backend(slot, bres, 0.0)
            status = JudgeStatus.BUDGET_EXCEEDED if "budget" in why and "deadline" not in why else JudgeStatus.TIMEOUT
            return _Outcome(status, slot, error=why)
        adm = await slot.limiter.acquire(est_prompt + max_sent, timeout=self._remaining(item))
        if adm is None:
            await self._settle(reservation, 0.0)
            await self._settle_backend(slot, bres, 0.0)
            return _Outcome(JudgeStatus.TIMEOUT, slot, error="deadline passed waiting for rate limit")
        if item.first_dispatch is None:
            item.first_dispatch = self.clock.now()
        t0 = self.clock.now()
        out = _Outcome(JudgeStatus.OK, slot, dispatched=True)
        resp: Optional[BackendResponse] = None
        try:
            if timeout:
                resp = await asyncio.wait_for(slot.backend.generate(call), timeout)
            else:
                resp = await slot.backend.generate(call)
        except asyncio.TimeoutError:
            out.status, out.error = JudgeStatus.TIMEOUT, f"no response within {timeout:.1f}s"
        except asyncio.CancelledError:
            slot.limiter.release(adm, None)
            await asyncio.shield(self._settle(reservation, 0.0))
            await asyncio.shield(self._settle_backend(slot, bres, 0.0))
            raise
        except Exception as e:  # noqa: BLE001 - typed below
            err = classify_exception(e)
            out.status, out.error, out.retry_after, out.retryable = err.status, err.message, err.retry_after, err.retryable
        out.latency_s = self.clock.now() - t0
        if resp is not None:
            out.response = resp
            out.usage = resp.usage
            slot.calibrate(raw_est, resp.usage.prompt_tokens)
            out.cost, out.cost_known = slot.costs.cost(slot.model_id, resp.usage, resp.cost_usd)
            self._interpret(req, resp, out)
        slot.limiter.release(adm, float(out.usage.total_tokens) if resp is not None else 0.0)
        await self._settle(reservation, out.cost)
        await self._settle_backend(slot, bres, out.cost)
        return out

    def _interpret(self, req: JudgeRequest, resp: BackendResponse, out: _Outcome) -> None:
        text = resp.text
        if text is not None and resp.reasoning_text is None:
            text, think = split_reasoning(text)
            resp.reasoning_text = think
        text = text or ""
        out.answer_text = text
        if resp.refusal or resp.finish_reason == "content_filter":
            out.status, out.error = JudgeStatus.REFUSAL, (resp.refusal or "content filter")[:300]
            return
        if not text.strip():
            if resp.finish_reason == "length":
                out.status, out.error = JudgeStatus.TRUNCATED, "token cap reached before any answer (reasoning?)"
            else:
                out.status, out.error = JudgeStatus.EMPTY_CONTENT, "empty answer"
            return
        schema = req.output_schema
        if schema is None:
            if resp.finish_reason == "length":
                out.status, out.error = JudgeStatus.TRUNCATED, "token cap reached (free-text answer)"
                return
            value: Any = text
            ok = True
        else:
            ok, value = parse_strict(text)
            if not ok and self.config.structured.lenient_fallback:
                value = extract_json(text)
                if value is not None:
                    ok, out.lenient = True, True
        if not ok:
            truncated = resp.finish_reason == "length"
            out.status = JudgeStatus.TRUNCATED if truncated else JudgeStatus.MALFORMED
            out.error = f"no parsable JSON (finish={resp.finish_reason}, chars={len(text)})"
            return
        parsed, errors = validate(schema, value)
        if not errors and req.validator is not None:
            try:
                msg = req.validator(parsed)
            except Exception as e:  # noqa: BLE001 - a crashing validator is a reject, not a crash
                msg = f"validator error: {e!r}"[:300]
            if msg:
                out.status, out.errors, out.correction = JudgeStatus.SEMANTIC_REJECT, [msg], msg
                out.error = msg[:300]
                return
        if errors:
            if resp.finish_reason == "length":
                out.status, out.error = JudgeStatus.TRUNCATED, "; ".join(errors)[:300]
                return
            out.status, out.errors = JudgeStatus.SEMANTIC_REJECT, errors
            out.correction = correction_message(errors)
            out.error = "; ".join(errors)[:300]
            return
        out.status, out.parsed = JudgeStatus.OK, parsed

    # ------------------------------------------------------------------ preflight / estimates / metrics
    async def preflight(self, strict: bool = False, timeout_s: float = 120.0,
                        backends: Optional[Sequence[str]] = None) -> PreflightReport:
        """Send one real request per backend and report auth / model / format / budget problems.

        Bypasses the queue, cache and breakers (but not rate limits, budgets or telemetry). With
        ``strict=True`` raises :class:`PreflightError` when any backend fails.
        """
        if not self._started:
            await self.start()
        entries, problems, warnings = [], [], []
        budget = self.config.budget_usd
        for name in backends or list(self.slots):
            slot = self.slots[name]
            entry = {"backend": name, "model": slot.model_id, "ok": False, "status": None, "error": None,
                     "latency_s": 0.0, "cost_usd": 0.0, "cost_known": True,
                     "structured": self._structured_mode(slot),
                     "reasoning_style": slot.backend.capabilities.reasoning_style}
            cfg_problems = list(getattr(slot.backend, "config_problems", lambda: [])())
            if cfg_problems:
                entry.update(status=JudgeStatus.AUTH.value, error=cfg_problems[0])
                problems.append(f"{name}: {cfg_problems[0]}")
                entries.append(entry)
                continue
            req = JudgeRequest(messages=[{"role": "user", "content": PREFLIGHT_PROMPT}], output_schema=PREFLIGHT_SCHEMA,
                               decoding={"max_tokens": 64}, tags={"purpose": "preflight"}, timeout_s=timeout_s,
                               use_cache=False)
            item = _Item(request=req, future=self._loop.create_future(), submitted=self.clock.now(),
                         deadline=self.clock.now() + timeout_s, seq=next(self._seq))
            dec, spec = self._effective(req)
            try:
                out = await self._attempt(item, slot, list(req.messages), int(dec.max_tokens), spec, dec, 1)
            except JudgeHalted as e:
                out = _Outcome(e.status, slot, error=e.reason)
            entry.update(status=out.status.value, error=out.error, latency_s=out.latency_s, cost_usd=out.cost,
                         cost_known=out.cost_known, ok=out.status == JudgeStatus.OK,
                         text=(out.answer_text or "")[:200])
            res = self._result(item, out.status, parsed=out.parsed, text=out.answer_text, usage=out.usage,
                               cost_usd=out.cost, cost_known=out.cost_known, attempts=1, backend=name,
                               model=slot.model_id, error=out.error, latency_s=out.latency_s,
                               attempt_log=[{"attempt": 1, "backend": name, "status": out.status.value}])
            self.telemetry.record(req, res)
            if out.status == JudgeStatus.OK:
                pass
            elif out.status in (JudgeStatus.MALFORMED, JudgeStatus.SEMANTIC_REJECT, JudgeStatus.TRUNCATED,
                                JudgeStatus.EMPTY_CONTENT):
                problems.append(f"{name}: reachable but the answer failed parsing ({out.status.value}: {out.error}); "
                                f"check structured / reasoning / max_tokens settings")
            else:
                problems.append(f"{name}: {out.status.value}: {out.error}")
            if not out.cost_known:
                warnings.append(f"{name}: cost unknown for model {slot.model_id!r}; set prices to enforce budgets")
            entries.append(entry)
        if budget is not None:
            if self._spent >= budget:
                problems.append(f"budget ${budget} already spent (${self._spent:.4f})")
            if any(not e["cost_known"] for e in entries):
                warnings.append("budget_usd is set but some backend costs are unknown: those calls count as $0")
        self.telemetry.event("preflight_problems", len(problems))
        report = PreflightReport(entries, problems, warnings)
        if strict:
            report.raise_if_failed()
        return report

    def estimate_cost(self, requests: Iterable[JudgeRequest], pool: Optional[str] = None) -> Dict[str, Any]:
        """Worst-case price of ``requests`` on the first backend of the pool, before launch.

        Useful to price prefix-style judges (one call per step over growing prefixes), whose prompt
        volume grows quadratically with trajectory length.
        """
        slot = self.slots[self.config.pool_members(pool)[0]] if self.config.backends else next(iter(self.slots.values()))
        n = prompt = completion = 0
        for r in requests:
            dec, spec = self._effective(r)
            n += 1
            prompt += estimate_prompt_tokens(r.messages)
            completion += int(dec.max_tokens) + reasoning_budget(spec, slot.cfg.reasoning_budget)
        usd = slot.costs.estimate(slot.model_id, prompt, completion)
        return {"requests": n, "prompt_tokens": prompt, "max_completion_tokens": completion,
                "usd_upper_bound": usd, "price_known": usd is not None, "backend": slot.name}

    def metrics(self, window: bool = False, reset_window: bool = False, detail: bool = True) -> Dict[str, float]:
        """Flat metrics for trainer logging (see :meth:`Telemetry.metrics`), plus live gauges."""
        m = self.telemetry.metrics(window=window, reset_window=reset_window, detail=detail)
        p = self.telemetry.prefix
        m[f"{p}/queue_depth"] = self.queue_depth
        m[f"{p}/inflight"] = self.inflight
        m[f"{p}/spent_usd"] = self._spent
        if self.config.budget_usd is not None:
            m[f"{p}/budget_usd"] = self.config.budget_usd
            m[f"{p}/budget_remaining_usd"] = max(0.0, self.config.budget_usd - self._spent)
        m[f"{p}/breaker_open"] = int(self.global_breaker.state != "closed")
        m[f"{p}/halted"] = int(self.halted is not None)
        for s in self.slots.values():
            m[f"{p}/backend/{s.name}/breaker_open"] = int(s.breaker.state != "closed")
            m[f"{p}/backend/{s.name}/rate_wait_s"] = s.limiter.waited_s
        if self.cache is not None:
            st = self.cache.stats()
            m[f"{p}/cache/entries_written"] = st["writes"]
            m[f"{p}/cache/lease_waits"] = st["lease_waits"]
        return m

    def session(self, **kwargs: Any):
        from judgerl.judge.session import JudgeSession

        return JudgeSession(self, **kwargs)
