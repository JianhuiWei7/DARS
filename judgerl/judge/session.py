"""Watermark sessions and the synchronous facade.

A trainer submits judge requests as trajectories finish and, before the optimizer step, waits for a
reward watermark::

    session = client.session()
    for traj in finished_trajectories():          # overlaps with rollout
        futs.append(session.submit(make_request(traj)))
    report = await session.barrier(watermark=1.0, timeout=600)
    # report.results: request_id -> JudgeResult (ok or a typed failure; cancelled if unresolved)

``barrier`` returns once ``ceil(watermark * n)`` of the requests submitted since the previous barrier
have resolved (successfully or not), or at ``timeout``. Requests still unresolved at that point are
cancelled (``cancel_unresolved=True``) and reported, so no label from this batch can leak into the
next one.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import math
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Union

from judgerl.judge.types import JudgeHalted, JudgeRequest, JudgeResult


@dataclass
class BarrierReport:
    n_submitted: int
    n_resolved: int                    # resolved before the barrier returned (before cancellation)
    watermark: float
    watermark_reached: bool
    timed_out: bool
    elapsed_s: float
    by_status: Dict[str, int] = field(default_factory=dict)
    cancelled_ids: List[str] = field(default_factory=list)
    results: Dict[str, JudgeResult] = field(default_factory=dict)

    @property
    def resolved_fraction(self) -> float:
        return self.n_resolved / self.n_submitted if self.n_submitted else 1.0

    def metrics(self, prefix: str = "judge/barrier") -> Dict[str, float]:
        m = {f"{prefix}/submitted": self.n_submitted, f"{prefix}/resolved": self.n_resolved,
             f"{prefix}/resolved_fraction": self.resolved_fraction, f"{prefix}/timed_out": int(self.timed_out),
             f"{prefix}/cancelled": len(self.cancelled_ids), f"{prefix}/wait_s": self.elapsed_s}
        for k, v in self.by_status.items():
            m[f"{prefix}/{k}"] = v
        return m


class JudgeSession:
    """Collects the requests of one batch and waits for their watermark. Use from the client's loop."""

    def __init__(self, client, default_tags: Optional[Mapping[str, Any]] = None):
        self.client = client
        self.default_tags = dict(default_tags or {})
        self._batch: Dict[str, asyncio.Future] = {}
        self._enqueue_tasks: List[asyncio.Task] = []
        self._resolved = 0
        self._changed: Optional[asyncio.Event] = None

    def _tag(self, request: JudgeRequest) -> JudgeRequest:
        if self.default_tags:
            tags = dict(self.default_tags)
            tags.update(request.tags)
            request.tags = tags
        return request

    def _track(self, request_id: str, outer: asyncio.Future) -> None:
        if self._changed is None:
            self._changed = asyncio.Event()
        self._batch[request_id] = outer

        def done(_f: asyncio.Future) -> None:
            self._resolved += 1
            self._changed.set()

        outer.add_done_callback(done)

    def _register(self, request: JudgeRequest) -> asyncio.Future:
        """Create the caller's future and count it in the current batch BEFORE queue admission, so a
        barrier always sees every submitted request (even one still waiting for queue space)."""
        outer: asyncio.Future = asyncio.get_running_loop().create_future()
        rid = request.request_id
        self._track(rid, outer)

        def outer_done(f: asyncio.Future) -> None:
            if f.cancelled():
                self.client.cancel(rid, "cancelled by caller")

        outer.add_done_callback(outer_done)
        return outer

    async def _admit(self, request: JudgeRequest, outer: asyncio.Future) -> None:
        """Enqueue (with backpressure) and relay the client's future into ``outer``."""
        try:
            inner = await self.client.submit(request)
        except BaseException as e:  # JudgeHalted, closed client, cancelled
            if not outer.done():
                if isinstance(e, asyncio.CancelledError):
                    outer.cancel()
                else:
                    outer.set_exception(e)
                    outer.exception()
            if isinstance(e, asyncio.CancelledError):
                raise
            return

        def relay(f: asyncio.Future) -> None:
            if outer.done():
                return
            if f.cancelled():
                outer.cancel()
            elif f.exception() is not None:
                outer.set_exception(f.exception())
                outer.exception()
            else:
                outer.set_result(f.result())

        if outer.done():  # cancelled by the caller while waiting for admission
            self.client.cancel(request.request_id, "cancelled by caller")
        inner.add_done_callback(relay)

    def submit(self, request: JudgeRequest) -> asyncio.Future:
        """Non-blocking submit: returns a future at once; enqueueing (with backpressure) runs as a task."""
        request = self._tag(request)
        outer = self._register(request)
        self._enqueue_tasks.append(asyncio.get_running_loop().create_task(self._admit(request, outer)))
        return outer

    async def asubmit(self, request: JudgeRequest) -> asyncio.Future:
        """Submit with backpressure: waits while the client queue is full. The request is counted by
        the next barrier from the moment this is called. Raises :class:`JudgeHalted` if halted."""
        request = self._tag(request)
        outer = self._register(request)
        await self._admit(request, outer)
        if outer.done() and not outer.cancelled() and isinstance(outer.exception(), JudgeHalted):
            raise outer.exception()
        return outer

    @property
    def pending(self) -> int:
        return sum(1 for f in self._batch.values() if not f.done())

    async def barrier(self, watermark: float = 1.0, timeout: Optional[float] = None,
                      cancel_unresolved: bool = True, cancel_grace_s: float = 10.0) -> BarrierReport:
        """Wait until ``watermark`` of this batch has resolved (or ``timeout``); then start a new batch.

        Raises :class:`JudgeHalted` if the client halted (breaker/budget policy ``stop``).
        """
        if not 0.0 <= watermark <= 1.0:
            raise ValueError("watermark must be in [0, 1]")
        t0 = time.monotonic()
        batch, self._batch = self._batch, {}
        tasks, self._enqueue_tasks = self._enqueue_tasks, []
        if self._changed is None:
            self._changed = asyncio.Event()
        n = len(batch)
        need = math.ceil(watermark * n)
        timed_out = False
        deadline = None if timeout is None else t0 + timeout
        while sum(1 for f in batch.values() if f.done()) < need:
            if self.client.halted is not None:
                break
            self._changed.clear()
            if sum(1 for f in batch.values() if f.done()) >= need:
                break
            rem = None if deadline is None else deadline - time.monotonic()
            if rem is not None and rem <= 0:
                timed_out = True
                break
            try:
                await asyncio.wait_for(self._changed.wait(), rem)
            except asyncio.TimeoutError:
                timed_out = sum(1 for f in batch.values() if f.done()) < need
                break
        n_resolved = sum(1 for f in batch.values() if f.done())
        cancelled: List[str] = []
        unresolved = [rid for rid, f in batch.items() if not f.done()]
        if unresolved and cancel_unresolved and self.client.halted is None:
            for rid in unresolved:
                cancelled.append(rid)
                if not self.client.cancel(rid, "unresolved at barrier"):
                    batch[rid].cancel()  # never reached the client queue
            await asyncio.wait([batch[r] for r in unresolved], timeout=cancel_grace_s)
        if tasks:
            await asyncio.wait(tasks, timeout=cancel_grace_s)
        results: Dict[str, JudgeResult] = {}
        by_status: Dict[str, int] = {}
        halted: Optional[BaseException] = None
        for rid, f in batch.items():
            if not f.done() or f.cancelled():
                continue
            exc = f.exception()
            if exc is not None:
                halted = halted or exc
                continue
            r = f.result()
            results[rid] = r
            by_status[r.status.value] = by_status.get(r.status.value, 0) + 1
        if self.client.halted is not None:
            raise self.client.halted
        if isinstance(halted, JudgeHalted):
            raise halted
        return BarrierReport(n_submitted=n, n_resolved=n_resolved, watermark=watermark,
                             watermark_reached=n_resolved >= need, timed_out=timed_out,
                             elapsed_s=time.monotonic() - t0, by_status=by_status, cancelled_ids=cancelled,
                             results=results)


# ----------------------------------------------------------------------------------------- sync facade
class SyncJudgeClient:
    """Thread-safe synchronous facade: runs a :class:`JudgeClient` on a private event-loop thread.

    ``SyncJudgeClient(config)`` accepts the same arguments as :class:`JudgeClient`.
    """

    def __init__(self, config=None, **kwargs: Any):
        from judgerl.judge.client import JudgeClient

        self._pending: set = set()  # concurrent futures handed to callers, resolved before the loop stops
        self._pending_lock = threading.Lock()
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, name="judge-loop", daemon=True)
        self._thread.start()

        async def build():
            c = JudgeClient(config, **kwargs)
            await c.start()
            return c

        self.client = self._run(build())

    def _run(self, coro, timeout: Optional[float] = None):
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result(timeout)

    def _track(self, fut: "concurrent.futures.Future") -> "concurrent.futures.Future":
        with self._pending_lock:
            self._pending.add(fut)
        fut.add_done_callback(self._untrack)
        return fut

    def _untrack(self, fut) -> None:
        with self._pending_lock:
            self._pending.discard(fut)

    def submit(self, request: JudgeRequest) -> "concurrent.futures.Future[JudgeResult]":
        async def go():
            return await (await self.client.submit(request))

        return self._track(asyncio.run_coroutine_threadsafe(go(), self._loop))

    def judge(self, request: JudgeRequest, timeout: Optional[float] = None) -> JudgeResult:
        return self.submit(request).result(timeout)

    def judge_many(self, requests: Iterable[JudgeRequest], timeout: Optional[float] = None) -> List[JudgeResult]:
        return self._run(self.client.judge_many(list(requests)), timeout)

    def cancel(self, request_id: str) -> bool:
        async def go():
            return self.client.cancel(request_id)

        return self._run(go())

    def preflight(self, strict: bool = False, **kw: Any):
        return self._run(self.client.preflight(strict=strict, **kw))

    def metrics(self, **kw: Any) -> Dict[str, float]:
        async def go():
            return self.client.metrics(**kw)

        return self._run(go())

    def session(self, **kw: Any) -> "SyncJudgeSession":
        async def go():
            return self.client.session(**kw)

        return SyncJudgeSession(self, self._run(go()))

    def close(self) -> None:
        if not self._loop.is_running():
            return
        try:
            self._run(self.client.close(), timeout=60)
            with self._pending_lock:
                pending = list(self._pending)
            concurrent.futures.wait(pending, timeout=10)  # relay coroutines finish on the live loop
        finally:
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=10)

    def __enter__(self) -> "SyncJudgeClient":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


class SyncJudgeSession:
    def __init__(self, owner: SyncJudgeClient, session: JudgeSession):
        self._owner = owner
        self.session = session

    def submit(self, request: JudgeRequest) -> "concurrent.futures.Future[JudgeResult]":
        async def go():
            return await (await self.session.asubmit(request))

        return self._owner._track(asyncio.run_coroutine_threadsafe(go(), self._owner._loop))

    def barrier(self, watermark: float = 1.0, timeout: Optional[float] = None, **kw: Any) -> BarrierReport:
        return self._owner._run(self.session.barrier(watermark=watermark, timeout=timeout, **kw))
