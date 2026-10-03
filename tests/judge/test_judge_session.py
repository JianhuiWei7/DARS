"""Watermark sessions: barrier at 100% / partial watermark / timeout, cancellation reporting, halts,
batches, tags and the synchronous session facade."""
from __future__ import annotations

import asyncio
import threading
import time

import pytest

from judge_helpers import make_client, req
from judgerl.judge import JudgeHalted, JudgeStatus, SyncJudgeClient
from judgerl.judge.backends.scripted import ScriptedBackend


def slow_for(prefix: str, delay: float):
    async def responder(call):
        if call.messages[0]["content"].startswith(prefix):
            await asyncio.sleep(delay)
        return {"score": 1}

    return responder


async def test_barrier_full_watermark_collects_everything():
    be = ScriptedBackend(latency_s=(0.0, 0.02), faults={"malformed": 0.2}, seed=3)
    async with make_client(be, retry={"max_attempts": 1}) as c:
        s = c.session()
        futs = [s.submit(req(f"t{i}")) for i in range(50)]
        rep = await s.barrier(watermark=1.0, timeout=10)
    assert rep.n_submitted == 50 and rep.n_resolved == 50 and rep.watermark_reached and not rep.timed_out
    assert len(rep.results) == 50 and sum(rep.by_status.values()) == 50 and not rep.cancelled_ids
    assert rep.by_status.get("malformed", 0) > 0  # failed labels are resolved too
    assert all(f.done() for f in futs)


async def test_partial_watermark_cancels_stragglers_and_reports_them():
    be = ScriptedBackend(responder=slow_for("slow", 30.0))
    async with make_client(be) as c:
        s = c.session()
        for i in range(8):
            s.submit(req(f"fast{i}"))
        for i in range(2):
            s.submit(req(f"slow{i}"))
        t0 = time.monotonic()
        rep = await s.barrier(watermark=0.8, timeout=20)
        dt = time.monotonic() - t0
        m = c.metrics()
    assert dt < 5 and rep.watermark_reached and not rep.timed_out and rep.n_resolved == 8
    assert len(rep.cancelled_ids) == 2 and rep.by_status == {"ok": 8, "cancelled": 2}
    assert m["judge/cancelled"] == 2 and m["judge/ok"] == 8


async def test_timeout_cancels_unresolved():
    be = ScriptedBackend(responder=slow_for("slow", 30.0))
    async with make_client(be) as c:
        s = c.session()
        for i in range(3):
            s.submit(req(f"fast{i}"))
        for i in range(3):
            s.submit(req(f"slow{i}"))
        rep = await s.barrier(watermark=1.0, timeout=0.3)
    assert rep.timed_out and not rep.watermark_reached and rep.n_resolved == 3
    assert sorted(rep.by_status.items()) == [("cancelled", 3), ("ok", 3)]
    assert all(rep.results[r].status == JudgeStatus.CANCELLED for r in rep.cancelled_ids)


async def test_barrier_without_cancellation_leaves_requests_running():
    be = ScriptedBackend(responder=slow_for("slow", 0.3))
    async with make_client(be) as c:
        s = c.session()
        s.submit(req("fast"))
        slow = s.submit(req("slow"))
        rep = await s.barrier(watermark=0.5, timeout=5, cancel_unresolved=False)
        assert rep.n_resolved == 1 and not rep.cancelled_ids and not slow.done()
        assert (await slow).ok


async def test_batches_are_separated_by_barriers():
    be = ScriptedBackend()
    async with make_client(be) as c:
        s = c.session(default_tags={"channel": "process"})
        for i in range(3):
            s.submit(req(f"a{i}", tags={"step": 1}))
        r1 = await s.barrier()
        for i in range(2):
            s.submit(req(f"b{i}", tags={"step": 2}))
        r2 = await s.barrier()
        m = c.metrics()
    assert r1.n_submitted == 3 and r2.n_submitted == 2
    assert all(r.tags["channel"] == "process" for r in r1.results.values())
    assert m["judge/tag/step=1/ok"] == 3 and m["judge/tag/step=2/ok"] == 2 and m["judge/tag/channel=process/total"] == 5


async def test_empty_barrier_returns_immediately():
    async with make_client(ScriptedBackend()) as c:
        rep = await c.session().barrier(timeout=1)
    assert rep.n_submitted == 0 and rep.watermark_reached and rep.resolved_fraction == 1.0


async def test_barrier_raises_when_client_halts():
    be = ScriptedBackend(sequence=["backend_error"])
    c = make_client(be, retry={"max_attempts": 1}, queue={"max_inflight": 1},
                    breaker={"enabled": True, "consecutive_failures": 2, "failure_rate": None, "policy": "stop"})
    async with c:
        s = c.session()
        for i in range(10):
            s.submit(req(f"q{i}"))
        with pytest.raises(JudgeHalted):
            await s.barrier(timeout=10)
    assert len(be.calls) == 2


async def test_asubmit_applies_backpressure():
    be = ScriptedBackend(latency_s=0.02)
    async with make_client(be, queue={"max_inflight": 1, "max_queue": 2}) as c:
        s = c.session()
        t0 = time.monotonic()
        for i in range(6):
            await s.asubmit(req(f"q{i}"))
        waited = time.monotonic() - t0
        rep = await s.barrier(timeout=10)
    assert rep.n_resolved == 6 and waited >= 0.04


def test_sync_session_from_trainer_threads():
    cfg = {"backends": [{"type": "scripted", "name": "s", "latency_s": 0.005}], "breaker": {"enabled": False}}
    with SyncJudgeClient(cfg) as sync:
        sess = sync.session()
        futs = []

        def producer(k):
            for i in range(20):
                futs.append(sess.submit(req(f"w{k}-{i}")))

        threads = [threading.Thread(target=producer, args=(k,)) for k in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        results = [f.result(timeout=10) for f in futs]
        rep = sess.barrier(watermark=1.0, timeout=10)
        assert all(r.ok for r in results) and rep.n_submitted == 80 and rep.by_status == {"ok": 80}
