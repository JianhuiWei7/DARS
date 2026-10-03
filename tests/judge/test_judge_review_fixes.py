"""Regression tests for four review findings: producers stranded on a full queue at close/halt,
session requests untracked while waiting for admission, cache lease fencing / renewal, and
per-backend cost caps under concurrency."""
from __future__ import annotations

import asyncio
import concurrent.futures
import time

import pytest

from judge_helpers import make_client, req
from judgerl.judge import JudgeCache, JudgeHalted, JudgeStatus, SyncJudgeClient
from judgerl.judge.backends.scripted import ScriptedBackend


# ----------------------------------------------------------------------------------------- 1. queue close / halt
async def test_close_wakes_producers_blocked_on_full_queue():
    be = ScriptedBackend(latency_s=5.0)
    c = make_client(be, queue={"max_inflight": 1, "max_queue": 1})
    await c.start()
    await c.submit(req("running"))
    await asyncio.sleep(0.01)  # the single worker picks it up
    await c.submit(req("queued"))  # fills the queue
    producers = [asyncio.create_task(c.submit(req(f"blocked{i}"))) for i in range(3)]
    await asyncio.sleep(0.02)
    assert not any(p.done() for p in producers)
    await asyncio.wait_for(c.close(), 5)
    done, pending = await asyncio.wait(producers, timeout=2)
    assert not pending, "producers blocked in put() were never woken"
    for p in producers:
        fut = p.result()
        assert fut.done() and fut.result().status == JudgeStatus.CANCELLED
    m = c.telemetry.metrics()
    assert m["judge/total"] == 5 and m["judge/cancelled"] == 5


async def test_halt_fails_blocked_producers_fast():
    be = ScriptedBackend(sequence=["backend_error"], latency_s=0.05)
    c = make_client(be, retry={"max_attempts": 1}, queue={"max_inflight": 1, "max_queue": 1},
                    breaker={"enabled": True, "consecutive_failures": 1, "failure_rate": None, "policy": "stop"})
    async with c:
        first = await c.submit(req("first"))
        await asyncio.sleep(0.01)
        await c.submit(req("queued"))
        producers = [asyncio.create_task(c.submit(req(f"blocked{i}"))) for i in range(3)]
        await asyncio.wait_for(asyncio.wait({first}), 2)
        done, pending = await asyncio.wait(producers, timeout=2)
        assert not pending
        assert all(isinstance(p.exception(), JudgeHalted) for p in producers)
    assert len(be.calls) == 1


def test_sync_close_under_saturation_resolves_every_future():
    cfg = {"backends": [{"type": "scripted", "name": "s", "latency_s": 5.0}], "breaker": {"enabled": False},
           "queue": {"max_inflight": 1, "max_queue": 1}}
    sync = SyncJudgeClient(cfg)
    futs = [sync.submit(req(f"q{i}")) for i in range(6)]
    time.sleep(0.1)
    sync.close()
    done, not_done = concurrent.futures.wait(futs, timeout=5)
    assert not not_done, f"{len(not_done)} sync futures leaked at shutdown"
    assert all(f.result().status == JudgeStatus.CANCELLED for f in futs)


# ----------------------------------------------------------------------------------------- 2. session tracking
def test_sync_session_barrier_counts_requests_waiting_for_admission():
    cfg = {"backends": [{"type": "scripted", "name": "s", "latency_s": 0.02}], "breaker": {"enabled": False},
           "queue": {"max_inflight": 1, "max_queue": 1}}
    with SyncJudgeClient(cfg) as sync:
        sess = sync.session()
        futs = [sess.submit(req(f"q{i}")) for i in range(10)]
        rep = sess.barrier(watermark=1.0, timeout=30)
        assert rep.n_submitted == 10 and rep.n_resolved == 10 and rep.by_status == {"ok": 10}
        assert all(f.done() for f in futs)


async def test_asubmit_is_tracked_before_admission():
    be = ScriptedBackend(latency_s=0.02)
    async with make_client(be, queue={"max_inflight": 1, "max_queue": 1}) as c:
        s = c.session()
        tasks = [asyncio.create_task(s.asubmit(req(f"q{i}"))) for i in range(8)]
        await asyncio.sleep(0)  # every asubmit has started; most wait for queue space
        rep = await s.barrier(watermark=1.0, timeout=30)
        await asyncio.gather(*tasks)
    assert rep.n_submitted == 8 and rep.n_resolved == 8 and rep.by_status == {"ok": 8}


# ----------------------------------------------------------------------------------------- 3. lease fencing
def test_stale_lease_holder_cannot_renew_overwrite_or_release(tmp_cache):
    a = JudgeCache(tmp_cache, lease_ttl_s=0.1)
    b = JudgeCache(tmp_cache, lease_ttl_s=5.0)
    ta = a.acquire_lease("k")
    assert ta
    time.sleep(0.15)  # a is paused past its TTL; b takes over
    tb = b.acquire_lease("k")
    assert tb and tb > ta
    assert a.renew_lease("k", ta) is False
    assert b.put("k", {"parsed": "new"}, token=tb) is True
    assert a.put("k", {"parsed": "stale"}, token=ta) is False
    a.release_lease("k", ta)
    assert b.lease_holder("k") == b.owner
    assert b.get("k")["parsed"] == "new"
    assert b.renew_lease("k", tb) is True
    b.release_lease("k", tb)
    assert b.lease_holder("k") is None
    a.close()
    b.close()


def test_lease_ttl_must_leave_room_for_renewal(tmp_cache):
    with pytest.raises(ValueError):
        JudgeCache(tmp_cache, lease_ttl_s=0)
    c = JudgeCache(tmp_cache, lease_ttl_s=0.2)
    assert 0 < c.renew_interval_s < 0.2 / 2
    c.close()


async def test_lease_renewal_outpaces_short_ttl(tmp_cache):
    b1 = ScriptedBackend(latency_s=0.6)
    b2 = ScriptedBackend(latency_s=0.6)
    cache = {"path": tmp_cache, "lease_ttl_s": 0.2, "poll_s": 0.02}
    c1 = make_client(b1, cache=dict(cache))
    c2 = make_client(b2, cache=dict(cache))
    async with c1, c2:
        f1 = asyncio.create_task(c1.judge(req("same")))
        await asyncio.sleep(0.35)  # past the TTL: only renewal keeps the lease
        r2 = await c2.judge(req("same"))
        r1 = await f1
    assert r1.ok and r2.ok and r2.cache_hit
    assert len(b1.calls) + len(b2.calls) == 1


async def test_lost_lease_result_is_not_written(tmp_cache):
    be = ScriptedBackend(latency_s=0.3, default_output={"score": 1})
    c = make_client(be, cache={"path": tmp_cache})
    thief = JudgeCache(tmp_cache)
    request = req("contested")
    async with c:
        task = asyncio.create_task(c.judge(request))
        await asyncio.sleep(0.1)
        key = c._key(request, c.slots["scripted"])
        with thief._lock:  # the holder looks dead (e.g. paused past its TTL): force expiry
            thief._conn.execute("UPDATE leases SET expires=0 WHERE key=?", (key,))
        tok = thief.acquire_lease(key)
        assert tok and thief.put(key, {"parsed": {"score": 9}, "backend": "other"}, token=tok)
        r = await task
        m = c.metrics()
    assert r.ok and r.parsed == {"score": 1}
    assert thief.get(key)["parsed"] == {"score": 9}  # the newer result survives
    assert m["judge/event/cache_lease_lost"] == 1
    thief.close()


# ----------------------------------------------------------------------------------------- 4. backend cost caps
def _capped(name, cap):
    return {name: {"breaker": {"enabled": True, "cost_budget_usd": cap, "failure_rate": None,
                               "consecutive_failures": None}}}


async def test_backend_cost_cap_holds_under_concurrency_with_unknown_prices():
    pricey = ScriptedBackend(name="pricey", cost_per_call=1.0, latency_s=0.05)
    cheap = ScriptedBackend(name="cheap", cost_per_call=0.0)
    c = make_client(pricey, cheap, queue={"max_inflight": 4}, backend_configs=_capped("pricey", 1.0))
    async with c:
        rs = await c.judge_many([req(f"q{i}") for i in range(4)])
    assert all(r.ok for r in rs)
    assert len(pricey.calls) == 1 and sum(r.cost_usd for r in rs) == 1.0
    assert c.slots["pricey"].breaker.cost_usd == 1.0 and c.slots["pricey"].breaker.budget_tripped


async def test_backend_cost_cap_reserves_worst_case_with_known_prices():
    # worst case = actual = 100 completion tokens * $4000/Mtok = $0.40 per call; cap $1 -> 2 calls
    pricey = ScriptedBackend(name="pricey", completion_tokens=100, cost_per_call=None, latency_s=0.05)
    cheap = ScriptedBackend(name="cheap")
    cfg = _capped("pricey", 1.0)
    cfg["pricey"]["prices"] = {"input_per_mtok": 0.0, "output_per_mtok": 4000.0}
    c = make_client(pricey, cheap, queue={"max_inflight": 4}, defaults={"max_tokens": 100}, backend_configs=cfg)
    async with c:
        rs = await c.judge_many([req(f"q{i}") for i in range(4)])
    assert all(r.ok for r in rs)
    assert len(pricey.calls) == 2 and c.slots["pricey"].breaker.cost_usd == pytest.approx(0.8)
    assert [r.backend for r in rs].count("cheap") == 2


# ----------------------------------------------------------------------------------------- found by the soak
async def test_late_future_cancel_is_accounted_as_cancelled():
    """The caller cancels its future after the work finished but before the worker delivered it."""
    from judgerl.judge.client import _Item
    from judgerl.judge.types import JudgeResult

    async with make_client(ScriptedBackend()) as c:
        r = req()
        item = _Item(request=r, future=asyncio.get_running_loop().create_future(), submitted=0.0, deadline=None,
                     seq=0, started=True)
        item.future.cancel()
        c._finish(item, JudgeResult(request_id=r.request_id, status=JudgeStatus.OK, cost_usd=0.5))
        m = c.metrics()
    assert m["judge/cancelled"] == 1 and m["judge/ok"] == 0 and m["judge/cost_usd"] == 0.5
