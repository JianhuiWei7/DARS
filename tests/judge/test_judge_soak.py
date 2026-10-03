"""Fault-injected soak: 10,000 calls over two "remote" backends and one "local" backend with every
fault type, per-backend breakers, rate limits, retries, failover, single-flight and a durable cache.
Must finish with no hangs and exact status accounting."""
from __future__ import annotations

import asyncio
import random
from collections import Counter

from judge_helpers import SCORE_SCHEMA, make_client, req
from judgerl.judge import JudgeStatus
from judgerl.judge.backends.scripted import ScriptedBackend

REMOTE_FAULTS = {"timeout": 0.03, "rate_limited": 0.05, "backend_error": 0.04, "empty_content": 0.02,
                 "truncated": 0.02, "malformed": 0.04, "semantic_reject": 0.03, "refusal": 0.01, "hang": 0.005}


def _backends():
    remote_a = ScriptedBackend(name="remote_a", model="api/judge-a", faults=REMOTE_FAULTS, seed=1,
                               latency_s=(0.0, 0.002), retry_after_s=0.002, cost_per_call=0.0001)
    remote_b = ScriptedBackend(name="remote_b", model="api/judge-b", faults=REMOTE_FAULTS, seed=2,
                               latency_s=(0.0, 0.002), retry_after_s=0.002, cost_per_call=0.0002)
    local = ScriptedBackend(name="local", model="hf/small-judge", structured="none",
                            faults={"malformed": 0.05, "timeout": 0.02}, seed=3, latency_s=(0.0, 0.001))
    return remote_a, remote_b, local


async def test_soak_10k_calls_exact_accounting(tmp_path):
    backends = _backends()
    breaker = {"enabled": True, "window_s": 5.0, "min_calls": 200, "failure_rate": 0.6, "consecutive_failures": 40,
               "cooldown_s": 0.05}
    c = make_client(*backends, queue={"max_inflight": 256, "max_queue": 2000},
                    retry={"max_attempts": 4, "attempts_per_backend": 2},
                    cache={"path": str(tmp_path / "soak.sqlite")},
                    backend_configs={b.name: {"breaker": dict(breaker), "rate_limit": {"max_concurrency": 128}}
                                     for b in backends},
                    telemetry={"jsonl_path": str(tmp_path / "soak.jsonl")})
    rng = random.Random(0)
    n = 10_000
    contents = [f"trajectory {rng.randrange(8_000)}" for _ in range(n)]  # ~20% duplicates
    async with c:
        s = c.session()
        submit_order = []
        for i, content in enumerate(contents):
            r = req(content, tags={"step": i % 5}, timeout_s=0.05, priority=i % 3)
            submit_order.append(r.request_id)
            await s.asubmit(r)
        rep = await asyncio.wait_for(s.barrier(watermark=1.0, timeout=300), 400)
        m = c.metrics()
        leftover = (c.inflight, c.queue_depth, len(c._items))
    assert leftover == (0, 0, 0)
    assert rep.n_submitted == n and rep.n_resolved == n and not rep.cancelled_ids
    results = rep.results
    assert set(results) == set(submit_order)
    by_status = Counter(r.status.value for r in results.values())
    assert sum(by_status.values()) == n and by_status == Counter(rep.by_status)
    # telemetry agrees with the results, status by status
    for st in JudgeStatus:
        assert m[f"judge/{st.value}"] == by_status.get(st.value, 0), st
    assert m["judge/total"] == n
    # every backend call is an attempt of exactly one non-shared result
    dispatched = [r for r in results.values() if not (r.cache_hit or r.coalesced)]
    calls = sum(len(b.calls) for b in backends)
    assert calls == sum(r.attempts for r in dispatched)
    assert all(len(r.attempt_log) == r.attempts for r in dispatched)
    assert sum(v for k, v in m.items() if k.startswith("judge/attempt/")) == calls
    # cost is exactly the sum of per-call costs
    exp_cost = sum(0.0001 if a["backend"] == "remote_a" else 0.0002 if a["backend"] == "remote_b" else 0.0
                   for r in dispatched for a in r.attempt_log if a["status"] not in (
                       "timeout", "rate_limited", "backend_error"))
    assert abs(m["judge/cost_usd"] - exp_cost) < 1e-9
    # retries and failover actually happened, and most requests succeeded
    assert m["judge/retries"] > 500 and m["judge/failover"] > 0 and by_status["ok"] > 0.9 * n
    assert m["judge/cache_hit"] + m["judge/coalesced"] > 0
    assert all(r.parsed is not None for r in results.values() if r.ok)
    assert m["judge/tag/step=0/total"] == n // 5
    with open(tmp_path / "soak.jsonl") as f:
        assert sum(1 for _ in f) == n


async def test_soak_with_random_cancellations_and_deadlines():
    backends = _backends()
    c = make_client(*backends, queue={"max_inflight": 64, "max_queue": 500},
                    retry={"max_attempts": 3})
    rng = random.Random(1)
    n = 2_000
    async with c:
        futs, ids = [], []
        for i in range(n):
            r = req(f"x{i}", timeout_s=0.05, deadline_s=rng.choice([None, 0.02, 0.5, 2.0]))
            futs.append(await c.submit(r))
            ids.append(r.request_id)
            if rng.random() < 0.1:
                c.cancel(ids[rng.randrange(len(ids))])
            if rng.random() < 0.02:
                futs[rng.randrange(len(futs))].cancel()
        done, pending = await asyncio.wait(futs, timeout=120)
        assert not pending
        await asyncio.sleep(0.05)
        m = c.metrics()
        assert (c.inflight, c.queue_depth, len(c._items)) == (0, 0, 0)
    got = Counter(f.result().status.value for f in futs if not f.cancelled())
    n_future_cancelled = sum(f.cancelled() for f in futs)
    assert sum(got.values()) + n_future_cancelled == n
    assert m["judge/total"] == n
    assert m["judge/cancelled"] == got.get("cancelled", 0) + n_future_cancelled
    for st in JudgeStatus:
        if st != JudgeStatus.CANCELLED:
            assert m[f"judge/{st.value}"] == got.get(st.value, 0)
    assert SCORE_SCHEMA  # schema used by req()
