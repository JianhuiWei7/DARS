"""Durable cache: keys, hits/misses, restart persistence, TTL, failures not cached, provenance,
redaction, in-process and cross-process single-flight."""
from __future__ import annotations

import asyncio
import json
import multiprocessing as mp
import time

from judge_helpers import SCORE_SCHEMA, make_client, req
from judgerl.judge import JudgeCache, JudgeStatus, cache_key
from judgerl.judge.backends.scripted import ScriptedBackend


def _key(**over):
    base = dict(messages=[{"role": "user", "content": "x"}], output_schema=SCORE_SCHEMA, prompt_version="1",
                schema_version="1", parser_version="1", model="m", revision=None,
                decoding={"max_tokens": 10, "temperature": 0.0}, reasoning={"mode": "off", "budget_tokens": None},
                structured="json_schema", sample_index=0)
    base.update(over)
    return cache_key(**base)


def test_cache_key_covers_everything_that_changes_the_answer():
    k = _key()
    assert k == _key()  # deterministic
    assert _key(messages=[{"content": "x", "role": "user"}]) == k  # canonical (key order)
    for change in (dict(messages=[{"role": "user", "content": "y"}]), dict(output_schema={"type": "object"}),
                   dict(prompt_version="2"), dict(schema_version="2"), dict(parser_version="2"), dict(model="m2"),
                   dict(revision="abc"), dict(decoding={"max_tokens": 11, "temperature": 0.0}),
                   dict(reasoning={"mode": "on", "budget_tokens": None}), dict(structured="none"),
                   dict(sample_index=1), dict(extra_params={"logprobs": True})):
        assert _key(**change) != k, change


async def test_hit_miss_and_cost_accounting(tmp_cache):
    be = ScriptedBackend(cost_per_call=0.01)
    async with make_client(be, cache={"path": tmp_cache}) as c:
        r1 = await c.judge(req("same", policy_version=1, tags={"step": 1}))
        r2 = await c.judge(req("same", policy_version=2, tags={"step": 2}))  # policy/tags not in the key
        r3 = await c.judge(req("different"))
        m = c.metrics()
    assert (r1.cache_hit, r2.cache_hit, r3.cache_hit) == (False, True, False)
    assert r2.ok and r2.parsed == r1.parsed and r2.cost_usd == 0.0 and r2.cache_key == r1.cache_key
    assert len(be.calls) == 2 and m["judge/cache_hit"] == 1 and m["judge/cost_usd"] == 0.02


async def test_cache_survives_restart_and_stores_raw_provenance(tmp_cache):
    be1 = ScriptedBackend(default_output={"score": 8})
    async with make_client(be1, cache={"path": tmp_cache}) as c:
        r1 = await c.judge(req("persist me"))
    be2 = ScriptedBackend(default_output={"score": 0})
    async with make_client(be2, cache={"path": tmp_cache}) as c:
        r2 = await c.judge(req("persist me"))
    assert r2.cache_hit and r2.parsed == {"score": 8} and len(be2.calls) == 0
    cache = JudgeCache(tmp_cache)
    rec = cache.get(r1.cache_key)
    assert rec["raw"] == {"scripted": True, "text": json.dumps({"score": 8})} and rec["backend"] == "scripted"
    assert len(cache) == 1
    cache.close()


async def test_failures_are_not_cached(tmp_cache):
    be = ScriptedBackend(sequence=["malformed"])
    async with make_client(be, cache={"path": tmp_cache}, retry={"max_attempts": 1}) as c:
        r1 = await c.judge(req("flaky"))
        be.sequence = ["ok"]
        r2 = await c.judge(req("flaky"))
        r3 = await c.judge(req("flaky"))
    assert r1.status == JudgeStatus.MALFORMED and r2.ok and not r2.cache_hit and r3.cache_hit
    assert len(be.calls) == 2


async def test_ttl_expiry(tmp_cache):
    be = ScriptedBackend()
    async with make_client(be, cache={"path": tmp_cache, "ttl_s": 0.05}) as c:
        await c.judge(req("t"))
        assert (await c.judge(req("t"))).cache_hit
        await asyncio.sleep(0.1)
        assert not (await c.judge(req("t"))).cache_hit
    assert len(be.calls) == 2


async def test_use_cache_false_bypasses(tmp_cache):
    be = ScriptedBackend()
    async with make_client(be, cache={"path": tmp_cache}) as c:
        await c.judge(req("x"))
        r = await c.judge(req("x", use_cache=False))
    assert not r.cache_hit and len(be.calls) == 2


async def test_redaction_applies_before_send_and_hash(tmp_cache):
    def redact(r):
        msgs = [dict(m, content=m["content"].split("|SECRET:")[0]) for m in r.messages]
        return r.with_(messages=msgs)

    be = ScriptedBackend()
    async with make_client(be, cache={"path": tmp_cache}, redact=redact) as c:
        a = await c.judge(req("trajectory|SECRET: success=1"))
        b = await c.judge(req("trajectory|SECRET: success=0"))
    assert all("SECRET" not in m["content"] for call in be.calls for m in call.messages)
    assert len(be.calls) == 1 and b.cache_hit and a.cache_key == b.cache_key


async def test_in_process_single_flight_coalesces_identical_requests():
    be = ScriptedBackend(latency_s=0.05)
    async with make_client(be) as c:
        rs = await c.judge_many([req("dup") for _ in range(10)])
        m = c.metrics()
    assert len(be.calls) == 1 and all(r.ok for r in rs) and sum(r.coalesced for r in rs) == 9
    assert m["judge/coalesced"] == 9 and m["judge/total"] == 10
    be2 = ScriptedBackend(latency_s=0.01)
    async with make_client(be2) as c:
        await c.judge_many([req("dup", use_cache=False) for _ in range(5)])
    assert len(be2.calls) == 5


async def test_distinct_sample_index_is_not_coalesced(tmp_cache):
    be = ScriptedBackend(latency_s=0.01)
    async with make_client(be, cache={"path": tmp_cache}) as c:
        rs = await c.judge_many([req("draw", sample_index=i) for i in range(4)])
    assert len(be.calls) == 4 and len({r.cache_key for r in rs}) == 4


def test_lease_exclusion_and_stale_lease_takeover(tmp_cache):
    a = JudgeCache(tmp_cache, lease_ttl_s=0.2)
    b = JudgeCache(tmp_cache, lease_ttl_s=0.2)
    ta = a.acquire_lease("k")
    assert ta and b.acquire_lease("k") is None
    assert b.lease_holder("k") == a.owner
    a.release_lease("k", ta)
    assert b.acquire_lease("k") > ta  # fencing tokens increase monotonically
    time.sleep(0.25)  # b "crashed": its lease expires
    assert a.acquire_lease("k")
    a.close()
    b.close()


def test_cross_process_single_flight(tmp_path):
    cache_path = str(tmp_path / "shared.sqlite")
    call_log = str(tmp_path / "calls.jsonl")
    ctx = mp.get_context("spawn")
    n = 4
    barrier = ctx.Barrier(n)
    q = ctx.Queue()
    from judge_mp_worker import run_worker

    procs = [ctx.Process(target=run_worker, args=(cache_path, call_log, "shared prompt", barrier, q))
             for _ in range(n)]
    for p in procs:
        p.start()
    outs = [q.get(timeout=120) for _ in range(n)]
    for p in procs:
        p.join(timeout=60)
        assert p.exitcode == 0
    with open(call_log) as f:
        calls = [json.loads(line) for line in f]
    assert len(calls) == 1, calls  # exactly one process called the model
    assert all(o["status"] == "ok" and o["parsed"] == {"score": 6} for o in outs)
    assert sum(o["cache_hit"] for o in outs) == n - 1
    assert sum(o["lease_waits"] for o in outs) >= 1
