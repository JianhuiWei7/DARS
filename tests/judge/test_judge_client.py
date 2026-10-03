"""Client behaviour through the scripted backend: typed failures, retries, backoff, retry-after,
failover, structured output, corrective re-ask, deadlines, cancellation and priorities."""
from __future__ import annotations

import asyncio
import json

import pytest
from pydantic import BaseModel

from judge_helpers import SCORE_SCHEMA, make_client, req
from judgerl.judge import BackendError, BackendResponse, FakeClock, JudgeStatus, Usage
from judgerl.judge.backends.scripted import ScriptedBackend


# ----------------------------------------------------------------------------------------- scripted backend
async def test_scripted_backend_is_deterministic_per_request_and_attempt():
    a = ScriptedBackend(faults={"malformed": 0.3, "timeout": 0.3}, seed=7)
    b = ScriptedBackend(faults={"malformed": 0.3, "timeout": 0.3}, seed=7)
    from judgerl.judge.types import BackendCall

    outs = []
    for be in (a, b):
        seq = []
        for i in range(50):
            call = BackendCall(messages=[{"role": "user", "content": "x"}], max_tokens=10, request_id=f"r{i}", attempt=1)
            try:
                r = await be.generate(call)
                seq.append(r.text)
            except BackendError as e:
                seq.append(e.status.value)
        outs.append(seq)
    assert outs[0] == outs[1]
    assert set(a.fault_counts) <= {"ok", "malformed", "timeout"} and len(a.calls) == 50


async def test_scripted_responder_and_responses():
    be = ScriptedBackend(responses=[{"score": 2}, "{\"score\": 3}"], responder=None)
    async with make_client(be) as c:
        r1 = await c.judge(req("a"))
        r2 = await c.judge(req("b"))
        r3 = await c.judge(req("c"))
    assert (r1.parsed, r2.parsed, r3.parsed) == ({"score": 2}, {"score": 3}, {"score": 1})


# ----------------------------------------------------------------------------------------- statuses
@pytest.mark.parametrize("fault,status", [
    ("timeout", JudgeStatus.TIMEOUT), ("rate_limited", JudgeStatus.RATE_LIMITED),
    ("backend_error", JudgeStatus.BACKEND_ERROR), ("auth", JudgeStatus.AUTH), ("quota", JudgeStatus.QUOTA),
    ("refusal", JudgeStatus.REFUSAL), ("empty_content", JudgeStatus.EMPTY_CONTENT),
    ("truncated", JudgeStatus.TRUNCATED), ("malformed", JudgeStatus.MALFORMED),
    ("semantic_reject", JudgeStatus.SEMANTIC_REJECT),
])
async def test_every_fault_maps_to_its_typed_status(fault, status):
    be = ScriptedBackend(sequence=[fault])
    async with make_client(be, retry={"max_attempts": 3}) as c:
        r = await c.judge(req())
        m = c.metrics()
    assert r.status == status and not r.ok and r.parsed is None
    retried = status not in (JudgeStatus.AUTH, JudgeStatus.QUOTA, JudgeStatus.REFUSAL)
    assert r.attempts == (3 if retried else 1)
    assert m[f"judge/{status.value}"] == 1 and m["judge/total"] == 1 and m["judge/ok"] == 0
    assert m[f"judge/attempt/{status.value}"] == r.attempts


async def test_retry_then_success_records_attempt_log():
    be = ScriptedBackend(sequence=["backend_error", "malformed", "ok"])
    async with make_client(be) as c:
        r = await c.judge(req())
    assert r.ok and r.attempts == 3 and r.parsed == {"score": 1}
    assert [a["status"] for a in r.attempt_log] == ["backend_error", "malformed", "ok"]


async def test_exponential_backoff_and_retry_after_with_fake_clock():
    clock = FakeClock()
    be = ScriptedBackend(sequence=["backend_error", "backend_error", "rate_limited", "ok"], retry_after_s=7.0)
    c = make_client(be, clock=clock, retry={"base_delay_s": 1.0, "max_delay_s": 100.0, "jitter": 0.0,
                                            "max_attempts": 5})
    async with c:
        r = await c.judge(req(timeout_s=30))
    assert r.ok and r.attempts == 4
    backoffs = [s for s in clock.slept if s > 0]
    # 1s, 2s exponential; then retry-after 7s dominates 4s; the limiter also pauses the backend for 7s
    assert backoffs[:3] == [1.0, 2.0, 7.0]


async def test_retry_after_is_capped():
    clock = FakeClock()
    be = ScriptedBackend(sequence=["rate_limited", "ok"], retry_after_s=10_000)
    c = make_client(be, clock=clock, retry={"base_delay_s": 0.5, "jitter": 0.0, "max_retry_after_s": 60})
    async with c:
        r = await c.judge(req())
    assert r.ok and max(clock.slept) == 60


async def test_non_retryable_status_stops_immediately():
    be = ScriptedBackend(sequence=["refusal", "ok"])
    async with make_client(be) as c:
        r = await c.judge(req())
    assert r.status == JudgeStatus.REFUSAL and r.attempts == 1 and len(be.calls) == 1


async def test_retry_on_is_configurable():
    be = ScriptedBackend(sequence=["malformed", "ok"])
    async with make_client(be, retry={"retry_on": ["timeout"]}) as c:
        r = await c.judge(req())
    assert r.status == JudgeStatus.MALFORMED and r.attempts == 1


# ----------------------------------------------------------------------------------------- failover
async def test_failover_to_next_backend_after_per_backend_attempts():
    primary = ScriptedBackend(name="primary", sequence=["backend_error"])
    secondary = ScriptedBackend(name="secondary", default_output={"score": 9})
    async with make_client(primary, secondary, retry={"max_attempts": 4, "attempts_per_backend": 2}) as c:
        r = await c.judge(req())
    assert r.ok and r.backend == "secondary" and r.parsed == {"score": 9}
    assert len(primary.calls) == 2 and len(secondary.calls) == 1 and r.failovers == 1


async def test_auth_failure_disables_backend_and_fails_over_without_retry():
    bad = ScriptedBackend(name="bad", sequence=["auth"])
    good = ScriptedBackend(name="good")
    async with make_client(bad, good) as c:
        r = await c.judge(req())
    assert r.ok and r.backend == "good" and len(bad.calls) == 1


async def test_named_pools_select_backends():
    a = ScriptedBackend(name="a", default_output={"score": 1})
    b = ScriptedBackend(name="b", default_output={"score": 2})
    async with make_client(a, b, pools={"eval": ["b"]}) as c:
        r_default = await c.judge(req("x"))
        r_eval = await c.judge(req("y", pool="eval"))
    assert r_default.backend == "a" and r_eval.backend == "b" and r_eval.parsed == {"score": 2}


# ----------------------------------------------------------------------------------------- structured output
async def test_native_schema_is_sent_when_supported_and_instruction_otherwise():
    native = ScriptedBackend(name="native", structured="json_schema")
    plain = ScriptedBackend(name="plain", structured="none")
    async with make_client(native) as c:
        await c.judge(req("q"))
    async with make_client(plain) as c:
        await c.judge(req("q"))
    assert native.calls[0].json_schema == SCORE_SCHEMA and native.calls[0].messages[0]["content"] == "q"
    assert plain.calls[0].json_schema is None and "JSON Schema" in plain.calls[0].messages[0]["content"]


async def test_lenient_extraction_is_a_counted_fallback():
    be = ScriptedBackend(responder=lambda call: 'Here is my verdict: {"score": 4}. Thanks!')
    async with make_client(be) as c:
        r = await c.judge(req())
        m = c.metrics()
    assert r.ok and r.parsed == {"score": 4} and r.lenient_parse and m["judge/lenient_parse"] == 1
    be2 = ScriptedBackend(responder=lambda call: 'Here is my verdict: {"score": 4}.')
    async with make_client(be2, structured={"lenient_fallback": False}, retry={"max_attempts": 1}) as c:
        r2 = await c.judge(req())
    assert r2.status == JudgeStatus.MALFORMED


async def test_schema_violation_triggers_corrective_reask():
    answers = iter([{"score": 99}, {"score": 5}])
    be = ScriptedBackend(responder=lambda call: next(answers))
    async with make_client(be) as c:
        r = await c.judge(req("rate it"))
    assert r.ok and r.parsed == {"score": 5} and r.reasks == 1
    second = be.calls[1].messages
    assert second[-2]["role"] == "assistant" and json.loads(second[-2]["content"]) == {"score": 99}
    assert second[-1]["role"] == "user" and "rejected" in second[-1]["content"] and "score" in second[-1]["content"]


async def test_validator_reject_uses_its_message_and_final_status():
    be = ScriptedBackend(default_output={"score": 3})
    validator = lambda parsed: None if parsed["score"] % 2 == 0 else "score must be even"  # noqa: E731
    async with make_client(be, retry={"max_attempts": 2}) as c:
        r = await c.judge(req(validator=validator))
    assert r.status == JudgeStatus.SEMANTIC_REJECT and r.error == "score must be even" and r.reasks == 2
    assert be.calls[1].messages[-1]["content"] == "score must be even"


async def test_reask_disabled_retries_original_messages():
    answers = iter([{"score": 99}, {"score": 5}])
    be = ScriptedBackend(responder=lambda call: next(answers))
    async with make_client(be, structured={"reask": False}) as c:
        r = await c.judge(req("rate it"))
    assert r.ok and r.reasks == 0 and len(be.calls[1].messages) == 1


async def test_pydantic_output_schema_returns_model_instance():
    class Verdict(BaseModel):
        score: int
        reason: str

    be = ScriptedBackend(default_output={"score": 2, "reason": "fine"})
    async with make_client(be) as c:
        r = await c.judge(req(schema=Verdict))
    assert isinstance(r.parsed, Verdict) and r.parsed.reason == "fine"
    assert be.calls[0].json_schema["title"] == "Verdict" and be.calls[0].schema_name == "Verdict"


async def test_free_text_request_and_think_block_stripping():
    be = ScriptedBackend(responder=lambda call: "<think>let me see</think>The answer is fine.")
    async with make_client(be) as c:
        r = await c.judge(req(schema=None))
    assert r.ok and r.parsed == "The answer is fine." and r.reasoning_text == "let me see"


async def test_truncated_when_reasoning_eats_everything():
    be = ScriptedBackend(responder=lambda call: BackendResponse(text="<think>long reasoning", finish_reason="length",
                                                                usage=Usage(10, 100)))
    async with make_client(be, retry={"max_attempts": 1}) as c:
        r = await c.judge(req())
    assert r.status == JudgeStatus.TRUNCATED


async def test_truncation_growth_raises_answer_budget_on_retry():
    be = ScriptedBackend(sequence=["truncated", "ok"])
    async with make_client(be, retry={"truncation_growth": 2.0, "max_tokens_cap": 300},
                           defaults={"max_tokens": 200}) as c:
        r = await c.judge(req())
    assert r.ok and [call.max_tokens for call in be.calls] == [200, 300]


async def test_reasoning_budget_is_added_on_top_of_answer_budget():
    be = ScriptedBackend()
    async with make_client(be, defaults={"max_tokens": 100}, backend_configs={"scripted": {"reasoning_budget": 50}}) as c:
        await c.judge(req("a", reasoning=1000))
        await c.judge(req("b", reasoning="off"))
        await c.judge(req("c"))  # default -> backend reasoning_budget
        await c.judge(req("d", reasoning="on"))
    assert [(x.max_tokens, x.reasoning_budget) for x in be.calls] == [(1100, 1000), (100, 0), (150, 50), (150, 50)]
    assert be.calls[1].reasoning.mode == "off"


# ----------------------------------------------------------------------------------------- deadlines / cancellation
async def test_hanging_backend_is_cut_by_attempt_timeout():
    be = ScriptedBackend(sequence=["hang", "ok"])
    async with make_client(be) as c:
        r = await c.judge(req(timeout_s=0.05))
    assert r.ok and r.attempts == 2 and r.attempt_log[0]["status"] == "timeout"


async def test_request_deadline_bounds_retries():
    be = ScriptedBackend(sequence=["hang"])
    async with make_client(be, retry={"max_attempts": 10}) as c:
        r = await asyncio.wait_for(c.judge(req(timeout_s=0.05, deadline_s=0.2)), 5)
    assert r.status == JudgeStatus.TIMEOUT and r.attempts < 10


async def test_cancel_running_and_queued_requests():
    be = ScriptedBackend(latency_s=5.0)
    async with make_client(be, queue={"max_inflight": 1}) as c:
        r1, r2 = req("one"), req("two")
        f1 = await c.submit(r1)
        f2 = await c.submit(r2)
        await asyncio.sleep(0.05)
        assert c.cancel(r2.request_id) and c.cancel(r1.request_id)
        res1, res2 = await asyncio.wait_for(asyncio.gather(f1, f2), 2)
        m = c.metrics()
    assert res1.status == res2.status == JudgeStatus.CANCELLED
    assert m["judge/cancelled"] == 2 and m["judge/total"] == 2


async def test_cancelling_the_future_is_accounted():
    be = ScriptedBackend(latency_s=5.0)
    async with make_client(be) as c:
        f = await c.submit(req())
        await asyncio.sleep(0.02)
        f.cancel()
        for _ in range(50):
            await asyncio.sleep(0.01)
            if c.metrics()["judge/cancelled"] == 1:
                break
        assert c.metrics()["judge/cancelled"] == 1 and c.inflight == 0


async def test_priority_order_and_backpressure():
    order = []
    be = ScriptedBackend(responder=lambda call: (order.append(call.messages[0]["content"]), {"score": 1})[1],
                         latency_s=0.01)
    c = make_client(be, queue={"max_inflight": 1, "max_queue": 3})
    async with c:
        blocker = await c.submit(req("blocker"))
        await asyncio.sleep(0.001)  # the single worker picks up the blocker
        low = await c.submit(req("low", priority=0))
        high = await c.submit(req("high", priority=5))
        mid = await c.submit(req("mid", priority=1))
        # queue is full now (3 queued): the next submit must wait for space
        extra = asyncio.create_task(c.submit(req("extra")))
        await asyncio.sleep(0)
        assert not extra.done() or c.queue_depth <= 3
        await asyncio.gather(blocker, low, high, mid)
        await (await extra)
    assert order[:4] == ["blocker", "high", "mid", "low"]


async def test_backend_exception_is_backend_error_not_crash():
    def boom(call):
        raise RuntimeError("socket closed")

    be = ScriptedBackend(responder=boom)
    async with make_client(be, retry={"max_attempts": 2}) as c:
        r = await c.judge(req())
    assert r.status == JudgeStatus.BACKEND_ERROR and "socket closed" in r.error and r.attempts == 2


async def test_sync_facade_round_trip():
    from judgerl.judge import SyncJudgeClient

    be = ScriptedBackend()
    cfg = {"backends": [{"type": "scripted", "name": "s"}], "breaker": {"enabled": False}}
    sync = await asyncio.to_thread(SyncJudgeClient, cfg)
    try:
        r = await asyncio.to_thread(sync.judge, req())
        many = await asyncio.to_thread(sync.judge_many, [req("a"), req("b")])
        m = await asyncio.to_thread(sync.metrics)
    finally:
        await asyncio.to_thread(sync.close)
    assert r.ok and all(x.ok for x in many) and m["judge/ok"] == 3
    del be
