"""Rate limiting (fake clock), concurrency caps, circuit breakers (pause / stop) and budgets."""
from __future__ import annotations

import asyncio
import time

import pytest

from judge_helpers import make_client, req
from judgerl.judge import FakeClock, JudgeHalted, JudgeStatus
from judgerl.judge.backends.scripted import ScriptedBackend
from judgerl.judge.breaker import BreakerSettings, CircuitBreaker
from judgerl.judge.ratelimit import RateLimiter, TokenBucket


# ----------------------------------------------------------------------------------------- token buckets
def test_token_bucket_refill_with_fake_clock():
    clock = FakeClock()
    b = TokenBucket(rate_per_s=2.0, capacity=4.0, clock=clock)
    assert b.wait_time(4) == 0
    b.take(4)
    assert b.wait_time(1) == pytest.approx(0.5)
    clock.advance(1.0)
    assert b.wait_time(2) == 0
    assert b.wait_time(100) == pytest.approx(1.0)  # above capacity: wait for a full bucket


async def test_rpm_limit_spaces_requests():
    clock = FakeClock()
    lim = RateLimiter(rpm=60, clock=clock)  # 1 request per second, burst 1
    for _ in range(5):
        adm = await lim.acquire(0)
        lim.release(adm)
    assert clock.now() == pytest.approx(4.0)


async def test_tpm_limit_and_reconciliation():
    clock = FakeClock()
    lim = RateLimiter(tpm=600, clock=clock, burst_s=1.0)  # 10 tokens/s, capacity 100
    adm = await lim.acquire(100)
    lim.release(adm, actual_tokens=20)  # 80 refunded
    t0 = clock.now()
    adm = await lim.acquire(80)
    assert clock.now() == t0  # refund made it immediately available
    lim.release(adm, actual_tokens=80)
    adm = await lim.acquire(50)  # bucket empty: 50 tokens at 10 tokens/s
    assert clock.now() - t0 == pytest.approx(5.0)
    lim.release(adm, 50)


async def test_rate_limiter_timeout_returns_none():
    clock = FakeClock()
    lim = RateLimiter(rpm=1, clock=clock)
    lim.release(await lim.acquire(0))
    assert await lim.acquire(0, timeout=1.0) is None


async def test_retry_after_pauses_the_backend():
    clock = FakeClock()
    lim = RateLimiter(rpm=6000, clock=clock)
    lim.pause_for(5.0)
    lim.release(await lim.acquire(0))
    assert clock.now() >= 5.0


async def test_client_rpm_limit_real_clock():
    be = ScriptedBackend()
    rl = {"rpm": 1200, "burst_s": 0.05}  # 20/s, bucket of one request
    async with make_client(be, backend_configs={"scripted": {"rate_limit": rl}}) as c:
        t0 = time.monotonic()
        rs = await c.judge_many([req(f"q{i}") for i in range(6)])
        dt = time.monotonic() - t0
    assert all(r.ok for r in rs) and dt >= 0.2


async def test_client_max_concurrency_per_backend():
    active = {"now": 0, "peak": 0}

    async def responder(call):
        active["now"] += 1
        active["peak"] = max(active["peak"], active["now"])
        await asyncio.sleep(0.01)
        active["now"] -= 1
        return {"score": 1}

    be = ScriptedBackend(responder=responder)
    async with make_client(be, backend_configs={"scripted": {"rate_limit": {"max_concurrency": 3}}}) as c:
        rs = await c.judge_many([req(f"q{i}") for i in range(30)])
    assert all(r.ok for r in rs) and active["peak"] == 3


# ----------------------------------------------------------------------------------------- breaker unit
def test_breaker_consecutive_failures_and_half_open_recovery():
    clock = FakeClock()
    br = CircuitBreaker("b", BreakerSettings(consecutive_failures=3, failure_rate=None, cooldown_s=10), clock)
    for _ in range(3):
        assert br.allow()
        br.record(JudgeStatus.BACKEND_ERROR)
    assert br.state == "open" and not br.allow() and br.wait_time() == pytest.approx(10)
    clock.advance(10)
    assert br.allow() and br.state == "half_open" and not br.allow()  # one probe only
    br.record(JudgeStatus.OK)
    assert br.state == "closed"


def test_breaker_failure_rate_window_and_ignored_statuses():
    clock = FakeClock()
    br = CircuitBreaker("b", BreakerSettings(window_s=60, min_calls=10, failure_rate=0.5, consecutive_failures=None),
                        clock)
    for i in range(9):
        br.record(JudgeStatus.OK if i % 2 else JudgeStatus.TIMEOUT)
    assert br.state == "closed"  # below min_calls
    br.record(JudgeStatus.MALFORMED)
    assert br.state == "open" and "failure rate" in br.reason
    br2 = CircuitBreaker("b2", BreakerSettings(consecutive_failures=2), clock)
    for _ in range(5):
        br2.record(JudgeStatus.CANCELLED)
    assert br2.state == "closed"
    clock.advance(0)
    br3 = CircuitBreaker("b3", BreakerSettings(window_s=1, min_calls=2, failure_rate=0.5, consecutive_failures=None),
                         clock)
    br3.record(JudgeStatus.TIMEOUT)
    clock.advance(5)  # old failure leaves the window
    br3.record(JudgeStatus.OK)
    br3.record(JudgeStatus.OK)
    assert br3.state == "closed"


def test_breaker_fatal_status_and_stop_policy_is_permanent():
    clock = FakeClock()
    br = CircuitBreaker("b", BreakerSettings(policy="stop", cooldown_s=1), clock)
    br.record(JudgeStatus.AUTH)
    clock.advance(100)
    assert br.state == "open" and br.permanent and br.wait_time() is None and not br.allow()


def test_breaker_cost_budget_trip_never_recovers():
    clock = FakeClock()
    br = CircuitBreaker("b", BreakerSettings(cost_budget_usd=1.0, cooldown_s=1), clock)
    br.add_cost(0.6)
    assert br.state == "closed"
    br.add_cost(0.5)
    clock.advance(100)
    assert br.permanent and not br.allow()


# ----------------------------------------------------------------------------------------- breaker in the client
async def test_backend_breaker_pause_fails_over_then_recovers():
    clock_cooldown = 0.2
    flaky = ScriptedBackend(name="flaky", sequence=["backend_error"])
    backup = ScriptedBackend(name="backup")
    c = make_client(flaky, backup, retry={"max_attempts": 2, "attempts_per_backend": 1},
                    backend_configs={"flaky": {"breaker": {"consecutive_failures": 2, "failure_rate": None,
                                                           "cooldown_s": clock_cooldown}}})
    async with c:
        for i in range(4):
            assert (await c.judge(req(f"q{i}"))).backend == "backup"
        assert len(flaky.calls) == 2  # breaker opened after two failures: no more calls to flaky
        assert c.slots["flaky"].breaker.state == "open"
        flaky.sequence = ["ok"]
        await asyncio.sleep(clock_cooldown + 0.05)
        r = await c.judge(req("after cooldown"))
        assert r.backend == "flaky" and c.slots["flaky"].breaker.state == "closed"


async def test_backend_breaker_stop_without_alternative_gives_circuit_open():
    be = ScriptedBackend(sequence=["backend_error"])
    c = make_client(be, retry={"max_attempts": 1},
                    backend_configs={"scripted": {"breaker": {"consecutive_failures": 2, "failure_rate": None,
                                                              "policy": "stop"}}})
    async with c:
        s = [(await c.judge(req(f"q{i}"))).status for i in range(4)]
    assert s == [JudgeStatus.BACKEND_ERROR, JudgeStatus.BACKEND_ERROR, JudgeStatus.CIRCUIT_OPEN,
                 JudgeStatus.CIRCUIT_OPEN]
    assert len(be.calls) == 2


async def test_global_breaker_pause_holds_queue_until_cooldown():
    be = ScriptedBackend(sequence=["malformed"])
    c = make_client(be, retry={"max_attempts": 1},
                    breaker={"enabled": True, "consecutive_failures": 3, "failure_rate": None, "cooldown_s": 0.3,
                             "policy": "pause"})
    async with c:
        rs = await c.judge_many([req(f"q{i}") for i in range(3)])
        assert all(r.status == JudgeStatus.MALFORMED for r in rs)
        assert c.global_breaker.state == "open"
        be.sequence = ["ok"]
        t0 = time.monotonic()
        r = await c.judge(req("waits"))
        waited = time.monotonic() - t0
    assert r.ok and waited >= 0.2  # dispatch paused (request kept) until the half-open probe


async def test_global_breaker_stop_halts_run():
    be = ScriptedBackend(sequence=["timeout"], latency_s=0.001)
    c = make_client(be, retry={"max_attempts": 1}, queue={"max_inflight": 1},
                    breaker={"enabled": True, "consecutive_failures": 5, "failure_rate": None, "policy": "stop"})
    async with c:
        futs = [await c.submit(req(f"q{i}")) for i in range(20)]
        done = await asyncio.gather(*futs, return_exceptions=True)
        with pytest.raises(JudgeHalted):
            await c.submit(req("late"))
        m = c.metrics()
    statuses = [d.status for d in done if not isinstance(d, BaseException)]
    halted = [d for d in done if isinstance(d, JudgeHalted)]
    assert statuses == [JudgeStatus.TIMEOUT] * 5 and len(halted) == 15
    assert len(be.calls) == 5  # dispatch stopped within the batch
    assert m["judge/total"] == 20 and m["judge/timeout"] == 5 and m["judge/circuit_open"] == 15
    assert m["judge/halted"] == 1


async def test_budget_stop_with_known_prices_never_overruns():
    # actual completion = the cap, so each call costs about its reserved worst case
    be = ScriptedBackend(completion_tokens=100, cost_per_call=None)
    prices = {"input_per_mtok": 1.0, "output_per_mtok": 1.0}
    c = make_client(be, budget_usd=0.0005, defaults={"max_tokens": 100},
                    backend_configs={"scripted": {"prices": prices}}, queue={"max_inflight": 4})
    async with c:
        futs = [await c.submit(req(f"q{i}")) for i in range(40)]
        done = await asyncio.gather(*futs, return_exceptions=True)
        m = c.metrics()
    ok = [d for d in done if not isinstance(d, BaseException) and d.ok]
    halted = [d for d in done if isinstance(d, JudgeHalted)]
    assert len(ok) + len(halted) == 40 and halted and halted[0].status == JudgeStatus.BUDGET_EXCEEDED
    assert c.spent_usd <= 0.0005 and len(ok) == 4 and len(be.calls) == 4
    assert m["judge/cost_usd"] == pytest.approx(c.spent_usd)
    assert m["judge/budget_exceeded"] == len(halted) == 36


async def test_budget_estimate_calibrates_to_observed_prompt_tokens():
    # the provider reports far more prompt tokens than the character heuristic predicts
    be = ScriptedBackend(prompt_tokens=1000, completion_tokens=100, cost_per_call=None)
    prices = {"input_per_mtok": 1.0, "output_per_mtok": 1.0}
    c = make_client(be, budget_usd=0.0105, defaults={"max_tokens": 100},
                    backend_configs={"scripted": {"prices": prices}}, queue={"max_inflight": 1})
    async with c:
        futs = [await c.submit(req(f"q{i}")) for i in range(40)]
        done = await asyncio.gather(*futs, return_exceptions=True)
    ok = [d for d in done if not isinstance(d, BaseException) and d.ok]
    assert len(ok) == 9 and c.spent_usd == pytest.approx(0.0099) and c.spent_usd <= 0.0105
    assert c.slots["scripted"].prompt_ratio > 50


async def test_budget_pause_returns_budget_exceeded_results():
    be = ScriptedBackend(cost_per_call=0.4)
    c = make_client(be, budget_usd=1.0, budget_policy="pause", queue={"max_inflight": 1})
    async with c:
        rs = await c.judge_many([req(f"q{i}") for i in range(5)])
    # prices unknown before the call: budget enforced post hoc (3 calls reach $1.2, then refused)
    assert [r.status for r in rs] == [JudgeStatus.OK] * 3 + [JudgeStatus.BUDGET_EXCEEDED] * 2
    assert len(be.calls) == 3


async def test_per_backend_cost_budget_fails_over():
    pricey = ScriptedBackend(name="pricey", cost_per_call=1.0)
    cheap = ScriptedBackend(name="cheap", cost_per_call=0.0)
    c = make_client(pricey, cheap, backend_configs={"pricey": {"breaker": {"enabled": True, "cost_budget_usd": 2.0,
                                                                           "failure_rate": None}}})
    async with c:
        rs = [await c.judge(req(f"q{i}")) for i in range(4)]
    assert [r.backend for r in rs] == ["pricey", "pricey", "cheap", "cheap"]
