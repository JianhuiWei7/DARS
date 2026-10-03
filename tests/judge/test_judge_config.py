"""Config loading and backend registry, preflight, telemetry (metrics + JSONL), cost estimation and
the offline batch job."""
from __future__ import annotations

import json

import pytest

from judge_helpers import make_client, req
from judgerl.judge import (BackendConfig, JudgeClient, JudgeConfig, JudgeStatus, PreflightError,
                           register_backend)
from judgerl.judge.backends.batch import BatchJob, run_batch
from judgerl.judge.backends.scripted import ScriptedBackend
from judgerl.judge.telemetry import CostModel, Prices
from judgerl.judge.types import Usage


# ----------------------------------------------------------------------------------------- config
def test_config_from_dict_with_type_specific_options():
    cfg = JudgeConfig.from_dict({
        "backends": [
            {"type": "litellm", "model": "deepseek/deepseek-chat", "api_key_env": "DEEPSEEK_API_KEY",
             "rate_limit": {"rpm": 600, "max_concurrency": 16}, "prices": {"input_per_mtok": 0.27,
                                                                             "output_per_mtok": 1.1}},
            {"type": "hf_server", "model": "Qwen/Qwen3-8B", "engine": "vllm", "gpus": "0",
             "reasoning_budget": 2048},
        ],
        "pools": {"train": ["deepseek/deepseek-chat", "Qwen/Qwen3-8B"]},
        "budget_usd": 25, "retry": {"max_attempts": 3}, "breaker": {"policy": "stop"},
        "cache": {"path": "outputs/c.sqlite"},
    })
    lit, hf = cfg.backends
    assert lit.options == {"api_key_env": "DEEPSEEK_API_KEY"} and lit.rate_limit.rpm == 600
    assert lit.prices.input_per_mtok == 0.27
    assert hf.options == {"engine": "vllm", "gpus": "0"} and hf.reasoning_budget == 2048
    assert cfg.breaker.policy == "stop" and cfg.breaker.min_calls == 50  # global defaults kept
    assert cfg.pool_members("train") == ["deepseek/deepseek-chat", "Qwen/Qwen3-8B"]
    assert cfg.pool_members(None) == ["deepseek/deepseek-chat", "Qwen/Qwen3-8B"]
    kw = hf.constructor_kwargs()
    assert kw == {"name": "Qwen/Qwen3-8B", "model": "Qwen/Qwen3-8B", "engine": "vllm", "gpus": "0"}
    json.dumps(cfg.to_dict())


def test_config_rejects_unknown_keys_and_bad_values():
    with pytest.raises(ValueError, match="unknown JudgeConfig keys"):
        JudgeConfig.from_dict({"backendz": []})
    with pytest.raises(ValueError, match="unknown RetryConfig keys"):
        JudgeConfig.from_dict({"retry": {"attempts": 3}})
    with pytest.raises(ValueError):
        JudgeConfig.from_dict({"breaker": {"policy": "explode"}})
    with pytest.raises(ValueError, match="duplicate backend names"):
        JudgeConfig.from_dict({"backends": [{"type": "scripted"}, {"type": "scripted"}]})
    with pytest.raises(ValueError, match="unknown backends"):
        JudgeConfig.from_dict({"backends": [{"type": "scripted"}], "pools": {"p": ["nope"]}})
    with pytest.raises(ValueError, match="needs a 'type'"):
        BackendConfig.from_dict({"model": "x"})
    with pytest.raises(ValueError, match="bad options"):
        JudgeClient({"backend": {"type": "scripted", "bogus": 1}})
    with pytest.raises(ValueError, match="bad options"):
        JudgeClient({"backend": {"type": "hf_local", "model": "m", "bogus": 1}})
    with pytest.raises(ValueError, match="bad options"):
        JudgeClient({"backend": {"type": "hf_server", "model": "m", "bogus": 1}})


def test_config_from_yaml_with_overrides(tmp_path):
    p = tmp_path / "judge.yaml"
    p.write_text("backend: {type: scripted, name: s, faults: {malformed: 0.1}}\nbudget_usd: 3\n"
                 "defaults: {max_tokens: 256, reasoning: 'off'}\n")
    cfg = JudgeConfig.from_yaml(str(p), overrides={"budget_usd": 4, "retry": {"max_attempts": 2}})
    assert cfg.budget_usd == 4 and cfg.retry.max_attempts == 2 and cfg.defaults.max_tokens == 256
    assert cfg.backends[0].options == {"faults": {"malformed": 0.1}}


async def test_client_builds_registered_backends_from_config():
    @register_backend("my_scripted")
    def factory(name, model=None, **kw):
        return ScriptedBackend(name=name, model=model or "custom", default_output={"score": 9}, **kw)

    c = JudgeClient({"backends": [{"type": "my_scripted", "name": "mine"},
                                  {"type": "scripted", "name": "plain", "structured": "none"}],
                     "breaker": {"enabled": False}})
    async with c:
        r = await c.judge(req())
    assert r.ok and r.parsed == {"score": 9} and r.backend == "mine"
    assert c.backend("plain").capabilities.structured == "none"
    with pytest.raises(KeyError):
        JudgeClient({"backend": {"type": "does_not_exist"}})


# ----------------------------------------------------------------------------------------- preflight
async def test_preflight_reports_each_backend():
    good = ScriptedBackend(name="good", responder=lambda call: {"ok": True})
    denied = ScriptedBackend(name="denied", sequence=["auth"])
    garbled = ScriptedBackend(name="garbled", sequence=["malformed"])
    async with make_client(good, denied, garbled) as c:
        report = await c.preflight()
        with pytest.raises(PreflightError):
            await c.preflight(strict=True)
        m = c.metrics()
    by = {e["backend"]: e for e in report.entries}
    assert by["good"]["ok"] and by["denied"]["status"] == "auth" and by["garbled"]["status"] == "malformed"
    assert not report.ok and len(report.problems) == 2 and "parsing" in report.problems[1]
    assert "FAIL" in str(report) and len(good.calls) == 2  # one real request per backend per preflight
    assert m["judge/tag/purpose=preflight/total"] == 6


async def test_preflight_ok_and_budget_warning():
    be = ScriptedBackend(responder=lambda call: {"ok": True}, cost_per_call=None, model="unpriced-model")
    async with make_client(be, budget_usd=1.0) as c:
        report = await c.preflight(strict=True)
    assert report.ok and any("cost unknown" in w for w in report.warnings)


# ----------------------------------------------------------------------------------------- telemetry
async def test_metrics_keys_windows_and_jsonl(tmp_path):
    path = tmp_path / "judge.jsonl"
    be = ScriptedBackend(sequence=["timeout", "ok"], cost_per_call=0.5)
    async with make_client(be, telemetry={"jsonl_path": str(path), "log_raw": True}) as c:
        await c.judge(req("a", tags={"channel": "process", "domain": "alfworld"}))
        m1 = c.metrics(window=True, reset_window=True)
        await c.judge(req("b", tags={"channel": "outcome"}))
        m2 = c.metrics(window=True)
        total = c.metrics()
    for k in ("judge/ok", "judge/truncated", "judge/cost_usd", "judge/p50_latency_s", "judge/p99_latency_s",
              "judge/retries", "judge/queue_depth", "judge/inflight", "judge/breaker_open"):
        assert k in total, k
    assert m1["judge/total"] == 1 and m2["judge/total"] == 1 and total["judge/total"] == 2
    assert total["judge/retries"] == 2 and total["judge/attempt/timeout"] == 2
    assert total["judge/cost_usd"] == 1.0 and total["judge/tag/channel=process/cost_usd"] == 0.5
    assert total["judge/tag/domain=alfworld/total"] == 1 and total["judge/backend/scripted/ok"] == 2
    lines = [json.loads(x) for x in path.read_text().splitlines()]
    assert len(lines) == 2 and lines[0]["request"]["tags"]["channel"] == "process"
    assert lines[0]["result"]["status"] == "ok" and lines[0]["result"]["raw"]["scripted"] is True
    assert [a["status"] for a in lines[0]["result"]["attempt_log"]] == ["timeout", "ok"]


def test_cost_model_precedence():
    cm = CostModel({"m": Prices(1.0, 2.0)}, use_litellm=False)
    assert cm.cost("m", Usage(1_000_000, 1_000_000), reported=99.0) == (3.0, True)  # configured prices win
    assert cm.cost("other", Usage(10, 10), reported=0.25) == (0.25, True)
    assert cm.cost("other", Usage(10, 10)) == (0.0, False)
    assert cm.cost("other", Usage(0, 0)) == (0.0, True)
    assert cm.estimate("m", 500_000, 0) == 0.5 and cm.estimate("other", 1, 1) is None


def test_cost_from_litellm_tables():
    cm = CostModel()
    usd, known = cm.cost("gpt-4o-mini", Usage(1_000_000, 0))
    assert known and usd > 0


async def test_estimate_cost_prices_quadratic_prefix_judges():
    be = ScriptedBackend()
    steps = ["step %d: " % i + "observation " * 50 for i in range(20)]
    prefix_requests = [req("\n".join(steps[: t + 1])) for t in range(len(steps))]  # one call per step
    single = [req("\n".join(steps))]
    async with make_client(be, backend_configs={"scripted": {"prices": {"input_per_mtok": 1.0}}},
                           defaults={"max_tokens": 50}) as c:
        prefix = c.estimate_cost(prefix_requests)
        once = c.estimate_cost(single)
    assert prefix["price_known"] and prefix["requests"] == 20
    assert prefix["prompt_tokens"] > 9 * once["prompt_tokens"]  # ~n/2 times the prompt volume


# ----------------------------------------------------------------------------------------- batch
async def test_batch_job_writes_jsonl_and_resumes(tmp_path):
    out = tmp_path / "batch.jsonl"
    be = ScriptedBackend(responder=lambda call: "garbage" if "bad" in call.messages[0]["content"] else {"score": 1})
    requests = [req(f"good {i}", request_id=f"g{i}") for i in range(5)] + [req("bad", request_id="b0")]
    client = make_client(be, retry={"max_attempts": 1})
    job = BatchJob(client, str(out), concurrency=3)
    job.extend(requests)
    s1 = await job.run()
    await client.close()
    assert s1.n_run == 6 and s1.by_status == {"ok": 5, "malformed": 1}
    assert len(out.read_text().splitlines()) == 6
    be2 = ScriptedBackend()  # a fixed backend: only the failed request is rerun
    job2 = BatchJob(be2, str(out))
    job2.extend(requests)
    s2 = await job2.run()
    assert s2.n_skipped == 5 and s2.n_run == 1 and s2.by_status == {"ok": 1} and len(be2.calls) == 1


def test_run_batch_sync_helper(tmp_path):
    out = tmp_path / "b.jsonl"
    s = run_batch(ScriptedBackend(), [req(f"x{i}", request_id=f"x{i}") for i in range(4)], str(out))
    assert s.by_status == {"ok": 4} and len(out.read_text().splitlines()) == 4
    assert JudgeStatus.OK.value == "ok"
