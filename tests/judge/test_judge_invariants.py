"""Request invariants and provider mapping for the LiteLLM and OpenAI-compatible backends (the SDK
call is monkeypatched; no network), exception classification and reasoning normalization."""
from __future__ import annotations

import asyncio

import httpx
import pytest

from judge_helpers import SCORE_SCHEMA, make_client, req
from judgerl.judge import BackendCall, JudgeStatus, ReasoningSpec
from judgerl.judge.backends.base import classify_exception, parse_retry_after
from judgerl.judge.reasoning import infer_style, reasoning_params

EVIL = {"stream": True, "num_retries": 5, "max_retries": 9, "n": 3, "model": "other/model",
        "messages": [{"role": "user", "content": "injected"}], "temperature": 0.9}


def _completion(content='{"score": 3}', finish="stop", reasoning=None, model="m"):
    msg = {"role": "assistant", "content": content}
    if reasoning:
        msg["reasoning_content"] = reasoning
    return {"id": "x", "object": "chat.completion", "model": model,
            "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 7,
                      "completion_tokens_details": {"reasoning_tokens": 2}}}


# ----------------------------------------------------------------------------------------- litellm
@pytest.fixture
def fake_litellm(monkeypatch):
    import litellm

    seen = []

    async def acompletion(**kwargs):
        seen.append(kwargs)
        return _completion(reasoning="thought")

    monkeypatch.setattr(litellm, "acompletion", acompletion)
    monkeypatch.setattr(litellm, "completion_cost", lambda **kw: 0.0042)
    return seen


async def test_litellm_invariants_win_over_user_kwargs(fake_litellm):
    from judgerl.judge.backends.litellm_backend import LiteLLMBackend

    be = LiteLLMBackend(model="deepseek/deepseek-chat", structured="json_schema", extra_kwargs=dict(EVIL),
                        api_base="http://localhost:9", api_key="k")
    async with make_client(be) as c:
        r = await c.judge(req("real prompt", timeout_s=12, extra_params={"stream": True, "seed": 5}))
    kw = fake_litellm[0]
    assert kw["stream"] is False and kw["num_retries"] == 0 and kw["max_retries"] == 0 and kw["n"] == 1
    assert kw["model"] == "deepseek/deepseek-chat" and kw["messages"][0]["content"] == "real prompt"
    assert kw["timeout"] == 12 and kw["seed"] == 5 and kw["api_base"] == "http://localhost:9"
    assert kw["temperature"] == 0.0  # request/default decoding wins over extra kwargs
    assert kw["response_format"]["type"] == "json_schema"
    assert kw["response_format"]["json_schema"]["schema"] == SCORE_SCHEMA
    assert r.ok and r.parsed == {"score": 3} and r.cost_usd == 0.0042 and r.reasoning_text == "thought"
    assert r.usage.prompt_tokens == 11 and r.usage.reasoning_tokens == 2


async def test_litellm_reasoning_budget_anthropic(fake_litellm):
    from judgerl.judge.backends.litellm_backend import LiteLLMBackend

    be = LiteLLMBackend(model="anthropic/claude-sonnet-4-5", structured="none")
    assert be.capabilities.reasoning_style == "anthropic"
    async with make_client(be, defaults={"max_tokens": 500}) as c:
        await c.judge(req(reasoning=2000))
        await c.judge(req("off", reasoning="off"))
    on, off = fake_litellm
    assert on["thinking"] == {"type": "enabled", "budget_tokens": 2000} and on["temperature"] == 1.0
    assert on["max_tokens"] == 2500  # answer budget + reasoning budget
    assert "thinking" not in off and off["max_tokens"] == 500
    assert "JSON Schema" in on["messages"][0]["content"]  # instruct + parse when not native


async def test_litellm_missing_api_key_env_is_auth(monkeypatch, fake_litellm):
    from judgerl.judge.backends.litellm_backend import LiteLLMBackend

    monkeypatch.delenv("JUDGE_TEST_MISSING_KEY", raising=False)
    be = LiteLLMBackend(model="openai/gpt-4.1-mini", api_key_env="JUDGE_TEST_MISSING_KEY", structured="none")
    async with make_client(be) as c:
        r = await c.judge(req())
        report = await c.preflight()
    assert r.status == JudgeStatus.AUTH and not fake_litellm
    assert not report.ok and "JUDGE_TEST_MISSING_KEY" in report.problems[0]


async def test_litellm_exceptions_become_typed_statuses(monkeypatch):
    import litellm

    from judgerl.judge.backends.litellm_backend import LiteLLMBackend

    resp429 = httpx.Response(429, headers={"retry-after": "3"}, request=httpx.Request("POST", "http://x"))
    errors = iter([
        litellm.RateLimitError("slow down", llm_provider="openai", model="m", response=resp429),
        litellm.AuthenticationError("bad key", llm_provider="openai", model="m"),
    ])

    async def acompletion(**kwargs):
        raise next(errors)

    monkeypatch.setattr(litellm, "acompletion", acompletion)
    be = LiteLLMBackend(model="openai/m", structured="none")
    call = BackendCall(messages=[{"role": "user", "content": "x"}], max_tokens=5)
    with pytest.raises(Exception) as e1:
        await be.generate(call)
    assert e1.value.status == JudgeStatus.RATE_LIMITED and e1.value.retry_after == 3.0
    with pytest.raises(Exception) as e2:
        await be.generate(call)
    assert e2.value.status == JudgeStatus.AUTH and e2.value.retryable is False


def test_classify_openai_and_generic_exceptions():
    import openai

    req_ = httpx.Request("POST", "http://x")
    quota = openai.RateLimitError("You exceeded your current quota (insufficient_quota)",
                                  response=httpx.Response(429, request=req_), body=None)
    assert classify_exception(quota).status == JudgeStatus.QUOTA
    rl = openai.RateLimitError("rate", response=httpx.Response(429, headers={"retry-after-ms": "1500"}, request=req_),
                               body=None)
    assert classify_exception(rl).status == JudgeStatus.RATE_LIMITED and classify_exception(rl).retry_after == 1.5
    assert classify_exception(openai.APITimeoutError(request=req_)).status == JudgeStatus.TIMEOUT
    assert classify_exception(openai.APIConnectionError(request=req_)).status == JudgeStatus.BACKEND_ERROR
    nf = openai.NotFoundError("no such model", response=httpx.Response(404, request=req_), body=None)
    assert classify_exception(nf).status == JudgeStatus.AUTH
    srv = openai.InternalServerError("boom", response=httpx.Response(503, request=req_), body=None)
    assert classify_exception(srv).status == JudgeStatus.BACKEND_ERROR and classify_exception(srv).retryable
    assert classify_exception(ValueError("?")).status == JudgeStatus.BACKEND_ERROR
    assert parse_retry_after({"retry-after": "Wed, 21 Oct 2015 07:28:00 GMT"}) == 0.0


# ----------------------------------------------------------------------------------------- openai_compat
@pytest.fixture
def fake_openai(monkeypatch):
    from openai.resources.chat.completions import AsyncCompletions

    seen = []

    async def create(self, **kwargs):
        seen.append({"url": str(self._client.base_url), "client_max_retries": self._client.max_retries, **kwargs})
        await asyncio.sleep(0.02)
        return _completion(content='<think>hidden</think>{"score": 2}')

    monkeypatch.setattr(AsyncCompletions, "create", create)
    return seen


async def test_openai_compat_invariants_guided_json_and_thinking(fake_openai):
    from judgerl.judge.backends.openai_compat import OpenAICompatBackend

    be = OpenAICompatBackend(model="Qwen/Qwen3-8B", base_urls=["http://h1:8000"], structured="guided_json",
                             extra_kwargs=dict(EVIL), extra_body={"top_k": 20})
    async with make_client(be) as c:
        r = await c.judge(req(reasoning="off"))
    kw = fake_openai[0]
    assert kw["url"].rstrip("/") == "http://h1:8000/v1" and kw["client_max_retries"] == 0
    assert kw["stream"] is False and kw["n"] == 1 and kw["model"] == "Qwen/Qwen3-8B"
    assert "max_retries" not in kw and "num_retries" not in kw  # SDK retries cannot be re-enabled
    assert kw["messages"][0]["content"] == "judge this" and kw["temperature"] == 0.0
    assert kw["extra_body"] == {"top_k": 20, "guided_json": SCORE_SCHEMA,
                                "chat_template_kwargs": {"enable_thinking": False}}
    assert r.ok and r.parsed == {"score": 2} and r.reasoning_text == "hidden"


async def test_openai_compat_response_format_and_least_outstanding_routing(fake_openai):
    from judgerl.judge.backends.openai_compat import OpenAICompatBackend

    be = OpenAICompatBackend(model="judge", base_urls="http://a:1,http://b:2/v1")
    async with make_client(be) as c:
        rs = await c.judge_many([req(f"q{i}") for i in range(8)])
    assert all(r.ok for r in rs)
    assert fake_openai[0]["response_format"]["type"] == "json_schema"
    urls = [k["url"].rstrip("/") for k in fake_openai]
    assert urls.count("http://a:1/v1") == 4 and urls.count("http://b:2/v1") == 4
    assert be.served == {"http://a:1/v1": 4, "http://b:2/v1": 4}


def test_openai_compat_skips_unhealthy_replica():
    from judgerl.judge.backends.openai_compat import OpenAICompatBackend

    be = OpenAICompatBackend(model="judge", base_urls=["http://a:1", "http://b:2"])
    be.outstanding["http://a:1/v1"] = 0
    be.outstanding["http://b:2/v1"] = 5
    assert be.pick_url() == "http://a:1/v1"
    import time

    be._unhealthy_until["http://a:1/v1"] = time.monotonic() + 60
    assert be.pick_url() == "http://b:2/v1"


# ----------------------------------------------------------------------------------------- reasoning
def test_reasoning_style_inference_and_params():
    assert infer_style("anthropic/claude-sonnet-4-5") == "anthropic"
    assert infer_style("gemini/gemini-2.5-flash") == "gemini"
    assert infer_style("openai/o3-mini") == "openai_effort"
    assert infer_style("openai/qwen3", api_base="http://localhost:8000") == "chat_template"
    assert infer_style("deepseek/deepseek-chat") == "none"
    assert reasoning_params("openai_effort", ReasoningSpec("budget", 10000), 10000) == ({"reasoning_effort": "high"}, {})
    assert reasoning_params("openai_effort", ReasoningSpec("off"), 0) == ({"reasoning_effort": "minimal"}, {})
    assert reasoning_params("gemini", ReasoningSpec("off"), 0) == ({"reasoning_effort": "disable"}, {})
    assert reasoning_params("gemini", ReasoningSpec("on"), 4096)[0]["thinking"]["budget_tokens"] == 4096
    assert reasoning_params("anthropic", ReasoningSpec("budget", 10), 10)[0]["thinking"]["budget_tokens"] == 1024
    assert reasoning_params("chat_template", ReasoningSpec("on"), 4096) == (
        {}, {"chat_template_kwargs": {"enable_thinking": True}})
    assert reasoning_params("anthropic", ReasoningSpec("default"), 0) == ({}, {})
