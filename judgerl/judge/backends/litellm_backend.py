"""Any API provider through LiteLLM (OpenAI, Anthropic, DeepSeek, Gemini, Mistral, ...).

The model string is LiteLLM's (``deepseek/deepseek-chat``, ``anthropic/claude-sonnet-4-5``,
``openai/gpt-4.1-mini``, ``gemini/gemini-2.5-flash``). Request invariants always win over user
``extra_kwargs``: ``stream=False``, ``n=1``, ``num_retries=0`` and ``max_retries=0`` (the judge client
owns retries, so it can count them, honor ``retry-after`` and fail over).
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

from judgerl.judge.backends.base import (BaseBackend, apply_invariants, classify_exception, deep_update,
                                         response_format_for, response_from_openai)
from judgerl.judge.reasoning import infer_style, reasoning_params
from judgerl.judge.types import BackendCall, BackendError, BackendResponse, JudgeStatus


class LiteLLMBackend(BaseBackend):
    def __init__(self, model: str, name: Optional[str] = None, revision: Optional[str] = None,
                 structured: str = "auto", reasoning_style: str = "auto", api_base: Optional[str] = None,
                 api_key: Optional[str] = None, api_key_env: Optional[str] = None,
                 extra_kwargs: Optional[Dict[str, Any]] = None, extra_body: Optional[Dict[str, Any]] = None,
                 strict_schema: bool = False, drop_params: bool = True, keep_raw: bool = True):
        if reasoning_style == "auto":
            reasoning_style = infer_style(model, api_base)
        super().__init__(name=name or model, model=model, revision=revision, structured=structured,
                         reasoning_style=reasoning_style)
        self.api_base = api_base
        self.api_key_env = api_key_env
        self.api_key = api_key or (os.environ.get(api_key_env) if api_key_env else None)
        self.extra_kwargs = dict(extra_kwargs or {})
        self.extra_body = dict(extra_body or {})
        self.strict_schema = strict_schema
        self.drop_params = drop_params
        self.keep_raw = keep_raw
        self._structured_resolved = structured != "auto"

    def config_problems(self) -> list:
        """Problems detectable without a request (used by preflight)."""
        if self.api_key_env and not self.api_key:
            return [f"environment variable {self.api_key_env!r} (api_key_env) is empty or unset"]
        return []

    def _resolve_structured(self) -> None:
        if self._structured_resolved:
            return
        mode = "none"
        try:
            import litellm

            if litellm.supports_response_schema(model=self.model_id):
                mode = "json_schema"
        except Exception:
            mode = "json_schema" if self.api_base else "none"
        self.capabilities.structured = mode
        self._structured_resolved = True

    async def start(self) -> None:
        self._resolve_structured()

    def build_kwargs(self, call: BackendCall) -> Dict[str, Any]:
        """The exact kwargs passed to ``litellm.acompletion`` (exposed for tests and audits)."""
        self._resolve_structured()
        kwargs: Dict[str, Any] = {"drop_params": self.drop_params}
        kwargs.update(self.extra_kwargs)
        kwargs.update(call.extra_params)
        extra_body = deep_update(self.extra_body, kwargs.pop("extra_body", None) or {})
        kwargs["max_tokens"] = call.max_tokens
        for k in ("temperature", "top_p", "seed"):
            v = getattr(call, k)
            if v is not None:
                kwargs[k] = v
        if call.stop:
            kwargs["stop"] = list(call.stop)
        rk, rb = reasoning_params(self.capabilities.reasoning_style, call.reasoning, call.reasoning_budget)
        kwargs.update(rk)
        rf = response_format_for(call, self.capabilities.structured, self.strict_schema)
        extra_body = deep_update(extra_body, rf.pop("extra_body", {}))
        kwargs.update(rf)
        extra_body = deep_update(extra_body, rb)
        if extra_body:
            kwargs["extra_body"] = extra_body
        if self.api_base:
            kwargs["api_base"] = self.api_base
        if self.api_key:
            kwargs["api_key"] = self.api_key
        invariants: Dict[str, Any] = {"model": self.model_id, "messages": call.messages, "stream": False, "n": 1,
                                      "num_retries": 0, "max_retries": 0}
        if call.timeout_s is not None:
            invariants["timeout"] = call.timeout_s
        return apply_invariants(kwargs, invariants, f"litellm[{self.name}]")

    async def generate(self, call: BackendCall) -> BackendResponse:
        problems = self.config_problems()
        if problems:
            raise BackendError(JudgeStatus.AUTH, problems[0], retryable=False)
        import litellm

        kwargs = self.build_kwargs(call)
        try:
            resp = await litellm.acompletion(**kwargs)
        except Exception as e:  # noqa: BLE001 - every SDK error becomes a typed status
            raise classify_exception(e) from e
        out = response_from_openai(resp, keep_raw=self.keep_raw)
        try:
            out.cost_usd = float(litellm.completion_cost(completion_response=resp))
        except Exception:  # unknown price: the telemetry cost model decides
            out.cost_usd = None
        return out

    def describe(self) -> Dict[str, Any]:
        d = super().describe()
        d.update(api_base=self.api_base, api_key_env=self.api_key_env)
        return d
