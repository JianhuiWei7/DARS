"""Self-hosted OpenAI-compatible servers (vLLM, SGLang, ...), one or many replicas.

Requests are routed to the replica with the fewest outstanding requests (ties rotate); a replica
that fails at the connection level is skipped for ``unhealthy_cooldown_s``. Structured output uses
``response_format`` with a JSON schema (``structured: json_schema``, supported by vLLM and SGLang)
or vLLM's ``extra_body.guided_json`` (``structured: guided_json``). Reasoning uses the chat template
flag ``enable_thinking`` by default (Qwen3-style models).
"""
from __future__ import annotations

import itertools
import os
import time
from typing import Any, Dict, List, Optional, Sequence, Union

from judgerl.judge.backends.base import (BaseBackend, apply_invariants, log, classify_exception, deep_update,
                                         response_format_for, response_from_openai)
from judgerl.judge.reasoning import reasoning_params
from judgerl.judge.types import BackendCall, BackendResponse, JudgeStatus


_SDK_ONLY = ("max_retries", "num_retries", "drop_params")


def _normalize_url(u: str) -> str:
    u = u.rstrip("/")
    return u if u.endswith("/v1") else u + "/v1"


class OpenAICompatBackend(BaseBackend):
    def __init__(self, model: str, base_urls: Union[str, Sequence[str], None] = None, base_url: Optional[str] = None,
                 name: Optional[str] = None, revision: Optional[str] = None, structured: str = "json_schema",
                 reasoning_style: str = "chat_template", api_key: Optional[str] = None,
                 api_key_env: Optional[str] = None, served_model_name: Optional[str] = None,
                 extra_kwargs: Optional[Dict[str, Any]] = None, extra_body: Optional[Dict[str, Any]] = None,
                 strict_schema: bool = False, keep_raw: bool = True, unhealthy_cooldown_s: float = 10.0,
                 default_headers: Optional[Dict[str, str]] = None):
        if structured == "auto":
            structured = "json_schema"
        super().__init__(name=name or model, model=model, revision=revision, structured=structured,
                         reasoning_style=reasoning_style)
        urls: List[str] = []
        if base_url:
            urls.append(base_url)
        if isinstance(base_urls, str):
            urls.extend(u for u in base_urls.split(",") if u.strip())
        elif base_urls:
            urls.extend(base_urls)
        if not urls:
            raise ValueError("openai_compat backend needs base_url or base_urls")
        self.base_urls = [_normalize_url(u.strip()) for u in urls]
        self.served_model_name = served_model_name or model
        self.api_key = api_key or (os.environ.get(api_key_env) if api_key_env else None) or "EMPTY"
        self.extra_kwargs = dict(extra_kwargs or {})
        self.extra_body = dict(extra_body or {})
        self.strict_schema = strict_schema
        self.keep_raw = keep_raw
        self.unhealthy_cooldown_s = unhealthy_cooldown_s
        self.default_headers = default_headers
        self.outstanding: Dict[str, int] = {u: 0 for u in self.base_urls}
        self.served: Dict[str, int] = {u: 0 for u in self.base_urls}
        self._unhealthy_until: Dict[str, float] = {u: 0.0 for u in self.base_urls}
        self._rr = itertools.count()
        self._clients: Dict[str, Any] = {}

    # ------------------------------------------------------------------ routing
    def pick_url(self) -> str:
        """Least outstanding requests among healthy replicas (all replicas if none is healthy)."""
        now = time.monotonic()
        healthy = [u for u in self.base_urls if self._unhealthy_until[u] <= now] or list(self.base_urls)
        low = min(self.outstanding[u] for u in healthy)
        cands = [u for u in healthy if self.outstanding[u] == low]
        return cands[next(self._rr) % len(cands)]

    def _client(self, url: str):
        c = self._clients.get(url)
        if c is None:
            import openai

            kw: Dict[str, Any] = {"base_url": url, "api_key": self.api_key, "max_retries": 0}
            if self.default_headers:
                kw["default_headers"] = self.default_headers
            c = openai.AsyncOpenAI(**kw)
            self._clients[url] = c
        return c

    def build_kwargs(self, call: BackendCall) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = dict(self.extra_kwargs)
        kwargs.update(call.extra_params)
        for k in _SDK_ONLY:  # retries belong to the judge client; the SDK client is built with max_retries=0
            if k in kwargs:
                log.warning("openai_compat[%s]: dropping %s=%r (retries are owned by the judge client)",
                            self.name, k, kwargs.pop(k))
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
        invariants: Dict[str, Any] = {"model": self.served_model_name, "messages": call.messages, "stream": False,
                                      "n": 1}
        if call.timeout_s is not None:
            invariants["timeout"] = call.timeout_s
        return apply_invariants(kwargs, invariants, f"openai_compat[{self.name}]")

    async def generate(self, call: BackendCall) -> BackendResponse:
        url = self.pick_url()
        kwargs = self.build_kwargs(call)
        client = self._client(url)
        self.outstanding[url] += 1
        try:
            resp = await client.chat.completions.create(**kwargs)
        except Exception as e:  # noqa: BLE001
            err = classify_exception(e)
            if err.status in (JudgeStatus.BACKEND_ERROR, JudgeStatus.TIMEOUT) and err.http_status is None:
                self._unhealthy_until[url] = time.monotonic() + self.unhealthy_cooldown_s
            raise err from e
        finally:
            self.outstanding[url] -= 1
        self.served[url] += 1
        out = response_from_openai(resp, keep_raw=self.keep_raw)
        if out.raw is not None:
            out.raw.setdefault("_base_url", url)
        return out

    async def close(self) -> None:
        for c in self._clients.values():
            close = getattr(c, "close", None)
            if close is not None:
                try:
                    await close()
                except Exception:  # pragma: no cover - best effort
                    pass
        self._clients.clear()

    def describe(self) -> Dict[str, Any]:
        d = super().describe()
        d.update(base_urls=self.base_urls, served_model_name=self.served_model_name, served=dict(self.served))
        return d
