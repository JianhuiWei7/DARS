"""Deterministic backend for tests and dry runs, with fault injection.

Outputs come from (first match wins):
- ``responder(call)``: a sync or async callable returning a str, a dict (JSON-encoded), a
  :class:`BackendResponse`, or an exception instance to raise;
- ``responses``: a list consumed in order (items as for ``responder``);
- ``default_output``.

Faults: ``sequence`` forces the status of attempt k of every request (``["rate_limited", "ok"]``);
otherwise ``faults`` maps a status to a probability, drawn from an RNG seeded by
``(seed, request_id, attempt)`` so a run is reproducible regardless of scheduling order.
"""
from __future__ import annotations

import asyncio
import inspect
import json
import os
import random
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from judgerl.judge.backends.base import BaseBackend
from judgerl.judge.types import BackendCall, BackendError, BackendResponse, JudgeStatus, Usage

FAULTS = ("timeout", "rate_limited", "backend_error", "auth", "quota", "refusal", "empty_content",
          "truncated", "malformed", "semantic_reject", "hang")


class ScriptedBackend(BaseBackend):
    def __init__(self, name: str = "scripted", model: str = "scripted-judge", revision: Optional[str] = None,
                 structured: str = "json_schema", reasoning_style: str = "none",
                 responder: Optional[Callable[[BackendCall], Any]] = None,
                 responses: Optional[Sequence[Any]] = None,
                 default_output: Any = None,
                 faults: Optional[Dict[str, float]] = None,
                 sequence: Optional[Sequence[str]] = None,
                 latency_s: Union[float, Tuple[float, float]] = 0.0,
                 retry_after_s: Optional[float] = 0.0,
                 seed: int = 0,
                 prompt_tokens: Optional[int] = None,
                 completion_tokens: Optional[int] = None,
                 cost_per_call: Optional[float] = 0.0,
                 call_log_path: Optional[str] = None):
        super().__init__(name=name, model=model, revision=revision, structured=structured,
                         reasoning_style=reasoning_style)
        for k in (faults or {}):
            if k not in FAULTS:
                raise ValueError(f"unknown scripted fault {k!r}; known: {FAULTS}")
        if sum((faults or {}).values()) > 1.0 + 1e-9:
            raise ValueError("scripted fault probabilities sum to more than 1")
        self.responder = responder
        self._responses: List[Any] = list(responses or [])
        self.default_output = {"score": 1} if default_output is None else default_output
        self.faults = dict(faults or {})
        self.sequence = list(sequence or [])
        self.latency_s = latency_s
        self.retry_after_s = retry_after_s
        self.seed = seed
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.cost_per_call = cost_per_call
        self.call_log_path = call_log_path
        self.calls: List[BackendCall] = []
        self.fault_counts: Dict[str, int] = {}
        self.started = False
        self.closed = False

    async def start(self) -> None:
        self.started = True

    async def close(self) -> None:
        self.closed = True

    # ------------------------------------------------------------------ fault selection
    def _rng(self, call: BackendCall) -> random.Random:
        return random.Random(f"{self.seed}:{call.request_id}:{call.attempt}")

    def _pick_fault(self, call: BackendCall, rng: random.Random) -> str:
        if self.sequence:
            idx = min(call.attempt - 1, len(self.sequence) - 1)
            return self.sequence[idx]
        u = rng.random()
        acc = 0.0
        for status, p in sorted(self.faults.items()):
            acc += p
            if u < acc:
                return status
        return "ok"

    async def _output(self, call: BackendCall) -> Any:
        if self.responder is not None:
            out = self.responder(call)
            if inspect.isawaitable(out):
                out = await out
            return out
        if self._responses:
            return self._responses.pop(0)
        return self.default_output

    def _usage(self, call: BackendCall, text: str) -> Usage:
        pt = self.prompt_tokens if self.prompt_tokens is not None else sum(
            len(str(m.get("content", ""))) for m in call.messages) // 4 + 1
        ct = self.completion_tokens if self.completion_tokens is not None else len(text) // 4 + 1
        return Usage(prompt_tokens=pt, completion_tokens=ct)

    # ------------------------------------------------------------------ generate
    async def generate(self, call: BackendCall) -> BackendResponse:
        self.calls.append(call)
        if self.call_log_path:
            with open(self.call_log_path, "a") as f:
                f.write(json.dumps({"pid": os.getpid(), "request_id": call.request_id, "attempt": call.attempt}) + "\n")
        rng = self._rng(call)
        lat = self.latency_s
        if isinstance(lat, (tuple, list)):
            lat = rng.uniform(lat[0], lat[1])
        fault = self._pick_fault(call, rng)
        self.fault_counts[fault] = self.fault_counts.get(fault, 0) + 1
        if fault == "hang":  # never returns; the client's per-attempt timeout must cut it
            await asyncio.sleep(3600)
        if lat:
            await asyncio.sleep(lat)
        if fault == "timeout":
            raise BackendError(JudgeStatus.TIMEOUT, "scripted timeout")
        if fault == "rate_limited":
            raise BackendError(JudgeStatus.RATE_LIMITED, "scripted 429", retry_after=self.retry_after_s,
                               http_status=429)
        if fault == "backend_error":
            raise BackendError(JudgeStatus.BACKEND_ERROR, "scripted 503", http_status=503, retryable=True)
        if fault == "auth":
            raise BackendError(JudgeStatus.AUTH, "scripted 401", http_status=401, retryable=False)
        if fault == "quota":
            raise BackendError(JudgeStatus.QUOTA, "scripted insufficient_quota", http_status=429, retryable=False)
        if fault == "refusal":
            return BackendResponse(text="", finish_reason="content_filter", refusal="I can't help with that.",
                                   usage=self._usage(call, ""), model=self.model_id, cost_usd=self.cost_per_call)
        if fault == "empty_content":
            return BackendResponse(text="", finish_reason="stop", usage=self._usage(call, ""), model=self.model_id,
                                   cost_usd=self.cost_per_call)
        if fault == "malformed":
            text = "I think the answer is good but I will not format it {"
            return BackendResponse(text=text, finish_reason="stop", usage=self._usage(call, text),
                                   model=self.model_id, cost_usd=self.cost_per_call)
        if fault == "semantic_reject":
            text = json.dumps({"__scripted_invalid__": True})
            return BackendResponse(text=text, finish_reason="stop", usage=self._usage(call, text),
                                   model=self.model_id, cost_usd=self.cost_per_call)

        out = await self._output(call)
        if isinstance(out, BaseException):
            raise out
        if isinstance(out, BackendResponse):
            if out.cost_usd is None:
                out.cost_usd = self.cost_per_call
            return out
        text = out if isinstance(out, str) else json.dumps(out)
        if fault == "truncated":
            cut = text[: max(1, len(text) // 2)]
            return BackendResponse(text=cut, finish_reason="length", usage=self._usage(call, cut),
                                   model=self.model_id, cost_usd=self.cost_per_call)
        return BackendResponse(text=text, finish_reason="stop", usage=self._usage(call, text), model=self.model_id,
                               cost_usd=self.cost_per_call, raw={"scripted": True, "text": text})
