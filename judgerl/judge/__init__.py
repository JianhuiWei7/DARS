"""Judge control plane: one async client for API, self-hosted, in-process and colocated LLM judges.

Quick start::

    from judgerl.judge import JudgeClient, JudgeRequest

    client = JudgeClient({"backend": {"type": "litellm", "model": "deepseek/deepseek-chat"},
                          "budget_usd": 5, "cache": {"path": "outputs/judge_cache.sqlite"}})
    result = await client.judge(JudgeRequest(messages=[...], output_schema=MySchema))

See ``docs/judge.md``. Nothing here imports torch, transformers or a trainer at import time.
"""
from judgerl.judge.backends.base import BaseBackend, JudgeBackend
from judgerl.judge.backends.registry import register_backend
from judgerl.judge.cache import JudgeCache, cache_key
from judgerl.judge.client import JudgeClient, PreflightError, PreflightReport
from judgerl.judge.clock import Clock, FakeClock
from judgerl.judge.config import BackendConfig, JudgeConfig
from judgerl.judge.session import BarrierReport, JudgeSession, SyncJudgeClient, SyncJudgeSession
from judgerl.judge.telemetry import CostModel, Prices, Telemetry
from judgerl.judge.types import (BackendCall, BackendError, BackendResponse, Decoding, JudgeHalted, JudgeRequest,
                                 JudgeResult, JudgeStatus, ReasoningSpec, Usage)

__all__ = [
    "BackendCall", "BackendConfig", "BackendError", "BackendResponse", "BarrierReport", "BaseBackend", "Clock",
    "CostModel", "Decoding", "FakeClock", "JudgeBackend", "JudgeCache", "JudgeClient", "JudgeConfig", "JudgeHalted",
    "JudgeRequest", "JudgeResult", "JudgeSession", "JudgeStatus", "PreflightError", "PreflightReport", "Prices",
    "ReasoningSpec", "SyncJudgeClient", "SyncJudgeSession", "Telemetry", "Usage", "cache_key", "register_backend",
]
