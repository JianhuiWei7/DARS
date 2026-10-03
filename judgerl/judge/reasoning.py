"""Provider-neutral reasoning control -> provider-specific request parameters.

The request says ``off`` / ``on`` / ``budget N`` / ``default``. The client turns that into a token
budget that is ADDED to the answer budget (so reasoning can never starve the answer), and the
backend's ``reasoning_style`` turns it into request parameters:

==============  ============================================================================
style           parameters
==============  ============================================================================
none            nothing is sent (models that always or never reason); the budget still counts
anthropic       ``thinking={"type": "enabled", "budget_tokens": N}`` (N >= 1024), temperature 1
openai_effort   ``reasoning_effort`` low / medium / high by budget; ``off`` -> ``minimal``
gemini          ``thinking={"type": "enabled", "budget_tokens": N}``; ``off`` -> ``reasoning_effort="disable"``
chat_template   ``extra_body.chat_template_kwargs.enable_thinking`` (Qwen3-style templates on
                vLLM / SGLang servers and the in-process HF backend)
==============  ============================================================================
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from judgerl.judge.types import ReasoningSpec

STYLES = ("auto", "none", "anthropic", "openai_effort", "gemini", "chat_template")
DEFAULT_ON_BUDGET = 4096


def infer_style(model: str, api_base: Optional[str] = None) -> str:
    """Best guess of the reasoning style from a LiteLLM model string."""
    m = (model or "").lower()
    provider, _, name = m.partition("/") if "/" in m else ("", "", m)
    if provider in ("anthropic", "bedrock") and "claude" in name or m.startswith("claude"):
        return "anthropic"
    if provider in ("gemini", "vertex_ai") and "gemini" in name or m.startswith("gemini"):
        return "gemini"
    if provider in ("hosted_vllm", "sglang") or (provider == "openai" and api_base):
        return "chat_template"
    if provider in ("openai", "azure", "") and (name.startswith(("o1", "o3", "o4", "gpt-5"))):
        return "openai_effort"
    return "none"


def reasoning_budget(spec: ReasoningSpec, backend_default: int = 0) -> int:
    """Tokens to reserve for reasoning on top of the answer budget."""
    if spec.mode == "off":
        return 0
    if spec.mode == "budget":
        return int(spec.budget_tokens or 0)
    if spec.mode == "on":
        return backend_default or DEFAULT_ON_BUDGET
    return backend_default  # "default": whatever the backend is configured to assume


def reasoning_params(style: str, spec: ReasoningSpec, budget: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Returns ``(top_level_kwargs, extra_body)`` for the provider request."""
    if spec.mode == "default" or style in ("none", "auto"):
        return {}, {}
    on = spec.mode == "on" or (spec.mode == "budget" and budget > 0)
    if style == "anthropic":
        if not on:
            return {}, {}
        return {"thinking": {"type": "enabled", "budget_tokens": max(1024, budget)}, "temperature": 1.0}, {}
    if style == "openai_effort":
        if not on:
            return {"reasoning_effort": "minimal"}, {}
        effort = "low" if budget <= 2048 else "medium" if budget <= 8192 else "high"
        return {"reasoning_effort": effort}, {}
    if style == "gemini":
        if not on:
            return {"reasoning_effort": "disable"}, {}
        return {"thinking": {"type": "enabled", "budget_tokens": budget}}, {}
    if style == "chat_template":
        return {}, {"chat_template_kwargs": {"enable_thinking": bool(on)}}
    raise ValueError(f"unknown reasoning style {style!r}; expected one of {STYLES}")
