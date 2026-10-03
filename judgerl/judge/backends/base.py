"""The async backend protocol and helpers shared by the OpenAI-shaped backends."""
from __future__ import annotations

import email.utils
import logging
import time
from typing import Any, Dict, Mapping, Optional, Protocol, runtime_checkable

from judgerl.judge.schema import split_reasoning
from judgerl.judge.types import (BackendCall, BackendCapabilities, BackendError, BackendResponse, JudgeStatus,
                                 Usage)

log = logging.getLogger("judgerl.judge")


@runtime_checkable
class JudgeBackend(Protocol):
    """What the client needs from a backend.

    ``generate`` performs exactly ONE attempt (no internal retries) and either returns a
    :class:`BackendResponse` or raises :class:`BackendError` with a typed status; any other exception
    is treated as ``backend_error``. ``model_id`` and ``revision`` enter the cache key.
    """

    name: str
    model_id: str
    revision: Optional[str]
    capabilities: BackendCapabilities

    async def start(self) -> None: ...

    async def generate(self, call: BackendCall) -> BackendResponse: ...

    async def close(self) -> None: ...


class BaseBackend:
    """Convenience base class with no-op lifecycle hooks."""

    name: str = "backend"
    model_id: str = "unknown"
    revision: Optional[str] = None

    def __init__(self, name: Optional[str] = None, model: Optional[str] = None, revision: Optional[str] = None,
                 structured: str = "none", reasoning_style: str = "none"):
        if name:
            self.name = name
        if model:
            self.model_id = model
        self.revision = revision
        self.capabilities = BackendCapabilities(structured=structured, reasoning_style=reasoning_style)

    async def start(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def generate(self, call: BackendCall) -> BackendResponse:  # pragma: no cover - abstract
        raise NotImplementedError

    def describe(self) -> Dict[str, Any]:
        return {"name": self.name, "type": type(self).__name__, "model": self.model_id, "revision": self.revision,
                "structured": self.capabilities.structured, "reasoning_style": self.capabilities.reasoning_style}


# ----------------------------------------------------------------------------------------- invariants
_warned: set = set()


def apply_invariants(user: Mapping[str, Any], invariants: Mapping[str, Any], where: str) -> Dict[str, Any]:
    """User kwargs updated with invariants that must win (no streaming, SDK retries off, ...)."""
    out = dict(user)
    for k, v in invariants.items():
        if k in out and out[k] != v and (where, k) not in _warned:
            _warned.add((where, k))
            log.warning("%s: ignoring user setting %s=%r (the judge client requires %r)", where, k, out[k], v)
        out[k] = v
    return out


def deep_update(base: Dict[str, Any], extra: Mapping[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in extra.items():
        if isinstance(v, Mapping) and isinstance(out.get(k), Mapping):
            out[k] = deep_update(dict(out[k]), v)
        else:
            out[k] = v
    return out


# ----------------------------------------------------------------------------------------- errors
_BY_CLASS = {
    "ContentPolicyViolationError": (JudgeStatus.REFUSAL, False),
    "ContextWindowExceededError": (JudgeStatus.BACKEND_ERROR, False),
    "BudgetExceededError": (JudgeStatus.QUOTA, False),
    "AuthenticationError": (JudgeStatus.AUTH, False),
    "PermissionDeniedError": (JudgeStatus.AUTH, False),
    "NotFoundError": (JudgeStatus.AUTH, False),
    "RateLimitError": (JudgeStatus.RATE_LIMITED, True),
    "Timeout": (JudgeStatus.TIMEOUT, True),
    "APITimeoutError": (JudgeStatus.TIMEOUT, True),
    "TimeoutError": (JudgeStatus.TIMEOUT, True),
    "ReadTimeout": (JudgeStatus.TIMEOUT, True),
    "ConnectTimeout": (JudgeStatus.TIMEOUT, True),
    "UnsupportedParamsError": (JudgeStatus.BACKEND_ERROR, False),
    "UnprocessableEntityError": (JudgeStatus.BACKEND_ERROR, False),
    "BadRequestError": (JudgeStatus.BACKEND_ERROR, False),
    "ServiceUnavailableError": (JudgeStatus.BACKEND_ERROR, True),
    "InternalServerError": (JudgeStatus.BACKEND_ERROR, True),
    "APIConnectionError": (JudgeStatus.BACKEND_ERROR, True),
    "ConnectError": (JudgeStatus.BACKEND_ERROR, True),
    "APIError": (JudgeStatus.BACKEND_ERROR, True),
}
_QUOTA_WORDS = ("insufficient_quota", "quota", "insufficient balance", "billing", "credit", "payment required")


def parse_retry_after(headers: Any) -> Optional[float]:
    if not headers:
        return None
    try:
        get = headers.get
    except AttributeError:
        return None
    ms = get("retry-after-ms")
    if ms:
        try:
            return float(ms) / 1000.0
        except ValueError:
            pass
    v = get("retry-after")
    if not v:
        return None
    try:
        return max(0.0, float(v))
    except ValueError:
        try:
            dt = email.utils.parsedate_to_datetime(v)
            return max(0.0, dt.timestamp() - time.time())
        except (TypeError, ValueError):
            return None


def classify_exception(e: BaseException) -> BackendError:
    """Map an SDK exception (LiteLLM, OpenAI, httpx) to a typed :class:`BackendError`."""
    if isinstance(e, BackendError):
        return e
    msg = f"{type(e).__name__}: {e}"[:500]
    code = getattr(e, "status_code", None)
    response = getattr(e, "response", None)
    headers = getattr(e, "litellm_response_headers", None) or getattr(response, "headers", None)
    retry_after = parse_retry_after(headers)
    status, retryable = None, None
    for cls in type(e).__mro__:
        if cls.__name__ in _BY_CLASS:
            status, retryable = _BY_CLASS[cls.__name__]
            break
    if status is None and isinstance(code, int):
        if code in (401, 403, 404):
            status, retryable = JudgeStatus.AUTH, False
        elif code == 402:
            status, retryable = JudgeStatus.QUOTA, False
        elif code == 408:
            status, retryable = JudgeStatus.TIMEOUT, True
        elif code == 429:
            status, retryable = JudgeStatus.RATE_LIMITED, True
        elif code >= 500:
            status, retryable = JudgeStatus.BACKEND_ERROR, True
        elif code >= 400:
            status, retryable = JudgeStatus.BACKEND_ERROR, False
    if status is None:
        status, retryable = JudgeStatus.BACKEND_ERROR, True
    if status == JudgeStatus.RATE_LIMITED and (code == 402 or any(w in str(e).lower() for w in _QUOTA_WORDS)):
        status, retryable = JudgeStatus.QUOTA, False
    return BackendError(status, msg, retry_after=retry_after, http_status=code, retryable=retryable)


# ----------------------------------------------------------------------------------------- responses
def _to_dict(obj: Any) -> Optional[Dict[str, Any]]:
    for attr in ("model_dump", "dict", "to_dict"):
        f = getattr(obj, attr, None)
        if callable(f):
            try:
                d = f()
                if isinstance(d, dict):
                    return d
            except Exception:  # pragma: no cover - best effort provenance
                continue
    return obj if isinstance(obj, dict) else None


def _get(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    v = getattr(obj, name, default)
    if v is default:
        extra = getattr(obj, "model_extra", None) or {}
        v = extra.get(name, default) if isinstance(extra, Mapping) else default
    return v


def response_from_openai(resp: Any, strip_think: bool = True, keep_raw: bool = True) -> BackendResponse:
    """Normalize an OpenAI-shaped chat completion (OpenAI SDK object, LiteLLM ModelResponse or dict)."""
    choices = _get(resp, "choices") or []
    if not choices:
        raise BackendError(JudgeStatus.EMPTY_CONTENT, "response has no choices")
    choice = choices[0]
    msg = _get(choice, "message") or {}
    content = _get(msg, "content")
    if isinstance(content, list):  # content parts
        content = "".join(str(_get(p, "text", "") or "") for p in content)
    reasoning = _get(msg, "reasoning_content") or _get(msg, "reasoning")
    if strip_think and content:
        content, think = split_reasoning(content)
        reasoning = reasoning or think
    usage_obj = _get(resp, "usage")
    details = _get(usage_obj, "completion_tokens_details") if usage_obj is not None else None
    usage = Usage(
        prompt_tokens=int(_get(usage_obj, "prompt_tokens", 0) or 0) if usage_obj is not None else 0,
        completion_tokens=int(_get(usage_obj, "completion_tokens", 0) or 0) if usage_obj is not None else 0,
        reasoning_tokens=int(_get(details, "reasoning_tokens", 0) or 0) if details is not None else 0,
    )
    return BackendResponse(
        text=content, finish_reason=_get(choice, "finish_reason"), usage=usage, model=_get(resp, "model"),
        reasoning_text=reasoning if isinstance(reasoning, str) else None, refusal=_get(msg, "refusal"),
        raw=_to_dict(resp) if keep_raw else None,
    )


def response_format_for(call: BackendCall, mode: str, strict: bool = False) -> Dict[str, Any]:
    """Request fields that enforce ``call.json_schema`` natively for the given structured mode."""
    if call.json_schema is None or mode == "none":
        return {}
    if mode == "json_schema":
        return {"response_format": {"type": "json_schema", "json_schema": {
            "name": call.schema_name, "schema": call.json_schema, "strict": strict}}}
    if mode == "guided_json":
        return {"extra_body": {"guided_json": call.json_schema}}
    if mode == "json_object":
        return {"response_format": {"type": "json_object"}}
    raise ValueError(f"unknown structured-output mode {mode!r}")
