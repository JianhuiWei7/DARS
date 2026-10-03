"""Contracts of the judge control plane: requests, results, statuses and backend call records.

Everything here is plain dataclasses so records are cheap to build (tens of thousands per training
step) and serialize to JSON with :meth:`to_dict`. Nothing in this module imports a provider SDK.
"""
from __future__ import annotations

import enum
import uuid
from dataclasses import asdict, dataclass, field, fields, replace
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Union

Message = Dict[str, Any]  # OpenAI-style chat message: {"role": ..., "content": ...}


class JudgeStatus(str, enum.Enum):
    """Final (or per-attempt) outcome of a judge request. Every non-``ok`` result is counted."""

    OK = "ok"
    TIMEOUT = "timeout"                  # attempt or request deadline elapsed
    RATE_LIMITED = "rate_limited"        # provider 429 (retry-after honored)
    REFUSAL = "refusal"                  # model refused / content filter
    EMPTY_CONTENT = "empty_content"      # HTTP success but no answer text
    TRUNCATED = "truncated"              # hit the token cap before a complete answer
    MALFORMED = "malformed"              # no parsable JSON in the answer
    SEMANTIC_REJECT = "semantic_reject"  # parsed but failed the schema or the request validator
    AUTH = "auth"                        # bad key / permission / unknown model
    QUOTA = "quota"                      # account out of credit or quota
    BACKEND_ERROR = "backend_error"      # 5xx, connection error, unexpected exception
    CANCELLED = "cancelled"              # cancelled by the caller or at a watermark barrier
    BUDGET_EXCEEDED = "budget_exceeded"  # cost cap reached before dispatch
    CIRCUIT_OPEN = "circuit_open"        # every usable backend's breaker was open

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return self.value


ALL_STATUSES: tuple = tuple(JudgeStatus)

#: statuses retried by default (on the same backend, then the next one in the pool)
DEFAULT_RETRY_ON = frozenset({
    JudgeStatus.TIMEOUT, JudgeStatus.RATE_LIMITED, JudgeStatus.BACKEND_ERROR, JudgeStatus.EMPTY_CONTENT,
    JudgeStatus.MALFORMED, JudgeStatus.TRUNCATED, JudgeStatus.SEMANTIC_REJECT,
})
#: statuses that disable a backend for the rest of the request (failover without retry) and trip its breaker
FATAL_BACKEND_STATUSES = frozenset({JudgeStatus.AUTH, JudgeStatus.QUOTA})
#: statuses that are control-plane decisions rather than backend health signals
CONTROL_STATUSES = frozenset({JudgeStatus.CANCELLED, JudgeStatus.BUDGET_EXCEEDED, JudgeStatus.CIRCUIT_OPEN})


class JudgeHalted(RuntimeError):
    """Raised when a breaker with policy ``stop`` trips (failure rate, consecutive failures or budget)."""

    def __init__(self, reason: str, status: JudgeStatus = JudgeStatus.CIRCUIT_OPEN):
        super().__init__(reason)
        self.reason = reason
        self.status = status


class BackendError(Exception):
    """A typed failure raised by a backend. ``retry_after`` (seconds) is honored by the client."""

    def __init__(self, status: Union[JudgeStatus, str], message: str = "", *, retry_after: Optional[float] = None,
                 http_status: Optional[int] = None, retryable: Optional[bool] = None):
        self.status = JudgeStatus(status)
        self.message = message
        self.retry_after = retry_after
        self.http_status = http_status
        self.retryable = retryable
        super().__init__(f"{self.status.value}: {message}")


# ----------------------------------------------------------------------------------------- reasoning
@dataclass(frozen=True)
class ReasoningSpec:
    """Provider-neutral reasoning control.

    ``mode``: ``default`` (send nothing; the model's own default), ``off``, ``on`` (backend default
    budget) or ``budget`` (``budget_tokens`` reasoning tokens). The client always adds the reasoning
    budget on top of the answer budget ``max_tokens``, so reasoning can never eat the answer.
    """

    mode: str = "default"
    budget_tokens: Optional[int] = None

    def __post_init__(self):
        if self.mode not in ("default", "off", "on", "budget"):
            raise ValueError(f"reasoning mode must be default|off|on|budget, got {self.mode!r}")
        if self.mode == "budget" and (self.budget_tokens is None or self.budget_tokens < 0):
            raise ValueError("reasoning mode 'budget' needs budget_tokens >= 0")

    @classmethod
    def parse(cls, value: Union["ReasoningSpec", str, int, bool, Mapping[str, Any], None]) -> "ReasoningSpec":
        """Accepts ``None``/"default", "off"/False, "on"/True, an int budget, or a mapping."""
        if isinstance(value, ReasoningSpec):
            return value
        if value is None:
            return cls()
        if isinstance(value, bool):
            return cls("on" if value else "off")
        if isinstance(value, int):
            return cls("off") if value <= 0 else cls("budget", int(value))
        if isinstance(value, str):
            v = value.strip().lower()
            if v.isdigit():
                return cls.parse(int(v))
            return cls(v)
        if isinstance(value, Mapping):
            return cls(**dict(value))
        raise TypeError(f"cannot interpret reasoning setting {value!r}")

    @property
    def enabled(self) -> bool:
        return self.mode in ("on", "budget") and not (self.mode == "budget" and self.budget_tokens == 0)

    def to_dict(self) -> Dict[str, Any]:
        return {"mode": self.mode, "budget_tokens": self.budget_tokens}


# ----------------------------------------------------------------------------------------- decoding
@dataclass(frozen=True)
class Decoding:
    """Sampling parameters. ``max_tokens`` is the ANSWER budget (reasoning is added on top)."""

    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    stop: Optional[tuple] = None
    seed: Optional[int] = None

    @classmethod
    def parse(cls, value: Union["Decoding", Mapping[str, Any], None]) -> "Decoding":
        if isinstance(value, Decoding):
            return value
        d = dict(value or {})
        if d.get("stop") is not None:
            stop = d["stop"]
            d["stop"] = (stop,) if isinstance(stop, str) else tuple(stop)
        return cls(**d)

    def merged(self, defaults: "Decoding") -> "Decoding":
        """Fill unset fields from ``defaults``."""
        return Decoding(**{f.name: (getattr(self, f.name) if getattr(self, f.name) is not None
                                    else getattr(defaults, f.name)) for f in fields(Decoding)})

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["stop"] = list(self.stop) if self.stop else None
        return d


# ----------------------------------------------------------------------------------------- request
Validator = Callable[[Any], Optional[str]]


@dataclass
class JudgeRequest:
    """One judge call.

    ``output_schema`` is a JSON Schema dict or a pydantic model class; ``None`` means a free-text
    answer (``parsed`` is then the answer text). ``validator`` is an optional semantic check on the
    parsed output that returns a corrective message (re-asked if enabled) or ``None`` when valid; it
    is covered by ``parser_version`` in the cache key.

    ``timeout_s`` bounds one attempt; ``deadline_s`` bounds the whole request (queueing, retries and
    backoff) measured from submission. ``priority``: higher is served first. ``sample_index``
    distinguishes deliberate repeated draws of the same prompt in the cache.
    """

    messages: List[Message]
    output_schema: Any = None
    decoding: Decoding = field(default_factory=Decoding)
    reasoning: ReasoningSpec = field(default_factory=ReasoningSpec)
    validator: Optional[Validator] = None
    tags: Dict[str, Any] = field(default_factory=dict)
    priority: int = 0
    timeout_s: Optional[float] = None
    deadline_s: Optional[float] = None
    pool: Optional[str] = None
    prompt_version: str = "0"
    schema_version: str = "0"
    parser_version: str = "0"
    policy_version: Optional[Union[int, str]] = None
    sample_index: int = 0
    use_cache: bool = True
    extra_params: Dict[str, Any] = field(default_factory=dict)
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex)

    def __post_init__(self):
        if not isinstance(self.messages, (list, tuple)) or not self.messages:
            raise ValueError("JudgeRequest.messages must be a non-empty list of chat messages")
        for m in self.messages:
            if not isinstance(m, Mapping) or "role" not in m or "content" not in m:
                raise ValueError(f"every message needs 'role' and 'content', got {m!r}")
        self.messages = [dict(m) for m in self.messages]
        self.decoding = Decoding.parse(self.decoding)
        self.reasoning = ReasoningSpec.parse(self.reasoning)
        if self.timeout_s is not None and self.timeout_s <= 0:
            raise ValueError("timeout_s must be > 0")
        if self.deadline_s is not None and self.deadline_s <= 0:
            raise ValueError("deadline_s must be > 0")

    def with_(self, **changes: Any) -> "JudgeRequest":
        return replace(self, **changes)

    @property
    def versions(self) -> Dict[str, Any]:
        return {"prompt_version": self.prompt_version, "schema_version": self.schema_version,
                "parser_version": self.parser_version, "policy_version": self.policy_version}

    def to_dict(self) -> Dict[str, Any]:
        from judgerl.judge.schema import schema_to_dict

        return {
            "request_id": self.request_id, "messages": self.messages,
            "output_schema": schema_to_dict(self.output_schema), "decoding": self.decoding.to_dict(),
            "reasoning": self.reasoning.to_dict(), "tags": dict(self.tags), "priority": self.priority,
            "timeout_s": self.timeout_s, "deadline_s": self.deadline_s, "pool": self.pool,
            "sample_index": self.sample_index, "use_cache": self.use_cache,
            "extra_params": dict(self.extra_params), **self.versions,
        }


# ----------------------------------------------------------------------------------------- result
@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def __iadd__(self, other: "Usage") -> "Usage":
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.reasoning_tokens += other.reasoning_tokens
        return self


@dataclass
class JudgeResult:
    """Outcome of a request. ``cost_usd`` and ``usage`` sum over every attempt that reached a model."""

    request_id: str
    status: JudgeStatus
    parsed: Any = None
    text: Optional[str] = None
    reasoning_text: Optional[str] = None
    finish_reason: Optional[str] = None
    usage: Usage = field(default_factory=Usage)
    cost_usd: float = 0.0
    cost_known: bool = True
    latency_s: float = 0.0        # submission -> resolution
    queue_s: float = 0.0          # submission -> first dispatch
    attempts: int = 0
    backend: Optional[str] = None
    model: Optional[str] = None
    error: Optional[str] = None
    cache_hit: bool = False
    coalesced: bool = False       # answered by an identical in-flight request (single-flight)
    lenient_parse: bool = False   # JSON recovered by the lenient extractor (counted fallback)
    reasks: int = 0               # corrective re-asks sent
    failovers: int = 0            # switches to another backend of the pool
    cache_key: Optional[str] = None
    tags: Dict[str, Any] = field(default_factory=dict)
    versions: Dict[str, Any] = field(default_factory=dict)
    attempt_log: List[Dict[str, Any]] = field(default_factory=list)
    raw: Optional[Dict[str, Any]] = None

    @property
    def ok(self) -> bool:
        return self.status == JudgeStatus.OK

    def to_dict(self, include_raw: bool = True) -> Dict[str, Any]:
        parsed = self.parsed
        if hasattr(parsed, "model_dump"):
            parsed = parsed.model_dump()
        d = {f.name: getattr(self, f.name) for f in fields(self) if f.name not in ("usage", "parsed", "raw", "status")}
        d.update(status=self.status.value, parsed=parsed, usage=asdict(self.usage))
        if include_raw:
            d["raw"] = self.raw
        return d


# ----------------------------------------------------------------------------------------- backend level
@dataclass
class BackendCall:
    """What a backend receives for one attempt. ``max_tokens`` already includes the reasoning budget."""

    messages: List[Message]
    max_tokens: int
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    stop: Optional[Sequence[str]] = None
    seed: Optional[int] = None
    reasoning: ReasoningSpec = field(default_factory=ReasoningSpec)
    reasoning_budget: int = 0           # tokens of max_tokens reserved for reasoning
    json_schema: Optional[Dict[str, Any]] = None  # set only when the backend should enforce it natively
    schema_name: str = "judge_output"
    timeout_s: Optional[float] = None
    request_id: str = ""
    attempt: int = 1
    extra_params: Dict[str, Any] = field(default_factory=dict)
    tags: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BackendResponse:
    """What a backend returns. ``cost_usd`` is optional (provider-reported or SDK-computed)."""

    text: Optional[str]
    finish_reason: Optional[str] = None
    usage: Usage = field(default_factory=Usage)
    model: Optional[str] = None
    reasoning_text: Optional[str] = None
    refusal: Optional[str] = None
    cost_usd: Optional[float] = None
    raw: Optional[Dict[str, Any]] = None

    @classmethod
    def from_mapping(cls, d: Mapping[str, Any]) -> "BackendResponse":
        """Build from a plain dict (used by the colocated adapter)."""
        usage = d.get("usage")
        if not isinstance(usage, Usage):
            usage = Usage(prompt_tokens=int(d.get("prompt_tokens") or (usage or {}).get("prompt_tokens") or 0),
                          completion_tokens=int(d.get("completion_tokens") or (usage or {}).get("completion_tokens") or 0),
                          reasoning_tokens=int(d.get("reasoning_tokens") or (usage or {}).get("reasoning_tokens") or 0))
        return cls(text=d.get("text"), finish_reason=d.get("finish_reason"), usage=usage, model=d.get("model"),
                   reasoning_text=d.get("reasoning_text"), refusal=d.get("refusal"), cost_usd=d.get("cost_usd"),
                   raw=d.get("raw"))


@dataclass
class BackendCapabilities:
    """``structured``: how the backend enforces a JSON schema natively (``json_schema`` = OpenAI-style
    ``response_format``, ``guided_json`` = vLLM ``extra_body.guided_json``, ``none`` = instruct+parse)."""

    structured: str = "none"
    reasoning_style: str = "none"
