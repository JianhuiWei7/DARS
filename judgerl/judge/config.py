"""Typed configuration of the judge control plane, loadable from a dict, YAML file or Hydra node.

Unknown keys are rejected everywhere except inside a backend spec, where keys that are not common
backend fields are passed to the backend constructor as type-specific options (and rejected there if
the backend does not know them). Example::

    backends:
      - {type: litellm, model: deepseek/deepseek-chat, api_key_env: DEEPSEEK_API_KEY,
         rate_limit: {rpm: 600, max_concurrency: 32}, prices: {input_per_mtok: 0.27, output_per_mtok: 1.1}}
      - {type: hf_server, model: Qwen/Qwen3-8B, engine: vllm, gpus: "0"}
    budget_usd: 50
    cache: {path: outputs/judge_cache.sqlite}
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Mapping, Optional

from judgerl.judge.breaker import BreakerSettings
from judgerl.judge.telemetry import Prices
from judgerl.judge.types import DEFAULT_RETRY_ON, JudgeStatus, ReasoningSpec


def _to_plain(data: Any) -> Any:
    """Accept OmegaConf/Hydra nodes transparently."""
    if data is not None and type(data).__module__.startswith("omegaconf"):
        from omegaconf import OmegaConf  # type: ignore

        return OmegaConf.to_container(data, resolve=True)
    return data


def _build(cls, data: Optional[Mapping[str, Any]]):
    data = dict(_to_plain(data) or {})
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"unknown {cls.__name__} keys: {sorted(unknown)}")
    return cls(**data)


@dataclass
class RateLimitConfig:
    rpm: Optional[float] = None               # requests per minute
    tpm: Optional[float] = None               # tokens per minute (prompt estimate + completion cap)
    max_concurrency: Optional[int] = None     # in-flight requests to this backend
    burst_s: float = 1.0                      # bucket capacity in seconds of budget


@dataclass
class BreakerConfig:
    enabled: bool = True
    window_s: float = 60.0
    min_calls: int = 20
    failure_rate: Optional[float] = 0.5
    consecutive_failures: Optional[int] = 10
    cooldown_s: float = 30.0
    half_open_probes: int = 1
    policy: str = "pause"                     # pause | stop
    cost_budget_usd: Optional[float] = None   # per-backend spend cap (the run cap is JudgeConfig.budget_usd)
    fatal_statuses: List[str] = field(default_factory=lambda: ["auth", "quota"])
    ignore_statuses: List[str] = field(default_factory=lambda: ["cancelled", "budget_exceeded", "circuit_open"])

    def __post_init__(self):
        if self.policy not in ("pause", "stop"):
            raise ValueError(f"breaker.policy must be pause or stop, got {self.policy!r}")

    def settings(self) -> BreakerSettings:
        return BreakerSettings(window_s=self.window_s, min_calls=self.min_calls, failure_rate=self.failure_rate,
                               consecutive_failures=self.consecutive_failures, cooldown_s=self.cooldown_s,
                               half_open_probes=self.half_open_probes, policy=self.policy,
                               cost_budget_usd=self.cost_budget_usd,
                               fatal_statuses=tuple(JudgeStatus(s) for s in self.fatal_statuses),
                               ignore_statuses=tuple(JudgeStatus(s) for s in self.ignore_statuses),
                               enabled=self.enabled)


@dataclass
class PriceConfig:
    input_per_mtok: float = 0.0
    output_per_mtok: float = 0.0

    def prices(self) -> Prices:
        return Prices(self.input_per_mtok, self.output_per_mtok)


_BACKEND_COMMON = ("type", "name", "model", "revision", "rate_limit", "breaker", "prices", "structured",
                   "reasoning_style", "reasoning_budget", "timeout_s", "options")


@dataclass
class BackendConfig:
    """One backend. ``structured``: auto | json_schema | guided_json | json_object | none.
    ``reasoning_style``: auto | none | anthropic | openai_effort | gemini | chat_template.
    ``reasoning_budget``: tokens reserved for reasoning when a request does not set a budget and does
    not turn reasoning off (set it for always-reasoning models). Other keys go to ``options``."""

    type: str
    model: Optional[str] = None
    name: Optional[str] = None
    revision: Optional[str] = None
    rate_limit: RateLimitConfig = field(default_factory=RateLimitConfig)
    breaker: BreakerConfig = field(default_factory=BreakerConfig)
    prices: Optional[PriceConfig] = None
    structured: str = "auto"
    reasoning_style: str = "auto"
    reasoning_budget: int = 0
    timeout_s: Optional[float] = None
    options: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if isinstance(self.rate_limit, Mapping):
            self.rate_limit = _build(RateLimitConfig, self.rate_limit)
        if isinstance(self.breaker, Mapping):
            self.breaker = _build(BreakerConfig, self.breaker)
        if isinstance(self.prices, Mapping):
            self.prices = _build(PriceConfig, self.prices)
        if self.structured not in ("auto", "json_schema", "guided_json", "json_object", "none"):
            raise ValueError(f"backend structured must be auto|json_schema|guided_json|json_object|none, "
                             f"got {self.structured!r}")

    @property
    def display_name(self) -> str:
        return self.name or self.model or self.type

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BackendConfig":
        data = dict(_to_plain(data))
        if "type" not in data:
            raise ValueError(f"backend spec needs a 'type' (e.g. litellm, openai_compat, hf_server): {data!r}")
        common = {k: data.pop(k) for k in list(data) if k in _BACKEND_COMMON}
        options = dict(common.pop("options", None) or {})
        options.update(data)  # type-specific keys, e.g. engine / gpus / base_urls / api_key_env
        return cls(options=options, **common)

    def constructor_kwargs(self) -> Dict[str, Any]:
        kw: Dict[str, Any] = {"name": self.display_name}
        if self.model is not None:
            kw["model"] = self.model
        if self.revision is not None:
            kw["revision"] = self.revision
        if self.structured != "auto":
            kw["structured"] = self.structured
        if self.reasoning_style != "auto":
            kw["reasoning_style"] = self.reasoning_style
        kw.update(self.options)
        return kw


@dataclass
class RetryConfig:
    max_attempts: int = 6                          # total attempts per request, over all backends
    attempts_per_backend: Optional[int] = None     # before failing over (default: max_attempts)
    base_delay_s: float = 1.0
    max_delay_s: float = 30.0
    jitter: float = 0.2                            # multiplicative, uniform in [1, 1 + jitter]
    max_retry_after_s: float = 120.0               # cap on a provider's retry-after
    retry_on: List[str] = field(default_factory=lambda: sorted(s.value for s in DEFAULT_RETRY_ON))
    truncation_growth: float = 1.0                 # multiply the answer budget after a truncation
    max_tokens_cap: Optional[int] = None           # upper bound for that growth

    def __post_init__(self):
        if self.max_attempts < 1:
            raise ValueError("retry.max_attempts must be >= 1")
        for s in self.retry_on:
            JudgeStatus(s)


@dataclass
class QueueConfig:
    max_queue: int = 10_000     # queued (not yet dispatched) requests; submit() waits when full
    max_inflight: int = 64      # requests processed concurrently (all backends)


@dataclass
class CacheConfig:
    path: Optional[str] = None  # SQLite file; None disables the durable cache
    enabled: bool = True
    ttl_s: Optional[float] = None
    lease_ttl_s: float = 300.0
    cross_process: bool = True  # single-flight across processes through SQLite leases
    poll_s: float = 0.05


@dataclass
class TelemetryConfig:
    jsonl_path: Optional[str] = None   # every request/result as one JSON line
    log_raw: bool = False              # include raw provider responses in the JSONL
    latency_window: int = 100_000
    prefix: str = "judge"
    use_litellm_costs: bool = True


@dataclass
class StructuredConfig:
    lenient_fallback: bool = True      # recover JSON embedded in prose (counted as judge/lenient_parse)
    reask: bool = True                 # corrective re-ask on semantic reject
    reask_max_chars: int = 2000        # previous answer echoed back in the re-ask
    schema_instruction: str = "auto"   # auto (when not enforced natively) | always | never

    def __post_init__(self):
        if self.schema_instruction not in ("auto", "always", "never"):
            raise ValueError("structured.schema_instruction must be auto|always|never")


@dataclass
class DefaultsConfig:
    max_tokens: int = 1024             # answer budget (reasoning budget is added on top)
    temperature: Optional[float] = 0.0
    top_p: Optional[float] = None
    reasoning: Any = "default"         # default | off | on | <int budget>
    timeout_s: Optional[float] = 120.0  # per attempt
    deadline_s: Optional[float] = None  # per request, from submission


def _default_global_breaker() -> BreakerConfig:
    return BreakerConfig(window_s=120.0, min_calls=50, failure_rate=0.5, consecutive_failures=25, cooldown_s=30.0)


@dataclass
class JudgeConfig:
    backends: List[BackendConfig] = field(default_factory=list)
    pools: Dict[str, List[str]] = field(default_factory=dict)   # name -> ordered backend names (failover)
    default_pool: str = "default"
    queue: QueueConfig = field(default_factory=QueueConfig)
    retry: RetryConfig = field(default_factory=RetryConfig)
    breaker: BreakerConfig = field(default_factory=_default_global_breaker)   # global, over final results
    budget_usd: Optional[float] = None          # hard cap for the run (reserved worst case before dispatch)
    budget_policy: str = "stop"                 # stop: raise JudgeHalted | pause: return budget_exceeded
    cache: CacheConfig = field(default_factory=CacheConfig)
    telemetry: TelemetryConfig = field(default_factory=TelemetryConfig)
    structured: StructuredConfig = field(default_factory=StructuredConfig)
    defaults: DefaultsConfig = field(default_factory=DefaultsConfig)
    redact: Optional[str] = None                # "package.module:function" JudgeRequest -> JudgeRequest

    def __post_init__(self):
        self.backends = [b if isinstance(b, BackendConfig) else BackendConfig.from_dict(b) for b in self.backends]
        for name, cls in (("queue", QueueConfig), ("retry", RetryConfig), ("breaker", BreakerConfig),
                          ("cache", CacheConfig), ("telemetry", TelemetryConfig), ("structured", StructuredConfig),
                          ("defaults", DefaultsConfig)):
            v = getattr(self, name)
            if isinstance(v, Mapping):
                if name == "breaker":
                    base = dataclasses.asdict(_default_global_breaker())
                    base.update(dict(_to_plain(v)))
                    v = base
                setattr(self, name, _build(cls, v))
        if self.budget_policy not in ("stop", "pause"):
            raise ValueError("budget_policy must be stop or pause")
        ReasoningSpec.parse(self.defaults.reasoning)
        names = [b.display_name for b in self.backends]
        dupes = {n for n in names if names.count(n) > 1}
        if dupes:
            raise ValueError(f"duplicate backend names {sorted(dupes)}; set 'name' to disambiguate")
        for pool, members in self.pools.items():
            missing = [m for m in members if m not in names]
            if missing and self.backends:
                raise ValueError(f"pool {pool!r} references unknown backends {missing}")

    @classmethod
    def from_dict(cls, data: Optional[Mapping[str, Any]]) -> "JudgeConfig":
        data = dict(_to_plain(data) or {})
        if "backend" in data:  # single-backend shorthand
            if "backends" in data:
                raise ValueError("give either 'backend' or 'backends', not both")
            data["backends"] = [data.pop("backend")]
        return _build(cls, data)

    @classmethod
    def from_yaml(cls, path: str, overrides: Optional[Mapping[str, Any]] = None) -> "JudgeConfig":
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls.from_dict(deep_merge(data, dict(overrides or {})))

    @classmethod
    def coerce(cls, value: Any) -> "JudgeConfig":
        if value is None:
            return cls()
        if isinstance(value, JudgeConfig):
            return value
        if isinstance(value, str):
            return cls.from_yaml(value)
        return cls.from_dict(value)

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def pool_members(self, pool: Optional[str]) -> List[str]:
        pool = pool or self.default_pool
        if pool in self.pools:
            return list(self.pools[pool])
        if pool == self.default_pool:
            return [b.display_name for b in self.backends]
        raise KeyError(f"unknown judge pool {pool!r}; known: {sorted(self.pools) or [self.default_pool]}")


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, Mapping) and isinstance(out.get(k), Mapping):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out
