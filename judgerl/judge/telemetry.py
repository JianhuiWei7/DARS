"""Cost accounting and telemetry.

Cost of one model response, in order of precedence:
1. configured per-model prices (``input_per_mtok`` / ``output_per_mtok`` USD per million tokens);
2. a cost reported by the backend (e.g. computed by LiteLLM from its cost tables);
3. ``litellm.cost_per_token`` for the model, when LiteLLM is installed and knows it;
4. otherwise 0 and the response is counted under ``judge/cost_unknown``.

:class:`Telemetry` keeps cumulative and windowed (since the last ``metrics(reset_window=True)``)
aggregates; :meth:`Telemetry.metrics` returns a flat dict for trainer loggers.
"""
from __future__ import annotations

import json
import os
import threading
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, Optional, Tuple

from judgerl.judge.types import ALL_STATUSES, JudgeRequest, JudgeResult, JudgeStatus, Usage


@dataclass(frozen=True)
class Prices:
    input_per_mtok: float = 0.0
    output_per_mtok: float = 0.0

    def cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        return (prompt_tokens * self.input_per_mtok + completion_tokens * self.output_per_mtok) / 1e6


class CostModel:
    """Resolves a USD cost for (model, usage)."""

    def __init__(self, prices: Optional[Dict[str, Prices]] = None, use_litellm: bool = True):
        self.prices = dict(prices or {})
        self.use_litellm = use_litellm
        self._litellm_unknown: set = set()

    def price_for(self, model: str) -> Optional[Prices]:
        return self.prices.get(model)

    def _litellm_cost(self, model: str, usage: Usage) -> Optional[float]:
        if not self.use_litellm or model in self._litellm_unknown:
            return None
        try:
            import litellm  # noqa: WPS433 - optional at runtime

            litellm.suppress_debug_info = True
            p, c = litellm.cost_per_token(model=model, prompt_tokens=usage.prompt_tokens,
                                          completion_tokens=usage.completion_tokens)
            return float(p) + float(c)
        except Exception:  # unknown model or litellm missing
            self._litellm_unknown.add(model)
            return None

    def cost(self, model: str, usage: Usage, reported: Optional[float] = None) -> Tuple[float, bool]:
        """Returns ``(usd, known)``."""
        p = self.prices.get(model)
        if p is not None:
            return p.cost(usage.prompt_tokens, usage.completion_tokens), True
        if reported is not None:
            return float(reported), True
        c = self._litellm_cost(model, usage)
        if c is not None:
            return c, True
        return 0.0, usage.total_tokens == 0

    def estimate(self, model: str, prompt_tokens: int, completion_tokens: int) -> Optional[float]:
        """Worst-case cost for budgeting; ``None`` if the price is unknown."""
        p = self.prices.get(model)
        if p is not None:
            return p.cost(prompt_tokens, completion_tokens)
        c = self._litellm_cost(model, Usage(prompt_tokens, completion_tokens))
        return c


class _Agg:
    __slots__ = ("count", "status", "cost", "prompt_tokens", "completion_tokens", "reasoning_tokens",
                 "latency_sum", "cache_hits", "coalesced", "lenient", "reasks", "retries", "failovers",
                 "cost_unknown", "attempt_status", "latencies")

    def __init__(self, latency_window: int = 0):
        self.count = 0
        self.status: Dict[str, int] = defaultdict(int)
        self.attempt_status: Dict[str, int] = defaultdict(int)
        self.cost = 0.0
        self.prompt_tokens = self.completion_tokens = self.reasoning_tokens = 0
        self.latency_sum = 0.0
        self.cache_hits = self.coalesced = self.lenient = self.reasks = self.retries = 0
        self.failovers = self.cost_unknown = 0
        self.latencies: Optional[Deque[float]] = deque(maxlen=latency_window) if latency_window else None

    def add(self, r: JudgeResult) -> None:
        self.count += 1
        self.status[r.status.value] += 1
        self.cost += r.cost_usd
        self.prompt_tokens += r.usage.prompt_tokens
        self.completion_tokens += r.usage.completion_tokens
        self.reasoning_tokens += r.usage.reasoning_tokens
        self.latency_sum += r.latency_s
        self.cache_hits += int(r.cache_hit)
        self.coalesced += int(r.coalesced)
        self.lenient += int(r.lenient_parse)
        self.reasks += r.reasks
        self.retries += max(0, r.attempts - 1)
        self.failovers += r.failovers
        self.cost_unknown += int(not r.cost_known)
        for a in r.attempt_log:
            self.attempt_status[a.get("status", "?")] += 1
        if self.latencies is not None:
            self.latencies.append(r.latency_s)


def _percentile(values, q: float) -> float:
    if not values:
        return 0.0
    try:
        import numpy as np

        return float(np.percentile(np.fromiter(values, dtype=float), q))
    except ImportError:  # pragma: no cover
        s = sorted(values)
        return float(s[min(len(s) - 1, int(round(q / 100.0 * (len(s) - 1))))])


class Telemetry:
    """Thread-safe aggregation of final results by status, backend and tag."""

    def __init__(self, jsonl_path: Optional[str] = None, log_raw: bool = False, latency_window: int = 100_000,
                 prefix: str = "judge"):
        self.prefix = prefix
        self.jsonl_path = jsonl_path
        self.log_raw = log_raw
        self.latency_window = latency_window
        self._lock = threading.Lock()
        self._fh = None
        if jsonl_path:
            os.makedirs(os.path.dirname(os.path.abspath(jsonl_path)), exist_ok=True)
            self._fh = open(jsonl_path, "a", encoding="utf-8")
        self._reset_all()

    def _reset_all(self) -> None:
        self.total = _Agg(self.latency_window)
        self.window = _Agg(self.latency_window)
        self.by_backend: Dict[str, _Agg] = defaultdict(_Agg)
        self.by_tag: Dict[str, _Agg] = defaultdict(_Agg)
        self.events: Dict[str, int] = defaultdict(int)

    def event(self, name: str, n: int = 1) -> None:
        """Count a control-plane event (breaker trip, preflight failure, ...)."""
        with self._lock:
            self.events[name] += n

    def record(self, request: Optional[JudgeRequest], result: JudgeResult) -> None:
        with self._lock:
            self.total.add(result)
            self.window.add(result)
            self.by_backend[result.backend or "none"].add(result)
            for k, v in (result.tags or {}).items():
                self.by_tag[f"{k}={v}"].add(result)
            if self._fh is not None:
                rec = {"request": request.to_dict() if request is not None else {"request_id": result.request_id},
                       "result": result.to_dict(include_raw=self.log_raw)}
                self._fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
                self._fh.flush()

    # ------------------------------------------------------------------ reporting
    def _flat(self, agg: _Agg, p: str, latencies: bool = True) -> Dict[str, float]:
        out: Dict[str, float] = {f"{p}/total": agg.count}
        for st in ALL_STATUSES:
            out[f"{p}/{st.value}"] = agg.status.get(st.value, 0)
        n = max(agg.count, 1)
        out.update({
            f"{p}/ok_rate": agg.status.get(JudgeStatus.OK.value, 0) / n if agg.count else 0.0,
            f"{p}/cost_usd": agg.cost, f"{p}/prompt_tokens": agg.prompt_tokens,
            f"{p}/completion_tokens": agg.completion_tokens, f"{p}/reasoning_tokens": agg.reasoning_tokens,
            f"{p}/cache_hit": agg.cache_hits, f"{p}/cache_hit_rate": agg.cache_hits / n if agg.count else 0.0,
            f"{p}/coalesced": agg.coalesced, f"{p}/lenient_parse": agg.lenient, f"{p}/reask": agg.reasks,
            f"{p}/retries": agg.retries, f"{p}/failover": agg.failovers, f"{p}/cost_unknown": agg.cost_unknown,
            f"{p}/mean_latency_s": agg.latency_sum / n if agg.count else 0.0,
            f"{p}/cost_per_label_usd": agg.cost / max(1, agg.status.get("ok", 0)),
        })
        for st, c in agg.attempt_status.items():
            out[f"{p}/attempt/{st}"] = c
        if latencies and agg.latencies is not None:
            lat = list(agg.latencies)
            out[f"{p}/p50_latency_s"] = _percentile(lat, 50)
            out[f"{p}/p90_latency_s"] = _percentile(lat, 90)
            out[f"{p}/p99_latency_s"] = _percentile(lat, 99)
        return out

    def metrics(self, window: bool = False, reset_window: bool = False, detail: bool = True) -> Dict[str, float]:
        """Flat metrics: ``judge/ok``, ``judge/truncated``, ``judge/cost_usd``, ``judge/p50_latency_s``, ...

        ``window=True`` reports only results since the last reset (per training step);
        ``detail`` adds ``judge/backend/<name>/...`` and ``judge/tag/<k>=<v>/...`` keys.
        """
        with self._lock:
            agg = self.window if window else self.total
            out = self._flat(agg, self.prefix)
            if detail:
                for name, a in self.by_backend.items():
                    fl = self._flat(a, f"{self.prefix}/backend/{name}", latencies=False)
                    out.update({k: v for k, v in fl.items() if "/attempt/" not in k})
                for tag, a in self.by_tag.items():
                    fl = self._flat(a, f"{self.prefix}/tag/{tag}", latencies=False)
                    for key in ("total", "ok", "cost_usd", "prompt_tokens", "completion_tokens", "cache_hit"):
                        out[f"{self.prefix}/tag/{tag}/{key}"] = fl[f"{self.prefix}/tag/{tag}/{key}"]
            for name, c in self.events.items():
                out[f"{self.prefix}/event/{name}"] = c
            if reset_window:
                self.window = _Agg(self.latency_window)
        return out

    def status_counts(self) -> Dict[str, int]:
        with self._lock:
            return {st.value: self.total.status.get(st.value, 0) for st in ALL_STATUSES}

    @property
    def cost_usd(self) -> float:
        return self.total.cost

    def close(self) -> None:
        with self._lock:
            if self._fh is not None:
                self._fh.close()
                self._fh = None
