"""Reward programs: turn a finished episode into step rewards and/or an episode score.

A reward program receives the episode as a list of step records (one per action):

    {"step", "anchor", "observation", "action", "result", "env_reward", "invalid", "won"}

plus the task text, the environment's judge-only metadata (gold answers, rubrics, requirement
schemas, ...) and the episode outcome. It returns
a :class:`RewardRecord`. Programs that call an LLM judge use :mod:`judgerl.judge`, so they inherit its
backends, caching, budgets, breakers and telemetry.

Conventions:

* ``step_rewards=None`` keeps the environment's step rewards; ``episode_score=None`` keeps the
  environment's episode score.
* A judge failure never silently becomes a reward: the record carries ``status != "ok"`` and the
  environment rewards are kept for that episode. The failure is counted in the judge metrics, and
  the judge's circuit breakers stop the run if failures exceed the configured thresholds.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class RewardRecord:
    step_rewards: Optional[List[float]] = None
    episode_score: Optional[float] = None
    status: str = "ok"                       # ok | skipped | failed
    info: Dict[str, Any] = field(default_factory=dict)


class RewardProgram:
    """Base class. Subclasses implement :meth:`ascore`."""

    name = "base"

    async def ascore(self, *, task: str, metadata: Dict[str, Any], rows: List[Dict[str, Any]], success: bool,
                     episode_score: float) -> RewardRecord:
        raise NotImplementedError

    async def aclose(self) -> None:
        pass


def combine(env_rewards: List[float], credit: List[float], mode: str, alpha: float = 0.25, gamma: float = 0.95,
            missing: Optional[List[bool]] = None) -> List[float]:
    """Combine judge credit with environment step rewards: replace | add | redistribute
    (redistribute keeps every episode's discounted return equal to the environment's; see DARS)."""
    T = len(env_rewards)
    miss = list(missing or [])[:T] + [False] * max(0, T - len(missing or []))
    g = [0.0 if (t >= len(credit) or miss[t]) else float(credit[t]) for t in range(T)]
    if mode == "replace":
        return g
    if mode == "add":
        return [float(e) + x for e, x in zip(env_rewards, g)]
    if mode == "redistribute":
        if T == 0:
            return []
        q = [alpha * max(g[t], 0.0) for t in range(T - 1)]
        budget = sum((gamma ** t) * float(env_rewards[t]) for t in range(T))
        paid = sum((gamma ** t) * q[t] for t in range(T - 1))
        return q + [(budget - paid) / (gamma ** (T - 1))]
    raise ValueError(f"unknown combine mode {mode!r}")
