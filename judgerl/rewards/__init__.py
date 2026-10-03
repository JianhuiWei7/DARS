"""Reward programs (see :mod:`judgerl.rewards.base`).

    build_reward_program({"type": "outcome", "judge": {...}, "rubric": "..."})
    build_reward_program({"type": "my_pkg.rewards:MyProgram", ...})
"""
from __future__ import annotations

import importlib
from typing import Any, Dict

from judgerl.rewards.base import RewardProgram, RewardRecord, combine

BUILTIN = {
    "outcome": "judgerl.rewards.judged:OutcomeRubric",
    "process": "judgerl.rewards.judged:ProcessScores",
    "first_error": "judgerl.rewards.judged:FirstError",
    "dars": "judgerl.rewards.dars:DARSProgram",
}


def to_plain(x: Any) -> Any:
    """Deep-convert mapping/sequence containers (e.g. OmegaConf DictConfig/ListConfig that Hydra passes
    as keyword arguments) to plain dicts and lists, so specs are JSON-serializable."""
    from collections.abc import Mapping
    if isinstance(x, Mapping):
        return {str(k): to_plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)) or (hasattr(x, "__iter__") and type(x).__name__ == "ListConfig"):
        return [to_plain(v) for v in x]
    return x


def build_reward_program(spec: Dict[str, Any]) -> RewardProgram:
    spec = to_plain(spec)
    kind = spec.pop("type")
    path = BUILTIN.get(kind, kind)
    module, _, attr = path.partition(":")
    return getattr(importlib.import_module(module), attr)(**spec)


__all__ = ["RewardProgram", "RewardRecord", "build_reward_program", "combine", "to_plain"]
