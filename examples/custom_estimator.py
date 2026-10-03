"""A custom advantage estimator.

    judgerl-train judgerl.estimator=examples.custom_estimator:success_weighted_grpo \
                  ++judgerl.estimator_kwargs.bonus=0.5 ...

An estimator gets a :class:`judgerl.algos.advantages.StepBatch` (one entry per step row: uid, traj_id,
step, step_reward, score, anchor, invalid) and returns one advantage per row.
"""
from __future__ import annotations

import numpy as np

from judgerl.algos.advantages import StepBatch, grpo


def success_weighted_grpo(batch: StepBatch, bonus: float = 0.5, **kwargs) -> np.ndarray:
    """GRPO on episode scores, plus ``bonus`` for the rows of rollouts that beat their group mean."""
    adv = grpo(batch)
    return adv + bonus * (adv > 0)
