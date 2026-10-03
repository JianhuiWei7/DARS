"""Advantage estimators on per-step rows (framework-agnostic, NumPy only).

Every training row is one agent step. A row carries:

    uid          task group (all rollouts of one task)       -> episode-level grouping
    traj_id      one rollout                                  -> discounting, trajectory stats
    step         position of the row inside its rollout (0-based)
    anchor       hashable environment state the step was taken from -> GiGPO step groups
    step_reward  reward of this step (environment reward, or shaped credit such as DARS)
    score        episode-level score of the rollout this row belongs to
    invalid      1 if the action could not be executed (optional penalty)

The estimators return one scalar advantage per row; the trainer broadcasts it over the row's
response tokens. `gigpo` reproduces verl-agent's GiGPO (arXiv:2505.10978) exactly, including its
statistics conventions (see tests/algos/test_gigpo_parity.py).
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Callable, Dict, Hashable, List, Optional, Sequence

import numpy as np


@dataclass
class StepBatch:
    uid: np.ndarray
    traj_id: np.ndarray
    step: np.ndarray
    step_reward: np.ndarray
    score: np.ndarray
    anchor: Optional[np.ndarray] = None
    invalid: Optional[np.ndarray] = None

    def __post_init__(self):
        n = len(self.uid)
        for name in ("traj_id", "step", "step_reward", "score"):
            if len(getattr(self, name)) != n:
                raise ValueError(f"{name} has {len(getattr(self, name))} rows, expected {n}")
        self.step_reward = np.asarray(self.step_reward, dtype=np.float64)
        self.score = np.asarray(self.score, dtype=np.float64)
        if self.invalid is None:
            self.invalid = np.zeros(n, dtype=np.float64)
        else:
            self.invalid = np.asarray(self.invalid, dtype=np.float64)


def discounted_returns(batch: StepBatch, gamma: float) -> np.ndarray:
    """Per-row return-to-go of ``step_reward`` within each trajectory, ordered by ``step``."""
    returns = np.zeros(len(batch.uid), dtype=np.float64)
    rows_by_traj: Dict[Hashable, List[int]] = defaultdict(list)
    for i, t in enumerate(batch.traj_id):
        rows_by_traj[t].append(i)
    for rows in rows_by_traj.values():
        rows.sort(key=lambda i: batch.step[i])
        running = 0.0
        for i in reversed(rows):
            running = batch.step_reward[i] + gamma * running
            returns[i] = running
    return returns


def _group_normalize(values: np.ndarray, groups: Sequence[Hashable], remove_std: bool, eps: float,
                     singleton_mean_zero: bool) -> np.ndarray:
    """(v - mean_g) [/ (std_g + eps)] with the unbiased std; singleton groups use mean 0 (episode
    convention) or their own value (step convention) and std 1."""
    members: Dict[Hashable, List[int]] = defaultdict(list)
    for i, g in enumerate(groups):
        members[g].append(i)
    out = np.empty_like(values, dtype=np.float64)
    for rows in members.values():
        v = values[rows]
        if len(rows) == 1:
            mean, std = (0.0 if singleton_mean_zero else float(v[0])), 1.0
        else:
            mean, std = float(v.mean()), float(v.std(ddof=1))
        out[rows] = (v - mean) if remove_std else (v - mean) / (std + eps)
    return out


def step_groups(anchor: np.ndarray, uid: np.ndarray, similarity: Optional[float] = None) -> List[Hashable]:
    """GiGPO step groups: rows of the same task whose anchor states are equal (or, with
    ``similarity`` in (0, 1), whose text anchors match by sequence ratio, first-fit clustering)."""
    groups: List[Hashable] = [None] * len(uid)
    by_uid: Dict[Hashable, List[int]] = defaultdict(list)
    for i, u in enumerate(uid):
        by_uid[u].append(i)
    for u, rows in by_uid.items():
        if similarity is None:
            for i in rows:
                a = anchor[i]
                groups[i] = (u, a if isinstance(a, Hashable) else repr(a))
        else:
            reps: List[str] = []
            for i in rows:
                a = str(anchor[i])
                for k, rep in enumerate(reps):
                    if SequenceMatcher(None, a, rep).ratio() >= similarity:
                        groups[i] = (u, k)
                        break
                else:
                    groups[i] = (u, len(reps))
                    reps.append(a)
    return groups


def gigpo(batch: StepBatch, gamma: float = 0.95, step_weight: float = 1.0, mode: str = "mean_std_norm",
          invalid_penalty: float = 0.0, similarity: Optional[float] = None, eps: float = 1e-6) -> np.ndarray:
    """GiGPO advantage per row: episode-relative + ``step_weight`` x anchor-step-relative.

    Episode part: row scores normalized within the task group, with statistics taken over rows
    (steps), as in verl-agent. Step part: discounted step returns normalized within groups of rows
    that share (task, anchor state). ``invalid_penalty`` is subtracted from both the row score and
    the row's step return for invalid actions (verl-agent's invalid-action penalty).
    """
    if mode not in ("mean_std_norm", "mean_norm"):
        raise ValueError(f"unknown GiGPO mode {mode!r}")
    remove_std = mode == "mean_norm"
    if batch.anchor is None:
        raise ValueError("GiGPO needs anchor states")
    scores = batch.score - invalid_penalty * batch.invalid
    returns = discounted_returns(batch, gamma) - invalid_penalty * batch.invalid
    episode = _group_normalize(scores, list(batch.uid), remove_std, eps, singleton_mean_zero=True)
    step = _group_normalize(returns, step_groups(batch.anchor, batch.uid, similarity), remove_std, eps,
                            singleton_mean_zero=False)
    return episode + step_weight * step


def _trajectory_scores(batch: StepBatch, invalid_penalty: float):
    """One score per trajectory (its episode score minus penalties summed over its rows)."""
    per_traj: Dict[Hashable, float] = {}
    uid_of: Dict[Hashable, Hashable] = {}
    for i, t in enumerate(batch.traj_id):
        per_traj[t] = per_traj.get(t, float(batch.score[i])) - invalid_penalty * float(batch.invalid[i])
        uid_of[t] = batch.uid[i]
    return per_traj, uid_of


def grpo(batch: StepBatch, invalid_penalty: float = 0.0, norm_by_std: bool = True, eps: float = 1e-6) -> np.ndarray:
    """GRPO on trajectories: each trajectory's score normalized within its task group; every row of
    the trajectory receives the trajectory's advantage."""
    per_traj, uid_of = _trajectory_scores(batch, invalid_penalty)
    trajs = list(per_traj)
    adv = _group_normalize(np.array([per_traj[t] for t in trajs]), [uid_of[t] for t in trajs],
                           remove_std=not norm_by_std, eps=eps, singleton_mean_zero=True)
    by_traj = dict(zip(trajs, adv))
    return np.array([by_traj[t] for t in batch.traj_id], dtype=np.float64)


def rloo(batch: StepBatch, invalid_penalty: float = 0.0) -> np.ndarray:
    """RLOO on trajectories: score minus the mean of the other trajectories of the task."""
    per_traj, uid_of = _trajectory_scores(batch, invalid_penalty)
    members: Dict[Hashable, List[Hashable]] = defaultdict(list)
    for t, u in uid_of.items():
        members[u].append(t)
    by_traj = {}
    for ts in members.values():
        total = sum(per_traj[t] for t in ts)
        for t in ts:
            by_traj[t] = per_traj[t] - (total - per_traj[t]) / (len(ts) - 1) if len(ts) > 1 else 0.0
    return np.array([by_traj[t] for t in batch.traj_id], dtype=np.float64)


ESTIMATORS: Dict[str, Callable[..., np.ndarray]] = {"gigpo": gigpo, "grpo": grpo, "rloo": rloo}


def register_estimator(name: str, fn: Callable[..., np.ndarray]) -> None:
    """Register a custom estimator ``fn(batch: StepBatch, **kwargs) -> per-row advantages``."""
    ESTIMATORS[name] = fn
