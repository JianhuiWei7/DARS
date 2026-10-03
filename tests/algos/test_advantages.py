import numpy as np
import pytest

from judgerl.algos.advantages import StepBatch, discounted_returns, gigpo, grpo, rloo


def batch():
    # task u0: two rollouts; rollout a succeeds in 2 steps, rollout b fails in 3 steps; state s0 shared
    uid = np.array(["u0"] * 5, dtype=object)
    traj = np.array(["a", "a", "b", "b", "b"], dtype=object)
    step = np.array([0, 1, 0, 1, 2])
    reward = np.array([0.0, 10.0, 0.0, 0.0, 0.0])
    score = np.array([10.0, 10.0, 0.0, 0.0, 0.0])
    anchor = np.array(["s0", "s1", "s0", "s2", "s3"], dtype=object)
    return StepBatch(uid, traj, step, reward, score, anchor)


def test_discounted_returns_follow_step_order():
    b = batch()
    b.step = np.array([1, 0, 0, 1, 2])  # rows of trajectory a given out of order
    r = discounted_returns(b, 0.5)
    assert r[0] == pytest.approx(0.0) and r[1] == pytest.approx(10.0)   # row 1 is step 0


def test_gigpo_groups_shared_anchor_only():
    adv = gigpo(batch(), gamma=0.95, mode="mean_norm")
    ep = np.array([10, 10, 0, 0, 0]) - 4.0
    step = np.array([9.5 - 4.75, 0, 0 - 4.75, 0, 0])   # only s0 is shared; singleton groups give 0
    assert adv == pytest.approx(ep + step)


def test_grpo_and_rloo_per_trajectory():
    assert grpo(batch(), norm_by_std=False) == pytest.approx([5, 5, -5, -5, -5])
    assert rloo(batch()) == pytest.approx([10, 10, -10, -10, -10])


def test_invalid_penalty_hits_score_and_return():
    b = batch()
    b.invalid = np.array([0, 0, 1, 0, 0], dtype=float)
    base = gigpo(batch(), mode="mean_norm")
    pen = gigpo(b, mode="mean_norm", invalid_penalty=1.0)
    assert pen[2] < base[2]
