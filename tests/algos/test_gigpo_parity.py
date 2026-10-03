"""judgerl's GiGPO against verl-agent's reference implementation on randomized batches.

Needs a verl-agent checkout ($VERL_AGENT_ROOT) and torch; skipped otherwise.
"""
import importlib.util
import os
import random
import sys
import types

import numpy as np
import pytest

from judgerl.algos.advantages import StepBatch, gigpo

ROOT = os.environ.get("VERL_AGENT_ROOT")
torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not ROOT, reason="set $VERL_AGENT_ROOT to a verl-agent checkout")


def _upstream():
    saved = sys.modules.get("verl")
    stub = types.ModuleType("verl")
    stub.DataProto = object          # core_gigpo only uses it for type hints
    sys.modules["verl"] = stub
    try:
        spec = importlib.util.spec_from_file_location("core_gigpo", os.path.join(ROOT, "gigpo", "core_gigpo.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        if saved is None:
            sys.modules.pop("verl", None)
        else:
            sys.modules["verl"] = saved
    mod.print = lambda *a, **k: None
    return mod


def _batch(rng):
    uid, traj, step, anchor, rew, score, inv = [], [], [], [], [], [], []
    for g in range(rng.randint(1, 4)):
        for k in range(rng.randint(1, 8)):
            ep = rng.choice([0.0, 10.0, 3.0])
            for t in range(rng.randint(1, 10)):
                uid.append(f"u{g}"); traj.append(f"u{g}t{k}"); step.append(t)
                anchor.append(f"s{rng.randint(0, 4)}"); rew.append(rng.choice([0.0, 0.0, 10.0, 0.3, -0.2]))
                score.append(ep); inv.append(float(rng.random() < 0.15))
    return uid, traj, step, anchor, rew, score, inv


def test_gigpo_matches_verl_agent():
    G = _upstream()
    rng = random.Random(0)
    for _ in range(400):
        uid, traj, step, anchor, rew, score, inv = _batch(rng)
        n = len(uid)
        gamma, w = rng.choice([0.95, 1.0]), rng.choice([1.0, 0.5])
        mode, pen = rng.choice(["mean_std_norm", "mean_norm"]), rng.choice([0.0, 0.1, 1.0])
        uid_a, traj_a, anchor_a = (np.array(x, dtype=object) for x in (uid, traj, anchor))
        # verl-agent: discounted returns per trajectory, then the invalid penalty, then the advantage
        rets = np.zeros(n)
        for t in set(traj):
            run = 0.0
            for i in reversed([i for i in range(n) if traj[i] == t]):
                run = rew[i] + gamma * run
                rets[i] = run
        mask = torch.ones(n, 3)
        tls = torch.zeros(n, 3)
        tls[:, -1] = torch.tensor(score) - pen * torch.tensor(inv)
        sr = torch.tensor(rets, dtype=torch.float32) - pen * torch.tensor(inv)
        up, _ = G.compute_gigpo_outcome_advantage(tls, sr, mask, anchor_a, uid_a, traj_a, step_advantage_w=w, mode=mode)
        mine = gigpo(StepBatch(uid_a, traj_a, np.array(step), np.array(rew), np.array(score), anchor_a, np.array(inv)),
                     gamma=gamma, step_weight=w, mode=mode, invalid_penalty=pen)
        np.testing.assert_allclose(mine, up[:, 0].numpy(), atol=1e-4)
