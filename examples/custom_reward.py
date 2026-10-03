"""Custom reward programs.

A reward program turns a finished episode into step rewards and/or an episode score. Point the
trainer at it by import path from a YAML file:

    # my_reward.yaml
    type: examples.custom_reward:Efficiency
    penalty: 0.5
    judge: {backend: {type: litellm, model: deepseek/deepseek-chat, api_key_env: DEEPSEEK_API_KEY}}

    judgerl-train judgerl.reward_program_file=my_reward.yaml ...

Each row of ``rows`` is one step: {"step", "anchor", "observation", "action", "result", "env_reward",
"invalid", "won", "state_key"}. Return ``RewardRecord(step_rewards=..., episode_score=...)``; ``None``
keeps the environment's value. On a judge failure return ``status="failed"``: the episode keeps its
environment rewards and the failure is counted in the judge metrics.
"""
from __future__ import annotations

from judgerl.rewards import RewardProgram, RewardRecord, combine
from judgerl.rewards.judged import render_episode
from judgerl.rewards.client import shared_client


class LengthPenalty(RewardProgram):
    """No judge: subtract a small cost per step, so shorter successful episodes score higher."""

    name = "length_penalty"

    def __init__(self, cost: float = 0.05):
        self.cost = cost

    async def ascore(self, *, task, metadata, rows, success, episode_score):
        return RewardRecord(step_rewards=[float(r["env_reward"]) - self.cost for r in rows])


class Efficiency(RewardProgram):
    """A judge marks the steps that were redundant (repeated or useless); each costs ``penalty``."""

    name = "efficiency"
    schema = {"type": "object", "properties": {"redundant_steps": {"type": "array", "items": {"type": "integer"}}},
              "required": ["redundant_steps"]}

    def __init__(self, judge, penalty: float = 0.5):
        self.judge_cfg, self.penalty = judge, penalty

    async def ascore(self, *, task, metadata, rows, success, episode_score):
        from judgerl.judge import JudgeRequest
        client = await shared_client(self.judge_cfg)    # one client per process: shared cache, budget, metrics

        def validate(parsed):                            # a returned string is sent back to the judge as a correction
            bad = [k for k in parsed["redundant_steps"] if not 1 <= k <= len(rows)]
            return f"step numbers out of range: {bad}" if bad else None

        r = await client.judge(JudgeRequest(
            messages=[{"role": "system", "content": "List the 1-based numbers of the steps that repeat an earlier "
                                                   "step or cannot help with the task. Return ONLY JSON "
                                                   "{\"redundant_steps\": [...]}."},
                      {"role": "user", "content": render_episode(task, rows)}],
            output_schema=self.schema, validator=validate, tags={"program": self.name}, prompt_version="efficiency-v1"))
        if not r.ok:
            return RewardRecord(status="failed", info={"judge_status": str(r.status), "error": r.error})
        bad = set(r.parsed["redundant_steps"])
        credit = [-self.penalty if t + 1 in bad else 0.0 for t in range(len(rows))]
        return RewardRecord(step_rewards=combine([float(x["env_reward"]) for x in rows], credit, "add"),
                            info={"redundant_steps": sorted(bad)})
