"""Group-level reward programs: judge the rollouts of one task *together*.

A per-episode program sees one trajectory at a time; a group program sees all rollouts of a task
(a GRPO/GiGPO group) and can compare them, which is what relative judgments are good at. They run in
the trainer after the batch is collected and before advantages are computed:

    judgerl:
      group_reward_program:
        type: listwise              # listwise | pairwise | package.module:Class
        judge: {backend: {type: litellm, model: deepseek/deepseek-chat}}
        weight: 0.5                 # episode score = weight * scale * judged + (1 - weight) * environment score

A group program implements ``ascore_group(task, metadata, episodes) -> list of RewardRecord`` (one per
episode, same order). Each episode is ``{"rows": [...step records...], "success": bool, "episode_score": float}``.
"""
from __future__ import annotations

import asyncio
import itertools
from typing import Any, Dict, List, Optional

from judgerl.rewards.base import RewardRecord
from judgerl.rewards.client import shared_client
from judgerl.rewards.judged import render_steps


class GroupRewardProgram:
    name = "group"

    async def ascore_group(self, *, task: str, metadata: Dict[str, Any],
                           episodes: List[Dict[str, Any]]) -> List[RewardRecord]:
        raise NotImplementedError


def _mix(judged: Optional[float], episode: Dict[str, Any], weight: float, scale: float, info) -> RewardRecord:
    if judged is None:
        return RewardRecord(status="failed", info=info)
    return RewardRecord(episode_score=weight * scale * judged + (1 - weight) * float(episode["episode_score"]),
                        info=dict(info, judge_score=scale * judged))


class _GroupJudged(GroupRewardProgram):
    def __init__(self, judge: Dict[str, Any], weight: float = 0.5, scale: float = 10.0, max_obs_chars: int = 300,
                 show_outcome: bool = False, criteria: str = "", prompt: Optional[str] = None, **request):
        self.judge_cfg, self.weight, self.scale = judge, weight, scale
        self.max_obs_chars, self.show_outcome, self.criteria = max_obs_chars, show_outcome, criteria
        self.prompt = prompt or self.default_prompt
        self.request = request

    default_prompt = ""

    def _render(self, task: str, episode: Dict[str, Any]) -> str:
        body = render_steps(episode["rows"], self.max_obs_chars)
        if self.show_outcome:
            body += f"\nOUTCOME: {'SUCCEEDED' if episode['success'] else 'FAILED'}"
        return body

    def _system(self) -> str:
        return self.prompt + (f"\n\nCRITERIA:\n{self.criteria}" if self.criteria else "")

    async def _judge(self, messages, schema, validator=None, tags=None):
        from judgerl.judge import JudgeRequest
        client = await shared_client(self.judge_cfg)
        return await client.judge(JudgeRequest(messages=messages, output_schema=schema, validator=validator,
                                               tags=dict(tags or {}, program=self.name),
                                               prompt_version=f"{self.name}-v1", **self.request))


class Listwise(_GroupJudged):
    """One judge call per group: the judge sees every rollout of the task and scores each in [0, 1]."""

    name = "listwise"
    default_prompt = ("You compare several attempts by an agent at the same task. Judge how well each attempt "
                      "accomplished the task and how efficiently, relative to the others. Return ONLY JSON "
                      "{\"scores\": [one number between 0 and 1 per attempt, in the order given]}.")

    async def ascore_group(self, *, task, metadata, episodes):
        n = len(episodes)
        if n == 0:
            return []
        body = f"TASK: {task}\n\n" + "\n\n".join(f"=== ATTEMPT {k + 1} ===\n{self._render(task, e)}"
                                                 for k, e in enumerate(episodes))
        schema = {"type": "object", "properties": {"scores": {"type": "array", "items": {"type": "number",
                                                                                         "minimum": 0, "maximum": 1}}},
                  "required": ["scores"]}

        def validate(parsed):
            return None if len(parsed["scores"]) == n else f"give exactly {n} scores, one per attempt"

        r = await self._judge([{"role": "system", "content": self._system()}, {"role": "user", "content": body}],
                              schema, validate)
        if not r.ok:
            info = {"judge_status": str(r.status), "error": r.error}
            return [RewardRecord(status="failed", info=info) for _ in episodes]
        return [_mix(float(s), e, self.weight, self.scale, {}) for s, e in zip(r.parsed["scores"], episodes)]


class Pairwise(_GroupJudged):
    """Round-robin pairwise preferences inside the group; an episode's judged score is its win rate.

    Each pair is judged in both orders when ``swap=True`` (position-bias control; a tie when the two
    orders disagree). ``max_pairs`` caps the judge calls per group by sampling pairs deterministically.
    """

    name = "pairwise"
    default_prompt = ("You compare two attempts by an agent at the same task. Decide which attempt accomplished "
                      "the task better (and, if both are equal on that, more efficiently). Return ONLY JSON "
                      "{\"winner\": \"A\" | \"B\" | \"tie\"}.")

    def __init__(self, judge, swap: bool = True, max_pairs: int = 0, **kw):
        super().__init__(judge, **kw)
        self.swap, self.max_pairs = swap, max_pairs

    async def _prefer(self, task, a, b) -> Optional[float]:
        """1.0 if a wins, 0.0 if b wins, 0.5 tie, None on judge failure."""
        schema = {"type": "object", "properties": {"winner": {"type": "string", "enum": ["A", "B", "tie"]}},
                  "required": ["winner"]}
        body = f"TASK: {task}\n\n=== ATTEMPT A ===\n{self._render(task, a)}\n\n=== ATTEMPT B ===\n{self._render(task, b)}"
        r = await self._judge([{"role": "system", "content": self._system()}, {"role": "user", "content": body}], schema)
        if not r.ok:
            return None
        return {"A": 1.0, "B": 0.0, "tie": 0.5}[r.parsed["winner"]]

    async def ascore_group(self, *, task, metadata, episodes):
        n = len(episodes)
        if n < 2:
            return [RewardRecord(status="skipped") for _ in episodes]
        pairs = list(itertools.combinations(range(n), 2))
        if self.max_pairs and len(pairs) > self.max_pairs:
            import random
            pairs = sorted(random.Random(n * 7919 + len(task)).sample(pairs, self.max_pairs))

        async def judge_pair(i, j):
            ab = await self._prefer(task, episodes[i], episodes[j])
            if not self.swap or ab is None:
                return ab
            ba = await self._prefer(task, episodes[j], episodes[i])
            if ba is None:
                return None
            return (ab + (1.0 - ba)) / 2
        results = await asyncio.gather(*(judge_pair(i, j) for i, j in pairs))
        wins, games = [0.0] * n, [0] * n
        failed = 0
        for (i, j), p in zip(pairs, results):
            if p is None:
                failed += 1
                continue
            wins[i] += p; wins[j] += 1.0 - p
            games[i] += 1; games[j] += 1
        out = []
        for k, e in enumerate(episodes):
            info = {"games": games[k], "failed_pairs": failed}
            out.append(_mix(wins[k] / games[k] if games[k] else None, e, self.weight, self.scale, info))
        return out


BUILTIN_GROUP = {"listwise": "judgerl.rewards.group:Listwise", "pairwise": "judgerl.rewards.group:Pairwise"}


def build_group_program(spec: Dict[str, Any]) -> GroupRewardProgram:
    import importlib
    from judgerl.rewards import to_plain
    spec = to_plain(spec)
    path = BUILTIN_GROUP.get(spec["type"], spec["type"])
    spec.pop("type")
    module, _, attr = path.partition(":")
    return getattr(importlib.import_module(module), attr)(**spec)
