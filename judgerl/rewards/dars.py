"""DARS (Dependency-Aware Reward Shaping) on the Judge RL judge client.

Uses the released ``dars`` package for everything that defines the reward: domain prompts and
rendering, annotation parsing and validation, graph replay, clipping/accounting and optional terms.
The judge client replaces the package's LiteLLM annotator, so DARS gets the judge plane's backends,
cache, budgets and breakers.

    reward_program:
      type: dars
      config: ${DARS_ROOT}/configs/alfworld.yaml    # a DARS config (its `annotator` block is ignored);
                                                    # environment variables are expanded
      overrides: {kappa: 0.3}
      judge: {backend: {type: litellm, model: deepseek/deepseek-chat}, cache: {path: ...}}
      request: {decoding: {max_tokens: 8192}, reasoning: "off", timeout_s: 240}
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from judgerl.rewards.base import RewardProgram, RewardRecord, combine
from judgerl.rewards.client import shared_client


class DARSProgram(RewardProgram):
    name = "dars"

    def __init__(self, judge: Dict[str, Any], config: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None,
                 request: Optional[Dict[str, Any]] = None):
        try:
            from dars import DARSConfig, DARSReward
        except ImportError as e:  # pragma: no cover
            raise ImportError("the dars reward program needs the DARS package: pip install judgerl[dars]") from e
        if config:
            config = os.path.expanduser(os.path.expandvars(config))
            if "$" in config:
                raise ValueError(f"dars config path has an unset environment variable: {config!r}")
        cfg = DARSConfig.from_yaml(config, overrides or {}) if config else DARSConfig.from_dict(overrides or {})
        self.config = cfg
        self.reward = DARSReward(cfg, annotator=object())     # replay/terms only; annotation goes through the judge
        self.domain = self.reward.domain
        self.judge_cfg = judge
        self.request = dict(request or {})

    def trajectory(self, task: str, metadata: Dict[str, Any], rows: List[Dict[str, Any]], success: bool):
        from dars import Trajectory, Turn
        meta = dict(metadata or {})
        turns = []
        for i, r in enumerate(rows):
            last = i == len(rows) - 1
            result = r.get("result") if (not last or self.config.include_terminal_result) else None
            turns.append(Turn(observation=r.get("observation", ""), action=r.get("action", ""), result=result))
        return Trajectory(task=task, turns=turns, success=success,
                          reference=meta.get("reference"), meta=meta)

    def _parse(self, traj, text):
        """Annotation from an annotator reply (re-parsed for every caller, so cache hits and coalesced
        requests get the same annotation as the caller that made the request)."""
        from dars.annotator import extract_json
        return self.domain.parse(extract_json(text), len(traj.turns))

    async def _annotate(self, traj):
        from judgerl.judge import JudgeRequest
        client = await shared_client(self.judge_cfg)

        def validate(text):
            try:
                ann = self._parse(traj, text)
            except Exception as e:  # noqa: BLE001
                return f"Your answer could not be parsed ({e}). Return ONLY the JSON described."
            return self.domain.validate(traj, ann)

        req = JudgeRequest(messages=self.domain.build_messages(traj), output_schema=None, validator=validate,
                           tags={"program": "dars", "domain": self.domain.name}, prompt_version="dars-" + self.domain.name,
                           **self.request)
        r = await client.judge(req)
        if not r.ok:
            return None, r
        try:
            return self._parse(traj, r.text if r.text is not None else r.parsed), r
        except Exception:  # noqa: BLE001
            return None, r

    async def ascore(self, *, task, metadata, rows, success, episode_score):
        if self.config.annotate == "failed" and success:
            return RewardRecord(status="skipped")
        if self.config.annotation != "trajectory":
            raise NotImplementedError("prefix annotation is available through the verl-agent backend of the DARS package")
        metadata = dict(metadata or {})
        keys = [r.get("state_key") for r in rows]
        if any(k is not None for k in keys):
            metadata.setdefault("state_keys", keys)   # page state after each step (WebShop product scope)
        traj = self.trajectory(task, metadata, rows, success)
        annotation, r = await self._annotate(traj)
        if annotation is None:
            return RewardRecord(status="failed", info={"judge_status": str(r.status), "error": r.error})
        try:
            phis, credit = self.reward.rewards_from_annotation(traj, annotation)
            env_rewards = [float(x["env_reward"]) for x in rows]
            steps = combine(env_rewards, credit, self.config.combine, self.config.alpha, self.config.gamma,
                            annotation.missing)
        except Exception as e:  # noqa: BLE001 - a bad term/vector keeps the environment rewards
            return RewardRecord(status="failed", info={"error": f"reward computation: {e!r}"[:300]})
        return RewardRecord(step_rewards=steps, info={"potentials": phis, "nodes": annotation.nodes})
