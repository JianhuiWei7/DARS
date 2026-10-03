"""verl agent loop that plays one environment episode and emits one training row per step.

Registered as ``judgerl_env`` by ``judgerl/backends/verl/config/agent_loops.yaml``. Its settings come
from the ``judgerl`` node of the trainer config (``env``, ``env_kwargs``, ``max_steps``,
``reward_program``, ``reward_stage``, ``judge_val``, ``env_pool_size``, ``env_timeout_s``); keyword
arguments in a user-supplied agent-loop config file take precedence:

    - name: judgerl_env
      _target_: judgerl.backends.verl.agent_loop.EnvAgentLoop
      env: numberline                # registered env name or package.module:Class
      env_kwargs: {size: 10}

The dataset row supplies the task: ``extra_info.task`` (a dict passed to ``env.reset``) and
``extra_info.seed``. All rollouts of one dataset row share the task, so they form a GiGPO group.

Per-step metadata travels in ``AgentLoopOutput.extra_fields`` under ``judgerl`` and is read back by
:class:`judgerl.backends.verl.trainer.JudgeRLTrainerSync` to compute advantages.
"""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Dict, List, Optional
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopMetrics, AgentLoopOutput
from verl.utils.profiler import simple_timer

import judgerl.envs  # noqa: F401  (registers built-in environments)
from judgerl.envs.base import env_class, make_env
from judgerl.envs.pool import shared_pool

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

META_KEY = "judgerl"
_PROGRAMS: Dict[str, Any] = {}   # one reward program (and judge client) per worker process and config


def _reward_program(spec: Dict[str, Any]):
    """verl instantiates an agent loop per trajectory; the reward program and its judge client are
    shared by all trajectories of the worker process."""
    import json
    key = json.dumps(spec, sort_keys=True, default=str)
    if key not in _PROGRAMS:
        from judgerl.rewards import build_reward_program
        _PROGRAMS[key] = build_reward_program(spec)
    return _PROGRAMS[key]


class _ThreadEnv:
    """Async view of a thread-safe environment running in this process."""

    def __init__(self, env):
        self.env = env
        self.max_steps = env.max_steps

    async def reset(self, task, seed):
        return await asyncio.to_thread(self.env.reset, task, seed)

    async def step(self, action):
        return await asyncio.to_thread(self.env.step, action)

    async def task_text(self):
        return self.env.task_text()

    async def metadata(self):
        return self.env.metadata()


_UNSET = object()


class EnvAgentLoop(AgentLoopBase):
    def __init__(self, *args, env=_UNSET, env_kwargs=_UNSET, max_steps=_UNSET, reward_program=_UNSET,
                 env_pool_size=_UNSET, env_timeout_s=_UNSET, judge_val=_UNSET, with_task=_UNSET, **kwargs):
        super().__init__(*args, **kwargs)
        from omegaconf import OmegaConf
        node = self.config.get("judgerl", None)
        j = OmegaConf.to_container(node, resolve=True) if node is not None else {}

        def pick(value, key, default):     # yaml kwargs arrive as DictConfig; make them plain containers
            from judgerl.rewards import to_plain
            return to_plain(j.get(key, default) if value is _UNSET else value)

        self.env_name = pick(env, "env", "numberline")
        self.env_kwargs = dict(pick(env_kwargs, "env_kwargs", None) or {})
        pool = pick(env_pool_size, "env_pool_size", None)
        self.env_pool_size = int(pool) if pool else self._auto_pool_size()
        self.env_timeout_s = float(pick(env_timeout_s, "env_timeout_s", 300.0))
        self.max_steps_override = pick(max_steps, "max_steps", None)
        self.judge_val = bool(pick(judge_val, "judge_val", False))
        # episodes are judged here (reward_stage=rollout) or by the trainer (reward_stage=trainer, e.g. for a
        # judge colocated on the trainer's GPUs); group programs always run in the trainer
        trainer_stage = j.get("reward_stage", "rollout") == "trainer"
        spec = pick(reward_program, "reward_program", None)
        self.reward_program = _reward_program(spec) if spec and not trainer_stage else None
        # carry task text + metadata to the trainer when it judges
        self.with_task = bool(pick(with_task, "with_task", False) or (spec and trainer_stage)
                              or j.get("group_reward_program"))
        self.response_length = self.rollout_config.response_length

    def _auto_pool_size(self) -> int:
        """Enough environment workers that no episode of a batch waits for one: episodes per batch
        (training or validation, whichever is larger) divided by the number of agent-loop workers."""
        import math
        c, r = self.config, self.rollout_config
        train = int(c.data.get("train_batch_size") or 1) * int(r.get("n", 1) or 1)
        val_kwargs = r.get("val_kwargs", None)
        val = int(c.data.get("val_batch_size") or 0) * int((val_kwargs.get("n", 1) if val_kwargs else 1) or 1)
        workers = max(1, int(r.get("agent", {}).get("num_workers", 8) or 8))
        return max(1, min(256, math.ceil(1.25 * max(train, val) / workers)))   # 25% headroom for uneven spread

    async def run(self, sampling_params: Dict[str, Any], **kwargs) -> List[AgentLoopOutput]:
        extra = dict(kwargs.get("extra_info") or {})
        task = dict(extra.get("task") or {})
        seed = int(extra.get("seed", kwargs.get("index", 0)) or 0)
        if getattr(env_class(self.env_name), "thread_safe", False):
            env = make_env(self.env_name, **self.env_kwargs)
            try:
                return await self._episode(_ThreadEnv(env), task, seed, sampling_params, kwargs)
            finally:
                await asyncio.to_thread(env.close)
        # environments with global state run in dedicated worker processes (one episode per worker at a time)
        pool = shared_pool(self.env_pool_size, self.env_timeout_s)
        async with pool.episode(self.env_name, self.env_kwargs) as env:
            return await self._episode(env, task, seed, sampling_params, kwargs)

    async def _episode(self, env, task, seed, sampling_params, kwargs) -> List[AgentLoopOutput]:
        obs = await env.reset(task, seed)
        max_steps = self.max_steps_override or env.max_steps
        rows: List[Dict[str, Any]] = []
        for t in range(max_steps):
            metrics: Dict[str, float] = {}
            prompt_ids = await self.ct_build_initial_tokens(obs.prompt)
            # verl left-truncates prompts longer than rollout.prompt_length (dropping the task
            # instruction at the top); flag it so the trainer can report it
            truncated = len(prompt_ids) >= self.rollout_config.prompt_length
            with simple_timer("generate_sequences", metrics):
                out = await self.server_manager.generate(
                    request_id=uuid4().hex, prompt_ids=prompt_ids, sampling_params=sampling_params)
            merge, response_mask, response_logprobs = await self.ct_merge_assistant_token(
                prompt_ids, out.token_ids, [], [] if out.log_probs else None,
                assistant_logprobs=out.log_probs if out.log_probs else None)
            response_ids = merge.token_ids[-len(response_mask):] if response_mask else []
            prompt_ids = merge.token_ids[: len(merge.token_ids) - len(response_mask)]
            action = self.tokenizer.decode(response_ids, skip_special_tokens=True)
            result = await env.step(action)
            versions = {k: v for k, v in (getattr(out, "extra_fields", None) or {}).items()
                        if k in ("min_global_steps", "max_global_steps")}   # policy version used for this step
            rows.append({
                "versions": versions,
                "prompt_ids": prompt_ids,
                "response_ids": response_ids[: self.response_length],
                "response_mask": response_mask[: self.response_length],
                "response_logprobs": response_logprobs[: self.response_length] if response_logprobs else None,
                "metrics": metrics,
                "meta": {
                    "step": t,
                    "anchor": obs.anchor,
                    "observation": obs.text,
                    "action": action,
                    "result": result.observation.text,
                    "env_reward": float(result.reward),
                    "invalid": 0.0 if result.info.get("is_action_valid", True) else 1.0,
                    "won": bool(result.info.get("won", False)),
                    "state_key": result.info.get("state_key"),     # page state after the step (optional)
                    "prompt_truncated": truncated,
                },
            })
            obs = result.observation
            if result.done:
                break

        episode_score = float(sum(r["meta"]["env_reward"] for r in rows))
        won = bool(rows and rows[-1]["meta"]["won"])
        step_rewards = [r["meta"]["env_reward"] for r in rows]
        judge_info: Dict[str, Any] = {}
        reward_status = "none"
        is_val = (kwargs.get("extra_info") or {}).get("split") == "val"
        task_text = task_meta = None
        if self.with_task or self.reward_program is not None:
            task_text, task_meta = await env.task_text(), await env.metadata()
        if self.reward_program is not None and (self.judge_val or not is_val):
            record = await self.reward_program.ascore(task=task_text, metadata=task_meta,
                                                      rows=[r["meta"] for r in rows], success=won,
                                                      episode_score=episode_score)
            reward_status = record.status
            if record.status == "ok":
                step_rewards = record.step_rewards if record.step_rewards is not None else step_rewards
                if record.episode_score is not None:
                    episode_score = record.episode_score
            judge_info = record.info

        # reported by verl's validation per data source (e.g. val-aux/alfworld/success/mean@1)
        extra_info = {"success": float(won), "n_steps": float(len(rows))}
        outputs = []
        for r, step_reward in zip(rows, step_rewards):
            meta = dict(r["meta"], step_reward=float(step_reward), episode_score=episode_score, success=won,
                        n_steps=len(rows), reward_status=reward_status)
            if self.with_task and r is rows[0]:
                meta["task"], meta["task_metadata"] = task_text, task_meta
            if judge_info and r is rows[-1]:
                meta["judge"] = judge_info
            outputs.append(AgentLoopOutput(
                prompt_ids=r["prompt_ids"], response_ids=r["response_ids"], response_mask=r["response_mask"],
                response_logprobs=r["response_logprobs"], reward_score=episode_score, num_turns=2,
                metrics=AgentLoopMetrics(**r["metrics"]),
                # verl copies reward_extra_info from the final row to the others
                extra_fields={META_KEY: meta, "reward_extra_info": extra_info, **r["versions"]}))
        return outputs
