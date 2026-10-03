"""verl V1 trainer with Judge RL's per-step advantage estimators.

``EnvAgentLoop`` stores one row per environment step in verl's TransferQueue, keyed
``{uid}_{session}_{step}``, with step metadata in ``extra_fields["judgerl"]``. This trainer replaces
verl's advantage step: it reads those rows, builds a :class:`judgerl.algos.advantages.StepBatch`
(task group = uid, trajectory = uid_session, anchor, step reward, episode score, invalid flag),
computes per-row advantages (GiGPO / GRPO / RLOO / custom) and writes them back broadcast over each
row's response tokens. Everything else (rollout, log-probs, PPO update, checkpointing) is verl's.

Padding rows that verl appends for divisibility are excluded from all statistics and get zero
advantage.
"""
from __future__ import annotations

import importlib

import logging
import os
from typing import Any, Dict

import numpy as np
import torch

import transfer_queue as tq
from verl import DataProto
from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_add_to_batch
from verl.trainer.ppo.v1.trainer_base import register_trainer
from verl.trainer.ppo.v1.trainer_colocate_async import PPOTrainerColocateAsync
from verl.trainer.ppo.v1.trainer_separate_async import PPOTrainerSeparateAsync
from verl.trainer.ppo.v1.trainer_sync import PPOTrainerSync
from verl.workers.utils.padding import response_to_nested

from judgerl.algos.advantages import ESTIMATORS, StepBatch, register_estimator
from judgerl.backends.verl.agent_loop import META_KEY

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


def _cfg(node, key, default):
    try:
        v = node.get(key, default)
    except AttributeError:
        v = getattr(node, key, default)
    return default if v is None else v


class JudgeRLMixin:
    """Judge RL's advantage step (per-step rows, trainer-stage judging, row-sized mini-batches) on top
    of any verl V1 trainer. Registered for verl's three modes below."""

    # ---------------------------------------------------------------- mini-batches in step rows
    def _mini_batch_rows(self):
        rows = (self.config.get("judgerl", {}) or {}).get("mini_batch_rows")
        return int(rows) if rows else None

    def _get_required_batch_multiple(self, dp_size: int) -> int:
        rows = self._mini_batch_rows()
        if rows is None:
            return super()._get_required_batch_multiple(dp_size)
        import math
        return math.lcm(dp_size, rows)

    def _with_row_mini_batch(self, role: str, fn, batch, metrics):
        """Run verl's update with ppo_mini_batch_size set so that ppo_mini_batch_size x rollout.n equals
        ``judgerl.mini_batch_rows`` (verl sizes mini-batches in prompts' samples; with one row per
        environment step the natural unit is rows; verl-agent uses 256 step rows)."""
        rows = self._mini_batch_rows()
        if rows is None:
            return fn(batch, metrics)
        n = int(self.config.actor_rollout_ref.rollout.n)
        if rows % n:
            raise ValueError(f"judgerl.mini_batch_rows ({rows}) must be a multiple of rollout.n ({n})")
        from omegaconf import open_dict
        node = self.config.critic if role == "critic" else self.config.actor_rollout_ref.actor
        original = node.ppo_mini_batch_size
        with open_dict(node):
            node.ppo_mini_batch_size = rows // n
        try:
            return fn(batch, metrics)
        finally:
            with open_dict(node):
                node.ppo_mini_batch_size = original

    def _update_actor(self, batch, metrics: dict):
        return self._with_row_mini_batch("actor", super()._update_actor, batch, metrics)

    def _update_critic(self, batch, metrics: dict):
        return self._with_row_mini_batch("critic", super()._update_critic, batch, metrics)

    def _estimator_kwargs(self, name: str) -> Dict[str, Any]:
        jcfg = self.config.get("judgerl", {}) or {}
        penalty = float(_cfg(jcfg, "invalid_penalty", 0.0))
        if name == "gigpo":
            g = jcfg.get("gigpo", {}) or {}
            sim = _cfg(g, "similarity", None)
            return {"gamma": float(self.config.algorithm.gamma), "step_weight": float(_cfg(g, "step_weight", 1.0)),
                    "mode": str(_cfg(g, "mode", "mean_std_norm")), "invalid_penalty": penalty,
                    "similarity": float(sim) if sim is not None else None}
        if name == "grpo":
            return {"invalid_penalty": penalty,
                    "norm_by_std": bool(self.config.algorithm.get("norm_adv_by_std_in_grpo", True))}
        if name == "rloo":
            return {"invalid_penalty": penalty}
        return dict((jcfg.get("estimator_kwargs", {}) or {}))

    # ---------------------------------------------------------------- trainer-stage judging
    def _program(self, attr: str, key: str, build):
        """A reward program from the ``judgerl`` config node, built once (main.py inlines *_file specs)."""
        if not hasattr(self, attr):
            from omegaconf import OmegaConf
            node = (self.config.get("judgerl", {}) or {}).get(key)
            spec = OmegaConf.to_container(node, resolve=True) if node is not None and not isinstance(node, dict) else node
            setattr(self, attr, build(spec) if spec else None)
        return getattr(self, attr)

    def _episode_program(self):
        """Per-episode program run by the trainer (``judgerl.reward_stage=trainer``)."""
        if (self.config.get("judgerl", {}) or {}).get("reward_stage", "rollout") != "trainer":
            return None
        from judgerl.rewards import build_reward_program
        return self._program("_episode_prog", "reward_program", build_reward_program)

    def _group_program(self):
        from judgerl.rewards.group import build_group_program
        return self._program("_group_prog", "group_reward_program", build_group_program)

    def _judges_in_trainer(self) -> bool:
        return self._episode_program() is not None or self._group_program() is not None

    def _run_async(self, coro):
        """Run a coroutine on a private event loop thread (one loop for the trainer's lifetime, so the
        judge client, its connections and its cache live across steps)."""
        import asyncio
        if getattr(self, "_loop", None) is None:
            import threading
            self._loop = asyncio.new_event_loop()
            threading.Thread(target=self._loop.run_forever, name="judgerl-trainer-judge", daemon=True).start()
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()

    def _judge_rows(self, metas, metrics: dict) -> Dict[str, Dict[str, Any]]:
        """Judge the episodes of a batch in the trainer: per-episode program (reward_stage=trainer), then
        the group program on each task group. ``metas`` = [(row key, step meta)] of the real rows.
        Returns {trajectory id: {"score", "step_rewards", "status"}} for the episodes whose rewards changed."""
        import asyncio
        episodes: Dict[str, Dict[str, Any]] = {}
        for key, meta in metas:
            traj_id = key.rpartition("_")[0]
            e = episodes.setdefault(traj_id, {"uid": key.split("_")[0], "rows": [], "task": None, "metadata": {},
                                              "success": bool(meta.get("success", False)),
                                              "episode_score": float(meta["episode_score"])})
            e["rows"].append(meta)
            if "task" in meta:
                e["task"], e["metadata"] = meta["task"], meta.get("task_metadata") or {}
        for e in episodes.values():
            e["rows"].sort(key=lambda m: int(m["step"]))
            if e["task"] is None:
                raise RuntimeError("trainer-stage judging needs the task text in the rows; launch through "
                                   "judgerl.backends.verl.main")
        updates: Dict[str, Dict[str, Any]] = {}
        episode_prog, group_prog = self._episode_program(), self._group_program()

        async def per_episode():
            async def one(t, e):
                return t, await episode_prog.ascore(task=e["task"], metadata=e["metadata"], rows=e["rows"],
                                                    success=e["success"], episode_score=e["episode_score"])
            return await asyncio.gather(*(one(t, e) for t, e in episodes.items()))

        if episode_prog is not None:     # (status ratios are logged with the episode metrics)
            for t, rec in self._run_async(per_episode()):
                if rec.status != "ok":
                    updates.setdefault(t, {})["status"] = rec.status
                    continue
                u = updates.setdefault(t, {"status": "ok"})
                if rec.episode_score is not None:
                    episodes[t]["episode_score"] = u["score"] = float(rec.episode_score)
                if rec.step_rewards is not None:
                    u["step_rewards"] = list(rec.step_rewards)

        if group_prog is not None:
            groups: Dict[str, list] = {}
            for t, e in episodes.items():
                groups.setdefault(e["uid"], []).append(t)

            async def per_group():
                return await asyncio.gather(*(group_prog.ascore_group(
                    task=episodes[ts[0]]["task"], metadata=episodes[ts[0]]["metadata"],
                    episodes=[episodes[t] for t in ts]) for ts in groups.values()))

            n_ok = n_fail = 0
            judged = []
            for ts, records in zip(groups.values(), self._run_async(per_group())):
                if len(records) != len(ts):
                    raise RuntimeError(f"group program returned {len(records)} records for {len(ts)} episodes")
                for t, rec in zip(ts, records):
                    if rec.status != "ok":
                        n_fail += rec.status == "failed"
                        continue
                    n_ok += 1
                    u = updates.setdefault(t, {"status": "ok"})
                    if rec.episode_score is not None:
                        u["score"] = float(rec.episode_score)
                    if rec.step_rewards is not None:
                        u["step_rewards"] = list(rec.step_rewards)
                    if "judge_score" in rec.info:
                        judged.append(float(rec.info["judge_score"]))
            metrics["judgerl/group_program/ok_ratio"] = n_ok / max(1, len(episodes))
            metrics["judgerl/group_program/failed_ratio"] = n_fail / max(1, len(episodes))
            if judged:
                metrics["judgerl/group_program/judge_score_mean"] = float(np.mean(judged))
        return updates

    def _check_judge_failures(self, metrics: dict) -> None:
        """Stop the run when a step's judged rewards mostly failed. Judge clients live in many rollout
        processes, so their own circuit breakers each see only a slice of the failures; the trainer
        sees the whole batch. A failed episode keeps its environment rewards, so without this check a
        broken judge (bad key, model name, request parameters) silently trains on environment rewards."""
        limit = (self.config.get("judgerl", {}) or {}).get("max_reward_failure_ratio", 0.5)
        if limit is None:
            return
        for key in ("judgerl/reward_program/failed_ratio", "judgerl/group_program/failed_ratio"):
            ratio = metrics.get(key)
            if ratio is not None and ratio > float(limit):
                raise RuntimeError(
                    f"{key}={ratio:.2f} exceeds judgerl.max_reward_failure_ratio={limit}: most judged rewards of "
                    "this step failed (see the judge telemetry / logs for the error; `judgerl-check <spec> --live` "
                    "tests a judge config). Set judgerl.max_reward_failure_ratio=null to train on anyway.")

    def _read_metas(self, batch):
        data = tq.kv_batch_get(keys=batch.keys, partition_id=batch.partition_id, select_fields=["extra_fields"])
        raw = data["extra_fields"]
        extras = raw.tolist() if hasattr(raw, "tolist") else list(raw)
        return [(batch.keys[i], (extras[i] or {}).get(META_KEY)) for i, tag in enumerate(batch.tags)
                if not tag.get("is_padding", False)]

    def _compute_reward_colocate(self, batch, metrics: dict | None = None):
        """With a verl reward model colocated on the trainer's GPUs (reward.reward_model.enable=True and no
        separate resource pool), verl calls this after rollout. Judge RL wakes the reward model, runs the
        trainer-stage judge programs against it (``verl://reward_model``) and puts it back to sleep."""
        if not self._judges_in_trainer():
            return super()._compute_reward_colocate(batch, metrics)
        if batch.partition_id == "val":        # validation keeps environment rewards (judge_val needs reward_stage=rollout)
            return batch
        manager = self.reward_loop_manager.reward_model_manager
        manager.wake_up()
        try:
            self._judged = self._judge_rows(self._read_metas(batch), metrics if metrics is not None else {})
        finally:
            manager.sleep()
        return batch

    def _compute_advantage(self, batch, metrics: dict):
        name = str((self.config.get("judgerl", {}) or {}).get("estimator") or "")
        if not name:            # judgerl.estimator=null: verl's own advantage estimator (algorithm.adv_estimator)
            return super()._compute_advantage(batch, metrics)
        if name not in ESTIMATORS and ":" in name:     # package.module:function
            module, _, attr = name.partition(":")
            register_estimator(name, getattr(importlib.import_module(module), attr))
        if name not in ESTIMATORS:
            raise ValueError(f"unknown judgerl.estimator {name!r}; built-in: {sorted(ESTIMATORS)}, "
                             "or give a 'package.module:function' path")
        if self.config.algorithm.use_kl_in_reward:
            raise ValueError("algorithm.use_kl_in_reward is not supported with Judge RL estimators; "
                             "use the actor KL loss (actor_rollout_ref.actor.use_kl_loss) instead")

        data = tq.kv_batch_get(keys=batch.keys, partition_id=batch.partition_id,
                               select_fields=["response_mask", "extra_fields", "rollout_log_probs", "old_log_probs"])
        response_mask_nested = data["response_mask"]

        # verl's decoupled rollout correction (importance weights / rejection masks) is independent of
        # how advantages are computed, so it runs exactly as in verl before the estimator.
        corr_cfg = self.config.algorithm.get("rollout_correction", None)
        bypass = bool(corr_cfg and corr_cfg.get("bypass_mode", False))
        extra_out = {}
        if corr_cfg is not None and "rollout_log_probs" in data.keys() and not bypass:
            proto = DataProto(batch=data.select("response_mask", "rollout_log_probs", "old_log_probs").to_padded_tensor())
            proto, is_metrics = compute_rollout_correction_and_add_to_batch(proto, corr_cfg)
            metrics.update(is_metrics)
            extra_out["response_mask"] = proto.batch["response_mask"]
            if "rollout_is_weights" in proto.batch.keys():
                extra_out["rollout_is_weights"] = proto.batch["rollout_is_weights"]
        response_mask = (extra_out["response_mask"] if "response_mask" in extra_out
                         else response_mask_nested.to_padded_tensor(0.0)).float()
        raw = data["extra_fields"]
        # depending on the TransferQueue backend this is a numpy array, a NonTensorStack or a LinkedList
        extras = raw.tolist() if hasattr(raw, "tolist") else list(raw)
        real = [i for i, tag in enumerate(batch.tags) if not tag.get("is_padding", False)]

        uid, traj, step, anchor, reward, score, invalid, truncated = [], [], [], [], [], [], [], []
        per_traj: Dict[str, Dict[str, Any]] = {}
        for i in real:
            key = batch.keys[i]
            meta = (extras[i] or {}).get(META_KEY)
            if meta is None:
                raise RuntimeError(f"row {key} has no {META_KEY!r} metadata: is the judgerl_env agent loop in use?")
            traj_id, _, _ = key.rpartition("_")
            uid.append(key.split("_")[0])
            traj.append(traj_id)
            step.append(int(meta["step"]))
            anchor.append(str(meta["anchor"]))
            reward.append(float(meta["step_reward"]))
            score.append(float(meta["episode_score"]))
            invalid.append(float(meta.get("invalid", 0.0)))
            truncated.append(bool(meta.get("prompt_truncated", False)))
            per_traj[traj_id] = {"success": bool(meta.get("success", False)), "score": float(meta["episode_score"]),
                                 "steps": int(meta.get("n_steps", 0)), "status": meta.get("reward_status", "none")}

        judged = getattr(self, "_judged", None)
        self._judged = None
        if real and judged is None and self._judges_in_trainer():
            judged = self._judge_rows([(batch.keys[i], extras[i][META_KEY]) for i in real], metrics)
        for k, t in enumerate(traj if judged else []):
            u = judged.get(t)
            if not u:
                continue
            if "score" in u:
                score[k] = u["score"]
                per_traj[t]["score"] = u["score"]
            if "step_rewards" in u and step[k] < len(u["step_rewards"]):
                reward[k] = float(u["step_rewards"][step[k]])
            if "status" in u:
                per_traj[t]["status"] = u["status"]

        adv = np.zeros(len(batch.keys), dtype=np.float32)
        if real:
            sb = StepBatch(np.array(uid, dtype=object), np.array(traj, dtype=object), np.array(step),
                           np.array(reward), np.array(score), np.array(anchor, dtype=object), np.array(invalid))
            adv[real] = ESTIMATORS[name](sb, **self._estimator_kwargs(name)).astype(np.float32)
            metrics["judgerl/step_rows"] = float(len(real))
            metrics["judgerl/trajectories"] = float(len(set(traj)))
            metrics["judgerl/step_reward_mean"] = float(np.mean(reward))
            metrics["judgerl/invalid_ratio"] = float(np.mean(invalid))
            metrics["judgerl/prompt_truncated_ratio"] = float(np.mean(truncated))   # raise data.max_prompt_length if > 0
            # per-episode statistics (verl's own score metrics average over step rows)
            eps = list(per_traj.values())
            metrics["judgerl/episode/success_rate"] = float(np.mean([e["success"] for e in eps]))
            metrics["judgerl/episode/score_mean"] = float(np.mean([e["score"] for e in eps]))
            metrics["judgerl/episode/length_mean"] = float(np.mean([e["steps"] for e in eps]))
            for status in ("ok", "failed", "skipped"):     # per-episode reward-program outcome
                metrics[f"judgerl/reward_program/{status}_ratio"] = float(np.mean([e["status"] == status for e in eps]))
            self._check_judge_failures(metrics)

        advantages = torch.from_numpy(adv).unsqueeze(-1) * response_mask
        output = {"advantages": response_to_nested(advantages, response_mask_nested),
                  "returns": response_to_nested(advantages.clone(), response_mask_nested)}
        for key, value in extra_out.items():
            output[key] = response_to_nested(value, response_mask_nested)
        from tensordict import TensorDict
        # kv_batch_put returns the batch handle updated with the new fields
        return tq.kv_batch_put(keys=batch.keys, partition_id=batch.partition_id,
                               fields=TensorDict(output, batch_size=len(batch.keys)))


@register_trainer("judgerl_sync")
class JudgeRLTrainerSync(JudgeRLMixin, PPOTrainerSync):
    """verl's synchronous colocated trainer: rollout, then training, on the same GPUs."""


@register_trainer("judgerl_colocate_async")
class JudgeRLTrainerColocateAsync(JudgeRLMixin, PPOTrainerColocateAsync):
    """verl's colocated asynchronous trainer: generation of the next batch starts before training on
    the current one ends (partial rollouts; each step row records the policy version it came from)."""


@register_trainer("judgerl_separate_async")
class JudgeRLTrainerSeparateAsync(JudgeRLMixin, PPOTrainerSeparateAsync):
    """verl's asynchronous trainer with rollout on separate GPUs
    (actor_rollout_ref.rollout.nnodes / n_gpus_per_node)."""
