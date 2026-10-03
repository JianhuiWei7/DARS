"""Train with Judge RL on verl:  python -m judgerl.backends.verl.main [hydra overrides]

The Judge RL agent loop is registered through an agent-loop config file generated from the
``judgerl`` config node; the trainer is registered inside the Ray task runner.
"""
from __future__ import annotations

import os
from pprint import pprint

import hydra
import ray
import yaml
from omegaconf import DictConfig, OmegaConf

from verl.trainer.main_ppo import run_ppo
from verl.trainer.ppo.utils import need_critic, need_reference_policy
from verl.utils.config import validate_config
from verl.utils.device import auto_set_device


AGENT_LOOPS_YAML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config", "agent_loops.yaml")


def _prepare_config(config: DictConfig) -> None:
    """Inline reward-program files into the config (workers on other nodes need not see the files) and
    register the environment agent loop, unless the user supplied an agent-loop config of their own."""
    j = config.judgerl
    OmegaConf.set_struct(config, False)
    for key in ("reward_program", "group_reward_program"):
        path = j.get(f"{key}_file")
        if path:
            with open(path) as f:
                j[key] = yaml.safe_load(f)
            j[f"{key}_file"] = None
    stage = str(j.get("reward_stage", "rollout"))
    if stage not in ("rollout", "trainer"):
        raise ValueError(f"judgerl.reward_stage must be rollout|trainer, got {stage!r}")
    if stage == "trainer" and j.get("judge_val"):
        raise ValueError("judgerl.judge_val=true needs judgerl.reward_stage=rollout: trainer-stage judging "
                         "(e.g. a colocated judge) scores training batches only")
    if j.get("env_pool_size") is not None and int(j.env_pool_size) < 1:
        raise ValueError("judgerl.env_pool_size must be >= 1")
    if float(j.get("env_timeout_s", 300.0)) <= 0:
        raise ValueError("judgerl.env_timeout_s must be > 0")
    if j.get("max_steps") is not None and int(j.max_steps) < 1:
        raise ValueError("judgerl.max_steps must be >= 1 (or null for the environment's own limit)")
    agent = config.actor_rollout_ref.rollout.agent
    if not agent.get("agent_loop_config_path"):
        agent.agent_loop_config_path = AGENT_LOOPS_YAML
    OmegaConf.set_struct(config, True)


def _publish_reward_model(trainer, config: DictConfig):
    """Make verl's reward-model server reachable as ``verl://reward_model`` for judge backends."""
    manager = getattr(getattr(trainer, "reward_loop_manager", None), "reward_model_manager", None)
    if manager is None:
        return None
    from judgerl.backends.verl.services import publish
    return publish("reward_model", {"url": f"http://{manager.get_router_address()}",
                                    "model": str(config.reward.reward_model.model_path)})


@ray.remote(num_cpus=1)
class JudgeRLTaskRunner:
    """verl's V1 task runner, with the Judge RL trainer registered in the actor process."""

    def run(self, config: DictConfig):
        import transfer_queue as tq

        import judgerl.backends.verl.trainer  # noqa: F401  (registers trainer mode "judgerl_sync")
        from verl.trainer.ppo.v1 import AgentLoopManagerTQ, get_trainer_cls
        from verl.utils.import_utils import load_class_from_fqn
        from verl.utils.logging_utils import configure_verl_logging

        configure_verl_logging()
        trainer_cls = get_trainer_cls(config.trainer.v1.trainer_mode)
        config.transfer_queue.enable = True
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)
        tq.init(config.transfer_queue)
        trainer = None
        succeeded = False
        try:
            trainer = trainer_cls(config=config)
            trainer.init()
            services = _publish_reward_model(trainer, config)  # noqa: F841  (keeps the registry actor alive)
            fqn = config.actor_rollout_ref.rollout.get("agent", {}).get("agent_loop_manager_class")
            manager_cls = load_class_from_fqn(fqn, "AgentLoopManager") if fqn else AgentLoopManagerTQ
            manager = manager_cls.create(config=config, llm_client=trainer.get_llm_client(),
                                         teacher_client=trainer.get_teacher_client(),
                                         reward_loop_worker_handles=trainer.get_reward_handles())
            trainer.fit(manager)
            succeeded = True
        finally:
            try:
                tracking = getattr(trainer, "logger", None)
                if tracking is not None:
                    tracking.finish(exit_code=0 if succeeded else 1)
            finally:
                tq.close()


@hydra.main(config_path="config", config_name="judgerl_trainer", version_base=None)
def main(config: DictConfig):
    auto_set_device(config)
    _prepare_config(config)
    validate_config(config=config, use_reference_policy=need_reference_policy(config), use_critic=need_critic(config))
    run_ppo(config, task_runner_class=JudgeRLTaskRunner)


if __name__ == "__main__":
    main()
