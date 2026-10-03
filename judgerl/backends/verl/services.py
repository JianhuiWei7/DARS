"""Addresses of servers that verl starts at run time, for judge configurations.

verl can serve a generative reward model (``reward.reward_model.*``): on its own GPUs, or colocated
with the trainer and woken only while rewards are computed. A judge backend reaches it with the
placeholder ``verl://reward_model``:

    judge:
      backend: {type: openai_compat, base_url: verl://reward_model}   # model defaults to the served one

The Judge RL task runner publishes the reward model's router address in a named Ray actor after the
trainer starts; judge clients created in any Ray process of the run resolve the placeholder from it.
"""
from __future__ import annotations

import copy
from typing import Any, Dict, Optional

PLACEHOLDER = "verl://reward_model"
_ACTOR = "judgerl_services"


def _actor_cls():
    import ray

    @ray.remote(num_cpus=0)
    class Services:
        def __init__(self):
            self.values: Dict[str, Any] = {}

        def set(self, name: str, value: Any) -> None:
            self.values[name] = value

        def get(self, name: str) -> Any:
            return self.values.get(name)

    return Services


def publish(name: str, value: Any):
    """Store ``value`` under ``name``; returns the actor handle (keep it alive for the run)."""
    import ray
    actor = _actor_cls().options(name=_ACTOR, get_if_exists=True).remote()
    ray.get(actor.set.remote(name, value))
    return actor


def lookup(name: str) -> Optional[Any]:
    import ray
    try:
        actor = ray.get_actor(_ACTOR)
    except ValueError:
        return None
    return ray.get(actor.get.remote(name))


def uses_placeholder(config: Any) -> bool:
    return PLACEHOLDER in repr(config)


def resolve_judge_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Replace ``verl://reward_model`` in backend URLs (and a missing model name) with the running
    reward model's router address and model path."""
    if not uses_placeholder(config):
        return config
    rm = lookup("reward_model")
    if rm is None:
        raise RuntimeError("the judge uses verl://reward_model but no verl reward model is running: set "
                           "reward.reward_model.enable=True and reward.reward_model.model_path=<HF id>")
    cfg = copy.deepcopy(config)
    backends = ([cfg["backend"]] if cfg.get("backend") else []) + list(cfg.get("backends") or [])
    for b in backends:
        if b.get("base_url") == PLACEHOLDER:
            b["base_url"] = rm["url"]
        if isinstance(b.get("base_urls"), (list, tuple)):
            b["base_urls"] = [rm["url"] if u == PLACEHOLDER else u for u in b["base_urls"]]
        if b.get("model") in (None, PLACEHOLDER) and rm["url"] in (b.get("base_url"), *(b.get("base_urls") or [])):
            b["model"] = rm["model"]
    return cfg
