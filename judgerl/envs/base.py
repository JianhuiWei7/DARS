"""Environment API: one episode per instance, text (and optionally image) observations.

Judge RL runs every rollout as its own asynchronous task, so an environment instance serves
exactly one episode at a time. Environment calls are synchronous and are executed off the event
loop by the rollout backend.

An environment returns, at every step:

* ``prompt``: the chat messages the policy sees for this step (built by the environment from the
  task, the current observation and the memory of earlier steps),
* ``anchor``: a hashable description of the state the next action is taken from. GiGPO groups steps
  of the same task by equal anchors, so it must identify the state and nothing else,
* the reward of the previous action, ``done``, and an ``info`` dict (``won``, ``is_action_valid``, ...).

Register an environment with :func:`register_env` or the ``judgerl.envs`` entry point group.
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class Observation:
    prompt: List[Dict[str, Any]]          # chat messages for the policy
    anchor: str                           # state identity for step grouping
    text: str = ""                        # the raw observation (shown to judges as the step's state)
    images: Optional[list] = None
    info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StepResult:
    observation: Observation
    reward: float
    done: bool
    info: Dict[str, Any] = field(default_factory=dict)   # won, is_action_valid, task_score, ...


class Env:
    """Base class. ``task`` is the dataset row's task specification (a dict)."""

    #: maximum number of steps per episode (the rollout stops after this many actions)
    max_steps: int = 50
    #: True if many instances can run concurrently in threads of one process (no global state).
    #: Otherwise each episode runs in its own worker process (judgerl.envs.pool).
    thread_safe: bool = False

    def reset(self, task: Dict[str, Any], seed: int) -> Observation:
        raise NotImplementedError

    def step(self, action: str) -> StepResult:
        raise NotImplementedError

    def task_text(self) -> str:
        """Plain task description (shown to judges)."""
        return ""

    def metadata(self) -> Dict[str, Any]:
        """Task metadata for judges and reward programs (gold answer, requirement schema, ...).
        Never shown to the policy."""
        return {}

    def close(self) -> None:
        pass


class Memory:
    """The last ``k`` (observation, action) pairs, for per-step prompts (verl-agent's memory)."""

    def __init__(self, k: int = 2):
        self.k = k
        self.items: List[Dict[str, str]] = []

    def add(self, observation: str, action: str) -> None:
        self.items.append({"observation": observation, "action": action})

    def recent(self) -> List[Dict[str, str]]:
        return self.items[-self.k:] if self.k > 0 else []

    def __len__(self) -> int:
        return len(self.items)


_REGISTRY: Dict[str, Callable[..., Env]] = {}

#: built-in environments, imported only when first used (their dependencies are optional)
BUILTIN_ENVS: Dict[str, str] = {
    "numberline": "judgerl.envs.toy:NumberLineEnv",
    "alfworld": "judgerl.envs.alfworld:ALFWorldEnv",
    "webshop": "judgerl.envs.webshop:WebShopEnv",
    "search": "judgerl.envs.search:SearchEnv",
    "python_tool": "judgerl.envs.python_tool:PythonToolEnv",
    "open_ended": "judgerl.envs.open_ended:OpenEndedEnv",
}


def register_env(name: str, factory: Optional[Callable[..., Env]] = None):
    """Register an environment factory (usable as a decorator)."""
    def deco(f):
        _REGISTRY[name] = f
        return f
    return deco(factory) if factory is not None else deco


def env_class(name: str):
    """The registered factory/class for ``name`` (without instantiating it)."""
    if name not in _REGISTRY and name in BUILTIN_ENVS:
        module, _, attr = BUILTIN_ENVS[name].partition(":")
        _REGISTRY[name] = getattr(importlib.import_module(module), attr)
    if name not in _REGISTRY:
        _load_entry_points()
    if name in _REGISTRY:
        return _REGISTRY[name]
    if ":" in name:
        module, _, attr = name.partition(":")
        return getattr(importlib.import_module(module), attr)
    raise KeyError(f"unknown environment {name!r}; registered: {sorted(_REGISTRY)}")


def make_env(name: str, **kwargs) -> Env:
    """Create an environment by registered name or ``package.module:Class`` path."""
    if name not in _REGISTRY and name in BUILTIN_ENVS:
        module, _, attr = BUILTIN_ENVS[name].partition(":")
        _REGISTRY[name] = getattr(importlib.import_module(module), attr)
    if name not in _REGISTRY:
        _load_entry_points()
    if name in _REGISTRY:
        return _REGISTRY[name](**kwargs)
    if ":" in name:
        module, _, attr = name.partition(":")
        return getattr(importlib.import_module(module), attr)(**kwargs)
    raise KeyError(f"unknown environment {name!r}; registered: {sorted(_REGISTRY)}")


def _load_entry_points() -> None:
    try:
        from importlib.metadata import entry_points
        for ep in entry_points(group="judgerl.envs"):
            if ep.name not in _REGISTRY:
                _REGISTRY[ep.name] = ep.load()
    except Exception:
        pass
