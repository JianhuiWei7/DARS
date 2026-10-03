"""WebShop (text mode) as a per-episode Judge RL environment.

A port of verl-agent's ``WebshopEnvironmentManager`` + ``WebshopWorker`` (GiGPO repository), turned
into one episode per instance. Prompts, history memory, observation formatting, action parsing and
rewards are byte-identical to verl-agent:

* the task is the instruction text of the first observation (``parts[2]`` of the ``[SEP]`` split);
* observations are reformatted by ``format_obs``: everything after the instruction, each part quoted and
  re-joined with `` [SEP] ``;
* first step: the no-history template; later steps: task, the last ``history_length`` (observation,
  action) pairs (formatted observations), the current observation and the available actions
  (``search[<your query>]`` when the page has a search bar, then ``click[...]`` for every clickable).
  A history prompt longer than 13000 characters falls back to the no-history template, as upstream;
* an action is the lower-cased text inside ``<action>...</action>`` (without tags: the last 20
  characters of the lower-cased response); it is *valid* only when the tags are present, the response
  has ``<think>`` and ``</think>`` and no CJK characters. Invalid responses still reach the site;
* reward 10 when the episode ends with a purchase of task score 1.0, else 0; the task score is kept in
  ``info["task_score"]``; ``won`` mirrors the 10-point reward. The anchor is the formatted observation.
  verl-agent runs WebShop with ``max_steps=15`` (``examples/gigpo_trainer/run_webshop.sh``).

Tasks are ``{"goal_idx": int, "goal_seed": int}``: the WebShop simulator shuffles (and prices) its goal
list with the environment seed, so a goal index is only meaningful together with that seed. verl-agent's
split is ``range(500)`` for validation and ``range(500, len(goals))`` for training (see :func:`tasks`
and :func:`verl_agent_draws`, which reproduces verl-agent's exact per-worker sampling).

Needs the WebShop simulator (the ``web_agent_site`` package that verl-agent vendors under
``agent_system/environments/env_package/webshop/webshop``, with its data and search index). Point
``$WEBSHOP_ROOT`` (or ``webshop_root=``) at the directory that contains ``web_agent_site``.

With the DARS package installed, the environment also publishes page state keys
(``info["state_key"]``, the same keys the DARS verl-agent patch writes) and the goal's requirement schema
(``metadata()["goal_schema"]``). Both are hidden benchmark metadata for judges and reward programs and
never reach the prompt.
"""
from __future__ import annotations

import copy
import itertools
import os
import re
import sys
import threading
from typing import Any, Dict, List, Optional, Tuple

from judgerl.envs.base import Env, Observation, StepResult, register_env

# ---------------------------------------------------------------------------- verl-agent prompts
# Verbatim from verl-agent ``agent_system/environments/prompts/webshop.py`` (note the non-breaking
# hyphen in "e‑commerce" and the trailing spaces, which are part of the prompt).
WEBSHOP_TEMPLATE_NO_HIS = """
You are an expert autonomous agent operating in the WebShop e‑commerce environment. 
Your task is to: {task_description}.
Your current observation is: {current_observation}.
Your admissible actions of the current situation are: 
[
{available_actions}
].

Now it's your turn to take one action for the current step.
You should first reason step-by-step about the current situation, then think carefully which admissible action best advances the shopping goal. This reasoning process MUST be enclosed within <think> </think> tags. 
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""

WEBSHOP_TEMPLATE = """
You are an expert autonomous agent operating in the WebShop e‑commerce environment.
Your task is to: {task_description}.
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}.
Your admissible actions of the current situation are: 
[
{available_actions}
].

Now it's your turn to take one action for the current step.
You should first reason step-by-step about the current situation, then think carefully which admissible action best advances the shopping goal. This reasoning process MUST be enclosed within <think> </think> tags. 
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""

#: verl-agent's split of the (seed-shuffled) goal list: validation = first 500 goals, train = the rest.
NUM_TEST_GOALS = 500
#: WebShop's own default simulator seed (``WebAgentTextEnv`` uses ``kwargs.get('seed', 42)``).
DEFAULT_GOAL_SEED = 42
#: prompts longer than this fall back to the no-history template (verl-agent ``build_text_obs``).
MAX_PROMPT_CHARS = 13000

_CJK = re.compile(r'[一-鿿]')


# ---------------------------------------------------------------------------- pure helpers
def project_action(response: str) -> Tuple[str, bool]:
    """verl-agent's ``webshop_projection`` for one response: (action sent to the site, valid flag)."""
    original = response
    text = response.lower()
    start, end = text.find("<action>"), text.find("</action>")
    valid = False
    if start == -1 or end == -1:
        action = text[-20:]
    else:
        action = text[start + len("<action>"):end].strip().lower()
        valid = True
    if original.find("<think>") == -1 or original.find("</think>") == -1:
        valid = False
    if _CJK.search(original):
        valid = False
    return action, valid


def extract_task(first_obs: str) -> str:
    """The instruction of a WebShop text observation (verl-agent ``extract_task``)."""
    parts = first_obs.split(" [SEP] ")
    if len(parts) < 3 or parts[1] != "Instruction:":
        raise ValueError(f"not a WebShop instruction page: {first_obs[:200]!r}")
    return parts[2]


def format_obs(text_obs: str, task: str) -> str:
    """verl-agent ``format_obs``: the parts after the instruction, quoted and re-joined."""
    parts = text_obs.split(" [SEP] ")
    try:
        index = parts.index(task)
        return " [SEP] ".join(f"'{p}'" for p in parts[index + 1:])
    except ValueError:
        return text_obs


def format_available_actions(avail: Dict[str, Any]) -> List[str]:
    """verl-agent ``format_avail_actions``."""
    for key in avail.keys():
        if key not in ["has_search_bar", "clickables"]:
            raise ValueError(f"Unknown key in available actions: {key}")
    actions = []
    if avail["has_search_bar"]:
        actions.append("search[<your query>]")
    for txt in avail["clickables"]:
        actions.append(f"click[{txt}]")
    return actions


def format_history(history: List[Dict[str, str]], history_length: int) -> Tuple[str, int]:
    """verl-agent ``SimpleMemory.fetch`` for one environment: (context, number of pairs shown)."""
    recent = history[-history_length:]
    start = len(history) - len(recent)
    lines = [f"[Observation {start + j + 1}: '{r['text_obs']}', Action {start + j + 1}: '{r['action']}']"
             for j, r in enumerate(recent)]
    return "\n".join(lines), len(recent)


def build_prompt(task: str, observation: str, available: Dict[str, Any], history: List[Dict[str, str]],
                 history_length: int, init: bool) -> str:
    """verl-agent ``WebshopEnvironmentManager.build_text_obs`` for one environment."""
    actions = "\n".join(f"'{s}'," for s in format_available_actions(available))
    if init or history_length <= 0:
        return WEBSHOP_TEMPLATE_NO_HIS.format(task_description=task, current_observation=observation,
                                              available_actions=actions)
    context, valid_len = format_history(history, history_length)
    prompt = WEBSHOP_TEMPLATE.format(task_description=task, step_count=len(history), history_length=valid_len,
                                     action_history=context, current_step=len(history) + 1,
                                     current_observation=observation, available_actions=actions)
    if len(prompt) > MAX_PROMPT_CHARS:
        prompt = WEBSHOP_TEMPLATE_NO_HIS.format(task_description=task, current_observation=observation,
                                                available_actions=actions)
    return prompt


def _plain(x):
    """JSON-friendly deep copy of a goal (sets become sorted lists, numpy scalars become Python)."""
    if isinstance(x, dict):
        return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if isinstance(x, (set, frozenset)):
        return sorted(_plain(v) for v in x)
    if hasattr(x, "item") and callable(getattr(x, "item")):
        try:
            return x.item()
        except Exception:
            pass
    return x


def _dars_state():
    try:
        from dars.integrations import webshop_state
        return webshop_state
    except Exception:
        return None


# ---------------------------------------------------------------------------- simulator loading
def webshop_root(root: Optional[str] = None) -> Optional[str]:
    """Directory containing ``web_agent_site`` (``root`` argument, else ``$WEBSHOP_ROOT``)."""
    root = root or os.environ.get("WEBSHOP_ROOT")
    return os.path.abspath(os.path.expanduser(root)) if root else None


def _import_text_env(root: Optional[str] = None):
    root = webshop_root(root)
    if root and root not in sys.path:
        sys.path.append(root)            # verl-agent's WebshopWorker appends its vendored checkout
    try:
        from web_agent_site.envs import WebAgentTextEnv
    except ImportError as e:
        raise ImportError("the webshop environment needs the WebShop simulator (web_agent_site); set "
                          "$WEBSHOP_ROOT to the directory that contains web_agent_site") from e
    return WebAgentTextEnv


def data_paths(root: Optional[str] = None, use_small: bool = True) -> Tuple[Optional[str], Optional[str]]:
    """verl-agent's product / attribute files (``env.webshop.use_small``, default True)."""
    root = webshop_root(root)
    if root is None:
        return None, None
    suffix = "_1000" if use_small else ""
    return (os.path.join(root, "data", f"items_shuffle{suffix}.json"),
            os.path.join(root, "data", f"items_ins_v2{suffix}.json"))


# Simulator servers are expensive (products, goals, search index); instances in one process share them
# per (seed, data) key. Every instance uses its own session prefix, and calls into a shared server are
# serialized by the server's lock.
_SERVERS: Dict[tuple, Tuple[Any, threading.Lock]] = {}
_SERVERS_LOCK = threading.Lock()
_INSTANCE_COUNTER = itertools.count()


def _clear_server_cache() -> None:
    with _SERVERS_LOCK:
        _SERVERS.clear()


@register_env("webshop")
class WebShopEnv(Env):
    """One WebShop episode per instance.

    Args:
        history_length: number of (observation, action) pairs in the prompt (verl-agent default 2).
        max_steps: rollout step limit (verl-agent's WebShop scripts use 15).
        webshop_root: directory containing ``web_agent_site`` (default ``$WEBSHOP_ROOT``).
        use_small / human_goals / num_products / file_path / attr_path: simulator data, as verl-agent's
            ``env.webshop`` config (defaults: small product set, synthetic goals, all products).
        share_server: share simulator servers between instances in the process (keyed by seed and data).
        state_keys: publish DARS page state keys in ``info`` (``None``: when the DARS package is present).
        text_env_factory: ``callable(seed, **env_kwargs) -> WebAgentTextEnv``-like object (tests).
    """

    def __init__(self, history_length: int = 2, max_steps: int = 15, webshop_root: Optional[str] = None,
                 use_small: bool = True, human_goals: bool = False, num_products: Optional[int] = None,
                 file_path: Optional[str] = None, attr_path: Optional[str] = None, share_server: bool = True,
                 state_keys: Optional[bool] = None, text_env_factory=None):
        self.history_length = history_length
        self.max_steps = max_steps
        self.webshop_root = webshop_root
        default_file, default_attr = data_paths(webshop_root, use_small)
        self.env_kwargs: Dict[str, Any] = {"observation_mode": "text", "num_products": num_products,
                                           "human_goals": human_goals}
        if file_path or default_file:
            self.env_kwargs["file_path"] = file_path or default_file
        if attr_path or default_attr:
            self.env_kwargs["attr_path"] = attr_path or default_attr
        self.share_server = share_server
        self._ws_state = _dars_state() if state_keys is not False else None
        if state_keys and self._ws_state is None:
            raise ImportError("state_keys=True needs the DARS package (dars.integrations.webshop_state)")
        self._factory = text_env_factory
        self._prefix = f"jrl{os.getpid()}x{next(_INSTANCE_COUNTER)}-"
        self._env = None
        self._env_seed: Optional[int] = None
        self._lock: Any = threading.Lock()
        self._task = ""
        self._goal_idx: Optional[int] = None
        self._goal: Optional[Dict[str, Any]] = None
        self._history: List[Dict[str, str]] = []
        self._prev_obs = ""
        self._available: Dict[str, Any] = {"has_search_bar": False, "clickables": []}
        self._done = False

    # ---------------------------------------------------------------- simulator
    def _make_text_env(self, seed: int):
        kwargs = dict(self.env_kwargs, seed=seed)
        if self._factory is not None:
            return self._factory(**kwargs), threading.Lock()
        text_env_cls = _import_text_env(self.webshop_root)
        if not self.share_server:
            return text_env_cls(**kwargs), threading.Lock()
        key = (seed, kwargs.get("file_path"), kwargs.get("attr_path"), bool(kwargs.get("human_goals")),
               kwargs.get("num_products"))
        with _SERVERS_LOCK:
            entry = _SERVERS.get(key)
            if entry is None:
                # Build the first server exactly as verl-agent's worker does (the constructor seeds the
                # global RNGs before loading products and sampling goal prices), then share it.
                env = text_env_cls(session_prefix=self._prefix, **kwargs)
                _SERVERS[key] = (env.server, threading.Lock())
                return env, _SERVERS[key][1]
        server, lock = entry
        with lock:
            env = text_env_cls(server=server, session_prefix=self._prefix, **kwargs)
        return env, lock

    def _ensure_env(self, seed: int) -> None:
        if self._env is None or self._env_seed != seed:
            self.close()
            self._env, self._lock = self._make_text_env(seed)
            self._env_seed = seed

    def _drop_sessions(self) -> None:
        """Forget this instance's finished sessions on a shared server (they would accumulate)."""
        server = getattr(self._env, "server", None)
        sessions = getattr(server, "user_sessions", None)
        if isinstance(sessions, dict) and self.share_server and self._factory is None:
            for k in [k for k in sessions if isinstance(k, str) and k.startswith(self._prefix)]:
                sessions.pop(k, None)

    # ---------------------------------------------------------------- episode
    def reset(self, task: Dict[str, Any], seed: int = 0) -> Observation:
        goal_idx = int(task["goal_idx"])
        goal_seed = int(task.get("goal_seed", DEFAULT_GOAL_SEED))
        self._ensure_env(goal_seed)
        with self._lock:
            self._drop_sessions()
            obs, _ = self._env.reset(session=goal_idx)
            available = self._env.get_available_actions()
            goal = self._served_goal(goal_idx)
            state_key = self._state_key() if self._ws_state is not None else None
        self._goal_idx = goal_idx
        self._goal = goal
        self._task = extract_task(obs)
        self._history = []
        self._done = False
        text = format_obs(obs, self._task)
        self._prev_obs = text
        self._available = available
        info = {"available_actions": format_available_actions(available)}
        if state_key is not None:
            info["state_key"] = state_key
        return self._observe(text, first=True, info=info)

    def step(self, response: str) -> StepResult:
        if self._env is None:
            raise RuntimeError("reset() must be called before step()")
        action, valid = project_action(response)
        with self._lock:
            before = self._ws_state.snapshot(self._env) if self._ws_state is not None else None
            obs, score, done, _ = self._env.step(action)
            available = self._env.get_available_actions()
            state_key = None
            if self._ws_state is not None:
                if done:     # the text env resets itself after a purchase: key the page bought from
                    state_key = self._ws_state.state_key_from_url(before[0], done=True, session=before[1])
                else:
                    state_key = self._state_key()
        done = bool(done)
        won = bool(done and score == 1.0)
        reward = 10.0 if won else 0.0
        text = format_obs(obs, self._task)
        self._history.append({"text_obs": self._prev_obs, "action": action})
        self._prev_obs = text
        self._available = available
        self._done = done
        info = {"won": won, "is_action_valid": valid, "task_score": float(score or 0.0), "action": action,
                "goal_idx": self._goal_idx, "goal_seed": self._env_seed}
        if state_key is not None:
            info["state_key"] = state_key
        obs_info = {"available_actions": format_available_actions(available)}
        if state_key is not None:
            obs_info["state_key"] = state_key
        return StepResult(self._observe(text, info=obs_info), reward, done, info)

    # ---------------------------------------------------------------- helpers
    def _served_goal(self, goal_idx: int) -> Optional[Dict[str, Any]]:
        """The goal the simulator actually serves, checked against ``goals[goal_idx]``."""
        server = getattr(self._env, "server", None)
        goals = getattr(server, "goals", None)
        if goals is None:
            return None
        expected = goals[goal_idx]
        try:
            goal = server.user_sessions[self._env.session]["goal"]
        except Exception:
            goal = expected
        if goal.get("instruction_text") != expected.get("instruction_text"):
            raise RuntimeError(f"WebShop served a different goal than goals[{goal_idx}]")
        return goal

    def _state_key(self) -> str:
        url, fields = self._ws_state.snapshot(self._env)
        return self._ws_state.state_key_from_url(url, session=fields)

    def _observe(self, text: str, first: bool = False, info: Optional[Dict[str, Any]] = None) -> Observation:
        prompt = build_prompt(self._task, text, self._available, self._history, self.history_length, init=first)
        return Observation(prompt=[{"role": "user", "content": prompt}], anchor=text, text=text,
                           info=dict(info or {}))

    def task_text(self) -> str:
        return self._task

    def metadata(self) -> Dict[str, Any]:
        """Hidden goal metadata (never in the prompt): target attributes, options, price cap, the goal."""
        goal = _plain(copy.deepcopy(self._goal)) if self._goal is not None else None
        meta: Dict[str, Any] = {"goal_idx": self._goal_idx, "goal_seed": self._env_seed,
                                "instruction": self._task}
        if goal is not None:
            meta.update({"goal": goal, "attributes": goal.get("attributes"),
                         "goal_options": goal.get("goal_options"), "price_upper": goal.get("price_upper"),
                         "asin": goal.get("asin"), "product_category": goal.get("product_category")})
            if self._ws_state is not None:
                schema = self._ws_state.goal_schema(self._goal)
                if schema is not None:
                    meta["goal_schema"] = schema
        return meta

    def close(self) -> None:
        if self._env is not None:
            try:
                with self._lock:
                    self._drop_sessions()
                    self._env.close()
            finally:
                self._env = None
                self._env_seed = None


# ---------------------------------------------------------------------------- task providers
def num_goals(goal_seed: int = DEFAULT_GOAL_SEED, webshop_root: Optional[str] = None, use_small: bool = True,
              human_goals: bool = False, num_products: Optional[int] = None) -> int:
    """Size of the simulator's goal list (loads WebShop once; shared with the environments)."""
    env = WebShopEnv(webshop_root=webshop_root, use_small=use_small, human_goals=human_goals,
                     num_products=num_products, state_keys=False)
    env._ensure_env(goal_seed)
    try:
        return len(env._env.server.goals)
    finally:
        env.close()


def _split_range(split: str, n_goals: Optional[int]) -> range:
    if split in ("test", "val", "valid", "validation", "eval"):
        return range(NUM_TEST_GOALS)
    if split == "train":
        if n_goals is None:
            raise ValueError("the train split needs the number of goals")
        return range(NUM_TEST_GOALS, n_goals)
    raise ValueError(f"unknown WebShop split {split!r} (train / test)")


def tasks(split: str, limit: int = 0, seed: int = 0, goal_seed: int = DEFAULT_GOAL_SEED,
          n_goals: Optional[int] = None, **webshop_kwargs) -> List[Dict[str, Any]]:
    """Dataset task provider over verl-agent's goal split, for one simulator seed.

    ``test`` (alias ``val``) is goal indices ``0..499`` in order; ``train`` is ``500..len(goals)-1``,
    shuffled with ``seed``. All tasks share ``goal_seed`` (WebShop's default 42), so the set of goals is
    fixed. ``n_goals`` avoids loading WebShop just to count the training goals. For verl-agent's exact
    per-worker draws (each worker has its own seed), use :func:`verl_agent_draws`.
    """
    if split == "train" and n_goals is None:
        n_goals = num_goals(goal_seed, **webshop_kwargs)
    idxs = list(_split_range(split, n_goals))
    if split == "train":
        import random
        random.Random(seed).shuffle(idxs)
    idxs = idxs[:limit] if limit > 0 else idxs
    return [{"goal_idx": i, "goal_seed": goal_seed} for i in idxs]


def verl_agent_draws(split: str, env_seed: int = 0, batch_size: int = 128, group_n: int = 1, n_resets: int = 1,
                     n_goals: Optional[int] = None) -> List[List[Dict[str, Any]]]:
    """The goals verl-agent's ``WebshopMultiProcessEnv`` serves, reset by reset.

    verl-agent builds ``batch_size * group_n`` workers; worker ``i`` runs a simulator seeded
    ``seed + i // group_n`` (``seed = env.seed`` for training, ``env.seed + 1000`` for validation, where
    ``group_n`` is 1), and every ``reset()`` draws ``batch_size`` distinct goal indices with
    ``np.random.RandomState(seed).choice(split_range, batch_size, replace=False)`` (the RNG persists
    across resets), each repeated ``group_n`` times. Returns ``n_resets`` lists of per-worker tasks.
    """
    import numpy as np

    is_train = split == "train"
    if not is_train and group_n != 1:
        raise ValueError("verl-agent validation uses group_n == 1")
    base = env_seed if is_train else env_seed + 1000
    goal_idxs = _split_range(split, n_goals)
    rng = np.random.RandomState(base)
    out = []
    for _ in range(n_resets):
        idx = rng.choice(goal_idxs, size=batch_size, replace=False)
        idx = np.repeat(idx, group_n).tolist()
        out.append([{"goal_idx": int(g), "goal_seed": base + (i // group_n)} for i, g in enumerate(idx)])
    return out
