"""ALFWorld (text) as a per-episode Judge RL environment.

Prompts, history memory, action parsing and rewards follow verl-agent exactly, so results are
comparable with GiGPO's published ALFWorld setup:

* first step: the no-history template; later steps: task, the last ``history_length`` (observation,
  action) pairs, the current observation and the admissible actions (``help`` excluded);
* an action is the lower-cased text inside ``<action>...</action>``; it counts as *valid* only when the
  response also has ``<think>...</think>`` and no CJK characters (invalid responses still reach the
  game, which answers "Nothing happens.");
* reward 10 when the game is won, else 0; the anchor is the raw game observation.

Tasks are explicit game files (``{"gamefile": ".../game.tw-pddl"}``), so the train and evaluation
sets are fixed lists; :func:`list_games` enumerates them from ``$ALFWORLD_DATA``.
Requires ``pip install alfworld`` and ``alfworld-download``.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Sequence

from judgerl.envs.base import Env, Observation, StepResult, register_env

TEMPLATE_NO_HISTORY = """
You are an expert agent operating in the ALFRED Embodied Environment.
Your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""

TEMPLATE = """
You are an expert agent operating in the ALFRED Embodied Environment. Your task is to: {task_description}
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""

TASK_TYPES = ("pick_and_place", "pick_two_obj_and_place", "look_at_obj_in_light", "pick_heat_then_place_in_recep",
              "pick_cool_then_place_in_recep", "pick_clean_then_place_in_recep")
_CJK = re.compile(r"[一-鿿]")


def project_action(response: str):
    """verl-agent's ALFWorld projection: (action sent to the game, valid flag)."""
    text = response.lower()
    start, end = text.find("<action>"), text.find("</action>")
    if start == -1 or end == -1:
        return text[-30:], False
    action = text[start + len("<action>"):end].strip().lower()
    valid = "<think>" in response and "</think>" in response and not _CJK.search(response)
    return action, valid


def task_type(gamefile: str) -> Optional[str]:
    for t in TASK_TYPES:
        if t in gamefile:
            return t
    return None


TASK_TYPE_NAMES = ("pick_and_place_simple", "look_at_obj_in_light", "pick_clean_then_place_in_recep",
                   "pick_heat_then_place_in_recep", "pick_cool_then_place_in_recep", "pick_two_obj_and_place")


def list_games(split: str = "train", data_dir: Optional[str] = None,
               task_types: Sequence[str] = TASK_TYPE_NAMES) -> List[str]:
    """Game files of a split (``train``, ``valid_seen`` = in-distribution, ``valid_unseen`` =
    out-of-distribution), filtered exactly like ALFWorld's own loader (``AlfredTWEnv.collect_game_files``:
    no movable/sliced trajectories, configured task types only, game file present and marked solvable),
    sorted by path for a deterministic, machine-independent order.

    Unlike verl-agent, which lets each environment worker draw games itself, Judge RL trains and
    evaluates on explicit task lists: the evaluated games do not depend on the machine's directory
    order, and the same task file reproduces the same evaluation.
    """
    root = os.path.expanduser(os.path.expandvars(os.path.join(
        data_dir or os.environ.get("ALFWORLD_DATA", "~/.cache/alfworld"), "json_2.1.1", split)))
    games = []
    for dirpath, _, files in os.walk(root):
        if "traj_data.json" not in files or "movable" in dirpath or "Sliced" in dirpath:
            continue
        try:
            with open(os.path.join(dirpath, "traj_data.json")) as f:
                if json.load(f).get("task_type") not in task_types:
                    continue
            game = os.path.join(dirpath, "game.tw-pddl")
            if not os.path.exists(game):
                continue
            with open(game) as f:
                if not json.load(f).get("solvable", False):
                    continue
        except (OSError, ValueError):
            continue
        games.append(game)
    return sorted(games)


@register_env("alfworld")
class ALFWorldEnv(Env):
    def __init__(self, history_length: int = 2, max_steps: int = 50):
        self.history_length = history_length
        self.max_steps = max_steps
        self._env = None
        self._task = ""
        self._gamefile = ""
        self._history: List[Dict[str, str]] = []
        self._prev_obs = ""
        self._admissible: List[str] = []

    # ---------------------------------------------------------------- game
    def _make(self, gamefile: str):
        import textworld
        import textworld.gym
        from alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredInfos

        infos = textworld.EnvInfos(won=True, admissible_commands=True, extras=["gamefile"])
        env_id = textworld.gym.register_games([gamefile], infos, batch_size=1, asynchronous=False,
                                              max_episode_steps=self.max_steps,
                                              wrappers=[AlfredDemangler(shuffle=False), AlfredInfos])
        return textworld.gym.make(env_id)

    def reset(self, task: Dict[str, Any], seed: int) -> Observation:
        self.close()
        self._gamefile = task["gamefile"]
        self._env = self._make(self._gamefile)
        obs, infos = self._env.reset()
        text = obs[0]
        i = text.find("Your task is to: ")
        if i == -1:
            raise ValueError(f"no task description in the first observation of {self._gamefile}")
        self._task = text[i + len("Your task is to: "):].strip()
        self._history = []
        self._prev_obs = text
        self._admissible = list(infos["admissible_commands"][0])
        return self._observe(text, first=True)

    def step(self, response: str) -> StepResult:
        action, valid = project_action(response)
        obs, scores, dones, infos = self._env.step([action])
        text = obs[0]
        won = bool(infos["won"][0])
        self._history.append({"text_obs": self._prev_obs, "action": action})
        self._prev_obs = text
        self._admissible = list(infos["admissible_commands"][0])
        return StepResult(self._observe(text), 10.0 * float(won), bool(dones[0]),
                          {"won": won, "is_action_valid": valid, "gamefile": self._gamefile,
                           "task_type": task_type(self._gamefile)})

    # ---------------------------------------------------------------- prompt
    def _observe(self, text: str, first: bool = False) -> Observation:
        actions = "\n ".join(f"'{a}'" for a in self._admissible if a != "help")
        if first or self.history_length <= 0:
            prompt = TEMPLATE_NO_HISTORY.format(current_observation=text, admissible_actions=actions)
        else:
            recent = self._history[-self.history_length:]
            start = len(self._history) - len(recent)
            lines = [f"[Observation {start + j + 1}: '{r['text_obs']}', Action {start + j + 1}: '{r['action']}']"
                     for j, r in enumerate(recent)]
            prompt = TEMPLATE.format(task_description=self._task, step_count=len(self._history),
                                     history_length=len(recent), action_history="\n".join(lines),
                                     current_step=len(self._history) + 1, current_observation=text,
                                     admissible_actions=actions)
        return Observation(prompt=[{"role": "user", "content": prompt}], anchor=text, text=text,
                           info={"admissible_actions": list(self._admissible)})

    def task_text(self) -> str:
        return self._task

    def metadata(self) -> Dict[str, Any]:
        return {"gamefile": self._gamefile, "task_type": task_type(self._gamefile)}

    def close(self) -> None:
        if self._env is not None:
            try:
                self._env.close()
            finally:
                self._env = None


def tasks(split: str, limit: int = 0, seed: int = 0) -> List[Dict[str, Any]]:
    """Dataset task provider: one task per game file (train games shuffled with ``seed``)."""
    games = list_games(split)
    if split == "train":
        import random
        random.Random(seed).shuffle(games)
    games = games[:limit] if limit > 0 else games
    return [{"gamefile": g, "seed": i} for i, g in enumerate(games)]
