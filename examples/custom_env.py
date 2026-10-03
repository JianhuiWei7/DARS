"""A custom environment: guess a hidden word from yes/no feedback on letters.

Use it without registering anything, by import path (the module must be importable on every node):

    judgerl-data --env examples.custom_env:GuessWordEnv --tasks-fn examples.custom_env:tasks \
                 --train 0 --val 0 --out data/guess              # or --tasks my_tasks.jsonl
    judgerl-train judgerl.env=examples.custom_env:GuessWordEnv ++judgerl.env_kwargs.max_steps=8 ...

What matters for Judge RL:
  * ``Observation.anchor`` names the state the step was taken from. Rollouts of the same task that
    reach the same anchor form a GiGPO step group, so make it a canonical state id (not the prompt).
  * ``Observation.text`` is what a judge sees as the step's state; ``StepResult.observation.text`` is
    what it sees as the step's result.
  * ``info["is_action_valid"]`` drives the invalid-action penalty; ``info["won"]`` the success flag.
  * ``metadata()`` goes to reward programs and judges only, never to the policy.
  * ``thread_safe = True`` runs episodes as threads; leave it False if the env has global state (each
    episode then runs in a worker process).
"""
from __future__ import annotations

import random
import re
from typing import Any, Dict, List

from judgerl.envs.base import Env, Memory, Observation, StepResult

WORDS = ["apple", "brick", "cloud", "dance", "eagle", "flame", "grape", "house", "juice", "lemon"]
_ACTION = re.compile(r"<action>\s*(letter|guess)\s+([a-z]+)\s*</action>", re.I)


class GuessWordEnv(Env):
    thread_safe = True

    def __init__(self, max_steps: int = 8, history: int = 3):
        self.max_steps = max_steps
        self.memory = Memory(history)
        self.word, self.known = "", {}

    def reset(self, task: Dict[str, Any], seed: int) -> Observation:
        self.word = task.get("word") or random.Random(seed).choice(WORDS)
        self.known = {}                     # letter -> present?
        self.memory = Memory(self.memory.k)
        return self._observe()

    def _state(self) -> str:
        return ", ".join(f"{c}:{'yes' if v else 'no'}" for c, v in sorted(self.known.items())) or "nothing yet"

    def _observe(self) -> Observation:
        hist = "".join(f"You did: {m['action']} -> {m['observation']}\n" for m in self.memory.recent())
        prompt = (f"Guess a hidden 5-letter English word. Known letters: {self._state()}.\n{hist}"
                  "Answer with <action>letter x</action> to test a letter or <action>guess word</action>.")
        return Observation(prompt=[{"role": "user", "content": prompt}], anchor=self._state(), text=self._state())

    def step(self, action: str) -> StepResult:
        m = _ACTION.search(action or "")
        if not m:
            self.memory.add("invalid action", (action or "")[-30:])
            return StepResult(self._observe(), 0.0, False, {"is_action_valid": False, "won": False})
        kind, arg = m.group(1).lower(), m.group(2).lower()
        if kind == "letter":
            self.known[arg[0]] = arg[0] in self.word
            self.memory.add(f"{arg[0]} is {'in' if self.known[arg[0]] else 'not in'} the word", m.group(0))
            return StepResult(self._observe(), 0.0, False, {"is_action_valid": True, "won": False})
        won = arg == self.word
        self.memory.add("correct" if won else "wrong guess", m.group(0))
        return StepResult(self._observe(), 10.0 if won else 0.0, won, {"is_action_valid": True, "won": won})

    def task_text(self) -> str:
        return "guess the hidden 5-letter word"

    def metadata(self) -> Dict[str, Any]:
        return {"reference": self.word}


def tasks(split: str = "train", limit: int = 0, seed: int = 0) -> List[Dict[str, Any]]:
    """Task provider: one dict per task (passed to ``reset``); held-out words for validation."""
    words = WORDS[:8] if split == "train" else WORDS[8:]
    out = [{"word": w, "seed": i} for i, w in enumerate(words)]
    return out[:limit] if limit > 0 else out
