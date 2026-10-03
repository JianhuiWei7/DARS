"""NumberLine: a tiny text environment for smoke tests and CI (no external dependencies).

The agent stands at an integer position and must reach a target position by moving left or right.
It is solved in a few steps, states repeat across rollouts (so GiGPO step groups are non-trivial),
and invalid actions are easy to produce.
"""
from __future__ import annotations

import random
import re
from typing import Any, Dict

from judgerl.envs.base import Env, Memory, Observation, StepResult, register_env

_ACTION = re.compile(r"<action>\s*(left|right)\s*</action>", re.I)

PROMPT = (
    "You are on a number line at position {pos}. Your goal is to reach position {target}.\n"
    "{history}"
    "Think briefly, then answer with exactly one action: <action>left</action> (move -1) or "
    "<action>right</action> (move +1)."
)


@register_env("numberline")
class NumberLineEnv(Env):
    thread_safe = True
    def __init__(self, size: int = 10, max_steps: int = 12, history: int = 2):
        self.size = size
        self.max_steps = max_steps
        self.memory = Memory(history)
        self.pos = self.target = 0

    def reset(self, task: Dict[str, Any], seed: int) -> Observation:
        rng = random.Random(task.get("seed", seed))
        self.target = int(task.get("target", rng.randint(0, self.size)))
        self.pos = int(task.get("start", rng.randint(0, self.size)))
        self.memory = Memory(self.memory.k)
        return self._observe()

    def _observe(self) -> Observation:
        hist = "".join(f"Earlier: at {m['observation']} you chose {m['action']}.\n" for m in self.memory.recent())
        text = f"position {self.pos}, target {self.target}"
        return Observation(prompt=[{"role": "user", "content": PROMPT.format(pos=self.pos, target=self.target, history=hist)}],
                           anchor=f"pos={self.pos};target={self.target}", text=text)

    def step(self, action: str) -> StepResult:
        m = _ACTION.search(action or "")
        valid = m is not None
        before = f"position {self.pos}"
        if valid:
            self.pos = max(0, min(self.size, self.pos + (1 if m.group(1).lower() == "right" else -1)))
        self.memory.add(before, m.group(0) if m else "an invalid action")
        won = self.pos == self.target
        return StepResult(self._observe(), 10.0 if won else 0.0, won,
                          {"won": won, "is_action_valid": valid})

    def task_text(self) -> str:
        return f"reach position {self.target} on the number line"
