"""PythonTool: math / QA with a Python interpreter (ReAct-style tool use).

Each step the policy writes either

* ``<python>...</python>``: the code runs in a sandbox (:mod:`judgerl.sandbox`, a fresh interpreter
  per call, so variables do not persist between calls) and its output comes back as a
  ``<result>...</result>`` message, or
* ``<answer>...</answer>``: the episode ends and a programmatic verifier compares the answer with the
  gold answer: reward 10 when they are equivalent, else 0.

A response with neither tag is an invalid action (``is_action_valid=False``); the policy is reminded
of the format and the episode continues. The first complete tag after any ``<think>...</think>``
block counts. The episode also ends, with reward 0, after ``max_steps`` actions.

Tasks are ``{"question": ..., "answer": ...}`` rows. The gold answer is kept in :meth:`metadata`
(for judges and verifiers) and never appears in the policy prompt. The anchor is a hash of the
normalized prompt, so rollouts that reach the same visible state share a GiGPO step group.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from fractions import Fraction
from typing import Any, Dict, List, Optional

from judgerl.envs.base import Env, Observation, StepResult, register_env

SYSTEM_PROMPT = """You solve problems step by step and may use a Python interpreter.

Each of your turns must contain exactly one of:
- <python>code</python> to run Python code. Each call starts a fresh interpreter (variables do not persist between calls), so print() everything you need. The output is returned to you inside <result></result>.
- <answer>final answer</answer> to give your final answer. This ends the episode. Put only the final answer (a number or a short expression) inside the tags.

You may reason before the tag. You have at most {max_steps} turns."""

FORMAT_REMINDER = ("Invalid response: it contains neither <python>...</python> nor <answer>...</answer>. "
                   "Use exactly one of the two tags.")

_THINK = re.compile(r"<think>.*?</think>", re.S | re.I)
_TAG = re.compile(r"<(python|answer)>(.*?)</\1>", re.S | re.I)
_FENCE = re.compile(r"^\s*```(?:python|py)?\s*\n(.*?)\n?\s*```\s*$", re.S)

#: Environment variable holding the JSONL task file (``{split}`` in the path is replaced by the split).
TASKS_ENV = "JUDGERL_PYTHON_TOOL_TASKS"

SAMPLE_TASKS: List[Dict[str, Any]] = [
    {"id": "sample-0", "question": "What is the sum of the first 100 positive integers?", "answer": "5050"},
    {"id": "sample-1", "question": "What is 2 to the power of 20?", "answer": "1048576"},
    {"id": "sample-2", "question": "How many primes are there below 100?", "answer": "25"},
    {"id": "sample-3", "question": "Simplify 84/126 to lowest terms.", "answer": "\\frac{2}{3}"},
]


# ---------------------------------------------------------------------- parsing
def parse_action(response: str) -> tuple[str, str]:
    """Return ``(kind, content)`` with kind ``python``, ``answer`` or ``invalid``."""
    text = _THINK.sub("", response or "")
    m = _TAG.search(text)
    if m is None:
        return "invalid", ""
    kind, body = m.group(1).lower(), m.group(2)
    if kind == "python":
        fenced = _FENCE.match(body)
        body = (fenced.group(1) if fenced else body).strip("\n")
        if not body.strip():
            return "invalid", ""
        return "python", body
    return "answer", body.strip()


# ---------------------------------------------------------------------- verification
def _unbox(s: str) -> str:
    i = s.rfind("\\boxed")
    if i == -1:
        return s
    j = s.find("{", i)
    if j == -1:
        return s
    depth = 0
    for k in range(j, len(s)):
        depth += {"{": 1, "}": -1}.get(s[k], 0)
        if depth == 0:
            return s[j + 1:k]
    return s


def normalize_answer(s: str) -> str:
    """Normalization for string comparison: boxes, math delimiters, spacing, LaTeX spelling variants."""
    s = _unbox(str(s or "")).strip()
    s = s.strip("$").strip()
    s = re.sub(r"\\text\{\s*([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\(?:mathrm|mathbf)\{([^{}]*)\}", r"\1", s)
    for a, b in (("\\dfrac", "\\frac"), ("\\tfrac", "\\frac"), ("\\left", ""), ("\\right", ""), ("\\!", ""),
                 ("\\,", ""), ("\\;", ""), ("\\ ", ""), ("^\\circ", ""), ("^{\\circ}", ""), ("\\%", ""), ("%", "")):
        s = s.replace(a, b)
    s = re.sub(r"^(?:the\s+)?(?:final\s+)?answer\s*(?:is|:)\s*", "", s, flags=re.I)
    s = re.sub(r"(\d),(?=\d{3}\b)", r"\1", s)            # 1,000 -> 1000
    s = re.sub(r"\s+", "", s).rstrip(".").lower()
    return s


def _to_number(s: str) -> Optional[Fraction]:
    m = re.fullmatch(r"\\frac\{(-?[\d.]+)\}\{(-?[\d.]+)\}", s) or re.fullmatch(r"(-?[\d.]+)/(-?[\d.]+)", s)
    try:
        if m:
            den = Fraction(m.group(2))
            return Fraction(m.group(1)) / den if den else None
        m2 = re.fullmatch(r"-\\frac\{([\d.]+)\}\{([\d.]+)\}", s)
        if m2:
            den = Fraction(m2.group(2))
            return -Fraction(m2.group(1)) / den if den else None
        return Fraction(s) if re.fullmatch(r"-?\d+(?:\.\d+)?(?:e-?\d+)?", s) else None
    except (ValueError, ZeroDivisionError):
        return None


def fallback_equivalent(pred: str, gold: str, rel_tol: float = 1e-6) -> bool:
    """Normalized string equality, or numeric equality (integers, decimals, fractions)."""
    a, b = normalize_answer(pred), normalize_answer(gold)
    if not a:
        return False
    if a == b:
        return True
    x, y = _to_number(a), _to_number(b)
    if x is None or y is None:
        return False
    return abs(x - y) <= rel_tol * max(1, abs(y))


def _math_verify_equivalent(pred: str, gold: str) -> Optional[bool]:
    """``math_verify`` verdict, or ``None`` when it is not installed or fails. Its signal-based
    timeouts only work in the main thread, so they are disabled (environments run in threads)."""
    try:
        from math_verify import parse, verify
    except ImportError:
        return None
    try:
        def p(s: str):
            s = s.strip()
            return parse(s if "$" in s or "\\boxed" in s else f"${s}$", parsing_timeout=None)
        g, t = p(gold), p(pred)
        if not g or not t:
            return None
        return bool(verify(g, t, timeout_seconds=None))
    except Exception:  # noqa: BLE001  (any parser failure falls back to the string comparison)
        return None


def verify_answer(pred: str, gold: str) -> bool:
    """True when ``pred`` is equivalent to ``gold``: either the normalized comparison or
    ``math_verify`` (when installed) accepts it."""
    if fallback_equivalent(pred, gold):
        return True
    return bool(_math_verify_equivalent(pred, gold))


# ---------------------------------------------------------------------- helpers
def clip_middle(text: str, n: int) -> str:
    if n <= 0 or len(text) <= n:
        return text
    return text[: n // 2] + f"\n...[{len(text) - n} characters clipped]...\n" + text[-n // 2:]


def state_hash(messages: List[Dict[str, Any]]) -> str:
    """Stable anchor for a prompt: whitespace-normalized messages, hashed."""
    norm = [[m.get("role", ""), re.sub(r"\s+", " ", str(m.get("content", ""))).strip()] for m in messages]
    return hashlib.sha256(json.dumps(norm, ensure_ascii=False).encode("utf-8")).hexdigest()[:24]


def format_result(r, max_chars: int) -> str:
    """Tool output shown to the policy: stdout, then stderr, then notes on limits that were hit."""
    parts = []
    if r.stdout:
        parts.append(r.stdout.rstrip("\n"))
    if r.stderr:
        parts.append(r.stderr.rstrip("\n"))
    if r.timed_out:
        parts.append("[execution stopped: time limit exceeded]")
    if r.oom:
        parts.append("[execution stopped: memory limit exceeded]")
    if r.truncated:
        parts.append("[output truncated]")
    if not parts:
        parts.append("[no output; use print() to see values]" if r.exit_code == 0 else f"[exit code {r.exit_code}]")
    return clip_middle("\n".join(parts), max_chars)


# ---------------------------------------------------------------------- environment
@register_env("python_tool")
class PythonToolEnv(Env):
    """Args:
        max_steps: actions per episode (tool calls plus the answer).
        history: tool calls (code and result) kept in the prompt; older ones are summarized as a count.
        sandbox: sandbox configuration for :func:`judgerl.sandbox.shared_sandbox` (one pool per
            configuration and process).
        timeout_s: wall-clock limit per tool call.
        max_result_chars: tool output shown to the policy (middle clipped).
        max_code_chars: longer code blocks are rejected without running.
        correct_reward: reward of a correct answer.
    """

    thread_safe = True     # per-instance state only; the shared sandbox pool is lock-protected

    def __init__(self, max_steps: int = 6, history: int = 4, sandbox: Optional[Dict[str, Any]] = None,
                 timeout_s: float = 10.0, max_result_chars: int = 2000, max_code_chars: int = 20000,
                 correct_reward: float = 10.0):
        self.max_steps = max_steps
        self.history = history
        self.sandbox_config = dict(sandbox or {"backend": "process"})
        self.timeout_s = timeout_s
        self.max_result_chars = max_result_chars
        self.max_code_chars = max_code_chars
        self.correct_reward = correct_reward
        self._sandbox = None
        self._task: Dict[str, Any] = {}
        self._question = ""
        self._gold = ""
        self._calls: List[Dict[str, str]] = []
        self._steps = 0

    @property
    def sandbox(self):
        if self._sandbox is None:
            from judgerl.sandbox import shared_sandbox
            self._sandbox = shared_sandbox(self.sandbox_config)
        return self._sandbox

    def reset(self, task: Dict[str, Any], seed: int) -> Observation:
        question = task.get("question", task.get("problem"))
        gold = task.get("answer", task.get("gold"))
        if question is None or gold is None:
            raise ValueError("python_tool task needs 'question' and 'answer'")
        self._task = dict(task)
        self._question, self._gold = str(question), str(gold)
        self._calls = []
        self._steps = 0
        return self._observe(self._question)

    # ------------------------------------------------------------------ prompt
    def _messages(self) -> List[Dict[str, str]]:
        msgs = [{"role": "system", "content": SYSTEM_PROMPT.format(max_steps=self.max_steps)},
                {"role": "user", "content": self._question}]
        recent = self._calls[-self.history:] if self.history > 0 else []
        omitted = len(self._calls) - len(recent)
        if omitted:
            msgs.append({"role": "user", "content": f"({omitted} earlier turn(s) omitted.)"})
        for c in recent:
            msgs.append({"role": "assistant", "content": c["action"]})
            msgs.append({"role": "user", "content": c["observation"]})
        left = self.max_steps - self._steps
        if self._steps:
            msgs[-1] = dict(msgs[-1], content=msgs[-1]["content"] + f"\n\nTurns left: {left}."
                            + (" This is your last turn: give your <answer> now." if left == 1 else ""))
        return msgs

    def _observe(self, text: str) -> Observation:
        msgs = self._messages()
        return Observation(prompt=msgs, anchor=state_hash(msgs), text=text, info={"tool_calls": self.n_tool_calls})

    @property
    def n_tool_calls(self) -> int:
        return sum(1 for c in self._calls if c["kind"] == "python")

    # ------------------------------------------------------------------ step
    def step(self, action: str) -> StepResult:
        self._steps += 1
        kind, content = parse_action(action)
        info: Dict[str, Any] = {"action_type": kind, "is_action_valid": kind != "invalid", "won": False}
        if kind == "answer":
            correct = verify_answer(content, self._gold)
            info.update(won=correct, answer=content)
            text = f"Final answer submitted: {content}"
            return StepResult(self._observe(text), self.correct_reward if correct else 0.0, True, info)
        if kind == "python":
            if len(content) > self.max_code_chars:
                text = f"<result>\n[code rejected: longer than {self.max_code_chars} characters]\n</result>"
                info.update(exit_code=None)
            else:
                r = self.sandbox.run(content, timeout_s=self.timeout_s)
                text = f"<result>\n{format_result(r, self.max_result_chars)}\n</result>"
                info.update(exit_code=r.exit_code, timed_out=r.timed_out, oom=r.oom, truncated=r.truncated,
                            exec_s=r.duration_s)
            self._calls.append({"kind": "python", "action": f"<python>\n{content}\n</python>", "observation": text})
        else:
            text = FORMAT_REMINDER
            # a fixed placeholder (not the raw text) keeps the history short and anchors shareable
            self._calls.append({"kind": "invalid", "action": "(a response without a <python> or <answer> tag)",
                                "observation": text})
        done = self._steps >= self.max_steps
        if done:
            info["out_of_steps"] = True
        return StepResult(self._observe(text), 0.0, done, info)

    def task_text(self) -> str:
        return self._question

    def metadata(self) -> Dict[str, Any]:
        return {"id": self._task.get("id"), "question": self._question, "answer": self._gold,
                "reference": self._gold}


# ---------------------------------------------------------------------- tasks
def tasks(split: str = "train", limit: int = 0, seed: int = 0, path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Dataset task provider.

    Reads JSONL rows with ``question`` and ``answer`` (``problem`` / ``gold`` also accepted) from
    ``path`` or ``$JUDGERL_PYTHON_TOOL_TASKS``; ``{split}`` in the path is replaced by the split name.
    ``split="sample"`` returns the built-in :data:`SAMPLE_TASKS`. Train rows are shuffled with ``seed``.
    """
    if split == "sample" and path is None:
        rows = [dict(t) for t in SAMPLE_TASKS]
    else:
        path = path or os.environ.get(TASKS_ENV)
        if not path:
            raise ValueError(f"python_tool tasks: pass path= or set ${TASKS_ENV} (or use split='sample')")
        path = os.path.expanduser(path.replace("{split}", split))
        rows = []
        with open(path) as f:
            for i, line in enumerate(f):
                if not line.strip():
                    continue
                row = json.loads(line)
                q, a = row.get("question", row.get("problem")), row.get("answer", row.get("gold"))
                if q is None or a is None:
                    raise ValueError(f"{path}:{i + 1}: row needs 'question' and 'answer'")
                rows.append({"id": row.get("id", f"{split}-{i}"), "question": str(q), "answer": str(a)})
    if split == "train":
        import random
        random.Random(seed).shuffle(rows)
    rows = rows[:limit] if limit > 0 else rows
    return [dict(r, seed=i) for i, r in enumerate(rows)]
