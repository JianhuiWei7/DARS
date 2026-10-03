"""Judge-based reward programs: outcome rubric, per-step process scores, first-error localization.

All three render the episode the same way: each step shows the state it was taken from, the action,
and the environment's response to that action, so the judge credits the step that caused an effect.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from judgerl.rewards.base import RewardProgram, RewardRecord, combine
from judgerl.rewards.client import shared_client


def clip_text(text: str, n: int) -> str:
    text = text or ""
    return text if n <= 0 or len(text) <= n else text[: n // 2] + " ...[clipped]... " + text[-n // 2:]


def render_steps(rows: List[Dict[str, Any]], max_obs_chars: int = 600) -> str:
    lines = []
    for i, r in enumerate(rows, 1):
        lines.append(f"[Step {i}] State: {clip_text(r.get('observation', ''), max_obs_chars)}\n"
                     f"[Step {i}] Action: {(r.get('action') or '').strip()}\n"
                     f"[Step {i}] Result: {clip_text(r.get('result', ''), max_obs_chars)}")
    return f"TRAJECTORY ({len(rows)} steps):\n" + "\n".join(lines)


def render_episode(task: str, rows: List[Dict[str, Any]], max_obs_chars: int = 600) -> str:
    return f"TASK: {task}\n\n" + render_steps(rows, max_obs_chars)


class _JudgedProgram(RewardProgram):
    def __init__(self, judge: Dict[str, Any], max_obs_chars: int = 600, show_outcome: bool = True,
                 prompt: Optional[str] = None, tags: Optional[Dict[str, str]] = None, **judge_request):
        self.judge_cfg = judge
        self.max_obs_chars = max_obs_chars
        self.show_outcome = show_outcome
        self.prompt = prompt or self.default_prompt
        self.tags = dict(tags or {}, program=self.name)
        self.request_kwargs = judge_request      # decoding / reasoning / timeout_s overrides

    default_prompt = ""
    schema: Dict[str, Any] = {}

    def messages(self, task, metadata, rows, success) -> List[Dict[str, str]]:
        body = render_episode(task, rows, self.max_obs_chars)
        if self.show_outcome:
            body += f"\n\nOUTCOME: the environment scored this episode as {'SUCCEEDED' if success else 'FAILED'}."
        return [{"role": "system", "content": self.prompt}, {"role": "user", "content": body}]

    def validator(self, rows):
        """``parsed -> error message or None``; an error is sent back to the judge as a correction."""
        return None

    async def _judge(self, task, metadata, rows, success):
        from judgerl.judge import JudgeRequest
        client = await shared_client(self.judge_cfg)
        req = JudgeRequest(messages=self.messages(task, metadata, rows, success), output_schema=self.schema,
                           validator=self.validator(rows), tags=self.tags, prompt_version=f"{self.name}-v1",
                           **self.request_kwargs)
        return await client.judge(req)


class OutcomeRubric(_JudgedProgram):
    """Episode score from a rubric: the judge returns a score in [0, 1] (checklist-weighted), which
    replaces or is mixed into the environment's episode score."""

    name = "outcome"
    default_prompt = ("You grade how well an agent accomplished a task. Read the task, the trajectory, and "
                      "the rubric, then return ONLY JSON {\"score\": <number between 0 and 1>, \"reason\": \"...\"}.")
    schema = {"type": "object", "properties": {"score": {"type": "number", "minimum": 0, "maximum": 1},
                                               "reason": {"type": "string"}}, "required": ["score"]}

    def __init__(self, judge, rubric: str = "", scale: float = 10.0, weight: float = 1.0, **kw):
        super().__init__(judge, **kw)
        self.rubric, self.scale, self.weight = rubric, scale, weight

    def messages(self, task, metadata, rows, success):
        msgs = super().messages(task, metadata, rows, success)
        if self.rubric:
            msgs[1]["content"] += f"\n\nRUBRIC:\n{self.rubric}"
        return msgs

    async def ascore(self, *, task, metadata, rows, success, episode_score):
        r = await self._judge(task, metadata, rows, success)
        if not r.ok:
            return RewardRecord(status="failed", info={"judge_status": str(r.status), "error": r.error})
        judged = self.scale * float(r.parsed["score"])
        return RewardRecord(episode_score=self.weight * judged + (1 - self.weight) * episode_score,
                            info={"judge_score": judged, "reason": r.parsed.get("reason", "")})


class ProcessScores(_JudgedProgram):
    """Per-step scores: the judge rates every step in {-1, 0, 1} (harmful / neutral / progress); the
    scores times ``scale`` become step credit, combined with the environment rewards."""

    name = "process"
    default_prompt = ("You grade each step of an agent trajectory. For every step, judge by that step's Result "
                      "whether its Action made real progress toward the task (1), made no difference (0), or "
                      "was a mistake that set the agent back (-1). Return ONLY JSON {\"scores\": [one integer per "
                      "step, in order]}.")
    schema = {"type": "object", "properties": {"scores": {"type": "array", "items": {"type": "integer",
                                                                                      "minimum": -1, "maximum": 1}}},
              "required": ["scores"]}

    def __init__(self, judge, scale: float = 0.3, combine: str = "replace", alpha: float = 0.25,
                 gamma: float = 0.95, **kw):
        super().__init__(judge, **kw)
        self.scale, self.mode, self.alpha, self.gamma = scale, combine, alpha, gamma

    def validator(self, rows):
        n = len(rows)

        def check(parsed):
            k = len(parsed["scores"])
            return None if k == n else (f"You returned {k} scores but the trajectory has {n} steps. Return exactly "
                                        f"{n} integers, one per step in order.")
        return check

    async def ascore(self, *, task, metadata, rows, success, episode_score):
        r = await self._judge(task, metadata, rows, success)
        if not r.ok or len(r.parsed["scores"]) != len(rows):
            return RewardRecord(status="failed", info={"judge_status": str(r.status), "error": r.error or "length"})
        credit = [self.scale * float(s) for s in r.parsed["scores"]]
        env_rewards = [float(x["env_reward"]) for x in rows]
        return RewardRecord(step_rewards=combine(env_rewards, credit, self.mode, self.alpha, self.gamma),
                            info={"scores": r.parsed["scores"]})


class FirstError(_JudgedProgram):
    """First-error localization: on a failed episode, the judge names the first step that went wrong;
    steps before it receive ``credit`` each and later steps nothing. Successful episodes keep their
    environment rewards."""

    name = "first_error"
    default_prompt = ("An agent failed a task. Find the FIRST step whose action took it off a correct path (a "
                      "wrong or wasteful action given what it knew). Reasonable exploration is not an error. "
                      "Return ONLY JSON {\"first_error_step\": <1-based step>, \"reason\": \"...\"}.")
    schema = {"type": "object", "properties": {"first_error_step": {"type": "integer", "minimum": 1},
                                               "reason": {"type": "string"}}, "required": ["first_error_step"]}

    def __init__(self, judge, credit: float = 0.1, combine: str = "add", **kw):
        super().__init__(judge, **kw)
        self.credit, self.mode = credit, combine

    def validator(self, rows):
        n = len(rows)

        def check(parsed):
            k = parsed["first_error_step"]
            return None if 1 <= k <= n else f"first_error_step must be between 1 and {n} (the trajectory has {n} steps)."
        return check

    async def ascore(self, *, task, metadata, rows, success, episode_score):
        if success:
            return RewardRecord(status="skipped")
        r = await self._judge(task, metadata, rows, success)
        if not r.ok:
            return RewardRecord(status="failed", info={"judge_status": str(r.status), "error": r.error})
        k = max(1, min(len(rows), int(r.parsed["first_error_step"])))
        credit = [self.credit if t < k - 1 else 0.0 for t in range(len(rows))]
        env_rewards = [float(x["env_reward"]) for x in rows]
        return RewardRecord(step_rewards=combine(env_rewards, credit, self.mode), info={"first_error_step": k})
