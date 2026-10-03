"""OpenEnded environment and its per-task rubric reward program, with a scripted judge."""
from __future__ import annotations

import json

import pytest

from judgerl.envs import make_env
from judgerl.envs.open_ended import OpenEndedEnv, RubricOutcome, normalize_rubric, tasks
from judgerl.rewards import build_reward_program

TASK = {"id": "haiku", "instruction": "Write a haiku about autumn rain.",
        "rubric": [{"item": "Exactly three lines", "weight": 1}, {"item": "Mentions rain", "weight": 3}],
        "followups": ["Make it sadder."]}


def play(env, task, actions):
    obs = env.reset(task, seed=0)
    rows = []
    for a in actions:
        r = env.step(a)
        rows.append({"step": len(rows), "anchor": obs.anchor, "observation": obs.text, "action": a,
                     "result": r.observation.text, "env_reward": r.reward,
                     "invalid": 0.0 if r.info["is_action_valid"] else 1.0, "won": r.info["won"]})
        obs = r.observation
        if r.done:
            break
    return rows, r


def judge(responses, seed):
    # a distinct seed per test gives each test its own judge client (clients are shared per config)
    return {"backend": {"type": "scripted", "responses": responses, "seed": seed}}


def test_multi_turn_flow_rubric_hidden():
    env = make_env("open_ended", max_steps=3)
    assert isinstance(env, OpenEndedEnv)
    obs = env.reset(TASK, seed=0)
    text = "\n".join(m["content"] for m in obs.prompt)
    assert TASK["instruction"] in text and "Mentions rain" not in text and "<final>" in text
    r = env.step("<think>hidden idea</think>Rain on red leaves...")
    assert not r.done and r.reward == 0 and r.info["is_action_valid"] and r.observation.text == "Make it sadder."
    assert "hidden idea" not in "\n".join(m["content"] for m in r.observation.prompt)
    r = env.step("<final>Cold rain on bare trees\nleaves drown in grey puddles\nno one walks tonight</final>")
    assert r.done and r.reward == 0 and r.info["final_tagged"] and r.info["is_action_valid"]
    assert env.final_response.startswith("Cold rain")
    assert env.metadata()["rubric"] == [{"item": "Exactly three lines", "weight": 1.0},
                                        {"item": "Mentions rain", "weight": 3.0}]


def test_max_steps_without_final_and_single_turn():
    env = OpenEndedEnv(max_steps=2)
    rows, last = play(env, TASK, ["first draft", "second draft", "never reached"])
    assert len(rows) == 2 and last.done and not last.info["is_action_valid"] and env.final_response == "second draft"
    single = OpenEndedEnv(max_steps=1)
    rows, last = play(single, dict(TASK, followups=[]), ["<final>done</final>"])
    assert len(rows) == 1 and last.done and single.final_response == "done"
    rows, last = play(single, dict(TASK, max_turns=3), ["draft", "<final>x"])   # per-task turn limit
    assert len(rows) == 2 and rows[1]["invalid"] == 1.0 and not last.done       # unclosed <final>


def test_anchor_stability():
    a = [OpenEndedEnv().reset(TASK, seed=s).anchor for s in (0, 1)]
    assert a[0] == a[1]
    e1, e2 = OpenEndedEnv(), OpenEndedEnv()
    e1.reset(TASK, seed=0)
    e2.reset(TASK, seed=0)
    assert e1.step("<think>a</think>same  draft").observation.anchor == e2.step("same draft").observation.anchor


def test_normalize_rubric():
    assert normalize_rubric(["a", {"criterion": "b", "weight": 2}]) == [{"item": "a", "weight": 1.0},
                                                                         {"item": "b", "weight": 2.0}]
    for bad in ([{"item": ""}], [{"item": "x", "weight": -1}], [{"item": "x", "weight": 0}], [3]):
        with pytest.raises(ValueError):
            normalize_rubric(bad)
    assert len(tasks("sample")) == 2


def program(judge_cfg, **kw):
    return build_reward_program({"type": "judgerl.envs.open_ended:RubricOutcome", "judge": judge_cfg, **kw})


async def score(p, env, rows, metadata=None):
    # reward programs get the task text and the metadata captured at the end of the episode
    return await p.ascore(task=env.task_text(), metadata=env.metadata() if metadata is None else metadata,
                          rows=rows, success=False, episode_score=0.0)


async def test_rubric_program_checklist_uses_task_rubric():
    env = OpenEndedEnv(max_steps=2)
    rows, _ = play(env, TASK, ["draft", "<final>rain falls</final>"])
    p = program(judge([{"met": [False, True], "reason": "two lines"}], seed=101))
    assert isinstance(p, RubricOutcome)
    rec = await score(p, env, rows)
    assert rec.status == "ok" and rec.info["met"] == [False, True]
    assert rec.episode_score == pytest.approx(7.5)             # 10 * 3 / (1 + 3)
    assert rec.step_rewards == pytest.approx([0.0, 7.5])       # the judged score lands on the final step


async def test_rubric_prompt_contents():
    from judgerl.rewards.client import shared_client
    cfg = judge([{"met": [True, True]}, {"met": [True, False]}], seed=102)
    env = OpenEndedEnv(max_steps=1)
    rows, _ = play(env, TASK, ["<think>secret plan</think><final>rain</final>"])
    meta = env.metadata()
    assert meta["final_response"] == "rain" and json.loads(json.dumps(meta)) == meta   # plain, picklable data
    rec = await score(program(cfg), env, rows)
    assert rec.episode_score == pytest.approx(10.0)
    backend = (await shared_client(cfg)).backend("scripted")
    user = backend.calls[-1].messages[-1]["content"]
    assert "1. Exactly three lines (weight 1)" in user and "2. Mentions rain (weight 3)" in user
    assert "FINAL RESPONSE:\n<<<\nrain\n>>>" in user and "OUTCOME" not in user and "secret plan" not in user
    # without final_response in the metadata, the final response is recovered from the last action
    rec = await score(program(cfg), env, rows, metadata={"rubric": meta["rubric"]})
    assert rec.episode_score == pytest.approx(2.5)
    assert "FINAL RESPONSE:\n<<<\nrain\n>>>" in backend.calls[-1].messages[-1]["content"]


async def test_rubric_program_score_mode_and_failures():
    env = OpenEndedEnv(max_steps=1)
    rows, _ = play(env, TASK, ["<final>rain</final>"])
    rec = await score(program(judge([{"score": 0.4}], seed=103), mode="score"), env, rows)
    assert rec.status == "ok" and rec.episode_score == pytest.approx(4.0)
    rec = await score(program(judge([{"met": [True]}] * 5, seed=104)), env, rows)   # 1 verdict for 2 items
    assert rec.status == "failed" and rec.episode_score is None and rec.step_rewards is None
    down = {"backend": {"type": "scripted", "responses": [], "sequence": ["auth"], "seed": 105}}
    rec = await score(program(down), env, rows)
    assert rec.status == "failed" and rec.episode_score is None


def test_envs_are_thread_safe():
    from judgerl.envs.python_tool import PythonToolEnv
    assert OpenEndedEnv.thread_safe and PythonToolEnv.thread_safe
