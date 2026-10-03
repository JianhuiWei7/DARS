"""PythonTool environment with scripted actions (no network, no GPU)."""
from __future__ import annotations

import json

import pytest

from judgerl.envs import make_env
from judgerl.envs.python_tool import (SAMPLE_TASKS, TASKS_ENV, PythonToolEnv, fallback_equivalent, parse_action,
                                      tasks, verify_answer)

TASK = {"id": "t0", "question": "What is the sum of the first 100 positive integers?", "answer": "5050"}


@pytest.fixture
def env(tmp_path):
    root = tmp_path / "sbx"
    root.mkdir()
    return make_env("python_tool", max_steps=4, sandbox={"backend": "process", "tmp_root": str(root),
                                                         "max_concurrency": 2}, timeout_s=5)


def prompt_text(obs) -> str:
    return "\n".join(m["content"] for m in obs.prompt)


def test_registered_and_gold_hidden(env):
    assert isinstance(env, PythonToolEnv)
    obs = env.reset(TASK, seed=0)
    assert obs.prompt[0]["role"] == "system" and "<python>" in obs.prompt[0]["content"]
    assert TASK["question"] in prompt_text(obs) and "5050" not in prompt_text(obs)
    meta = env.metadata()
    assert meta["answer"] == "5050" and meta["reference"] == "5050" and env.task_text() == TASK["question"]


def test_tool_call_then_correct_answer(env):
    env.reset(TASK, seed=0)
    r1 = env.step("<think>Let me compute.</think>I will compute it. <python>print(sum(range(1, 101)))</python>")
    assert r1.reward == 0 and not r1.done and r1.info["is_action_valid"] and r1.info["action_type"] == "python"
    assert r1.info["exit_code"] == 0
    assert "<result>\n5050\n</result>" in r1.observation.text
    assert "print(sum(range(1, 101)))" in prompt_text(r1.observation) and "5050" in prompt_text(r1.observation)
    r2 = env.step("The tool says 5050. <answer>\\boxed{5050}</answer>")
    assert r2.done and r2.reward == 10.0 and r2.info["won"] and r2.info["action_type"] == "answer"


def test_wrong_answer_scores_zero(env):
    env.reset(TASK, seed=0)
    r = env.step("<answer>5000</answer>")
    assert r.done and r.reward == 0.0 and not r.info["won"] and r.info["is_action_valid"]


def test_invalid_action_flag_and_reminder(env):
    env.reset(TASK, seed=0)
    r = env.step("I think the answer is 5050.")
    assert not r.info["is_action_valid"] and r.info["action_type"] == "invalid" and not r.done and r.reward == 0
    assert "neither <python>" in r.observation.text
    r = env.step("<python>   </python>")                    # empty code is invalid too
    assert not r.info["is_action_valid"]


def test_errors_and_timeouts_are_observations(env):
    env.reset(TASK, seed=0)
    r = env.step("<python>\n```python\n1/0\n```\n</python>")
    assert "ZeroDivisionError" in r.observation.text and r.info["exit_code"] != 0
    r = env.step("<python>import time\ntime.sleep(30)</python>")
    assert r.info["timed_out"] and "time limit exceeded" in r.observation.text


def test_episode_ends_after_max_steps(env):
    env.reset(TASK, seed=0)
    results = [env.step("<python>print(1)</python>") for _ in range(4)]
    assert [r.done for r in results] == [False, False, False, True]
    assert results[-1].info.get("out_of_steps") and results[-1].reward == 0
    assert "last turn" in prompt_text(results[-2].observation)


def test_anchor_stability(tmp_path):
    def play(actions, **kw):
        env = PythonToolEnv(max_steps=5, sandbox={"backend": "process", "tmp_root": str(tmp_path)}, **kw)
        anchors = [env.reset(TASK, seed=0).anchor]
        anchors += [env.step(a).observation.anchor for a in actions]
        return anchors

    a = play(["<python>print(2+2)</python>", "thinking only"])
    b = play(["Some other reasoning.  <python>print(2+2)</python>", "<think>x</think>still no tag"])
    c = play(["<python>print(2+3)</python>"])
    assert a == b                        # same visible state -> same anchor, whatever the reasoning text
    assert a[0] == c[0] and a[1] != c[1]  # the initial state is shared; different results differ
    assert len(set(a)) == len(a)
    other = PythonToolEnv(sandbox={"backend": "process", "tmp_root": str(tmp_path)}).reset(
        dict(TASK, question="What is 1+1?"), seed=0)
    assert other.anchor != a[0]


def test_history_is_bounded(tmp_path):
    env = PythonToolEnv(max_steps=10, history=2, sandbox={"backend": "process", "tmp_root": str(tmp_path)})
    env.reset(TASK, seed=0)
    for i in range(4):
        obs = env.step(f"<python>print({i} * 11)</python>").observation
    text = prompt_text(obs)
    assert "2 earlier turn(s) omitted" in text and "print(0 * 11)" not in text and "print(3 * 11)" in text


def test_parse_action():
    assert parse_action("<answer>3</answer><python>x</python>") == ("answer", "3")
    assert parse_action("<think><answer>1</answer></think><python>print(1)</python>") == ("python", "print(1)")
    assert parse_action("<python>print(1)") == ("invalid", "")


@pytest.mark.parametrize("pred,gold,ok", [
    ("5050", "5050", True), ("\\boxed{5,050}", "5050", True), ("$5050.$", "5050", True),
    ("0.5", "\\frac{1}{2}", True), ("1/2", "\\dfrac{1}{2}", True), ("2/3", "\\frac{2}{3}", True),
    ("-\\frac{3}{4}", "-0.75", True), ("The answer is 12", "12", True), ("x+1", "x + 1", True),
    ("5000", "5050", False), ("", "0", False), ("\\text{Monday}", "monday", True),
])
def test_verifier(pred, gold, ok):
    assert fallback_equivalent(pred, gold) is ok
    assert verify_answer(pred, gold) is ok


def test_math_verify_path_when_installed():
    pytest.importorskip("math_verify")
    assert verify_answer("\\sqrt{2}/2", "\\frac{\\sqrt{2}}{2}")
    assert not verify_answer("\\sqrt{3}", "\\frac{\\sqrt{2}}{2}")
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(2) as ex:                  # environments verify from worker threads
        assert ex.submit(verify_answer, "\\sqrt{2}/2", "\\frac{\\sqrt{2}}{2}").result()


def test_tasks_provider(tmp_path, monkeypatch):
    assert len(tasks("sample")) == len(SAMPLE_TASKS) and all("answer" in t for t in tasks("sample"))
    path = tmp_path / "math_{split}.jsonl"
    (tmp_path / "math_test.jsonl").write_text("\n".join(json.dumps({"question": f"q{i}", "answer": str(i)})
                                                        for i in range(5)) + "\n\n")
    monkeypatch.setenv(TASKS_ENV, str(path))
    rows = tasks("test", limit=3)
    assert [r["question"] for r in rows] == ["q0", "q1", "q2"] and rows[0]["id"] == "test-0"
    monkeypatch.delenv(TASKS_ENV)
    with pytest.raises(ValueError):
        tasks("test")
    (tmp_path / "math_train.jsonl").write_text(json.dumps({"problem": "p", "gold": "1"}) + "\n")
    assert tasks("train", path=str(path))[0]["question"] == "p"
