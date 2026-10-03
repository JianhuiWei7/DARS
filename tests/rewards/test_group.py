import json

import pytest

from judgerl.rewards.group import build_group_program


def episode(tag, score=0.0, success=False, n=2):
    rows = [{"step": t, "observation": f"obs {t}", "action": f"{tag} act {t}", "result": f"res {t}",
             "env_reward": 0.0} for t in range(n)]
    return {"rows": rows, "success": success, "episode_score": score}


def scripted(responder):
    return {"backend": {"type": "scripted", "responder": responder}, "retry": {"max_attempts": 1}}


async def test_listwise_scores_each_episode_and_mixes():
    def responder(call):
        body = call.messages[-1]["content"]
        assert body.count("=== ATTEMPT") == 3 and "TASK: find the key" in body
        return json.dumps({"scores": [1.0, 0.5, 0.0]})
    p = build_group_program({"type": "listwise", "judge": scripted(responder), "weight": 0.5, "scale": 10.0})
    recs = await p.ascore_group(task="find the key", metadata={},
                                episodes=[episode("a", 10.0), episode("b"), episode("c")])
    assert [r.status for r in recs] == ["ok"] * 3
    assert [r.episode_score for r in recs] == pytest.approx([10.0, 2.5, 0.0])


async def test_listwise_wrong_length_fails_every_episode():
    p = build_group_program({"type": "listwise", "judge": scripted(lambda call: json.dumps({"scores": [1.0]}))})
    recs = await p.ascore_group(task="t", metadata={}, episodes=[episode("a"), episode("b")])
    assert [r.status for r in recs] == ["failed", "failed"] and all(r.episode_score is None for r in recs)


async def test_pairwise_win_rates_with_swap():
    # the judge prefers the attempt whose actions are tagged "good", whichever position it is in
    def responder(call):
        body = call.messages[-1]["content"]
        a, b = body.split("=== ATTEMPT B ===")
        if "good" in a and "good" not in b:
            return json.dumps({"winner": "A"})
        if "good" in b and "good" not in a:
            return json.dumps({"winner": "B"})
        return json.dumps({"winner": "tie"})
    p = build_group_program({"type": "pairwise", "judge": scripted(responder), "weight": 1.0, "scale": 1.0})
    recs = await p.ascore_group(task="t", metadata={}, episodes=[episode("good"), episode("bad1"), episode("bad2")])
    assert [r.episode_score for r in recs] == pytest.approx([1.0, 0.25, 0.25])
    assert all(r.info["games"] == 2 for r in recs)


async def test_pairwise_position_bias_becomes_tie():
    p = build_group_program({"type": "pairwise", "judge": scripted(lambda call: json.dumps({"winner": "A"})),
                             "weight": 1.0, "scale": 1.0})
    recs = await p.ascore_group(task="t", metadata={}, episodes=[episode("x"), episode("y")])
    assert [r.episode_score for r in recs] == pytest.approx([0.5, 0.5])


async def test_pairwise_single_episode_is_skipped():
    p = build_group_program({"type": "pairwise", "judge": scripted(lambda call: "{}")})
    assert [r.status for r in await p.ascore_group(task="t", metadata={}, episodes=[episode("x")])] == ["skipped"]
