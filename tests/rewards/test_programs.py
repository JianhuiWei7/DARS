import json

import pytest

from judgerl.rewards import build_reward_program, combine


class FakeEnv:
    def __init__(self, task="put a mug in desk", meta=None):
        self._task, self._meta = task, meta or {}

    def task_text(self):
        return self._task

    def metadata(self):
        return self._meta


def rows(n=3, won=False):
    return [{"step": t, "anchor": f"s{t}", "observation": f"obs {t}", "action": f"act {t}", "result": f"res {t}",
             "env_reward": 10.0 if (won and t == n - 1) else 0.0, "invalid": 0.0, "won": won and t == n - 1}
            for t in range(n)]


def judge(responses, **extra):
    return {"backend": {"type": "scripted", "responses": responses, **extra}}


async def test_outcome_rubric_replaces_score():
    p = build_reward_program({"type": "outcome", "judge": judge([{"score": 0.4, "reason": "partial"}]), "rubric": "r"})
    rec = await p.ascore(task=FakeEnv().task_text(), metadata={}, rows=rows(), success=False, episode_score=0.0)
    assert rec.status == "ok" and rec.episode_score == pytest.approx(4.0)


async def test_process_scores_and_length_check():
    p = build_reward_program({"type": "process", "judge": judge([{"scores": [1, 0, -1]}]), "scale": 0.5})
    rec = await p.ascore(task=FakeEnv().task_text(), metadata={}, rows=rows(), success=False, episode_score=0.0)
    assert rec.step_rewards == pytest.approx([0.5, 0.0, -0.5])
    bad = build_reward_program({"type": "process", "judge": judge([{"scores": [1]}], seed=1)})
    assert (await bad.ascore(task=FakeEnv().task_text(), metadata={}, rows=rows(), success=False, episode_score=0.0)).status == "failed"


async def test_first_error_skips_success_and_credits_prefix():
    p = build_reward_program({"type": "first_error", "judge": judge([{"first_error_step": 3}]), "credit": 0.1})
    assert (await p.ascore(task=FakeEnv().task_text(), metadata={}, rows=rows(won=True), success=True, episode_score=10.0)).status == "skipped"
    rec = await p.ascore(task=FakeEnv().task_text(), metadata={}, rows=rows(4), success=False, episode_score=0.0)
    assert rec.step_rewards == pytest.approx([0.1, 0.1, 0.0, 0.0])


async def test_judge_failure_keeps_env_rewards():
    p = build_reward_program({"type": "outcome", "judge": judge([], sequence=["auth"]), "rubric": ""})
    rec = await p.ascore(task=FakeEnv().task_text(), metadata={}, rows=rows(), success=False, episode_score=3.0)
    assert rec.status == "failed" and rec.episode_score is None and rec.step_rewards is None


@pytest.fixture
def require_dars():
    pytest.importorskip("dars")


async def test_dars_program_matches_dars_package(require_dars):
    ann = {"nodes": ["locate", "pick", "place"], "turns": [
        {"turn_index": 1, "verified": ["locate"]}, {"turn_index": 2, "verified": ["pick"]},
        {"turn_index": 3, "errors": ["pick"]}]}
    p = build_reward_program({"type": "dars", "judge": judge([json.dumps(ann)]), "overrides": {"domain": "alfworld"}})
    rec = await p.ascore(task=FakeEnv().task_text(), metadata={}, rows=rows(), success=False, episode_score=0.0)
    assert rec.status == "ok" and rec.step_rewards == pytest.approx([0.3, 0.3, -0.3])  # 1/3 steps clipped at kappa


async def test_dars_corrective_reask_for_two_objects(require_dars):
    blind = {"nodes": ["locate", "pick", "place"], "turns": []}
    fixed = {"nodes": ["locate", "pick", "place", "locate:2", "pick:2", "place:2"], "turns": []}
    p = build_reward_program({"type": "dars", "judge": judge([json.dumps(blind), json.dumps(fixed)], seed=2),
                              "overrides": {"domain": "alfworld"}})
    rec = await p.ascore(task=FakeEnv("put two mug in desk").task_text(), metadata={}, rows=rows(), success=False, episode_score=0.0)
    assert rec.status == "ok" and "place:2" in rec.info["nodes"]


def test_combine_modes():
    assert combine([0, 0, 10], [0.3, 0.3, 0.3], "replace") == pytest.approx([0.3, 0.3, 0.3])
    assert combine([0, 0, 10], [0.3, 0.3, 0.3], "add") == pytest.approx([0.3, 0.3, 10.3])
    r = combine([0, 0, 10], [0.4, -0.2, 0.3], "redistribute", alpha=0.25, gamma=0.95)
    assert sum(0.95 ** t * x for t, x in enumerate(r)) == pytest.approx(0.95 ** 2 * 10)


async def test_dars_cached_result_still_gets_annotation(tmp_path, require_dars):
    ann = {"nodes": ["locate", "pick", "place"], "turns": [{"turn_index": 1, "verified": ["locate"]}]}
    cfg = {"backend": {"type": "scripted", "responses": [json.dumps(ann)], "seed": 11},
           "cache": {"path": str(tmp_path / "c.sqlite")}}
    p = build_reward_program({"type": "dars", "judge": cfg, "overrides": {"domain": "alfworld"}})
    first = await p.ascore(task="put a mug in desk", metadata={}, rows=rows(), success=False, episode_score=0.0)
    second = await p.ascore(task="put a mug in desk", metadata={}, rows=rows(), success=False, episode_score=0.0)
    assert first.status == second.status == "ok" and first.step_rewards == second.step_rewards


async def test_dars_bad_term_keeps_env_rewards(require_dars):
    ann = {"nodes": ["locate", "pick", "place"], "turns": []}
    p = build_reward_program({"type": "dars", "judge": judge([json.dumps(ann)], seed=12),
                              "overrides": {"domain": "alfworld", "terms": ["tests.rewards.bad_terms:ShortTerm"]}})
    rec = await p.ascore(task="t", metadata={}, rows=rows(), success=False, episode_score=0.0)
    assert rec.status == "failed" and rec.step_rewards is None


async def test_dars_state_keys_reach_product_scope(require_dars):
    from dars.integrations.webshop_state import state_key_from_url
    goal = {"product_category": "shoes", "attributes": [], "goal_options": {"size": "xl"}, "price_upper": 50.0}
    from dars.integrations.webshop_state import goal_schema
    ann = {"nodes": ["type", "opt:size", "price"]}
    judge_cfg = judge([json.dumps({"verified": ["type"], "errors": [], "recovered": []})], seed=13)
    p = build_reward_program({"type": "dars", "judge": judge_cfg, "overrides": {
        "domain": "webshop_goal", "include_terminal_result": True}})
    r = rows(2)
    r[0]["state_key"] = state_key_from_url("http://h/item_page/abc/B0ABCDEFGH/kw/1/{}")
    r[1]["state_key"] = state_key_from_url("http://h/item_page/abc/B0ZZZZZZZZ/kw/1/{}")
    traj = p.trajectory("q", {"goal_schema": goal_schema(goal), "state_keys": [x["state_key"] for x in r]}, r, False)
    resets = p.domain.scope_resets(traj, goal_schema(goal)["nodes"])
    assert resets[1] == ["type", "opt:size", "price"]


def test_specs_from_hydra_become_plain_containers():
    """Hydra passes agent-loop kwargs as DictConfig; a nested DictConfig in a judge request is not
    JSON-serializable and made every judge call fail."""
    omegaconf = pytest.importorskip("omegaconf")
    spec = omegaconf.OmegaConf.create({"type": "process", "judge": judge([{"scores": [1]}]),
                                       "extra_params": {"extra_body": {"thinking": {"type": "disabled"}}}})
    p = build_reward_program(spec)
    json.dumps(p.request_kwargs) and json.dumps(p.judge_cfg)
    assert type(p.request_kwargs["extra_params"]["extra_body"]) is dict


async def test_process_scores_wrong_length_is_corrected_by_the_judge():
    calls = []

    def responder(call):
        calls.append(call)
        return json.dumps({"scores": [1]} if len(calls) == 1 else {"scores": [1, 0, -1]})
    p = build_reward_program({"type": "process", "scale": 1.0,
                              "judge": {"backend": {"type": "scripted", "responder": responder}}})
    rec = await p.ascore(task="t", metadata={}, rows=rows(), success=False, episode_score=0.0)
    assert rec.status == "ok" and rec.step_rewards == pytest.approx([1.0, 0.0, -1.0])
    assert len(calls) == 2 and "3 steps" in calls[1].messages[-1]["content"]
