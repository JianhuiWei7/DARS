"""WebShop port: prompts, projection, rewards and task draws against verl-agent.

The WebShop simulator is mocked. When a verl-agent checkout is available (``$VERL_AGENT_ROOT``), the
upstream ``WebshopEnvironmentManager`` + ``WebshopWorker`` + ``webshop_projection`` run on the same mock
and every prompt, anchor, reward and flag must match byte for byte.
"""
from __future__ import annotations

import ast
import json
import os
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest

from judgerl.envs import webshop as ws
from judgerl.envs.base import make_env

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------- upstream loading
def _find_upstream():
    for c in (os.environ.get("VERL_AGENT_ROOT"), os.path.join(REPO, "..", "verl-agent"),
              os.path.join(REPO, "third_party", "verl-agent")):
        if c and os.path.isfile(os.path.join(c, "agent_system", "environments", "env_manager.py")):
            return os.path.abspath(c)
    return None


UPSTREAM = _find_upstream()
_BLOCKED = {"torch", "ray", "gym", "omegaconf", "agent_system"}


def load_upstream(rel, ns=None):
    """Exec an upstream source file without its heavy / package imports (supplied through ``ns``)."""
    path = os.path.join(UPSTREAM, rel)
    with open(path, encoding="utf-8") as f:
        tree = ast.parse(f.read(), path)

    def blocked(n):
        if isinstance(n, ast.Import):
            return any(a.name.split(".")[0] in _BLOCKED for a in n.names)
        if isinstance(n, ast.ImportFrom):
            return n.level > 0 or (n.module or "").split(".")[0] in _BLOCKED
        return False

    tree.body = [n for n in tree.body if not blocked(n)]
    g = {"__name__": "upstream_" + os.path.basename(rel)[:-3]}
    g.update(ns or {})
    exec(compile(tree, path, "exec"), g)
    return g


@pytest.fixture(scope="module")
def up():
    if UPSTREAM is None:
        pytest.skip("verl-agent checkout not found (set VERL_AGENT_ROOT)")
    torch_stub = SimpleNamespace(Tensor=type("Tensor", (), {}))
    prompts = load_upstream("agent_system/environments/prompts/webshop.py")
    mem_base = load_upstream("agent_system/memory/base.py")
    mem = load_upstream("agent_system/memory/memory.py", {"BaseMemory": mem_base["BaseMemory"]})
    base = load_upstream("agent_system/environments/base.py", {"torch": torch_stub})
    em = load_upstream("agent_system/environments/env_manager.py", {
        "torch": torch_stub, "EnvironmentManagerBase": base["EnvironmentManagerBase"],
        "to_numpy": base["to_numpy"], "SimpleMemory": mem["SimpleMemory"], "SearchMemory": mem["SearchMemory"],
        "WEBSHOP_TEMPLATE_NO_HIS": prompts["WEBSHOP_TEMPLATE_NO_HIS"],
        "WEBSHOP_TEMPLATE": prompts["WEBSHOP_TEMPLATE"]})
    proj = load_upstream("agent_system/environments/env_package/webshop/projection.py")
    envs = load_upstream("agent_system/environments/env_package/webshop/envs.py",
                         {"gym": SimpleNamespace(Env=object), "ray": None})
    return SimpleNamespace(prompts=prompts, Manager=em["WebshopEnvironmentManager"],
                           projection=proj["webshop_projection"], Worker=envs["WebshopWorker"],
                           MultiEnv=envs["WebshopMultiProcessEnv"])


# ---------------------------------------------------------------------------- WebShop mock
GOALS = [
    {"asin": "B001", "category": "fashion", "query": "running shoes", "name": "Red Running Shoe",
     "product_category": "Clothing › Shoes › Running",
     "instruction_text": "i need red running shoes with size: large, and price lower than 40.00 dollars",
     "attributes": ["running", "red"], "price_upper": 40.0, "goal_options": {"size": "large"}, "weight": 1},
    {"asin": "B002", "category": "fashion", "query": "blue shoes", "name": "Blue Shoe",
     "product_category": "Clothing › Shoes", "instruction_text": "find me blue shoes",
     "attributes": ["blue"], "price_upper": 1000000, "goal_options": {}, "weight": 1},
    {"asin": "B003", "category": "home", "query": "lamp", "name": "Desk Lamp", "product_category": "Home",
     "instruction_text": "a desk lamp under 30 dollars", "attributes": ["desk"], "price_upper": 30.0,
     "goal_options": {"color": "white"}, "weight": 1},
]
LONG = "x" * 7000


class FakeServer:
    def __init__(self, seed):
        self.seed = seed
        self.goals = [dict(g) for g in GOALS]
        self.user_sessions = {}


class FakeTextEnv:
    """Pages, clickables and rewards shaped like WebShop's text mode (simple observation mode)."""
    instances = 0
    base_url = "http://127.0.0.1:3000"

    def __init__(self, server=None, session_prefix=None, seed=42, **kwargs):
        FakeTextEnv.instances += 1
        self.kwargs = dict(kwargs, seed=seed)
        self.server = server if server is not None else FakeServer(seed)
        self.session_prefix = session_prefix
        self.session = None
        self.closed = False
        self.reset(session="rnd")                      # WebAgentTextEnv resets in its constructor

    # -- pages
    def _s(self):
        return self.server.user_sessions[self.session]

    def _head(self):
        return f"WebShop [SEP] Instruction: [SEP] {self._s()['goal']['instruction_text']} [SEP] "

    @property
    def observation(self):
        s = self._s()
        p = s["pt"]
        if p == "index":
            return self._head() + "Search"
        if p == "search":
            return self._head() + ("Back to Search [SEP] Page 1 (Total results: 2) [SEP] Next > [SEP] B001 [SEP] "
                                   "Red Running Shoe [SEP] $10.99 [SEP] B002 [SEP] Blue Shoe [SEP] $24.00")
        if p == "item":
            return self._head() + ("Back to Search [SEP] < Prev [SEP] size [SEP] small [SEP] large [SEP] "
                                   f"{'Red Running Shoe' if s['asin'] == 'B001' else 'Blue Shoe'} [SEP] Price: $10.99 "
                                   "[SEP] Rating: N.A. [SEP] Description [SEP] Features [SEP] Reviews [SEP] Buy Now")
        if p == "desc":
            return self._head() + f"Back to Search [SEP] < Prev [SEP] {LONG}"
        raise AssertionError(p)

    @property
    def state(self):
        s = self._s()
        url = {"index": f"{self.base_url}/{self.session}",
               "search": f"{self.base_url}/search_results/{self.session}/{'+'.join(s['keywords'] or [])}/1",
               "item": f"{self.base_url}/item_page/{self.session}/{s['asin']}/{'+'.join(s['keywords'] or [])}/1/"
                       f"{json.dumps(s['options'])}",
               "desc": f"{self.base_url}/item_sub_page/{self.session}/{s['asin']}/{'+'.join(s['keywords'] or [])}/1/"
                       f"Description/{s['options']}"}[s["pt"]]
        return {"url": url, "html": "", "instruction_text": s["goal"]["instruction_text"]}

    def get_available_actions(self):
        p = self._s()["pt"]
        clickables = {"index": ["search"], "search": ["back to search", "next >", "b001", "b002"],
                      "item": ["back to search", "< prev", "small", "large", "description", "features", "reviews",
                               "buy now"],
                      "desc": ["back to search", "< prev"]}[p]
        self.text_to_clickable = {c: c for c in clickables}
        return {"has_search_bar": p == "index", "clickables": clickables}

    # -- gym API
    def reset(self, session=None):
        idx = session if isinstance(session, int) else 0
        self.session = str(session) if self.session_prefix is None else self.session_prefix + str(session)
        if self.session not in self.server.user_sessions:
            self.server.user_sessions[self.session] = {"goal": self.server.goals[idx]}
        self._s().update(pt="index", keywords=None, page=None, asin=None, options={})
        return self.observation, None

    def step(self, action):
        self.get_available_actions()
        s = self._s()
        name, _, arg = action.partition("[")
        arg = arg[:-1].lower() if arg.endswith("]") else None
        reward, done = 0.0, False
        if name == "search" and arg and s["pt"] == "index":
            s.update(pt="search", keywords=arg.split(" "), page=1)
        elif name == "click" and arg in self.text_to_clickable and arg != "search":
            if arg == "back to search":
                s.update(pt="index", keywords=None, asin=None, options={})
            elif arg in ("b001", "b002"):
                s.update(pt="item", asin=arg.upper(), options={})
            elif arg in ("small", "large"):
                s["options"]["size"] = arg
            elif arg == "description":
                s["pt"] = "desc"
            elif arg == "< prev":
                s["pt"] = "item" if s["pt"] == "desc" else "search"
            elif arg == "buy now":
                goal = s["goal"]
                ok = s["asin"] == goal["asin"] and all(s["options"].get(k) == v
                                                       for k, v in goal["goal_options"].items())
                reward, done = (1.0 if ok else 0.5), True
        if done:
            ob = ("Thank you for shopping with us! [SEP] Your code: [SEP] None [SEP] Purchased [SEP] asin [SEP] "
                  f"{s['asin']} [SEP] options [SEP] {s['options']} [SEP] Your score (min 0.0, max 1.0) [SEP] {reward}")
            self.reset(session="after-purchase")         # the text env resets itself after a purchase
        else:
            ob = self.observation
        return ob, reward, done, None

    def close(self):
        self.closed = True


def make_ours(**kw):
    return ws.WebShopEnv(text_env_factory=lambda **k: FakeTextEnv(**k), **kw)


SCRIPT = [
    "<think>search first</think><action>search[Red Running Shoes]</action>",
    "no tags here at all, only some rambling text",
    "<think>ok</think>\n<action> click[B001] </action>",
    "<action>click[large]</action>",                                  # no think: invalid, executed
    "<think>read</think><action>click[description]</action>",
    "<think>wait</think><action>click[nothing]</action>",             # no-op on the long page
    "<think>back</think><action>click[< prev]</action>",
    "<think>一</think><action>click[buy now]</action>",               # CJK: invalid, executed
]


# ---------------------------------------------------------------------------- fidelity vs upstream
def test_templates_are_verbatim(up):
    assert ws.WEBSHOP_TEMPLATE_NO_HIS == up.prompts["WEBSHOP_TEMPLATE_NO_HIS"]
    assert ws.WEBSHOP_TEMPLATE == up.prompts["WEBSHOP_TEMPLATE"]


PROJECTION_CASES = [
    "<think>a</think><action>click[B001]</action>",
    "<think>a</think><ACTION>Click[Buy Now]</ACTION>",
    "<THINK>a</THINK><action>search[x]</action>",
    "<think>a</think><action>search[x]",
    "</action> reversed <action>",
    "<action></action><think></think>",
    "<think>中文</think><action>click[x]</action>",
    "short",
    "",
    "<think>a</think>no action but a long tail of text here",
    "<think>x</think><action>  search[ Red  Shoes ]  </action> trailing",
    "<think>x</think><action>click[a]</action><action>click[b]</action>",
]


def test_projection_matches_upstream(up):
    exp_actions, exp_valid = up.projection(list(PROJECTION_CASES))
    got = [ws.project_action(r) for r in PROJECTION_CASES]
    assert [a for a, _ in got] == exp_actions
    assert [int(v) for _, v in got] == exp_valid


@pytest.mark.parametrize("history_length", [0, 1, 2, 3])
def test_episode_matches_upstream_manager(up, history_length):
    # upstream: manager -> vectorized env (batch 1) -> WebshopWorker -> mock text env
    worker = up.Worker.__new__(up.Worker)
    worker.env = FakeTextEnv(seed=7)

    class Vec:
        def reset(self):
            obs, info = worker.reset(0)
            return [obs], [info]

        def step(self, actions):
            o, r, d, i = worker.step(actions[0])
            return [o], [r], [d], [i]

    mgr = up.Manager(Vec(), up.projection, SimpleNamespace(env=SimpleNamespace(history_length=history_length)))
    up_obs, _ = mgr.reset({})

    env = make_ours(history_length=history_length, state_keys=False)
    ob = env.reset({"goal_idx": 0, "goal_seed": 7})
    assert ob.prompt == [{"role": "user", "content": up_obs["text"][0]}]
    assert ob.anchor == up_obs["anchor"][0] == ob.text
    assert env.task_text() == mgr.tasks[0]

    fallback_seen = False
    for response in SCRIPT:
        u_obs, u_rew, u_done, u_info = mgr.step([response])
        res = env.step(response)
        assert res.observation.prompt[0]["content"] == u_obs["text"][0]
        assert res.observation.anchor == u_obs["anchor"][0]
        assert res.reward == float(u_rew[0])
        assert res.done == bool(u_done[0])
        assert res.info["won"] == u_info[0]["won"]
        assert res.info["task_score"] == u_info[0]["task_score"]
        assert res.info["is_action_valid"] == bool(u_info[0]["is_action_valid"])
        if history_length > 0 and "Prior to this step" not in u_obs["text"][0]:
            fallback_seen = True
    assert res.done and res.reward == 10.0 and res.info["won"]
    if history_length > 0:
        assert fallback_seen          # the >13000-character fallback was exercised


def test_verl_agent_draws_match_upstream_reset(up):
    for is_train, group_n, env_seed in [(False, 1, 0), (True, 4, 3)]:
        n_goals = 520
        m = up.MultiEnv.__new__(up.MultiEnv)
        m._closed = True                     # no ray actors to tear down
        seed = env_seed if is_train else env_seed + 1000
        m.env_num, m.group_n = 5, group_n
        m._rng = np.random.RandomState(seed)
        m.goal_idxs = range(500, n_goals) if is_train else range(500)
        sent = []

        class W:
            def __init__(self, i):
                self.reset = SimpleNamespace(remote=lambda idx: sent.append((i, idx)) or ("o", {}))

        m._workers = [W(i) for i in range(5 * group_n)]
        ray_stub = SimpleNamespace(get=lambda futures: futures)
        m.reset.__globals__["ray"] = ray_stub
        draws = ws.verl_agent_draws("train" if is_train else "test", env_seed=env_seed, batch_size=5,
                                    group_n=group_n, n_resets=2, n_goals=n_goals)
        for r in range(2):
            sent.clear()
            m.reset()
            assert [d["goal_idx"] for d in draws[r]] == [idx for _, idx in sent]
            # worker i runs a simulator seeded seed + i // group_n
            assert [d["goal_seed"] for d in draws[r]] == [seed + i // group_n for i, _ in sent]


# ---------------------------------------------------------------------------- self-contained checks
def test_first_prompt_golden():
    env = make_ours()
    ob = env.reset({"goal_idx": 1, "goal_seed": 0})
    assert ob.anchor == "'Search'"
    expected = ("\nYou are an expert autonomous agent operating in the WebShop e‑commerce environment. \n"
                "Your task is to: find me blue shoes.\n"
                "Your current observation is: 'Search'.\n"
                "Your admissible actions of the current situation are: \n[\n'search[<your query>]',\n"
                "'click[search]',\n].\n\n"
                "Now it's your turn to take one action for the current step.\n"
                "You should first reason step-by-step about the current situation, then think carefully which "
                "admissible action best advances the shopping goal. This reasoning process MUST be enclosed "
                "within <think> </think> tags. \n"
                "Once you've finished your reasoning, you should choose an admissible action for current step and "
                "present it within <action> </action> tags.\n")
    assert ob.prompt == [{"role": "user", "content": expected}]
    assert ob.info["available_actions"] == ["search[<your query>]", "click[search]"]


def test_history_prompt_golden():
    env = make_ours(history_length=1)
    env.reset({"goal_idx": 1})
    env.step("<think>t</think><action>search[blue shoes]</action>")
    res = env.step("<think>t</think><action>click[b002]</action>")
    p = res.observation.prompt[0]["content"]
    assert "Prior to this step, you have already taken 2 step(s). Below are the most recent 1 observations" in p
    assert ("[Observation 2: ''Back to Search' [SEP] 'Page 1 (Total results: 2)' [SEP] 'Next >' [SEP] 'B001' "
            "[SEP] 'Red Running Shoe' [SEP] '$10.99' [SEP] 'B002' [SEP] 'Blue Shoe' [SEP] '$24.00'', "
            "Action 2: 'click[b002]']") in p
    assert "You are now at step 3 and your current observation is: 'Back to Search' [SEP] '< Prev'" in p
    assert "'click[buy now]',\n].\n" in p


@pytest.mark.parametrize("response,action,valid", [
    ("<think>a</think><action>Click[B001]</action>", "click[b001]", True),
    ("<action>click[b001]</action>", "click[b001]", False),
    ("<think>a</think><action>click[b001]</action>中", "click[b001]", False),
    ("<think>a</think> I will Click The First Product", "ck the first product", False),
    ("<THINK>a</THINK><action>x</action>", "x", False),
])
def test_projection_edge_cases(response, action, valid):
    assert ws.project_action(response) == (action, valid)


def test_reward_and_won_rule():
    env = make_ours()
    env.reset({"goal_idx": 0})
    for a in ("search[red shoes]", "click[b002]"):              # wrong product: task score 0.5
        res = env.step(f"<think>t</think><action>{a}</action>")
    res = env.step("<think>t</think><action>click[buy now]</action>")
    assert res.done and res.reward == 0.0 and not res.info["won"] and res.info["task_score"] == 0.5
    # the text env reset itself after the purchase; the prompt still uses the episode's own task
    assert "Thank you for shopping with us!" in res.observation.text
    assert "red running shoes" in res.observation.prompt[0]["content"]


def test_long_history_prompt_falls_back_to_no_history():
    avail = {"has_search_bar": False, "clickables": ["< prev"]}
    hist = [{"text_obs": LONG, "action": "click[description]"}]
    p = ws.build_prompt("task", LONG, avail, hist, 2, init=False)
    assert p == ws.WEBSHOP_TEMPLATE_NO_HIS.format(task_description="task", current_observation=LONG,
                                                  available_actions="'click[< prev]',")
    short = ws.build_prompt("task", "obs", avail, hist[:0] + [{"text_obs": "o", "action": "a"}], 2, init=False)
    assert short.startswith("\nYou are an expert autonomous agent operating in the WebShop e‑commerce "
                            "environment.\nYour task is to: task.\nPrior to this step, you have already taken 1")


def test_format_obs_and_task_extraction():
    obs = "WebShop [SEP] Instruction: [SEP] buy a hat [SEP] Search"
    assert ws.extract_task(obs) == "buy a hat"
    assert ws.format_obs(obs, "buy a hat") == "'Search'"
    assert ws.format_obs("no instruction [SEP] here", "buy a hat") == "no instruction [SEP] here"
    with pytest.raises(ValueError):
        ws.extract_task("Search")
    with pytest.raises(ValueError):
        ws.format_available_actions({"has_search_bar": True, "clickables": [], "other": 1})


def test_metadata_is_hidden_goal_data():
    env = make_ours()
    ob = env.reset({"goal_idx": 0, "goal_seed": 5})
    meta = env.metadata()
    assert meta["goal_idx"] == 0 and meta["goal_seed"] == 5
    assert meta["attributes"] == ["running", "red"]
    assert meta["goal_options"] == {"size": "large"}
    assert meta["price_upper"] == 40.0 and meta["asin"] == "B001"
    assert meta["goal"]["instruction_text"] == GOALS[0]["instruction_text"]
    json.dumps(meta)
    assert "B001" not in ob.prompt[0]["content"] and "Clothing" not in ob.prompt[0]["content"]


def test_served_goal_mismatch_raises():
    class Wrong(FakeTextEnv):
        def reset(self, session=None):
            out = super().reset(session)
            if isinstance(session, int):
                self._s()["goal"] = GOALS[2]
            return out

    env = ws.WebShopEnv(text_env_factory=lambda **k: Wrong(**k), state_keys=False)
    with pytest.raises(RuntimeError):
        env.reset({"goal_idx": 0})


def test_dars_state_keys_and_goal_schema():
    pytest.importorskip("dars.integrations.webshop_state")
    env = make_ours(state_keys=True)
    ob = env.reset({"goal_idx": 0})
    assert ob.info["state_key"].startswith("pt=index;")
    res = env.step("<think>t</think><action>search[red shoes]</action>")
    assert res.info["state_key"] == "pt=search;asin=-;opt=;kw=red+shoes;pg=1"
    env.step("<think>t</think><action>click[b001]</action>")
    env.step("<think>t</think><action>click[large]</action>")
    res = env.step("<think>t</think><action>click[buy now]</action>")
    assert res.done and res.info["won"]
    # keyed on the page the purchase was made from, not the page after the self-reset
    assert res.info["state_key"] == "pt=done;asin=B001;opt=size=large;kw=;pg=0"
    schema = env.metadata()["goal_schema"]
    assert schema["nodes"][0] == "type" and "price" in schema["nodes"] and "opt:size" in schema["nodes"]
    assert "state_key" not in res.observation.prompt[0]["content"]


def test_env_defaults_and_registry():
    env = make_env("webshop", text_env_factory=lambda **k: FakeTextEnv(**k))
    assert isinstance(env, ws.WebShopEnv) and env.max_steps == 15 and env.history_length == 2
    env.reset({"goal_idx": 2})
    assert env._env.kwargs["observation_mode"] == "text" and env._env.kwargs["human_goals"] is False
    env.close()


def test_simulator_rebuilt_only_when_seed_changes():
    made = []
    env = ws.WebShopEnv(text_env_factory=lambda **k: made.append(k["seed"]) or FakeTextEnv(**k))
    env.reset({"goal_idx": 0, "goal_seed": 1})
    env.reset({"goal_idx": 1, "goal_seed": 1})
    env.reset({"goal_idx": 1, "goal_seed": 2})
    assert made == [1, 2]


def test_shared_server_via_web_agent_site(monkeypatch):
    """The real loading path: web_agent_site.envs.WebAgentTextEnv, one server per seed, own sessions."""
    pkg = types.ModuleType("web_agent_site")
    mod = types.ModuleType("web_agent_site.envs")
    mod.WebAgentTextEnv = FakeTextEnv
    pkg.envs = mod
    monkeypatch.setitem(sys.modules, "web_agent_site", pkg)
    monkeypatch.setitem(sys.modules, "web_agent_site.envs", mod)
    ws._clear_server_cache()
    try:
        a, b = ws.WebShopEnv(state_keys=False), ws.WebShopEnv(state_keys=False)
        a.reset({"goal_idx": 0, "goal_seed": 9})
        b.reset({"goal_idx": 0, "goal_seed": 9})
        assert a._env.server is b._env.server
        assert a._env.session != b._env.session
        a.step("<think>t</think><action>search[red]</action>")
        assert b._env._s()["pt"] == "index" and a._env._s()["pt"] == "search"
        a.close()
        assert not any(k.startswith(a._prefix) for k in b._env.server.user_sessions)
        c = ws.WebShopEnv(state_keys=False)
        c.reset({"goal_idx": 0, "goal_seed": 10})
        assert c._env.server is not b._env.server
    finally:
        ws._clear_server_cache()


def test_task_providers():
    test = ws.tasks("test")
    assert [t["goal_idx"] for t in test] == list(range(500)) and {t["goal_seed"] for t in test} == {42}
    train = ws.tasks("train", n_goals=600, seed=3)
    assert sorted(t["goal_idx"] for t in train) == list(range(500, 600))
    assert train == ws.tasks("train", n_goals=600, seed=3)
    assert train != ws.tasks("train", n_goals=600, seed=4)
    assert len(ws.tasks("val", limit=7)) == 7
    draws = ws.verl_agent_draws("test", env_seed=0, batch_size=4, n_resets=2)
    first = np.random.RandomState(1000).choice(range(500), size=4, replace=False).tolist()
    assert [d["goal_idx"] for d in draws[0]] == first
    assert [d["goal_seed"] for d in draws[0]] == [1000, 1001, 1002, 1003]
    assert len({d["goal_idx"] for d in draws[1]}) == 4
    tr = ws.verl_agent_draws("train", env_seed=0, batch_size=2, group_n=3, n_goals=510)[0]
    assert [d["goal_seed"] for d in tr] == [0, 0, 0, 1, 1, 1]
    assert len({d["goal_idx"] for d in tr[:3]}) == 1 and all(500 <= d["goal_idx"] < 510 for d in tr)
    with pytest.raises(ValueError):
        ws.verl_agent_draws("test", group_n=2)
