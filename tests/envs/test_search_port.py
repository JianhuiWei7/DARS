"""Search port: prompts, memory, projection, retrieval formatting and scoring against verl-agent.

The retrieval server is mocked at the HTTP-session level. When a verl-agent checkout is available
(``$VERL_AGENT_ROOT``), the upstream ``SearchEnvironmentManager`` + ``SearchMultiProcessEnv`` + SkyRL
``SearchEnv`` / ``SearchToolGroup`` run against the same mock and every prompt, anchor, reward and flag
must match byte for byte.
"""
from __future__ import annotations

import ast
import json
import os
import re
from types import SimpleNamespace

import numpy as np
import pytest

from judgerl.envs import search as se
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
_SKY = "agent_system/environments/env_package/search/third_party/skyrl_gym/"


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


class _GenericStub:
    def __class_getitem__(cls, item):
        return cls


@pytest.fixture(scope="module")
def up():
    if UPSTREAM is None:
        pytest.skip("verl-agent checkout not found (set VERL_AGENT_ROOT)")
    pytest.importorskip("requests")
    torch_stub = SimpleNamespace(Tensor=type("Tensor", (), {}))
    prompts = load_upstream("agent_system/environments/prompts/search.py")
    mem_base = load_upstream("agent_system/memory/base.py")
    mem = load_upstream("agent_system/memory/memory.py", {"BaseMemory": mem_base["BaseMemory"]})
    base = load_upstream("agent_system/environments/base.py", {"torch": torch_stub})
    em = load_upstream("agent_system/environments/env_manager.py", {
        "torch": torch_stub, "EnvironmentManagerBase": base["EnvironmentManagerBase"],
        "to_numpy": base["to_numpy"], "SimpleMemory": mem["SimpleMemory"], "SearchMemory": mem["SearchMemory"],
        "SEARCH_TEMPLATE_NO_HIS": prompts["SEARCH_TEMPLATE_NO_HIS"], "SEARCH_TEMPLATE": prompts["SEARCH_TEMPLATE"]})
    proj = load_upstream("agent_system/environments/env_package/search/projection.py")
    utils = load_upstream(_SKY + "envs/search/utils.py")
    core = load_upstream(_SKY + "tools/core.py")
    tools = load_upstream(_SKY + "tools/search.py", {"tool": core["tool"], "ToolGroup": core["ToolGroup"]})
    btext = load_upstream(_SKY + "envs/base_text_env.py", {"Env": _GenericStub})
    sky = load_upstream(_SKY + "envs/search/env.py", {
        "BaseTextEnv": btext["BaseTextEnv"], "BaseTextEnvStepOutput": dict, "ConversationType": list,
        "compute_score": utils["compute_score"], "SearchToolGroup": tools["SearchToolGroup"], "DictConfig": dict})
    menv = load_upstream("agent_system/environments/env_package/search/envs.py",
                         {"gym": SimpleNamespace(Env=object), "DictConfig": dict, "ListConfig": list})
    return SimpleNamespace(prompts=prompts, Manager=em["SearchEnvironmentManager"], projection=proj["search_projection"],
                           utils=utils, ToolGroup=tools["SearchToolGroup"], SkyEnv=sky["SearchEnv"],
                           MultiEnv=menv["SearchMultiProcessEnv"])


# ---------------------------------------------------------------------------- retrieval mock
DOCS = {
    "capital of france": [{"document": {"contents": "\"Paris\"\nParis is the capital and largest city of France.  "},
                           "score": 0.91},
                          {"document": {"contents": "\"France\"\nFrance, officially the French Républic ..."},
                           "score": 0.72}],
    "paris population": [{"document": {"contents": "\"Paris\"\nPopulation 2.1 million (2020)."}, "score": 0.5}],
}


class FakeResponse:
    def __init__(self, status, payload=None, bad_json=False):
        self.status_code = status
        self._payload = payload
        self._bad_json = bad_json
        self.text = "<html>oops</html>" if bad_json else json.dumps(payload)

    def raise_for_status(self):
        if 400 <= self.status_code < 600:
            import requests
            raise requests.exceptions.HTTPError(f"{self.status_code} Client Error: for url: http://mock/retrieve")

    def json(self):
        if self._bad_json:
            raise json.JSONDecodeError("Expecting value", self.text, 0)
        return self._payload


class FakeSession:
    """Search-R1 /retrieve: POST {"query", "topk", "return_scores"} -> {"result": [[doc, ...]]}."""

    def __init__(self, script=None):
        self.calls = []
        self.script = list(script or [])

    def post(self, url, headers=None, json=None, timeout=None):
        self.calls.append({"url": url, "headers": headers, "json": json, "timeout": timeout})
        if self.script:
            item = self.script.pop(0)
            if isinstance(item, Exception):
                raise item
            return item
        q = json["query"]
        if q == "http 404":
            return FakeResponse(404, {"detail": "nope"})
        if q == "empty":
            return FakeResponse(200, {"result": []})
        return FakeResponse(200, {"result": [DOCS.get(q, [])[: json["topk"]]]})


def make_ours(session=None, **kw):
    kw.setdefault("success_reward", 1.0)
    client = se.RetrievalClient("http://mock/retrieve", session=session or FakeSession(), sleep=lambda s: None)
    return se.SearchEnv(retriever=client, **kw)


GT = {"target": np.array(["Paris", "City of Paris"], dtype=object)}
SCENARIOS = {
    "correct": ["<think>a</think><search> capital of france </search>",
                "<think>b</think><search>paris population</search> then <search>x</search>",
                "<think>c</think><ANSWER> Paris </ANSWER>"],
    "invalid_then_wrong": ["I do not know.", "<think>x</think><answer>the London.</answer>"],
    "max_steps": ["<search>capital of france</search>", "<search>paris population</search>",
                  "<search>unknown thing</search>", "<search>capital of france</search>"],
    "search_and_answer": ["<search>capital of france</search><answer>Lyon</answer>",
                          "<think>t</think><answer>the city of paris!</answer>"],
    "http_error": ["<search>http 404</search>", "<search>empty</search>", "<answer>Paris</answer>"],
    "answer_first": ["<answer>paris</answer>"],
}


# ---------------------------------------------------------------------------- fidelity vs upstream
def test_templates_are_verbatim(up):
    assert se.SEARCH_TEMPLATE_NO_HIS == up.prompts["SEARCH_TEMPLATE_NO_HIS"]
    assert se.SEARCH_TEMPLATE == up.prompts["SEARCH_TEMPLATE"]


PROJECTION_CASES = [
    "<search>q</search>", "<SEARCH> Q </SEARCH>", "<answer>a</answer>", "<Answer>A</ANSWER>",
    "<search>a</search><search>b</search>", "<answer>a</answer><answer>b</answer>",
    "<search>q</search><answer>a</answer>", "<answer>a</answer><search>q</search>",
    "<search>unterminated", "", "plain text", "<think>x</think>\n<search>\nmulti\nline\n</search>",
    "<search>q</SEARCH> tail </search>", "<search></search>", "<answer></answer> <search>",
]


def test_projection_matches_upstream(up):
    exp_actions, exp_valid = up.projection(list(PROJECTION_CASES))
    got = [se.project_action(r) for r in PROJECTION_CASES]
    assert [a for a, _ in got] == exp_actions
    assert [int(v) for _, v in got] == exp_valid


SCORE_CASES = [
    ("<answer>Paris</answer>", ["Paris"]), ("<answer> the  PARIS. </answer>", ["paris"]),
    ("<answer>a</answer> ... <answer>An Paris</answer>", ["paris"]), ("<answer>Lyon</answer>", ["Paris", "Lyon"]),
    ("no answer", ["x"]), ("<ANSWER>Paris</ANSWER>", ["Paris"]), ("<answer>\nNew-York\n</answer>", ["new york"]),
    ("<answer>newyork</answer>", ["new-york"]), ("<answer>Ünïcode</answer>", ["ünïcode"]),
]


@pytest.mark.parametrize("text,target", SCORE_CASES)
def test_scoring_matches_upstream(up, text, target):
    assert se.compute_score(text, {"target": target}) == up.utils["compute_score"](text, {"target": target})
    assert se.normalize_answer(text) == up.utils["normalize_answer"](text)


_RID = re.compile(r"\[Search Request ID: [0-9a-f-]+\] ")


def _run_upstream(up, responses, history_length, gt, question="What is the capital of France?"):
    tg = up.ToolGroup(search_url="http://mock/retrieve", topk=3, timeout=60, log_requests=False)
    tg.session = FakeSession()
    sky = up.SkyEnv.__new__(up.SkyEnv)
    up.SkyEnv.__mro__[1].__init__(sky)
    sky.tool_group = tg
    sky.init_tool_groups([tg])
    multi = up.MultiEnv.__new__(up.MultiEnv)
    multi.max_steps = 4
    multi._closed = True

    class Vec:
        def reset(self, kwargs):
            o, i = multi._sync_reset(sky, kwargs[0])
            return [o], [i]

        def step(self, actions):
            o, r, d, i = multi._sync_step(sky, actions[0])
            return [o], [r], [d], [i]

    mgr = up.Manager(Vec(), up.projection, SimpleNamespace(env=SimpleNamespace(history_length=history_length)))
    obs, infos = mgr.reset([{"question": question, "ground_truth": gt, "data_source": "nq"}])
    out = [(obs["text"][0], obs["anchor"][0], None, None, None, None)]
    for r in responses:
        o, rew, done, info = mgr.step([r])
        out.append((o["text"][0], o["anchor"][0], float(rew[0]), bool(done[0]), info[0]["won"],
                    bool(info[0]["is_action_valid"])))
        if done[0]:
            break
    return out


def _run_ours(responses, history_length, gt, question="What is the capital of France?"):
    env = make_ours(history_length=history_length)
    ob = env.reset({"question": question, "ground_truth": gt, "data_source": "nq"})
    out = [(ob.prompt[0]["content"], ob.anchor, None, None, None, None)]
    for r in responses:
        res = env.step(r)
        out.append((res.observation.prompt[0]["content"], res.observation.anchor, res.reward, res.done,
                    res.info["won"], res.info["is_action_valid"]))
        if res.done:
            break
    return out


@pytest.mark.parametrize("history_length", [0, 1, 2, 4])
@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_episode_matches_upstream_manager(up, name, history_length):
    expected = _run_upstream(up, SCENARIOS[name], history_length, GT)
    got = _run_ours(SCENARIOS[name], history_length, GT)
    norm = lambda rows: [tuple(_RID.sub("[ID] ", x) if isinstance(x, str) else x for x in row) for row in rows]
    assert norm(got) == norm(expected)


def test_upstream_scenario_outcomes(up):
    rows = {n: _run_ours(s, 4, GT) for n, s in SCENARIOS.items()}
    assert rows["correct"][-1][2:5] == (1.0, True, True)
    assert rows["invalid_then_wrong"][-1][2:5] == (0.0, True, False)
    assert rows["max_steps"][-1][2:5] == (0.0, True, False) and len(rows["max_steps"]) == 5
    assert rows["search_and_answer"][-1][2:5] == (1.0, True, True)
    assert rows["answer_first"][-1][2:5] == (1.0, True, True)


# ---------------------------------------------------------------------------- self-contained checks
def test_prompts_golden():
    env = make_ours()
    ob = env.reset({"question": "Who wrote Hamlet?", "ground_truth": {"target": ["Shakespeare"]}})
    assert ob.anchor == ob.text == "Who wrote Hamlet?"
    assert ob.prompt[0]["content"].startswith(
        "\nYou are an expert agent tasked with answering the given question step-by-step.\n"
        "Your question: Who wrote Hamlet?\n\nNow it's your turn to respond for the current step.\n")
    res = env.step("<think>t</think><search>capital of france</search>")
    info = ('<information>{"result": "Doc 1: \\"Paris\\"\\nParis is the capital and largest city of France.\\n'
            'Doc 2: \\"France\\"\\nFrance, officially the French R\\u00e9public ...\\n"}</information>')
    assert res.observation.text == res.observation.anchor == info
    p = res.observation.prompt[0]["content"]
    assert ("Prior to this step, you have already taken 1 step(s). Below is the interaction history where "
            "<search> </search> wrapped your past search queries") in p
    assert f"History:\nStep 1:<search>capital of france</search> {info}\n\n\nNow it's your turn" in p
    assert not res.done and res.reward == 0.0 and res.info["is_action_valid"]


def test_memory_format():
    hist = [{"search": "<search>a</search>", "information": "i1"}, {"search": "", "information": ""},
            {"search": "<search>c</search>", "information": "i3"}]
    assert se.format_memory(hist, 2) == "Step 2: \n\nStep 3:<search>c</search> i3\n"


@pytest.mark.parametrize("response,action,valid", [
    ("<think>x</think><search> q </search>", "<search>q</search>", True),
    ("<search>q</search><answer>a</answer>", "<search>q</search>", False),
    ("<answer>a</answer><answer>b</answer>", "<answer>a</answer>", False),
    ("no tags", "", False),
    ("<Search>Q</Search>", "<search>Q</search>", True),
])
def test_projection_edge_cases(response, action, valid):
    assert se.project_action(response) == (action, valid)


def test_reward_scale_done_and_max_steps():
    env = make_ours(success_reward=10.0)
    env.reset({"question": "q", "ground_truth": {"target": ["Paris"]}, "data_source": "hotpotqa"})
    res = env.step("<answer>paris</answer>")
    assert res.done and res.reward == 10.0 and res.info["won"] and res.info["em_score"] == 1.0
    assert res.observation.text == "" and res.info["data_source"] == "hotpotqa"

    env = make_ours(max_steps=2)
    env.reset({"question": "q", "ground_truth": ["Paris"]})
    assert not env.step("<search>capital of france</search>").done
    res = env.step("<search>capital of france</search>")    # the last allowed action ends the episode
    assert res.done and res.reward == 0.0 and not res.info["won"]
    assert len(env.retriever.session.calls) == 1


def test_invalid_action_observes_nothing_and_skips_retrieval():
    env = make_ours()
    env.reset({"question": "q", "ground_truth": "Paris"})
    res = env.step("I am thinking without tags")
    assert res.observation.text == "" and not res.info["is_action_valid"] and not res.done
    assert env.retriever.session.calls == []


def test_retriever_exception_becomes_observation():
    class Boom:
        def search(self, query):
            raise RuntimeError("backend down")

    env = se.SearchEnv(retriever=Boom())
    env.reset({"question": "q", "ground_truth": {"target": ["x"]}})
    res = env.step("<search>q</search>")
    assert res.observation.text == "backend down" and not res.done


def test_retrieval_client_request_and_retries():
    import requests
    sleeps = []
    ok = FakeResponse(200, {"result": [DOCS["paris population"]]})
    s = FakeSession([FakeResponse(503), requests.exceptions.ConnectionError("refused"), ok])
    c = se.RetrievalClient("http://h/retrieve", topk=5, timeout=7, session=s, sleep=sleeps.append)
    text = c.search("  paris population ")
    assert json.loads(text) == {"result": "Doc 1: \"Paris\"\nPopulation 2.1 million (2020).\n"}
    assert sleeps == [1.0, 2.0] and c.last_status == "success"
    assert s.calls[0]["json"] == {"query": "paris population", "topk": 5, "return_scores": True}
    assert s.calls[0]["headers"] == {"Content-Type": "application/json", "Accept": "application/json"}
    assert s.calls[0]["timeout"] == 7 and s.calls[0]["url"] == "http://h/retrieve"


def test_retrieval_client_failures():
    import requests
    sleeps = []
    s = FakeSession([requests.exceptions.Timeout("slow")] * 3)
    c = se.RetrievalClient("http://h", max_retries=3, session=s, sleep=sleeps.append)
    out = json.loads(c.search("q"))["result"]
    assert out.startswith("Search error: [Search Request ID: ") and "Timeout Error: slow" in out
    assert sleeps == [1.0, 2.0] and len(s.calls) == 3 and c.last_status == "api_error"

    s = FakeSession([FakeResponse(404, {})])                  # client errors are not retried
    c = se.RetrievalClient("http://h", session=s, sleep=sleeps.append)
    assert "API Request Error: 404" in c.search("q") and len(s.calls) == 1

    s = FakeSession([FakeResponse(200, bad_json=True)])
    c = se.RetrievalClient("http://h", session=s)
    assert "JSON Decode Error" in c.search("q") and len(s.calls) == 1

    c = se.RetrievalClient("http://h", session=FakeSession([FakeResponse(200, {"result": []})]))
    assert json.loads(c.search("q")) == {"result": "No search results found."}
    assert c.search(None) == ""


def test_url_round_robin_and_registry():
    a = se.SearchEnv(search_url=["http://a/retrieve", "http://b/retrieve"])
    b = se.SearchEnv(search_url="http://a/retrieve,http://b/retrieve")
    assert {a.retriever.url, b.retriever.url} == {"http://a/retrieve", "http://b/retrieve"}
    env = make_env("search", retriever=a.retriever)
    assert isinstance(env, se.SearchEnv) and env.max_steps == 4 and env.history_length == 4
    assert env.success_reward == 1.0


def test_metadata():
    env = make_ours()
    env.reset({"question": "q", "ground_truth": {"target": np.array(["Paris", "paris, france"])},
               "data_source": "nq"})
    meta = env.metadata()
    assert meta == {"reference": {"target": ["Paris", "paris, france"]}, "answers": ["Paris", "paris, france"],
                    "data_source": "nq"}
    json.dumps(meta)
    assert env.task_text() == "q"


def test_task_provider_jsonl(tmp_path):
    rows = [
        {"data_source": "nq", "env_kwargs": {"question": "q1", "ground_truth": {"target": ["a1"]},
                                             "data_source": "nq"}},
        {"data_source": "hotpotqa", "question": "q2", "reward_model": {"ground_truth": {"target": ["a2"]}}},
        {"data_source": "triviaqa", "question": "q3", "golden_answers": ["a3", "b3"]},
    ]
    p = tmp_path / "test.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows))
    ts = se.tasks("test", path=str(p))
    assert [t["question"] for t in ts] == ["q1", "q2", "q3"]
    assert ts[0] == {"question": "q1", "ground_truth": {"target": ["a1"]}, "data_source": "nq", "index": 0}
    assert ts[2]["ground_truth"] == {"target": ["a3", "b3"]}
    assert [t["question"] for t in se.tasks("test", path=str(p), data_sources=["hotpotqa", "triviaqa"])] == ["q2", "q3"]
    assert len(se.tasks("test", path=str(p), limit=2)) == 2


def test_task_provider_parquet(tmp_path):
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq
    table = pa.Table.from_pylist([{"data_source": "nq", "env_kwargs": {
        "question": "q1", "ground_truth": {"target": ["a1", "b1"]}, "data_source": "nq"}}])
    pq.write_table(table, str(tmp_path / "train.parquet"))
    ts = se.tasks("train", data_dir=str(tmp_path))
    assert ts == [{"question": "q1", "ground_truth": {"target": ["a1", "b1"]}, "data_source": "nq", "index": 0}]
