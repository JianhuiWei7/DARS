"""hf_local (fake torch/transformers), hf_server (mocked subprocess + readiness), colocated adapter,
and import hygiene (no torch/transformers/trainer imports)."""
from __future__ import annotations

import asyncio
import contextlib
import json
import subprocess
import sys
import types

import pytest

from judge_helpers import make_client, req
from judgerl.judge import JudgeClient, JudgeStatus

PAD, EOS, OFFSET = 0, 2, 10


# ----------------------------------------------------------------------------------------- fake torch / transformers
class FakeTensor:
    def __init__(self, rows):
        self.rows = [list(r) for r in rows] if rows and isinstance(rows[0], (list, tuple)) else list(rows)

    @property
    def shape(self):
        return (len(self.rows), len(self.rows[0]) if self.rows and isinstance(self.rows[0], list) else 0)

    def to(self, device):
        return self

    def __getitem__(self, i):
        return FakeTensor(self.rows[i])

    def tolist(self):
        return list(self.rows)


class FakeTokenizer:
    eos_token_id = EOS
    pad_token = None
    eos_token = "<eos>"
    pad_token_id = PAD

    def __init__(self):
        self.template_calls = []
        self.padding_side = "right"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kw):
        self.template_calls.append(kw)
        return "|".join(m["content"] for m in messages) + f"<think={kw.get('enable_thinking')}>"

    def __call__(self, prompts, return_tensors="pt", padding=True, add_special_tokens=False):
        enc = [[OFFSET + (ord(ch) % 50) for ch in p[:8]] for p in prompts]
        n = max(len(e) for e in enc)
        ids = [[PAD] * (n - len(e)) + e for e in enc]
        mask = [[0] * (n - len(e)) + [1] * len(e) for e in enc]
        return {"input_ids": FakeTensor(ids), "attention_mask": FakeTensor(mask)}

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(i - OFFSET) for i in ids if i >= OFFSET)


class FakeModel:
    device = "cpu"

    def __init__(self, answer_for):
        self.answer_for = answer_for
        self.generate_calls = []
        self.generation_config = types.SimpleNamespace(eos_token_id=[EOS])

    def eval(self):
        return self

    def to(self, device):
        return self

    def generate(self, input_ids, attention_mask, max_new_tokens, **kw):
        self.generate_calls.append({"batch": input_ids.shape[0], "max_new_tokens": max_new_tokens, **kw})
        rows = []
        for i, row in enumerate(input_ids.rows):
            text = self.answer_for(i)
            gen = [OFFSET + ord(ch) for ch in text][:max_new_tokens]
            if len(gen) < max_new_tokens:
                gen.append(EOS)
            rows.append(row + gen)
        n = max(len(r) for r in rows)
        return FakeTensor([r + [PAD] * (n - len(r)) for r in rows])


@pytest.fixture
def fake_hf(monkeypatch):
    state = {"answer": lambda i: '{"score": 7}', "tokenizer": None, "model": None, "loaded": []}
    torch_mod = types.ModuleType("torch")
    torch_mod.inference_mode = contextlib.nullcontext
    torch_mod.manual_seed = lambda s: None
    torch_mod.bfloat16 = "bf16"
    tf_mod = types.ModuleType("transformers")

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(name, revision=None, trust_remote_code=False):
            state["loaded"].append(("tokenizer", name, revision))
            state["tokenizer"] = FakeTokenizer()
            return state["tokenizer"]

    class AutoModelForCausalLM:
        @staticmethod
        def from_pretrained(name, revision=None, trust_remote_code=False, **kw):
            state["loaded"].append(("model", name, revision, kw))
            state["model"] = FakeModel(lambda i: state["answer"](i))
            return state["model"]

    tf_mod.AutoTokenizer = AutoTokenizer
    tf_mod.AutoModelForCausalLM = AutoModelForCausalLM
    monkeypatch.setitem(sys.modules, "torch", torch_mod)
    monkeypatch.setitem(sys.modules, "transformers", tf_mod)
    return state


async def test_hf_local_batches_requests_and_parses(fake_hf):
    from judgerl.judge.backends.hf_local import HFLocalBackend

    be = HFLocalBackend(model="org/tiny-judge", revision="abc123", max_batch_size=8, batch_wait_s=0.05,
                        device="cpu")
    async with make_client(be) as c:
        rs = await c.judge_many([req(f"q{i}", reasoning="off") for i in range(5)])
    assert all(r.ok and r.parsed == {"score": 7} for r in rs)
    assert be.batch_sizes == [5] and fake_hf["model"].generate_calls[0]["do_sample"] is False
    assert fake_hf["tokenizer"].padding_side == "left"
    assert all(kw.get("enable_thinking") is False for kw in fake_hf["tokenizer"].template_calls)
    assert ("model", "org/tiny-judge", "abc123", {"torch_dtype": "auto"}) in fake_hf["loaded"]
    assert rs[0].usage.completion_tokens == len('{"score": 7}') + 1 and rs[0].model == "org/tiny-judge"


async def test_hf_local_truncation_and_thinking_split(fake_hf):
    from judgerl.judge.backends.hf_local import HFLocalBackend

    fake_hf["answer"] = lambda i: "<think>" + "x" * 5000
    be = HFLocalBackend(model="org/tiny-judge", device="cpu")
    async with make_client(be, retry={"max_attempts": 1}, defaults={"max_tokens": 20}) as c:
        r = await c.judge(req(reasoning="on"))
    assert r.status == JudgeStatus.TRUNCATED and r.finish_reason == "length"
    assert fake_hf["tokenizer"].template_calls[0]["enable_thinking"] is True
    assert fake_hf["model"].generate_calls[0]["max_new_tokens"] == 20 + 4096  # reasoning budget added

    fake_hf["answer"] = lambda i: '<think>ok</think>{"score": 1}'
    be2 = HFLocalBackend(model="org/tiny-judge", device="cpu", enable_thinking=True)
    async with make_client(be2) as c:
        r2 = await c.judge(req())
    assert r2.ok and r2.reasoning_text == "ok"


async def test_hf_local_load_failure_is_backend_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "transformers", None)  # import fails
    from judgerl.judge.backends.hf_local import HFLocalBackend

    be = HFLocalBackend(model="org/missing", device="cpu")
    async with make_client(be, retry={"max_attempts": 1}) as c:
        r = await c.judge(req())
    assert r.status == JudgeStatus.BACKEND_ERROR


# ----------------------------------------------------------------------------------------- hf_server
class FakePopen:
    instances = []

    def __init__(self, cmd, env=None, stdout=None, stderr=None, stdin=None, start_new_session=False):
        self.cmd, self.env, self.start_new_session = cmd, env, start_new_session
        self.pid = 40_000 + len(FakePopen.instances)
        self.returncode = None
        self.exit_early = getattr(FakePopen, "exit_early", False)
        FakePopen.instances.append(self)
        if self.exit_early and stdout is not None:
            stdout.write(b"CUDA out of memory\n")
            stdout.flush()

    def poll(self):
        return 1 if self.exit_early else self.returncode

    def wait(self, timeout=None):
        self.returncode = -15
        return self.returncode


@pytest.fixture
def fake_server(monkeypatch, tmp_path):
    from judgerl.judge.backends import hf_server

    FakePopen.instances = []
    FakePopen.exit_early = False
    kills = []
    probes = {"n": 0}

    def probe(self, url):
        probes["n"] += 1
        return probes["n"] > 2  # ready on the third poll

    monkeypatch.setattr(hf_server.subprocess, "Popen", FakePopen)
    monkeypatch.setattr(hf_server.HFServer, "probe", probe)
    monkeypatch.setattr(hf_server.os, "killpg", lambda pid, sig: kills.append((pid, sig)))
    monkeypatch.setattr(hf_server.atexit, "register", lambda f: None)
    return {"kills": kills, "probes": probes, "log_dir": str(tmp_path / "logs")}


def test_hf_server_gpu_groups_and_commands():
    from judgerl.judge.backends.hf_server import HFServer

    s = HFServer("Qwen/Qwen3-8B", engine="vllm", gpus="0,1,2,3", tensor_parallel_size=2, reasoning_parser="qwen3",
                 gpu_memory_utilization=0.5, max_model_len=8192, extra_args=["--enforce-eager"])
    assert s.gpu_groups() == ["0,1", "2,3"]
    cmd = s.command("0,1", 8001)
    assert cmd[1:3] == ["-m", "vllm.entrypoints.openai.api_server"]
    for flag, val in (("--model", "Qwen/Qwen3-8B"), ("--port", "8001"), ("--tensor-parallel-size", "2"),
                      ("--reasoning-parser", "qwen3"), ("--gpu-memory-utilization", "0.5"),
                      ("--max-model-len", "8192")):
        assert cmd[cmd.index(flag) + 1] == val
    assert cmd[-1] == "--enforce-eager"
    g = HFServer("org/m", engine="sglang", gpus=["0", "3"])
    assert g.gpu_groups() == ["0", "3"]
    cmd = g.command("3", 9000)
    assert cmd[1:3] == ["-m", "sglang.launch_server"] and cmd[cmd.index("--model-path") + 1] == "org/m"
    assert cmd[cmd.index("--tp-size") + 1] == "1"
    with pytest.raises(ValueError):
        HFServer("m", gpus="0,1,2", tensor_parallel_size=2).gpu_groups()
    with pytest.raises(ValueError):
        HFServer("m", engine="tgi")


def test_hf_server_launch_ready_and_shutdown(fake_server):
    from judgerl.judge.backends.hf_server import launch_hf_server

    server, backend = launch_hf_server("Qwen/Qwen3-8B", engine="vllm", gpus="4,5", tensor_parallel_size=1,
                                       log_dir=fake_server["log_dir"], poll_s=0.001)
    procs = FakePopen.instances
    assert len(procs) == 2 and [p.env["CUDA_VISIBLE_DEVICES"] for p in procs] == ["4", "5"]
    assert all(p.start_new_session for p in procs)
    assert len(backend.base_urls) == 2 and all(u.endswith("/v1") for u in backend.base_urls)
    assert backend.served_model_name == "Qwen/Qwen3-8B" and backend.model_id == "Qwen/Qwen3-8B"
    server.stop()
    assert sorted(pid for pid, _ in fake_server["kills"]) == sorted(p.pid for p in procs)


def test_hf_server_early_exit_reports_log_tail(fake_server):
    from judgerl.judge.backends.hf_server import HFServer

    FakePopen.exit_early = True
    s = HFServer("org/m", gpus="0", log_dir=fake_server["log_dir"], poll_s=0.001)
    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        s.start()
    assert s.procs == []


def test_hf_server_ready_timeout_stops_processes(fake_server, monkeypatch):
    from judgerl.judge.backends import hf_server

    monkeypatch.setattr(hf_server.HFServer, "probe", lambda self, url: False)
    s = hf_server.HFServer("org/m", gpus="0", log_dir=fake_server["log_dir"], poll_s=0.001, ready_timeout_s=0.05)
    with pytest.raises(TimeoutError):
        s.start()
    assert fake_server["kills"] and s.procs == []


async def test_hf_server_backend_from_config_is_one_line(fake_server, monkeypatch):
    from openai.resources.chat.completions import AsyncCompletions

    seen = []

    async def create(self, **kwargs):
        seen.append((str(self._client.base_url), kwargs))
        return {"choices": [{"message": {"content": '{"score": 4}'}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2}, "model": "Qwen/Qwen3-8B"}

    monkeypatch.setattr(AsyncCompletions, "create", create)
    cfg = {"backend": {"type": "hf_server", "model": "Qwen/Qwen3-8B", "engine": "sglang", "gpus": "0",
                       "log_dir": fake_server["log_dir"], "poll_s": 0.001},
           "breaker": {"enabled": False}}
    client = JudgeClient(cfg)
    async with client:
        r = await client.judge(req(reasoning="off"))
        be = client.backend("Qwen/Qwen3-8B")
        assert be.describe()["engine"] == "sglang"
    assert r.ok and r.parsed == {"score": 4}
    assert FakePopen.instances[0].cmd[2] == "sglang.launch_server"
    assert seen[0][1]["response_format"]["type"] == "json_schema"
    assert seen[0][1]["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}
    assert fake_server["kills"]  # closed with the client


# ----------------------------------------------------------------------------------------- colocated
class FakeEngine:
    def __init__(self):
        self.awake = False
        self.events = []

    async def wake_up(self):
        self.awake = True
        self.events.append("wake")

    async def sleep(self):
        self.awake = False
        self.events.append("sleep")

    async def generate(self, call):
        assert self.awake, "generate called while asleep"
        return {"text": json.dumps({"score": 5}), "finish_reason": "stop", "prompt_tokens": 4,
                "completion_tokens": 3}


async def test_colocated_waits_while_asleep_and_resumes_on_wake():
    from judgerl.judge.backends.colocated import register_colocated_engine, unregister_colocated_engine

    engine = FakeEngine()
    register_colocated_engine("trainer_rm", engine)
    try:
        client = JudgeClient({"backend": {"type": "colocated", "name": "rm", "engine": "trainer_rm"},
                              "breaker": {"enabled": False}})
        async with client:
            s = client.session()
            futs = [s.submit(req(f"t{i}")) for i in range(4)]
            await asyncio.sleep(0.05)
            assert not any(f.done() for f in futs)  # engine asleep: requests wait
            await client.backend("rm").wake()
            rep = await s.barrier(timeout=5)
            await client.backend("rm").sleep()
        assert rep.by_status == {"ok": 4} and engine.events == ["wake", "sleep"]
        assert all(r.usage.completion_tokens == 3 for r in rep.results.values())
    finally:
        unregister_colocated_engine("trainer_rm")


async def test_colocated_late_attach_with_plain_function():
    client = JudgeClient({"backend": {"type": "colocated", "name": "rm"}, "breaker": {"enabled": False}})
    calls = []

    async def generate(call):
        calls.append(call)
        return '{"score": 2}'

    async with client:
        fut = await client.submit(req())
        await asyncio.sleep(0.02)
        assert not fut.done()
        client.attach_colocated("rm", generate)  # no sleep/wake hooks: starts awake
        r = await asyncio.wait_for(fut, 2)
    assert r.ok and r.parsed == {"score": 2} and calls[0].json_schema is None
    assert "JSON Schema" in calls[0].messages[-1]["content"]


# ----------------------------------------------------------------------------------------- import hygiene
def test_package_imports_without_torch_transformers_or_trainer():
    code = (
        "import sys\n"
        "import judgerl.judge\n"
        "import judgerl.judge.backends.hf_local, judgerl.judge.backends.hf_server\n"
        "import judgerl.judge.backends.colocated, judgerl.judge.backends.batch\n"
        "import judgerl.judge.backends.litellm_backend, judgerl.judge.backends.openai_compat\n"
        "bad = [m for m in ('torch', 'transformers', 'verl', 'litellm', 'openai') if m in sys.modules]\n"
        "assert not bad, bad\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         env={**__import__("os").environ, "PYTHONDONTWRITEBYTECODE": "1"},
                         cwd=str(__import__("pathlib").Path(__file__).resolve().parents[2]))
    assert out.returncode == 0, out.stderr
