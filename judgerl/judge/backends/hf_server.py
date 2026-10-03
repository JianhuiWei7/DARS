"""Serve any Hugging Face model id as a judge: launch vLLM or SGLang as a managed subprocess.

``{type: hf_server, model: Qwen/Qwen3-8B, engine: vllm, gpus: "0"}`` is the whole config. The
backend picks free ports, sets ``CUDA_VISIBLE_DEVICES``, waits until ``/v1/models`` answers (or the
process dies, with the log tail in the error), and then routes through :class:`OpenAICompatBackend`.
``gpus: "0,1,2,3"`` with ``tensor_parallel_size: 2`` starts two replicas (``"0,1"`` and ``"2,3"``);
a list such as ``["0", "1"]`` gives the GPU group of each replica explicitly.

The engine only has to be importable by ``python`` (default: this interpreter; set ``python`` to use
an engine installed in another environment).
"""
from __future__ import annotations

import asyncio
import atexit
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Sequence, Union

from judgerl.judge.backends.base import BaseBackend
from judgerl.judge.backends.openai_compat import OpenAICompatBackend
from judgerl.judge.types import BackendCall, BackendError, BackendResponse, JudgeStatus

ENGINES = ("vllm", "sglang")


def free_port(host: str = "127.0.0.1") -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return s.getsockname()[1]


class HFServer:
    """One or more replicas of an OpenAI-compatible server for a Hugging Face model id."""

    def __init__(self, model: str, engine: str = "vllm", gpus: Union[str, Sequence[str], None] = "0",
                 tensor_parallel_size: Optional[int] = None, host: str = "127.0.0.1", port: Optional[int] = None,
                 served_model_name: Optional[str] = None, revision: Optional[str] = None,
                 dtype: Optional[str] = None, max_model_len: Optional[int] = None,
                 gpu_memory_utilization: Optional[float] = None, reasoning_parser: Optional[str] = None,
                 trust_remote_code: bool = False, extra_args: Sequence[str] = (), python: Optional[str] = None,
                 env: Optional[Dict[str, str]] = None, log_dir: Optional[str] = None,
                 ready_timeout_s: float = 900.0, poll_s: float = 2.0, shutdown_grace_s: float = 30.0):
        if engine not in ENGINES:
            raise ValueError(f"engine must be one of {ENGINES}, got {engine!r}")
        self.model = model
        self.engine = engine
        self.gpus = gpus
        self.tensor_parallel_size = tensor_parallel_size
        self.host = host
        self.port = port
        self.served_model_name = served_model_name or model
        self.revision = revision
        self.dtype = dtype
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        self.reasoning_parser = reasoning_parser
        self.trust_remote_code = trust_remote_code
        self.extra_args = list(extra_args)
        self.python = python or sys.executable
        self.env = dict(env or {})
        self.log_dir = log_dir
        self.ready_timeout_s = ready_timeout_s
        self.poll_s = poll_s
        self.shutdown_grace_s = shutdown_grace_s
        self.procs: List[subprocess.Popen] = []
        self.urls: List[str] = []
        self.log_paths: List[str] = []
        self._atexit = False

    # ------------------------------------------------------------------ planning
    def gpu_groups(self) -> List[Optional[str]]:
        """GPU group (CUDA_VISIBLE_DEVICES value) of each replica; ``[None]`` = inherit."""
        if self.gpus is None or self.gpus == "":
            return [None]
        if isinstance(self.gpus, (list, tuple)):
            return [str(g) for g in self.gpus]
        ids = [g.strip() for g in str(self.gpus).split(",") if g.strip()]
        tp = self.tensor_parallel_size or len(ids)
        if tp <= 0 or len(ids) % tp:
            raise ValueError(f"{len(ids)} GPUs cannot be split into replicas of tensor_parallel_size={tp}")
        return [",".join(ids[i:i + tp]) for i in range(0, len(ids), tp)]

    def _tp_for(self, group: Optional[str]) -> int:
        if self.tensor_parallel_size:
            return self.tensor_parallel_size
        return len(group.split(",")) if group else 1

    def command(self, group: Optional[str], port: int) -> List[str]:
        tp = self._tp_for(group)
        if self.engine == "vllm":
            cmd = [self.python, "-m", "vllm.entrypoints.openai.api_server", "--model", self.model,
                   "--served-model-name", self.served_model_name, "--host", self.host, "--port", str(port),
                   "--tensor-parallel-size", str(tp)]
            if self.revision:
                cmd += ["--revision", self.revision]
            if self.dtype:
                cmd += ["--dtype", self.dtype]
            if self.max_model_len:
                cmd += ["--max-model-len", str(self.max_model_len)]
            if self.gpu_memory_utilization:
                cmd += ["--gpu-memory-utilization", str(self.gpu_memory_utilization)]
        else:
            cmd = [self.python, "-m", "sglang.launch_server", "--model-path", self.model,
                   "--served-model-name", self.served_model_name, "--host", self.host, "--port", str(port),
                   "--tp-size", str(tp)]
            if self.revision:
                cmd += ["--revision", self.revision]
            if self.dtype:
                cmd += ["--dtype", self.dtype]
            if self.max_model_len:
                cmd += ["--context-length", str(self.max_model_len)]
            if self.gpu_memory_utilization:
                cmd += ["--mem-fraction-static", str(self.gpu_memory_utilization)]
        if self.reasoning_parser:
            cmd += ["--reasoning-parser", self.reasoning_parser]
        if self.trust_remote_code:
            cmd += ["--trust-remote-code"]
        return cmd + self.extra_args

    # ------------------------------------------------------------------ lifecycle
    def probe(self, url: str) -> bool:
        """True when the server lists the served model (``GET {url}/v1/models``)."""
        try:
            with urllib.request.urlopen(f"{url}/v1/models", timeout=5) as r:
                if r.status != 200:
                    return False
                body = json.loads(r.read().decode("utf-8") or "{}")
        except (urllib.error.URLError, OSError, ValueError):
            return False
        ids = [m.get("id") for m in body.get("data", []) if isinstance(m, dict)]
        return not ids or self.served_model_name in ids

    def _log_tail(self, i: int, n: int = 40) -> str:
        try:
            with open(self.log_paths[i], errors="replace") as f:
                return "".join(f.readlines()[-n:])
        except (OSError, IndexError):
            return "<no log>"

    def start(self) -> List[str]:
        """Launch every replica and block until all are ready. Returns the base URLs."""
        if self.procs:
            return self.urls
        log_dir = self.log_dir or os.path.join(os.getcwd(), "outputs", "judge_servers")
        os.makedirs(log_dir, exist_ok=True)
        try:
            for i, group in enumerate(self.gpu_groups()):
                port = (self.port + i) if self.port else free_port(self.host)
                env = dict(os.environ)
                env.update(self.env)
                if group is not None:
                    env["CUDA_VISIBLE_DEVICES"] = group
                log_path = os.path.join(log_dir, f"{self.engine}_{port}.log")
                self.log_paths.append(log_path)
                logf = open(log_path, "ab")
                proc = subprocess.Popen(self.command(group, port), env=env, stdout=logf, stderr=subprocess.STDOUT,
                                        stdin=subprocess.DEVNULL, start_new_session=True)
                logf.close()
                self.procs.append(proc)
                self.urls.append(f"http://{self.host}:{port}")
            if not self._atexit:
                atexit.register(self.stop)
                self._atexit = True
            self._wait_ready()
        except BaseException:
            self.stop()
            raise
        return self.urls

    def _wait_ready(self) -> None:
        deadline = time.monotonic() + self.ready_timeout_s
        pending = set(range(len(self.procs)))
        while pending:
            for i in sorted(pending):
                rc = self.procs[i].poll()
                if rc is not None:
                    raise RuntimeError(f"{self.engine} server for {self.model} (replica {i}) exited with code {rc} "
                                       f"before becoming ready; log {self.log_paths[i]}:\n{self._log_tail(i)}")
                if self.probe(self.urls[i]):
                    pending.discard(i)
            if not pending:
                return
            if time.monotonic() > deadline:
                raise TimeoutError(f"{self.engine} server for {self.model} not ready after {self.ready_timeout_s}s; "
                                   f"logs: {self.log_paths}")
            time.sleep(self.poll_s)

    def stop(self) -> None:
        """Terminate every replica (process group), escalating to SIGKILL after the grace period."""
        for proc in self.procs:
            if proc.poll() is not None:
                continue
            try:
                if hasattr(os, "killpg"):
                    os.killpg(proc.pid, signal.SIGTERM)
                else:  # pragma: no cover - non-POSIX
                    proc.terminate()
            except (ProcessLookupError, PermissionError):
                continue
        deadline = time.monotonic() + self.shutdown_grace_s
        for proc in self.procs:
            try:
                proc.wait(timeout=max(0.1, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                try:
                    if hasattr(os, "killpg"):
                        os.killpg(proc.pid, signal.SIGKILL)
                    else:  # pragma: no cover
                        proc.kill()
                    proc.wait(timeout=10)
                except (ProcessLookupError, PermissionError, subprocess.TimeoutExpired):
                    pass
        self.procs, self.urls = [], []

    def backend(self, **kwargs: Any) -> OpenAICompatBackend:
        """An :class:`OpenAICompatBackend` over the running replicas."""
        if not self.urls:
            raise RuntimeError("server not started")
        kwargs.setdefault("name", self.model)
        return OpenAICompatBackend(model=self.model, base_urls=list(self.urls), served_model_name=self.served_model_name,
                                   revision=self.revision, **kwargs)

    def __enter__(self) -> "HFServer":
        self.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.stop()


def launch_hf_server(model: str, engine: str = "vllm", gpus: Union[str, Sequence[str], None] = "0",
                     backend_kwargs: Optional[Dict[str, Any]] = None, **server_kwargs: Any):
    """Start a server and return ``(server, backend)``; call ``server.stop()`` when done."""
    server = HFServer(model, engine=engine, gpus=gpus, **server_kwargs)
    server.start()
    return server, server.backend(**(backend_kwargs or {}))


_SERVER_KEYS = ("engine", "gpus", "tensor_parallel_size", "host", "port", "served_model_name", "dtype",
                "max_model_len", "gpu_memory_utilization", "reasoning_parser", "trust_remote_code", "extra_args",
                "python", "env", "log_dir", "ready_timeout_s", "poll_s", "shutdown_grace_s")


class HFServerBackend(BaseBackend):
    """Backend that owns an :class:`HFServer`; the server starts in :meth:`start` and stops in :meth:`close`."""

    def __init__(self, model: str, name: Optional[str] = None, revision: Optional[str] = None,
                 structured: str = "json_schema", reasoning_style: str = "chat_template",
                 api_key: Optional[str] = None, extra_kwargs: Optional[Dict[str, Any]] = None,
                 extra_body: Optional[Dict[str, Any]] = None, strict_schema: bool = False,
                 **server_kwargs: Any):
        unknown = set(server_kwargs) - set(_SERVER_KEYS)
        if unknown:
            raise TypeError(f"unknown hf_server options: {sorted(unknown)}")
        super().__init__(name=name or model, model=model, revision=revision,
                         structured="json_schema" if structured == "auto" else structured,
                         reasoning_style=reasoning_style)
        self.server = HFServer(model, revision=revision, **server_kwargs)
        self._backend_kwargs = dict(name=self.name, structured=self.capabilities.structured,
                                    reasoning_style=reasoning_style, api_key=api_key, extra_kwargs=extra_kwargs,
                                    extra_body=extra_body, strict_schema=strict_schema)
        self.inner: Optional[OpenAICompatBackend] = None
        self._start_lock: Optional[asyncio.Lock] = None

    async def start(self) -> None:
        if self.inner is not None:
            return
        if self._start_lock is None:
            self._start_lock = asyncio.Lock()
        async with self._start_lock:
            if self.inner is None:
                await asyncio.to_thread(self.server.start)
                self.inner = self.server.backend(**self._backend_kwargs)

    async def generate(self, call: BackendCall) -> BackendResponse:
        if self.inner is None:
            try:
                await self.start()
            except Exception as e:  # noqa: BLE001
                raise BackendError(JudgeStatus.BACKEND_ERROR, f"judge server failed to start: {e}"[:800],
                                   retryable=False) from e
        return await self.inner.generate(call)

    async def close(self) -> None:
        if self.inner is not None:
            await self.inner.close()
            self.inner = None
        await asyncio.to_thread(self.server.stop)

    def describe(self) -> Dict[str, Any]:
        d = super().describe()
        d.update(engine=self.server.engine, gpu_groups=self.server.gpu_groups(), urls=list(self.server.urls))
        return d
