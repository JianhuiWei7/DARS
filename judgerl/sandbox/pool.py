"""Bounded concurrency around a sandbox, and a process-wide pool per configuration."""
from __future__ import annotations

import json
import logging
import os
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Dict, Optional

from judgerl.sandbox.base import ExecResult, FileMap, Sandbox

logger = logging.getLogger(__name__)


class SandboxPool(Sandbox):
    """Runs at most ``max_concurrency`` programs at a time; further calls block until a slot frees.

    ``run`` is synchronous (environments call it from their worker threads); ``submit`` returns a
    :class:`concurrent.futures.Future`.
    """

    def __init__(self, sandbox: Sandbox, max_concurrency: Optional[int] = None):
        self.sandbox = sandbox
        self.name = f"pool[{sandbox.name}]"
        self.max_concurrency = max(1, int(max_concurrency or os.cpu_count() or 4))
        self._slots = threading.BoundedSemaphore(self.max_concurrency)
        self._lock = threading.Lock()
        self._executor: Optional[ThreadPoolExecutor] = None
        self.inflight = 0
        self.peak_inflight = 0
        self.completed = 0

    def run(self, code: str, stdin: str = "", timeout_s: Optional[float] = None,
            files: Optional[FileMap] = None) -> ExecResult:
        with self._slots:
            with self._lock:
                self.inflight += 1
                self.peak_inflight = max(self.peak_inflight, self.inflight)
            try:
                return self.sandbox.run(code, stdin=stdin, timeout_s=timeout_s, files=files)
            finally:
                with self._lock:
                    self.inflight -= 1
                    self.completed += 1

    def submit(self, code: str, stdin: str = "", timeout_s: Optional[float] = None,
               files: Optional[FileMap] = None) -> "Future[ExecResult]":
        with self._lock:
            if self._executor is None:
                self._executor = ThreadPoolExecutor(self.max_concurrency, thread_name_prefix="judgerl-sandbox")
        return self._executor.submit(self.run, code, stdin, timeout_s, files)

    def capabilities(self) -> Dict[str, bool]:
        return self.sandbox.capabilities()

    def stats(self) -> Dict[str, int]:
        with self._lock:
            return {"inflight": self.inflight, "peak_inflight": self.peak_inflight, "completed": self.completed,
                    "max_concurrency": self.max_concurrency}

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None
        self.sandbox.close()


def make_sandbox(config: Optional[Dict[str, Any]] = None) -> SandboxPool:
    """Build a pooled sandbox from ``{"backend": "process" | "docker" | "podman" | "auto", ...}``.

    ``max_concurrency`` sizes the pool; every other key goes to the backend. ``auto`` uses docker
    when it is available (daemon up and image present), else the process backend, with a warning.
    """
    from judgerl.sandbox.container import DockerSandbox, container_available
    from judgerl.sandbox.process import ProcessSandbox

    cfg = dict(config or {})
    backend = cfg.pop("backend", "process")
    max_concurrency = cfg.pop("max_concurrency", None)
    if backend == "auto":
        image = cfg.get("image", "python:3.11-slim")
        binary = next((b for b in ("docker", "podman") if container_available(b, image)), None)
        if binary:
            backend, cfg["binary"] = "docker", binary
        else:
            logger.warning("sandbox backend 'auto': no container runtime with image %s; using the process "
                           "backend (best-effort isolation)", image)
            backend = "process"
            for key in ("image", "binary", "cpus", "pids_limit", "tmpfs_mb", "user", "runtime", "start_overhead_s",
                        "extra_args"):
                cfg.pop(key, None)
    if backend == "process":
        sandbox: Sandbox = ProcessSandbox(**cfg)
    elif backend in ("docker", "podman"):
        cfg.setdefault("binary", backend)
        sandbox = DockerSandbox(**cfg)
    else:
        raise ValueError(f"unknown sandbox backend {backend!r}; use process, docker, podman or auto")
    return SandboxPool(sandbox, max_concurrency)


_SHARED: Dict[str, SandboxPool] = {}
_SHARED_LOCK = threading.Lock()


def shared_sandbox(config: Optional[Dict[str, Any]] = None) -> SandboxPool:
    """One pool per configuration and process, so the concurrency bound holds across all episodes."""
    key = json.dumps(config or {}, sort_keys=True, default=str)
    with _SHARED_LOCK:
        pool = _SHARED.get(key)
        if pool is None:
            pool = _SHARED[key] = make_sandbox(config)
        return pool
