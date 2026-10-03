"""Process-isolated environments.

Many environments (TextWorld/ALFWorld, WebShop, simulators) keep global state and are not safe to run
concurrently in threads of one process. :class:`ProcessEnvPool` keeps a fixed set of worker processes;
an episode borrows one worker for its whole lifetime and drives its environment through a pipe, so
episodes run in parallel with the same isolation verl-agent gets from one Ray actor per environment.

    pool = shared_pool(size=16)
    async with pool.episode("alfworld", {"history_length": 2}) as env:   # an env proxy
        obs = await env.reset(task, seed)
        result = await env.step(action)
"""
from __future__ import annotations

import asyncio
import multiprocessing as mp
import os
import traceback
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional, Tuple

_CTX = mp.get_context("spawn")


def _worker(conn, env_path: Optional[str]):
    """Worker loop: holds at most one environment; executes (method, args) requests."""
    if env_path:
        import sys
        sys.path[:0] = [p for p in env_path.split(os.pathsep) if p]
    from judgerl.envs.base import make_env
    env = None
    key = None
    while True:
        try:
            msg = conn.recv()
        except EOFError:
            break
        op = msg[0]
        try:
            if op == "shutdown":
                break
            if op == "make":
                name, kwargs = msg[1], msg[2]
                if env is None or key != (name, repr(sorted(kwargs.items()))):
                    if env is not None:
                        env.close()
                    env = make_env(name, **kwargs)
                    key = (name, repr(sorted(kwargs.items())))
                conn.send(("ok", {"max_steps": env.max_steps}))
            elif op == "call":
                method, args = msg[1], msg[2]
                conn.send(("ok", getattr(env, method)(*args)))
            else:
                conn.send(("error", f"unknown op {op!r}"))
        except Exception:  # noqa: BLE001 - reported to the caller
            conn.send(("error", traceback.format_exc()))
    if env is not None:
        try:
            env.close()
        except Exception:  # noqa: BLE001
            pass


class _Worker:
    def __init__(self):
        parent, child = _CTX.Pipe()
        self.conn = parent
        self.proc = _CTX.Process(target=_worker, args=(child, os.environ.get("PYTHONPATH")), daemon=True)
        self.proc.start()
        child.close()
        self.lock = asyncio.Lock()

    def alive(self) -> bool:
        return self.proc.is_alive()

    async def request(self, *msg, timeout: Optional[float] = None) -> Any:
        async with self.lock:
            await asyncio.to_thread(self.conn.send, msg)
            ready = await asyncio.to_thread(self.conn.poll, timeout)
            if not ready:
                raise TimeoutError(f"environment worker did not answer {msg[0]!r} within {timeout}s")
            status, payload = await asyncio.to_thread(self.conn.recv)
        if status != "ok":
            raise RuntimeError(f"environment error:\n{payload}")
        return payload

    def kill(self):
        try:
            self.proc.kill()
        finally:
            self.conn.close()


class EnvProxy:
    """Async view of an environment living in a worker process."""

    def __init__(self, worker: _Worker, max_steps: int, timeout: Optional[float]):
        self._w = worker
        self.max_steps = max_steps
        self.timeout = timeout

    async def reset(self, task: Dict[str, Any], seed: int):
        return await self._w.request("call", "reset", (task, seed), timeout=self.timeout)

    async def step(self, action: str):
        return await self._w.request("call", "step", (action,), timeout=self.timeout)

    async def task_text(self) -> str:
        return await self._w.request("call", "task_text", (), timeout=self.timeout)

    async def metadata(self) -> Dict[str, Any]:
        return await self._w.request("call", "metadata", (), timeout=self.timeout)


class ProcessEnvPool:
    def __init__(self, size: int = 16, timeout_s: Optional[float] = 300.0):
        if size < 1:
            raise ValueError(f"ProcessEnvPool size must be >= 1, got {size}")
        self.size = size
        self.timeout_s = timeout_s
        self._free: Optional[asyncio.Queue] = None
        self._workers = []

    def _ensure(self):
        if self._free is None:
            self._free = asyncio.Queue()
            for _ in range(self.size):
                w = _Worker()
                self._workers.append(w)
                self._free.put_nowait(w)

    @asynccontextmanager
    async def episode(self, name: str, kwargs: Dict[str, Any]):
        self._ensure()
        worker: _Worker = await self._free.get()
        healthy = True
        try:
            if not worker.alive():
                worker = self._replace(worker)
            info = await worker.request("make", name, dict(kwargs), timeout=self.timeout_s)
            yield EnvProxy(worker, info["max_steps"], self.timeout_s)
        except BaseException:
            healthy = False       # a timed-out or crashed worker may be mid-request: replace it
            raise
        finally:
            if not healthy:
                worker = self._replace(worker)
            self._free.put_nowait(worker)

    def _replace(self, worker: _Worker) -> _Worker:
        worker.kill()
        new = _Worker()
        self._workers = [new if w is worker else w for w in self._workers]
        return new

    def close(self):
        for w in self._workers:
            try:
                w.conn.send(("shutdown",))
            except Exception:  # noqa: BLE001
                pass
            w.kill()
        self._workers.clear()
        self._free = None


_POOLS: Dict[Tuple[int, Optional[float]], ProcessEnvPool] = {}


def shared_pool(size: int = 16, timeout_s: Optional[float] = 300.0) -> ProcessEnvPool:
    """One pool per process (and size/timeout), created on first use."""
    key = (size, timeout_s)
    if key not in _POOLS:
        _POOLS[key] = ProcessEnvPool(size, timeout_s)
    return _POOLS[key]
