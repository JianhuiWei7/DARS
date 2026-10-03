"""Sandbox interface and the pipe/timeout machinery shared by the backends.

A sandbox runs one untrusted Python program per call::

    result = sandbox.run(code, stdin="", timeout_s=5.0, files={"data.csv": "a,b\\n1,2\\n"})

and always returns an :class:`ExecResult`; it never raises because of what the program did (a crash,
a timeout, a memory error or an output flood are all reported in the result). It raises only for
invalid arguments (e.g. a file name that escapes the work directory) or a broken backend.
"""
from __future__ import annotations

import os
import selectors
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Callable, Dict, List, Mapping, Optional, Union

FileMap = Mapping[str, Union[str, bytes]]

#: Environment variables a sandboxed program may inherit (values come from the parent when set).
#: Everything else, in particular API keys, tokens and credentials, is dropped.
DEFAULT_ENV_ALLOWLIST = ("PATH", "LANG", "LC_ALL", "LC_CTYPE", "TZ")

#: Fixed environment of every sandboxed program (single-threaded numeric libraries, so the process
#: limit is not consumed by thread pools; no bytecode written into the work directory).
BASE_ENV = {
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONUNBUFFERED": "1",
    "PYTHONIOENCODING": "utf-8",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}


@dataclass
class ExecResult:
    stdout: str
    stderr: str
    exit_code: Optional[int]      # negative: killed by that signal (process backend); None: never started
    timed_out: bool = False       # wall-clock or CPU-time limit hit
    oom: bool = False             # memory limit hit (best-effort detection, see the backend docs)
    truncated: bool = False       # stdout or stderr exceeded the output cap (the program was then stopped)
    duration_s: float = 0.0

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not (self.timed_out or self.oom or self.truncated)


class Sandbox:
    """Interface of every backend."""

    name = "base"

    def run(self, code: str, stdin: str = "", timeout_s: Optional[float] = None,
            files: Optional[FileMap] = None) -> ExecResult:
        raise NotImplementedError

    def capabilities(self) -> Dict[str, bool]:
        """Which guarantees this backend actually enforces on this host."""
        return {}

    def close(self) -> None:
        pass


def check_files(files: Optional[FileMap], reserved: tuple = ()) -> Dict[str, bytes]:
    """Validate a ``{relative path: content}`` map; paths must stay inside the work directory."""
    out: Dict[str, bytes] = {}
    for name, content in (files or {}).items():
        p = PurePosixPath(name)
        if not name or p.is_absolute() or ".." in p.parts or "\\" in name or "\x00" in name:
            raise ValueError(f"sandbox file name must be a relative path inside the work dir: {name!r}")
        if str(p) in reserved:
            raise ValueError(f"sandbox file name {name!r} is reserved")
        out[str(p)] = content.encode("utf-8") if isinstance(content, str) else bytes(content)
    return out


def write_files(root: str, files: Dict[str, bytes], mode: int = 0o600) -> None:
    for name, data in files.items():
        path = os.path.join(root, name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_NOFOLLOW", 0), mode)
        with os.fdopen(fd, "wb") as f:
            f.write(data)


def scrubbed_env(allowlist: tuple, extra: Optional[Mapping[str, str]] = None) -> Dict[str, str]:
    env = {k: os.environ[k] for k in allowlist if k in os.environ}
    env.update(BASE_ENV)
    env.update(extra or {})
    return env


def kill_group(pid: int) -> None:
    """SIGKILL the process group led by ``pid`` (the sandbox is started in a new session)."""
    try:
        os.killpg(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


@dataclass
class PipeOutcome:
    stdout: bytes
    stderr: bytes
    returncode: Optional[int]
    timed_out: bool
    truncated: bool
    duration_s: float


def run_with_limits(cmd: List[str], *, cwd: str, env: Dict[str, str], stdin_path: Optional[str],
                    timeout_s: float, max_output_bytes: int, kill: Callable[[subprocess.Popen], None],
                    drain_grace_s: float = 1.0) -> PipeOutcome:
    """Start ``cmd`` in a new session, stream its output with per-stream caps and a wall-clock deadline.

    ``kill(proc)`` must stop everything the program started; it is called on timeout, on output
    overflow, and after the program exits (to reap background children that still hold the pipes).
    """
    start = time.monotonic()
    stdin = open(stdin_path, "rb") if stdin_path else subprocess.DEVNULL
    try:
        proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdin=stdin, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, close_fds=True, start_new_session=True)
    finally:
        if stdin_path:
            stdin.close()
    bufs = {"stdout": bytearray(), "stderr": bytearray()}
    timed_out = truncated = killed = False
    sel = selectors.DefaultSelector()
    for name, stream in (("stdout", proc.stdout), ("stderr", proc.stderr)):
        os.set_blocking(stream.fileno(), False)
        sel.register(stream, selectors.EVENT_READ, name)
    deadline = start + timeout_s
    drain_until: Optional[float] = None
    try:
        while sel.get_map():
            now = time.monotonic()
            if not killed and now >= deadline:
                timed_out = killed = True
                kill(proc)
            if not killed and proc.poll() is not None:
                killed = True        # the program exited: stop any background children holding the pipes
                kill(proc)
            if killed:
                drain_until = drain_until or now + drain_grace_s
                if now >= drain_until:
                    break            # a process outside our reach still holds a pipe: stop reading
            for key, _ in sel.select(timeout=0.05):
                try:
                    chunk = os.read(key.fileobj.fileno(), 65536)
                except BlockingIOError:
                    continue
                except OSError:
                    chunk = b""
                if not chunk:
                    sel.unregister(key.fileobj)
                    continue
                buf = bufs[key.data]
                room = max_output_bytes - len(buf)
                if len(chunk) > room:
                    buf.extend(chunk[:max(room, 0)])
                    if not truncated:
                        truncated = True
                        if not killed:
                            killed = True
                            kill(proc)
                else:
                    buf.extend(chunk)
    finally:
        sel.close()
        for stream in (proc.stdout, proc.stderr):
            stream.close()
        try:
            proc.wait(timeout=max(drain_grace_s, 1.0))
        except subprocess.TimeoutExpired:
            kill(proc)
            proc.kill()
            proc.wait()
    return PipeOutcome(bytes(bufs["stdout"]), bytes(bufs["stderr"]), proc.returncode, timed_out, truncated,
                       time.monotonic() - start)
