"""``process`` backend: best-effort isolation with plain POSIX mechanisms (no root, no daemon).

Each run gets:

* a fresh private temporary directory (mode 0700) as working directory and ``HOME``/``TMPDIR``,
  removed afterwards;
* a new session / process group; on the wall-clock timeout, on output overflow and after the program
  exits, the whole group is killed with SIGKILL;
* resource limits set by a small launcher before ``exec`` (no ``preexec_fn``, so it is safe to call
  from threads): ``RLIMIT_CPU`` (CPU seconds), ``RLIMIT_AS`` (address space), ``RLIMIT_NPROC``
  (processes), ``RLIMIT_FSIZE`` (largest file written), ``RLIMIT_NOFILE`` (open files), and
  ``RLIMIT_CORE = 0``;
* a scrubbed environment (only :data:`~judgerl.sandbox.base.DEFAULT_ENV_ALLOWLIST` plus explicitly
  allowed names; API keys and tokens are never inherited), no inherited file descriptors, stdin from a
  file, and ``python -I`` (ignores ``PYTHON*`` variables and the user site directory);
* network isolation when the host offers it without privileges: ``unshare`` with a user + network
  (and, when available, PID) namespace on Linux, ``sandbox-exec`` with a deny-network profile on
  macOS. :meth:`ProcessSandbox.capabilities` reports what is actually in force.

This is **not** a security boundary against a determined attacker. The program runs as the same
user as the trainer and can read every file that user can read; it can use the network when no
namespace tool is available; a process that leaves the process group (``setsid``) survives the
group kill unless a PID namespace is in force; ``RLIMIT_NPROC`` counts *all* processes (on Linux,
threads) of the user, so the limit is set to the user's current count plus ``max_processes`` and is
shared with everything else the user runs; ``RLIMIT_AS`` is not enforced on macOS. Use the
``docker`` backend for untrusted code.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from typing import Dict, List, Optional, Tuple

from judgerl.sandbox.base import (DEFAULT_ENV_ALLOWLIST, ExecResult, FileMap, Sandbox, check_files, kill_group,
                                  run_with_limits, scrubbed_env, write_files)

logger = logging.getLogger(__name__)

MAIN = "main.py"
_STDIN = ".stdin"

# Runs in the sandboxed process: apply the limits, then exec the program. Limits above the hard limit
# are clamped; a limit the platform rejects (RLIMIT_AS on macOS) is skipped and reported by the probe.
LAUNCHER = r"""
import json, os, resource, sys
spec = json.loads(sys.argv[1])
for name, value in spec["limits"].items():
    res = getattr(resource, name, None)
    if res is None or value is None:
        continue
    soft, hard = value
    cur_soft, cur_hard = resource.getrlimit(res)
    if cur_hard != resource.RLIM_INFINITY:
        hard = min(hard, cur_hard)
        soft = min(soft, hard)
    try:
        resource.setrlimit(res, (soft, hard))
    except (ValueError, OSError):
        pass
os.umask(0o077)
os.execv(spec["argv"][0], spec["argv"])
"""

_PROBE_LOCK = threading.Lock()
_NET_WRAPPERS: Dict[str, Tuple[List[str], Dict[str, bool]]] = {}
_MEMORY_PROBE: Dict[str, bool] = {}


def _net_wrapper(python: str) -> Tuple[List[str], Dict[str, bool]]:
    """The command prefix that isolates the network on this host (probed once per interpreter)."""
    with _PROBE_LOCK:
        if python in _NET_WRAPPERS:
            return _NET_WRAPPERS[python]
        candidates: List[Tuple[List[str], Dict[str, bool]]] = []
        unshare = shutil.which("unshare")
        if sys.platform.startswith("linux") and unshare:
            candidates += [([unshare, "--user", "--map-root-user", "--net", "--pid", "--fork", "--kill-child"],
                            {"network_isolated": True, "pid_namespace": True}),
                           ([unshare, "--user", "--map-root-user", "--net"],
                            {"network_isolated": True, "pid_namespace": False})]
        sbx = shutil.which("sandbox-exec")
        if sys.platform == "darwin" and sbx:
            candidates.append(([sbx, "-p", "(version 1)(allow default)(deny network*)"],
                               {"network_isolated": True, "pid_namespace": False}))
        chosen: Tuple[List[str], Dict[str, bool]] = ([], {"network_isolated": False, "pid_namespace": False})
        probe = "import socket\ntry:\n socket.create_connection(('127.0.0.1', 9), timeout=1)\nexcept OSError:\n pass\n"
        for prefix, caps in candidates:
            try:
                r = subprocess.run(prefix + [python, "-I", "-c", probe], capture_output=True, timeout=20,
                                   stdin=subprocess.DEVNULL)
            except (OSError, subprocess.TimeoutExpired):
                continue
            if r.returncode == 0:
                chosen = (prefix, caps)
                break
        if not chosen[0]:
            logger.warning("process sandbox: no unprivileged network isolation on this host; sandboxed code "
                           "can reach the network (use the docker backend for isolation)")
        _NET_WRAPPERS[python] = chosen
        return chosen


def _count_user_tasks() -> int:
    """Processes (Linux: threads) of the current real uid, as counted by ``RLIMIT_NPROC``."""
    uid = os.getuid()
    if os.path.isdir("/proc/self/task"):
        n = 0
        for pid in os.listdir("/proc"):
            if not pid.isdigit():
                continue
            try:
                with open(f"/proc/{pid}/status") as f:
                    status = f.read()
            except OSError:
                continue
            fields = dict(line.split(":", 1) for line in status.splitlines() if ":" in line)
            if fields.get("Uid", "").split()[:1] == [str(uid)]:
                n += int(fields.get("Threads", "1").strip() or 1)
        return n
    try:
        out = subprocess.run(["ps", "-U", str(uid), "-o", "pid="], capture_output=True, text=True, timeout=10,
                             stdin=subprocess.DEVNULL).stdout
        return len(out.split())
    except (OSError, subprocess.TimeoutExpired):
        return 0


class ProcessSandbox(Sandbox):
    """Subprocess backend (see the module docstring for its guarantees and limits).

    Args:
        python: interpreter that runs the code (default: the current one, so its packages are usable).
        timeout_s: default wall-clock limit per run.
        cpu_s: CPU-time limit (default: ``ceil(timeout_s)``).
        memory_mb: address-space limit (Linux; not enforced on macOS).
        max_processes: processes/threads the program may add to the user's current count.
        max_file_mb: largest file the program may write.
        max_open_files: open file descriptors.
        max_output_bytes: cap per stream (stdout, stderr); beyond it the program is stopped.
        network: ``"deny"`` (isolate when possible, warn otherwise), ``"require"`` (raise when the
            host cannot isolate), or ``"allow"``.
        env_allowlist: extra environment variable names to pass through.
        tmp_root: parent directory of the per-run work directories.
    """

    name = "process"

    def __init__(self, python: Optional[str] = None, timeout_s: float = 10.0, cpu_s: Optional[float] = None,
                 memory_mb: int = 1024, max_processes: int = 32, max_file_mb: int = 16, max_open_files: int = 64,
                 max_output_bytes: int = 65536, network: str = "deny", env_allowlist: Tuple[str, ...] = (),
                 tmp_root: Optional[str] = None):
        if network not in ("deny", "require", "allow"):
            raise ValueError(f"network must be deny|require|allow, got {network!r}")
        self.python = python or sys.executable
        self.timeout_s = float(timeout_s)
        self.cpu_s = cpu_s
        self.memory_mb = int(memory_mb)
        self.max_processes = int(max_processes)
        self.max_file_mb = int(max_file_mb)
        self.max_open_files = int(max_open_files)
        self.max_output_bytes = int(max_output_bytes)
        self.network = network
        self.env_allowlist = tuple(DEFAULT_ENV_ALLOWLIST) + tuple(env_allowlist)
        self.tmp_root = tmp_root
        self._nproc_cache: Tuple[float, int] = (0.0, 0)
        self._nproc_lock = threading.Lock()
        if network == "require" and not self.capabilities()["network_isolated"]:
            raise RuntimeError("process sandbox: network isolation required but not available on this host")

    # ------------------------------------------------------------------ capabilities
    def _wrapper(self) -> Tuple[List[str], Dict[str, bool]]:
        if self.network == "allow":
            return [], {"network_isolated": False, "pid_namespace": False}
        return _net_wrapper(self.python)

    def capabilities(self) -> Dict[str, bool]:
        prefix, caps = self._wrapper()
        return {**caps, "memory_limit": self._memory_enforced(), "process_group_kill": True,
                "filesystem_isolated": False}

    def _memory_enforced(self) -> bool:
        """Whether RLIMIT_AS actually stops an allocation on this host (probed once per interpreter)."""
        with _PROBE_LOCK:
            if self.python in _MEMORY_PROBE:
                return _MEMORY_PROBE[self.python]
        probe = ProcessSandbox(self.python, timeout_s=20, memory_mb=128, network="allow", tmp_root=self.tmp_root)
        r = probe.run("x = bytearray(512 * 1024 * 1024)\nprint('allocated')")
        enforced = "allocated" not in r.stdout
        with _PROBE_LOCK:
            _MEMORY_PROBE[self.python] = enforced
        return enforced

    def _nproc_limit(self) -> int:
        with self._nproc_lock:
            at, count = self._nproc_cache
            if time.monotonic() - at > 2.0:
                count = _count_user_tasks()
                self._nproc_cache = (time.monotonic(), count)
        return count + self.max_processes

    # ------------------------------------------------------------------ run
    def command(self, workdir: str) -> List[str]:
        cpu = int(self.cpu_s if self.cpu_s is not None else max(1.0, self.timeout_s + 0.999))
        mem = self.memory_mb << 20
        fsize = self.max_file_mb << 20
        nproc = self._nproc_limit()
        spec = {
            "limits": {
                "RLIMIT_CPU": [cpu, cpu + 1],
                "RLIMIT_AS": [mem, mem],
                "RLIMIT_NPROC": [nproc, nproc],
                "RLIMIT_FSIZE": [fsize, fsize],
                "RLIMIT_NOFILE": [self.max_open_files, self.max_open_files],
                "RLIMIT_CORE": [0, 0],
            },
            "argv": [self.python, "-I", os.path.join(workdir, MAIN)],
        }
        prefix, _ = self._wrapper()
        return prefix + [self.python, "-I", "-c", LAUNCHER, json.dumps(spec)]

    def run(self, code: str, stdin: str = "", timeout_s: Optional[float] = None,
            files: Optional[FileMap] = None) -> ExecResult:
        payload = check_files(files, reserved=(MAIN, _STDIN))
        timeout = float(timeout_s if timeout_s is not None else self.timeout_s)
        workdir = tempfile.mkdtemp(prefix="judgerl-sbx-", dir=self.tmp_root)
        try:
            payload[MAIN] = code.encode("utf-8")
            payload[_STDIN] = stdin.encode("utf-8")
            write_files(workdir, payload)
            env = scrubbed_env(self.env_allowlist, {"HOME": workdir, "TMPDIR": workdir})
            out = run_with_limits(self.command(workdir), cwd=workdir, env=env,
                                  stdin_path=os.path.join(workdir, _STDIN), timeout_s=timeout,
                                  max_output_bytes=self.max_output_bytes, kill=lambda p: kill_group(p.pid))
        finally:
            _remove_tree(workdir)
        stderr = out.stderr.decode("utf-8", "replace")
        cpu_killed = out.returncode == -signal.SIGXCPU
        oom = (not out.timed_out) and ("MemoryError" in stderr[-4000:] or "Cannot allocate memory" in stderr[-4000:])
        return ExecResult(stdout=out.stdout.decode("utf-8", "replace"), stderr=stderr, exit_code=out.returncode,
                          timed_out=out.timed_out or cpu_killed, oom=oom, truncated=out.truncated,
                          duration_s=out.duration_s)


def _remove_tree(path: str) -> None:
    """Remove a work directory even if the program made parts of it read-only."""
    def retry(func, p, _exc):
        try:
            os.chmod(os.path.dirname(p), 0o700)
            os.chmod(p, 0o700)
            func(p)
        except OSError:
            pass
    if sys.version_info >= (3, 12):
        shutil.rmtree(path, onexc=retry)
    else:
        shutil.rmtree(path, onerror=retry)
