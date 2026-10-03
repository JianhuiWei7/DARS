"""``docker`` / ``podman`` backend: one throw-away container per run.

Every run starts::

    docker run --rm -i --init --network none --read-only --tmpfs /tmp:rw,nosuid,nodev,size=<tmpfs_mb>m
        --pids-limit <n> --memory <m> --memory-swap <m> --cpus <c> --user 65534:65534
        --cap-drop ALL --security-opt no-new-privileges --ulimit nofile=... --ulimit fsize=...
        -v <run dir>:/work:ro <image> python -I -c <bootstrap>

The code and input files are mounted read-only at ``/work``; the bootstrap copies them to a writable
``/tmp/work`` (bounded by the tmpfs size) and runs ``main.py`` there. Guarantees, as enforced by the
container runtime: no network interfaces except loopback, a read-only root filesystem, no access to
the host filesystem beyond the run directory (read-only), an unprivileged user with no capabilities,
a PID limit (fork bombs are contained), a hard memory limit without swap, a CPU quota, and a
wall-clock timeout after which the container is killed (``docker kill``) and removed.

Limits: the isolation is that of the container runtime (a shared kernel; use gVisor or Kata through
``runtime=`` for a stronger boundary); the container start adds ~0.3-1 s per run (covered by
``start_overhead_s``, which is added to the wall-clock deadline); the image must already be present
(it is never pulled implicitly: ``--pull never``); the run directory must be visible to the daemon
(set ``tmp_root`` to a shared path when the daemon runs in a VM).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import threading
import uuid
from typing import Dict, List, Optional, Tuple

from judgerl.sandbox.base import ExecResult, FileMap, Sandbox, check_files, run_with_limits, write_files
from judgerl.sandbox.process import MAIN, _remove_tree

BOOTSTRAP = ("import os, shutil, sys\n"
             "shutil.copytree('/work', '/tmp/work', dirs_exist_ok=True)\n"
             "os.chdir('/tmp/work')\n"
             "os.execv(sys.executable, [sys.executable, '-I', 'main.py'])\n")

_AVAILABLE: Dict[Tuple[str, str], bool] = {}
_AVAILABLE_LOCK = threading.Lock()


def container_available(binary: str = "docker", image: Optional[str] = None) -> bool:
    """True when ``binary`` exists, its daemon/service answers, and (if given) ``image`` is present."""
    key = (binary, image or "")
    with _AVAILABLE_LOCK:
        if key in _AVAILABLE:
            return _AVAILABLE[key]
    path = shutil.which(binary)
    ok = False
    if path:
        try:
            ok = subprocess.run([path, "info"], capture_output=True, timeout=15,
                                stdin=subprocess.DEVNULL).returncode == 0
            if ok and image:
                ok = subprocess.run([path, "image", "inspect", image], capture_output=True, timeout=15,
                                    stdin=subprocess.DEVNULL).returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            ok = False
    with _AVAILABLE_LOCK:
        _AVAILABLE[key] = ok
    return ok


class DockerSandbox(Sandbox):
    """Container backend (see the module docstring). ``binary="podman"`` uses podman.

    Args:
        image: image with a ``python`` on ``PATH`` (must be pulled beforehand).
        timeout_s: default wall-clock limit per run (the container start is added on top, see
            ``start_overhead_s``).
        memory_mb / cpus / pids_limit / tmpfs_mb / max_open_files / max_file_mb: container limits.
        max_output_bytes: cap per stream; beyond it the container is killed.
        user: uid:gid inside the container (default ``nobody``).
        runtime: optional OCI runtime (e.g. ``runsc`` for gVisor).
        python: interpreter inside the image.
        tmp_root: parent directory of the per-run directories mounted into the container.
    """

    name = "docker"

    def __init__(self, image: str = "python:3.11-slim", binary: str = "docker", timeout_s: float = 10.0,
                 memory_mb: int = 1024, cpus: float = 1.0, pids_limit: int = 64, tmpfs_mb: int = 64,
                 max_open_files: int = 256, max_file_mb: int = 16, max_output_bytes: int = 65536,
                 user: str = "65534:65534", runtime: Optional[str] = None, python: str = "python",
                 start_overhead_s: float = 5.0, tmp_root: Optional[str] = None, extra_args: Tuple[str, ...] = ()):
        self.image = image
        self.binary = binary
        self.name = binary
        self.timeout_s = float(timeout_s)
        self.memory_mb = int(memory_mb)
        self.cpus = float(cpus)
        self.pids_limit = int(pids_limit)
        self.tmpfs_mb = int(tmpfs_mb)
        self.max_open_files = int(max_open_files)
        self.max_file_mb = int(max_file_mb)
        self.max_output_bytes = int(max_output_bytes)
        self.user = user
        self.runtime = runtime
        self.python = python
        self.start_overhead_s = float(start_overhead_s)
        self.tmp_root = tmp_root
        self.extra_args = tuple(extra_args)

    def available(self) -> bool:
        return container_available(self.binary, self.image)

    def capabilities(self) -> Dict[str, bool]:
        return {"network_isolated": True, "pid_namespace": True, "memory_limit": True,
                "process_group_kill": True, "filesystem_isolated": True}

    def command(self, workdir: str, name: str, timeout_s: float) -> List[str]:
        cpu = max(1, int(timeout_s + 0.999))
        fsize = self.max_file_mb << 20
        cmd = [shutil.which(self.binary) or self.binary, "run", "--rm", "-i", "--init", "--pull", "never",
               "--name", name, "--network", "none", "--read-only",
               "--tmpfs", f"/tmp:rw,nosuid,nodev,size={self.tmpfs_mb}m",
               "--pids-limit", str(self.pids_limit), "--memory", f"{self.memory_mb}m",
               "--memory-swap", f"{self.memory_mb}m", "--cpus", str(self.cpus), "--user", self.user,
               "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
               "--ulimit", f"nofile={self.max_open_files}:{self.max_open_files}",
               "--ulimit", f"fsize={fsize}:{fsize}", "--ulimit", f"cpu={cpu}:{cpu + 1}", "--ulimit", "core=0",
               "--env", "HOME=/tmp", "--env", "TMPDIR=/tmp", "--env", "PYTHONDONTWRITEBYTECODE=1",
               "--env", "PYTHONUNBUFFERED=1", "--env", "OMP_NUM_THREADS=1", "--env", "OPENBLAS_NUM_THREADS=1",
               "--volume", f"{workdir}:/work:ro", "--workdir", "/tmp"]
        if self.runtime:
            cmd += ["--runtime", self.runtime]
        cmd += list(self.extra_args)
        return cmd + [self.image, self.python, "-I", "-c", BOOTSTRAP]

    def _kill(self, name: str, proc: subprocess.Popen) -> None:
        try:
            subprocess.run([shutil.which(self.binary) or self.binary, "kill", name], capture_output=True,
                           timeout=15, stdin=subprocess.DEVNULL)
        except (OSError, subprocess.TimeoutExpired):
            pass
        if proc.poll() is None:
            proc.kill()

    def run(self, code: str, stdin: str = "", timeout_s: Optional[float] = None,
            files: Optional[FileMap] = None) -> ExecResult:
        payload = check_files(files, reserved=(MAIN,))
        timeout = float(timeout_s if timeout_s is not None else self.timeout_s)
        root = tempfile.mkdtemp(prefix="judgerl-sbx-", dir=self.tmp_root)
        name = f"judgerl-sbx-{uuid.uuid4().hex[:16]}"
        try:
            workdir = os.path.join(root, "work")
            os.mkdir(workdir, 0o755)
            os.chmod(workdir, 0o755)        # readable by the unprivileged container user
            payload[MAIN] = code.encode("utf-8")
            write_files(workdir, payload, mode=0o644)
            for d, _, _ in os.walk(workdir):
                os.chmod(d, 0o755)
            stdin_path = os.path.join(root, "stdin")
            write_files(root, {"stdin": stdin.encode("utf-8")})
            env = {k: os.environ[k] for k in ("PATH", "HOME", "DOCKER_HOST", "DOCKER_CONFIG", "DOCKER_CONTEXT",
                                               "XDG_RUNTIME_DIR", "CONTAINER_HOST") if k in os.environ}

            def exited_or_killed(proc: subprocess.Popen) -> None:
                if proc.poll() is None:
                    self._kill(name, proc)

            out = run_with_limits(self.command(workdir, name, timeout), cwd=root, env=env, stdin_path=stdin_path,
                                  timeout_s=timeout + self.start_overhead_s, max_output_bytes=self.max_output_bytes,
                                  kill=exited_or_killed)
        finally:
            _remove_tree(root)
        stderr = out.stderr.decode("utf-8", "replace")
        # 137 = SIGKILL inside the container: with no timeout/overflow on our side, the memory cgroup
        oom = out.returncode == 137 and not out.timed_out and not out.truncated
        cpu_killed = out.returncode == 128 + 24       # SIGXCPU: the CPU-time ulimit
        return ExecResult(stdout=out.stdout.decode("utf-8", "replace"), stderr=stderr, exit_code=out.returncode,
                          timed_out=out.timed_out or cpu_killed, oom=oom or "MemoryError" in stderr[-4000:],
                          truncated=out.truncated, duration_s=out.duration_s)
