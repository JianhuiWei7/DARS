"""Sandbox backends (no network, no GPU). Docker tests run only when a daemon and the image exist."""
from __future__ import annotations

import os
import socket
import subprocess
import threading
import time

import pytest

from judgerl.sandbox import DockerSandbox, ProcessSandbox, SandboxPool, container_available, make_sandbox

pytestmark = pytest.mark.skipif(os.name != "posix", reason="the sandbox needs POSIX")


@pytest.fixture
def root(tmp_path):
    d = tmp_path / "sbx"
    d.mkdir()
    return str(d)


@pytest.fixture
def sandbox(root):
    return ProcessSandbox(timeout_s=10, tmp_root=root)


def processes_matching(marker: str) -> list:
    out = subprocess.run(["ps", "-A", "-o", "pid=,command="], capture_output=True, text=True).stdout
    return [line for line in out.splitlines() if marker in line and "ps -A" not in line]


def test_runs_code_with_stdin_and_files(sandbox):
    r = sandbox.run("import sys\nprint(sys.stdin.read().upper())\nprint(open('data/x.txt').read())",
                    stdin="hello", files={"data/x.txt": "payload"})
    assert r.ok and r.stdout == "HELLO\npayload\n" and r.exit_code == 0


def test_errors_are_results_not_exceptions(sandbox):
    r = sandbox.run("raise ValueError('boom')")
    assert r.exit_code == 1 and "ValueError: boom" in r.stderr and not r.ok and not r.timed_out


@pytest.mark.parametrize("name", ["/etc/passwd", "../escape.txt", "a/../../b", "main.py"])
def test_rejects_unsafe_file_names(sandbox, name):
    with pytest.raises(ValueError):
        sandbox.run("pass", files={name: "x"})


def test_wall_timeout_kills_the_whole_group(sandbox, root):
    code = ("import os, time\n"
            "for _ in range(3):\n"
            "    if os.fork() == 0:\n"
            "        time.sleep(60)\n"
            "        os._exit(0)\n"
            "time.sleep(60)\n")
    t = time.monotonic()
    r = sandbox.run(code, timeout_s=1.0)
    assert r.timed_out and time.monotonic() - t < 8
    time.sleep(0.2)
    assert processes_matching(root) == []


def test_background_children_are_reaped_after_exit(sandbox, root):
    code = "import os, time\nif os.fork() == 0:\n    time.sleep(60)\nprint('parent done')\n"
    r = sandbox.run(code, timeout_s=5)
    assert r.stdout == "parent done\n" and not r.timed_out and r.duration_s < 4.5
    time.sleep(0.2)
    assert processes_matching(root) == []


def test_fork_bomb_is_contained(root):
    sandbox = ProcessSandbox(timeout_s=3, max_processes=16, tmp_root=root)
    code = "import os\nwhile True:\n    os.fork()\n"
    t = time.monotonic()
    r = sandbox.run(code)
    assert time.monotonic() - t < 10
    assert r.timed_out or "Resource temporarily unavailable" in r.stderr or r.exit_code != 0
    time.sleep(0.3)
    assert processes_matching(root) == []
    assert sandbox.run("print('alive')").stdout == "alive\n"        # the host can still fork


def test_cpu_limit(root):
    r = ProcessSandbox(timeout_s=30, cpu_s=1, tmp_root=root).run("while True:\n    pass\n")
    assert r.timed_out and r.duration_s < 20


def test_memory_cap(root):
    sandbox = ProcessSandbox(timeout_s=20, memory_mb=256, tmp_root=root)
    if not sandbox.capabilities()["memory_limit"]:
        pytest.skip("RLIMIT_AS is not enforced on this platform")
    r = sandbox.run("x = bytearray(1024 * 1024 * 1024)\nprint('allocated')")
    assert "allocated" not in r.stdout and r.oom and not r.ok
    assert sandbox.run("x = bytearray(16 * 1024 * 1024)\nprint('small ok')").stdout == "small ok\n"


def test_output_truncation(root):
    sandbox = ProcessSandbox(timeout_s=10, max_output_bytes=1000, tmp_root=root)
    r = sandbox.run("import sys\nwhile True:\n    sys.stdout.write('x' * 4096)\n")
    assert r.truncated and len(r.stdout) == 1000 and not r.timed_out and r.duration_s < 5
    r = sandbox.run("import sys\nsys.stderr.write('e' * 5000)\n")
    assert r.truncated and len(r.stderr) == 1000


def test_file_size_limit(root):
    sandbox = ProcessSandbox(timeout_s=10, max_file_mb=1, tmp_root=root)
    r = sandbox.run("with open('big', 'wb') as f:\n    f.write(b'0' * (4 << 20))\nprint('wrote')")
    assert "wrote" not in r.stdout and r.exit_code != 0


def test_secret_env_vars_are_not_visible(sandbox, monkeypatch):
    monkeypatch.setenv("JUDGERL_TEST_SECRET", "hunter2")
    monkeypatch.setenv("PROVIDER_API_KEY", "sk-test")
    monkeypatch.setenv("PYTHONPATH", "/somewhere")
    r = sandbox.run("import os\nprint(sorted(os.environ.items()))")
    assert r.ok
    for leaked in ("hunter2", "sk-test", "JUDGERL_TEST_SECRET", "PROVIDER_API_KEY", "PYTHONPATH"):
        assert leaked not in r.stdout


def test_env_allowlist_passes_named_vars(root, monkeypatch):
    monkeypatch.setenv("JUDGERL_ALLOWED", "yes")
    r = ProcessSandbox(tmp_root=root, env_allowlist=("JUDGERL_ALLOWED",)).run(
        "import os\nprint(os.environ.get('JUDGERL_ALLOWED'))")
    assert r.stdout == "yes\n"


def test_no_inherited_file_descriptors(sandbox, tmp_path):
    f = open(tmp_path / "held.txt", "w")
    try:
        os.set_inheritable(f.fileno(), True)
        r = sandbox.run("import os\nprint(sorted(int(x) for x in os.listdir('/dev/fd')))")
        fds = [int(x) for x in r.stdout.strip("[]\n").split(", ")]
        assert f.fileno() > 3 and all(fd <= 3 for fd in fds)   # 0-2 plus the one listdir opens
    finally:
        f.close()


def test_work_dir_is_private_and_removed(sandbox, root):
    r = sandbox.run("import os, stat\nprint(os.getcwd())\nprint(oct(stat.S_IMODE(os.stat('.').st_mode)))\n"
                    "open('out.txt', 'w').write('x')\nos.mkdir('sub')\nos.chmod('sub', 0o500)")
    cwd, mode = r.stdout.split()
    assert os.path.realpath(cwd).startswith(os.path.realpath(root)) and mode == "0o700"
    assert not os.path.exists(cwd) and os.listdir(root) == []
    sandbox.run("import time\ntime.sleep(30)", timeout_s=0.5)
    assert os.listdir(root) == []


def test_network_is_blocked_when_isolation_is_available(sandbox):
    if not sandbox.capabilities()["network_isolated"]:
        pytest.skip("no unprivileged network isolation on this host")
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    port = server.getsockname()[1]
    try:
        r = sandbox.run(f"import socket\ns = socket.create_connection(('127.0.0.1', {port}), timeout=2)\n"
                        "print('connected')")
        assert "connected" not in r.stdout and r.exit_code != 0
    finally:
        server.close()


def test_network_require_raises_without_isolation(root, monkeypatch):
    from judgerl.sandbox import process
    monkeypatch.setitem(process._NET_WRAPPERS, "/nonexistent/python",
                        ([], {"network_isolated": False, "pid_namespace": False}))
    monkeypatch.setitem(process._MEMORY_PROBE, "/nonexistent/python", False)
    with pytest.raises(RuntimeError):
        ProcessSandbox(python="/nonexistent/python", network="require", tmp_root=root)


def test_concurrent_runs_are_bounded_and_isolated(root):
    pool = SandboxPool(ProcessSandbox(timeout_s=10, tmp_root=root), max_concurrency=3)
    results = {}

    def job(i):
        results[i] = pool.run(f"import time\ntime.sleep(0.3)\nopen('f', 'w').write('{i}')\nprint(open('f').read())")

    threads = [threading.Thread(target=job, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert {i: r.stdout.strip() for i, r in results.items()} == {i: str(i) for i in range(8)}
    stats = pool.stats()
    assert stats["peak_inflight"] <= 3 and stats["completed"] == 8 and stats["inflight"] == 0
    futures = [pool.submit("print(6 * 7)") for _ in range(4)]
    assert [f.result().stdout for f in futures] == ["42\n"] * 4
    pool.close()
    assert os.listdir(root) == []


def test_make_sandbox_config(root):
    pool = make_sandbox({"backend": "process", "max_concurrency": 2, "timeout_s": 3, "tmp_root": root})
    assert isinstance(pool.sandbox, ProcessSandbox) and pool.max_concurrency == 2
    assert pool.run("print(1)").stdout == "1\n"
    with pytest.raises(ValueError):
        make_sandbox({"backend": "vm"})


def test_docker_command_flags(tmp_path):
    sbx = DockerSandbox(image="python:3.11-slim", memory_mb=256, pids_limit=32, cpus=0.5)
    cmd = sbx.command(str(tmp_path), "judgerl-sbx-test", timeout_s=5)
    joined = " ".join(cmd)
    for flag in ("--rm", "--network none", "--read-only", "--pids-limit 32", "--memory 256m", "--memory-swap 256m",
                 "--cpus 0.5", "--user 65534:65534", "--cap-drop ALL", "no-new-privileges", "--pull never",
                 f"{tmp_path}:/work:ro"):
        assert flag in joined
    assert "--tmpfs" in cmd and cmd[cmd.index("python:3.11-slim") + 1:][:2] == ["python", "-I"]


# ---------------------------------------------------------------------- docker (skipped when absent)
DOCKER_IMAGE = os.environ.get("JUDGERL_TEST_DOCKER_IMAGE", "python:3.11-slim")
needs_docker = pytest.mark.skipif(not container_available("docker", DOCKER_IMAGE),
                                  reason=f"docker daemon or image {DOCKER_IMAGE} not available")


@pytest.fixture
def docker_sbx(tmp_path):
    return DockerSandbox(image=DOCKER_IMAGE, timeout_s=10, memory_mb=128, pids_limit=32, max_output_bytes=2000,
                         tmp_root=str(tmp_path))


@needs_docker
def test_docker_runs_and_isolates(docker_sbx, monkeypatch):
    monkeypatch.setenv("JUDGERL_TEST_SECRET", "hunter2")
    r = docker_sbx.run("import os, sys\nprint(sys.stdin.read())\nprint(open('in.txt').read())\n"
                       "print(os.getuid())\nprint(dict(os.environ))", stdin="hi", files={"in.txt": "data"})
    lines = r.stdout.splitlines()
    assert r.exit_code == 0 and lines[:3] == ["hi", "data", "65534"] and "hunter2" not in r.stdout


@needs_docker
def test_docker_network_timeout_memory_pids_output(docker_sbx):
    r = docker_sbx.run("import socket\nsocket.create_connection(('1.1.1.1', 53), timeout=2)\nprint('connected')")
    assert "connected" not in r.stdout
    assert docker_sbx.run("import time\ntime.sleep(60)", timeout_s=1).timed_out
    r = docker_sbx.run("x = bytearray(1024 * 1024 * 1024)\nprint('allocated')")
    assert "allocated" not in r.stdout and r.oom
    r = docker_sbx.run("import os\nn = 0\ntry:\n    while True:\n        if os.fork() == 0:\n"
                       "            import time; time.sleep(30); os._exit(0)\n        n += 1\n"
                       "except OSError:\n    print('limited', n)\n", timeout_s=10)
    assert "limited" in r.stdout
    assert docker_sbx.run("print('x' * 100000)").truncated
    assert docker_sbx.run("open('/etc/evil', 'w')").exit_code != 0
