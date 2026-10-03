"""Code execution for tool environments.

    from judgerl.sandbox import make_sandbox
    sandbox = make_sandbox({"backend": "process", "timeout_s": 5, "memory_mb": 512, "max_concurrency": 8})
    r = sandbox.run("print(sum(range(10)))")      # ExecResult(stdout='45\\n', exit_code=0, ...)

Backends: ``process`` (best-effort isolation with POSIX limits, see :mod:`judgerl.sandbox.process`)
and ``docker`` / ``podman`` (container isolation, see :mod:`judgerl.sandbox.container`). The security
guarantees and limits of each are documented in ``docs/environments.md``.
"""
from judgerl.sandbox.base import ExecResult, Sandbox
from judgerl.sandbox.container import DockerSandbox, container_available
from judgerl.sandbox.pool import SandboxPool, make_sandbox, shared_sandbox
from judgerl.sandbox.process import ProcessSandbox

__all__ = ["DockerSandbox", "ExecResult", "ProcessSandbox", "Sandbox", "SandboxPool", "container_available",
           "make_sandbox", "shared_sandbox"]
