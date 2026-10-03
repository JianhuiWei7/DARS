import asyncio
import os

import pytest

from judgerl.envs.pool import ProcessEnvPool


async def test_episodes_run_in_separate_processes_and_reuse_workers():
    pool = ProcessEnvPool(size=2, timeout_s=60)
    try:
        async def episode(i):
            async with pool.episode("numberline", {"size": 4}) as env:
                obs = await env.reset({"start": 0, "target": 1}, i)
                r = await env.step("<action>right</action>")
                return obs.anchor, r.reward, r.done, env.max_steps
        results = await asyncio.gather(*[episode(i) for i in range(6)])
        assert all(res == ("pos=0;target=1", 10.0, True, 12) for res in results)
        pids = {w.proc.pid for w in pool._workers}
        assert len(pids) == 2 and os.getpid() not in pids
    finally:
        pool.close()


async def test_env_error_is_raised_and_worker_replaced():
    pool = ProcessEnvPool(size=1, timeout_s=60)
    try:
        with pytest.raises(RuntimeError, match="environment error"):
            async with pool.episode("numberline", {}) as env:
                before = pool._workers[0].proc.pid
                await env.reset({"start": "abc"}, 0)       # int("abc") raises inside the worker
        assert pool._workers[0].proc.pid != before          # the worker was replaced
        async with pool.episode("numberline", {}) as env:
            obs = await env.reset({"start": 2, "target": 2}, 0)
            assert obs.anchor == "pos=2;target=2"
    finally:
        pool.close()
