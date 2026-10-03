"""Child-process entry point for the cross-process single-flight test (spawned by multiprocessing)."""
from __future__ import annotations

import asyncio


def run_worker(cache_path: str, call_log: str, content: str, barrier, out_q) -> None:
    from judgerl.judge import JudgeClient, JudgeRequest
    from judgerl.judge.backends.scripted import ScriptedBackend

    async def main():
        be = ScriptedBackend(latency_s=1.0, call_log_path=call_log, default_output={"score": 6})
        client = JudgeClient.from_backends(be, cache={"path": cache_path, "poll_s": 0.02},
                                           breaker={"enabled": False})
        async with client:
            await asyncio.to_thread(barrier.wait)  # all processes submit at the same moment
            r = await client.judge(JudgeRequest(messages=[{"role": "user", "content": content}],
                                                output_schema={"type": "object"}))
            return {"status": r.status.value, "parsed": r.parsed, "cache_hit": r.cache_hit,
                    "lease_waits": client.cache.lease_waits}

    out_q.put(asyncio.run(main()))
