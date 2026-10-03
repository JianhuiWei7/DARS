"""One judge client per process and configuration, shared by all episodes of a rollout worker."""
from __future__ import annotations

import asyncio
import json
from typing import Any, Dict

_CLIENTS: Dict[str, Any] = {}
_LOCK = asyncio.Lock()


async def shared_client(config: Dict[str, Any]):
    """Return a started :class:`judgerl.judge.JudgeClient` for ``config`` (created once per process)."""
    from judgerl.judge import JudgeClient
    if "verl://" in json.dumps(config, default=str):      # a server verl started (see backends/verl/services.py)
        from judgerl.backends.verl.services import resolve_judge_config
        config = await asyncio.to_thread(resolve_judge_config, config)
    key = json.dumps(config, sort_keys=True, default=str)
    client = _CLIENTS.get(key)
    if client is None:
        async with _LOCK:
            client = _CLIENTS.get(key)
            if client is None:
                client = JudgeClient(config)
                await client.start()
                _CLIENTS[key] = client
    return client


def all_metrics() -> Dict[str, float]:
    """Judge metrics of every client in this process (for logging)."""
    out: Dict[str, float] = {}
    for c in _CLIENTS.values():
        out.update(c.metrics())
    return out
