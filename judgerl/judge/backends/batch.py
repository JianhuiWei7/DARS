"""Offline batch judging: accumulate requests, run them with bounded concurrency, write JSONL.

No provider batch API is needed: requests go through a :class:`JudgeClient` (so caching, retries,
failover, budgets and telemetry all apply) or through a bare backend wrapped in a default client.
Each result is appended as one line ``{"request_id", "request", "result"}`` as soon as it resolves,
and ``resume=True`` skips requests whose id already has an ``ok`` line, so an interrupted job can be
rerun as is (give requests stable ``request_id`` values for this).
"""
from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Union

from judgerl.judge.types import JudgeHalted, JudgeRequest


@dataclass
class BatchSummary:
    out_path: str
    n_total: int
    n_skipped: int
    n_run: int
    by_status: Dict[str, int] = field(default_factory=dict)
    cost_usd: float = 0.0
    halted: Optional[str] = None


def completed_ids(path: str) -> set:
    done = set()
    if not os.path.exists(path):
        return done
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except ValueError:
                continue  # a torn last line from an interrupted run
            if rec.get("result", {}).get("status") == "ok":
                done.add(rec.get("request_id"))
    return done


class BatchJob:
    def __init__(self, target: Any, out_path: str, concurrency: int = 16, resume: bool = True,
                 include_raw: bool = False):
        """``target``: a :class:`JudgeClient` or a backend instance."""
        self.target = target
        self.out_path = out_path
        self.concurrency = max(1, int(concurrency))
        self.resume = resume
        self.include_raw = include_raw
        self.requests: List[JudgeRequest] = []

    def add(self, request: JudgeRequest) -> None:
        self.requests.append(request)

    def extend(self, requests: Iterable[JudgeRequest]) -> None:
        self.requests.extend(requests)

    async def run(self) -> BatchSummary:
        from judgerl.judge.client import JudgeClient

        own = not isinstance(self.target, JudgeClient)
        client = JudgeClient.from_backends(self.target) if own else self.target
        os.makedirs(os.path.dirname(os.path.abspath(self.out_path)), exist_ok=True)
        done = completed_ids(self.out_path) if self.resume else set()
        todo = [r for r in self.requests if r.request_id not in done]
        summary = BatchSummary(self.out_path, len(self.requests), len(self.requests) - len(todo), len(todo))
        sem = asyncio.Semaphore(self.concurrency)
        lock = asyncio.Lock()
        fh = open(self.out_path, "a" if self.resume else "w", encoding="utf-8")

        async def one(req: JudgeRequest) -> None:
            async with sem:
                if summary.halted:
                    return
                try:
                    res = await client.judge(req)
                except JudgeHalted as e:
                    summary.halted = e.reason
                    return
            async with lock:
                summary.by_status[res.status.value] = summary.by_status.get(res.status.value, 0) + 1
                summary.cost_usd += res.cost_usd
                fh.write(json.dumps({"request_id": req.request_id, "request": req.to_dict(),
                                     "result": res.to_dict(include_raw=self.include_raw)},
                                    ensure_ascii=False, default=str) + "\n")
                fh.flush()

        try:
            await client.start()
            await asyncio.gather(*(one(r) for r in todo))
        finally:
            fh.close()
            if own:
                await client.close()
        return summary

    def run_sync(self) -> BatchSummary:
        return asyncio.run(self.run())


def run_batch(target: Any, requests: Iterable[JudgeRequest], out_path: str, concurrency: int = 16,
              resume: bool = True) -> BatchSummary:
    """Synchronous one-shot helper."""
    job = BatchJob(target, out_path, concurrency=concurrency, resume=resume)
    job.extend(requests)
    return job.run_sync()
