"""Build verl datasets whose rows are environment tasks.

    judgerl-data --env numberline --train 256 --val 64 --out data/numberline
    judgerl-data --env examples.custom_env:GuessWordEnv --tasks-fn examples.custom_env:tasks --out data/guess

Each row has a placeholder prompt (the agent loop builds real per-step prompts from the environment),
``agent_name: judgerl_env`` and ``extra_info = {task, seed, split, index}``. Environments that need
specific tasks (game files, goals, questions) provide them through ``tasks_fn`` or a JSONL file.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, Iterable, List


TASK_PROVIDERS = {
    "alfworld": "judgerl.envs.alfworld:tasks",
    "webshop": "judgerl.envs.webshop:tasks",
    "search": "judgerl.envs.search:tasks",
    "python_tool": "judgerl.envs.python_tool:tasks",
    "open_ended": "judgerl.envs.open_ended:tasks",
}


def provider_tasks(env: str, split: str, limit: int, fn: str = None, **kwargs):
    """Tasks from a provider ``tasks(split, limit, **kwargs)``: ``fn`` (``package.module:function``) or
    the environment's built-in provider."""
    import importlib
    module, _, attr = (fn or TASK_PROVIDERS[env]).partition(":")
    return getattr(importlib.import_module(module), attr)(split, limit, **kwargs)


def rows(tasks: Iterable[Dict[str, Any]], env: str, split: str) -> List[Dict[str, Any]]:
    out = []
    for i, task in enumerate(tasks):
        out.append({
            "data_source": env,
            "prompt": [{"role": "user", "content": f"{env} task {i}"}],
            "agent_name": "judgerl_env",
            "extra_info": {"task": task, "seed": int(task.get("seed", i)), "split": split, "index": i},
        })
    return out


def write_parquet(records: List[Dict[str, Any]], path: str) -> None:
    import pandas as pd
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    pd.DataFrame(records).to_parquet(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True)
    ap.add_argument("--train", type=int, default=256)
    ap.add_argument("--val", type=int, default=64)
    ap.add_argument("--tasks", help="JSONL of task dicts (default: seeds 0..N-1)")
    ap.add_argument("--tasks-fn", help="task provider package.module:function(split, limit, **kwargs) "
                                       "(for custom environments)")
    ap.add_argument("--train-split", default="train", help="split name for environments with a task provider")
    ap.add_argument("--val-split", default="valid_seen")
    ap.add_argument("--provider-arg", action="append", default=[], metavar="KEY=VALUE",
                    help="extra keyword argument for the task provider (e.g. path=data/math.jsonl)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if (a.tasks_fn or a.env in TASK_PROVIDERS) and not a.tasks:
        extra = dict(kv.split("=", 1) for kv in a.provider_arg)
        train = provider_tasks(a.env, a.train_split, a.train, a.tasks_fn, **extra)
        val = provider_tasks(a.env, a.val_split, a.val, a.tasks_fn, **extra)
    elif a.tasks:
        with open(a.tasks) as f:
            tasks = [json.loads(l) for l in f if l.strip()]
        train, val = tasks[: a.train], tasks[a.train: a.train + a.val]
    else:
        train = [{"seed": i} for i in range(a.train)]
        val = [{"seed": 10_000 + i} for i in range(a.val)]
    write_parquet(rows(train, a.env, "train"), os.path.join(a.out, "train.parquet"))
    write_parquet(rows(val, a.env, "val"), os.path.join(a.out, "val.parquet"))
    print(f"wrote {len(train)} train / {len(val)} val rows to {a.out}")


if __name__ == "__main__":
    main()
