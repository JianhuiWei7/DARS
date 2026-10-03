"""Command-line entry points.

    judgerl-data  --env alfworld --train 0 --val 128 --out data/alfworld     # build verl datasets
    judgerl-train [hydra overrides]                                         # train (verl backend)
    judgerl-check recipes/rewards/process.yaml [--live]                     # validate a reward program
"""
from __future__ import annotations

import argparse
import asyncio
import sys


def data():
    from judgerl.backends.verl.data import main
    main()


def train():
    from judgerl.backends.verl.main import main
    main()


def check():
    """Build a reward program from YAML, validate its judge config and, with ``--live``, send one real
    request per judge backend (auth, model name, output format, budget)."""
    ap = argparse.ArgumentParser(prog="judgerl-check")
    ap.add_argument("spec", help="reward-program YAML")
    ap.add_argument("--live", action="store_true", help="send one real request per judge backend")
    a = ap.parse_args()
    import yaml

    from judgerl.rewards import build_reward_program
    from judgerl.rewards.group import BUILTIN_GROUP, GroupRewardProgram, build_group_program
    with open(a.spec) as f:
        spec = yaml.safe_load(f)
    try:
        program = (build_group_program if spec.get("type") in BUILTIN_GROUP else build_reward_program)(spec)
    except TypeError:
        program = build_group_program(spec)
    kind = "group reward program" if isinstance(program, GroupRewardProgram) else "reward program"
    print(f"{kind}: {type(program).__name__} ({spec.get('type')})")
    judge = spec.get("judge")
    if not judge:
        print("no judge block: nothing else to check")
        return
    from judgerl.judge import JudgeClient
    if not a.live:       # a config check creates no cache or telemetry files
        judge = {k: v for k, v in judge.items() if k not in ("cache", "telemetry")}
    client = JudgeClient(judge)
    print(f"judge backends: {', '.join(client.slots)}")
    if not a.live:
        print("config ok (add --live to send one request per backend)")
        return

    async def run():
        async with client:
            return await client.preflight(strict=False)
    report = asyncio.run(run())
    print(report)
    sys.exit(0 if report.ok else 1)
