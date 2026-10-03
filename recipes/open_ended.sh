#!/usr/bin/env bash
# Open-ended instruction following, scored entirely by an LLM judge against a per-task rubric.
# TASKS=<jsonl with instruction/rubric; {split} is substituted>; without TASKS the built-in samples are used.
MAX_RESPONSE=${MAX_RESPONSE:-1024}
source "$(dirname "$0")/common.sh"
ENV=open_ended DATA=${DATA:-data/open_ended}
if [ ! -f "$DATA/train.parquet" ]; then
  if [ -n "${TASKS:-}" ]; then judgerl-data --env open_ended --train 0 --val 128 --provider-arg "path=$TASKS" --val-split test --out "$DATA"
  else judgerl-data --env open_ended --train 64 --val 16 --train-split sample --val-split sample --out "$DATA"; fi
fi
judgerl_train open_ended_rubric data.train_batch_size=16 data.val_batch_size=64 \
  judgerl.estimator=grpo judgerl.judge_val=true judgerl.reward_program_file="$(dirname "$0")/rewards/open_ended_rubric.yaml" "$@"
