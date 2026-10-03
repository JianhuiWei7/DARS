#!/usr/bin/env bash
# Math with a Python tool (sandboxed). TASKS=<jsonl with question/answer; {split} is substituted>.
# REWARD=env (default: exact-answer reward) | outcome (rubric judge) | process | first_error | <reward-program YAML>.
MAX_RESPONSE=${MAX_RESPONSE:-1024}
source "$(dirname "$0")/common.sh"
ENV=python_tool DATA=${DATA:-data/python_tool}
REWARD=${REWARD:-env}
if [ ! -f "$DATA/train.parquet" ]; then
  if [ -n "${TASKS:-}" ]; then judgerl-data --env python_tool --train 0 --val 256 --provider-arg "path=$TASKS" --val-split test --out "$DATA"
  else judgerl-data --env python_tool --train 64 --val 16 --train-split sample --val-split sample --out "$DATA"; fi
fi
EXTRA=()
case "$REWARD" in
  env) ;;
  outcome|process|first_error) EXTRA=(judgerl.reward_program_file="$(dirname "$0")/rewards/$REWARD.yaml") ;;
  *) EXTRA=(judgerl.reward_program_file="$REWARD") ;;
esac
judgerl_train "python_tool_${REWARD##*/}" data.train_batch_size=16 data.val_batch_size=64 \
  ${EXTRA[@]+"${EXTRA[@]}"} "$@"
