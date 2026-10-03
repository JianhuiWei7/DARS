#!/usr/bin/env bash
# Smoke test (minutes, 1-2 GPUs, no downloads besides the model): GiGPO on the NumberLine toy env.
#   bash recipes/numberline.sh            MODEL=Qwen/Qwen2.5-0.5B-Instruct NGPUS=1 bash recipes/numberline.sh
STEPS=${STEPS:-30}
MAX_PROMPT=${MAX_PROMPT:-1024} MAX_RESPONSE=${MAX_RESPONSE:-256}
source "$(dirname "$0")/common.sh"
ENV=numberline DATA=${DATA:-data/numberline}
[ -f "$DATA/train.parquet" ] || judgerl-data --env numberline --train 512 --val 64 --out "$DATA"
judgerl_train numberline_gigpo \
  data.train_batch_size=16 data.val_batch_size=64 \
  judgerl.mini_batch_rows=null actor_rollout_ref.actor.ppo_mini_batch_size=16 "$@"
