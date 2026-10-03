#!/usr/bin/env bash
# ALFWorld, verl-agent / GiGPO settings (16 tasks x 8 rollouts, 50 steps max, history 2, gamma 0.95).
#   REWARD=gigpo  (default) environment rewards + GiGPO
#   REWARD=dars   DARS step credit from an LLM annotator (recipes/rewards/dars_alfworld.yaml)
#   REWARD=<file> any reward-program YAML
# Needs ALFWorld data: `alfworld-download` and $ALFWORLD_DATA. `--train 0` = every training game.
source "$(dirname "$0")/common.sh"
ENV=alfworld DATA=${DATA:-data/alfworld}
REWARD=${REWARD:-gigpo}
[ -f "$DATA/val.parquet" ] || judgerl-data --env alfworld --train 0 --val 128 --val-split valid_seen --out "$DATA"
EXTRA=()
case "$REWARD" in
  gigpo) ;;
  dars)  EXTRA=(judgerl.reward_program_file="$(dirname "$0")/rewards/dars_alfworld.yaml") ;;
  *)     EXTRA=(judgerl.reward_program_file="$REWARD") ;;
esac
judgerl_train "alfworld_${REWARD##*/}" \
  data.train_batch_size=16 data.val_batch_size=128 \
  ++judgerl.env_kwargs.history_length=2 ${EXTRA[@]+"${EXTRA[@]}"} "$@"
