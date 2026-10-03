#!/usr/bin/env bash
# WebShop, verl-agent / GiGPO settings. Needs the WebShop simulator: set $WEBSHOP_ROOT
# (the directory containing web_agent_site). REWARD=gigpo | dars | <reward-program YAML>.
source "$(dirname "$0")/common.sh"
ENV=webshop DATA=${DATA:-data/webshop}
REWARD=${REWARD:-gigpo}
[ -f "$DATA/train.parquet" ] || judgerl-data --env webshop --train 0 --val 128 --train-split train --val-split test --out "$DATA"
EXTRA=()
case "$REWARD" in
  gigpo) ;;
  dars)  EXTRA=(judgerl.reward_program_file="$(dirname "$0")/rewards/dars_webshop.yaml") ;;
  *)     EXTRA=(judgerl.reward_program_file="$REWARD") ;;
esac
judgerl_train "webshop_${REWARD##*/}" \
  data.train_batch_size=16 data.val_batch_size=128 \
  ++judgerl.env_kwargs.history_length=2 ++judgerl.env_kwargs.max_steps=15 ${EXTRA[@]+"${EXTRA[@]}"} "$@"
