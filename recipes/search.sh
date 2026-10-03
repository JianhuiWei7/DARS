#!/usr/bin/env bash
# Search-R1 style multi-turn retrieval QA. Needs a retrieval server (verl-agent's `retrieval_server`
# or any endpoint with the same API) at $RETRIEVER_URL and the preprocessed data at
# ~/data/searchR1_processed_direct (or DATA_DIR=...). REWARD=gigpo | dars | <reward-program YAML>.
source "$(dirname "$0")/common.sh"
ENV=search DATA=${DATA:-data/search}
REWARD=${REWARD:-gigpo}
RETRIEVER_URL=${RETRIEVER_URL:-http://127.0.0.1:8000/retrieve}
PA=(); [ -n "${DATA_DIR:-}" ] && PA=(--provider-arg "data_dir=$DATA_DIR")
[ -f "$DATA/train.parquet" ] || judgerl-data --env search --train 0 --val 512 --train-split train --val-split test ${PA[@]+"${PA[@]}"} --out "$DATA"
EXTRA=()
case "$REWARD" in
  gigpo) ;;
  dars)  EXTRA=(judgerl.reward_program_file="$(dirname "$0")/rewards/dars_search.yaml") ;;
  *)     EXTRA=(judgerl.reward_program_file="$REWARD") ;;
esac
judgerl_train "search_${REWARD##*/}" \
  data.train_batch_size=16 data.val_batch_size=512 \
  ++judgerl.env_kwargs.search_url="$RETRIEVER_URL" ++judgerl.env_kwargs.max_steps=4 ${EXTRA[@]+"${EXTRA[@]}"} "$@"
