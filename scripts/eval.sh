#!/usr/bin/env bash
# Evaluate a policy on a recipe's validation set, without training.
#
#   CKPT=outputs/alfworld_gigpo/global_step_150 bash scripts/eval.sh alfworld     # a Judge RL / verl checkpoint
#   MODEL=Qwen/Qwen2.5-1.5B-Instruct            bash scripts/eval.sh alfworld     # any HF model id or path
#
#   N        samples per task (default 4); success is averaged over them   (val-aux/<env>/success/mean@N)
#   TEMP     sampling temperature (default 0.4, as during training validation; 0 = greedy)
#   DATA     validation data dir (default: the recipe's)
#
# With CKPT, the checkpoint's actor weights are loaded; MODEL must then be the base model it was
# trained from (default Qwen/Qwen2.5-1.5B-Instruct). Metrics print to the console (add LOGGER=...).
set -euo pipefail
RECIPE=${1:?usage: scripts/eval.sh <recipe name, e.g. alfworld> [hydra overrides]}; shift
HERE=$(cd "$(dirname "$0")/.." && pwd)
N=${N:-4}; TEMP=${TEMP:-0.4}
ARGS=(trainer.val_only=True trainer.val_before_train=True
      actor_rollout_ref.rollout.val_kwargs.n="$N" actor_rollout_ref.rollout.val_kwargs.temperature="$TEMP"
      actor_rollout_ref.rollout.val_kwargs.do_sample="$([ "$TEMP" = 0 ] && echo False || echo True)")
if [ -n "${CKPT:-}" ]; then
  ARGS+=(trainer.resume_mode=resume_path trainer.resume_from_path="$(cd "$CKPT" && pwd)")
fi
EXP=${EXP:-eval_${RECIPE}_$(basename "${CKPT:-${MODEL:-base}}")} STEPS=1 \
  bash "$HERE/recipes/$RECIPE.sh" ${ARGS[@]+"${ARGS[@]}"} "$@"
