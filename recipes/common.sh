#!/usr/bin/env bash
# Shared launcher for the recipes. A recipe sets ENV, DATA and its own overrides, then calls `judgerl_train`.
#
# Environment knobs (all optional):
#   MODEL        policy model: any Hugging Face id or local path   (default Qwen/Qwen2.5-1.5B-Instruct)
#   NGPUS        GPUs on this node                                  (default 2)
#   TP           rollout tensor parallel size (default 1: one rollout replica per GPU is fastest for
#                models that fit on one GPU; raise it for models that do not)
#   ROLLOUT      sglang | vllm                                      (default sglang)
#   STEPS        training steps                                     (default 150)
#   EXP          experiment name                                    (default: recipe name)
#   OUT          output root                                        (default outputs/)
#   LOGGER       verl logger list, e.g. '[console,wandb]'           (default '[console]')
#   MAX_PROMPT / MAX_RESPONSE   per-step prompt / response token limits (default 4096 / 512); prompts
#                longer than MAX_PROMPT are left-truncated by verl (judgerl/prompt_truncated_ratio)
#   ATTN         auto | flash | sdpa. flash = FlashAttention with padding removal (fast); sdpa = PyTorch
#                attention on padded batches (slower; for machines where flash-attn cannot be installed).
#                auto (default) uses flash when `import flash_attn` works.
#   MAX_TOKENS   tokens per GPU per micro-batch for the actor update (dynamic batching). Default by model
#                size: 32768 (<3B), 16384 (3-10B), 8192 (larger); halved with sdpa (padded batches).
#   LOGPROB_TOKENS  the same for the log-prob passes (no gradients) (default: 2 x MAX_TOKENS with flash)
#                Lower both if a step runs out of memory (LoRA or offloading also frees memory).
#   REF_OFFLOAD  1 keeps the reference model on CPU between uses (saves GPU memory, costs time; default 0)
#   GRAD_CKPT    gradient checkpointing (default 1; 0 is faster when activations fit)
#   BYPASS_OLD_LOGPROB  1 skips recomputing old log-probs and uses the rollout engine's log-probs instead
#                (verl's rollout-correction bypass mode; saves one forward pass over the batch per step)
#   TRAINER_MODE sync | colocate_async | separate_async: verl's V1 trainer modes (default sync); the
#                async modes overlap generation with training
#   LORA_RANK    >0 trains a LoRA adapter (all linear layers) instead of the full model; LR defaults to 3e-5
#   LR           actor learning rate                                (default 1e-6, or 3e-5 with LoRA)
#   COLOCATED_JUDGE  a Hugging Face model id: verl serves it on the trainer's GPUs (asleep during rollout
#                and training) and judge configs reach it as `base_url: verl://reward_model`; judging then
#                runs in the trainer after each rollout (judgerl.reward_stage=trainer)
#   JUDGE_TP     tensor parallel size of the colocated judge        (default: TP)
set -euo pipefail

MODEL=${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}
NGPUS=${NGPUS:-2}
TP=${TP:-1}
ROLLOUT=${ROLLOUT:-sglang}
STEPS=${STEPS:-150}
OUT=${OUT:-outputs}
LOGGER=${LOGGER:-'[console]'}
MAX_PROMPT=${MAX_PROMPT:-4096}
MAX_RESPONSE=${MAX_RESPONSE:-512}

ATTN=${ATTN:-auto}
if [ "$ATTN" = auto ]; then
  if python -c "import flash_attn" 2>/dev/null; then ATTN=flash; else ATTN=sdpa; fi
fi
# default token budget by model size (billions of parameters, estimated from the model config):
# < 3B: 32768, 3-10B: 16384, larger: 8192 tokens per GPU for the update, twice that for log-prob passes
if [ -z "${MAX_TOKENS:-}" ]; then
  PARAMS_B=$(python - "$MODEL" <<'PY' 2>/dev/null || echo 7
import sys
from transformers import AutoConfig
c = AutoConfig.from_pretrained(sys.argv[1])
c = getattr(c, "text_config", None) or c
h, L, v = c.hidden_size, c.num_hidden_layers, c.vocab_size
ff = getattr(c, "intermediate_size", 4 * h)
print(int((L * (4 * h * h + 3 * h * ff) + 2 * v * h) / 1e9))
PY
)
  if [ "$PARAMS_B" -lt 3 ]; then MAX_TOKENS=32768; elif [ "$PARAMS_B" -le 10 ]; then MAX_TOKENS=16384; else MAX_TOKENS=8192; fi
  [ "$ATTN" = sdpa ] && MAX_TOKENS=$((MAX_TOKENS / 2))    # padded batches use more memory than their token count
fi
case "$ATTN" in
  flash) LOGPROB_TOKENS=${LOGPROB_TOKENS:-$((2 * MAX_TOKENS))}
         ATTN_ARGS=(actor_rollout_ref.model.use_remove_padding=True) ;;
  sdpa)  LOGPROB_TOKENS=${LOGPROB_TOKENS:-$MAX_TOKENS}
         echo "judgerl: flash-attn not available; using SDPA on padded batches (slower). See README > Install." >&2
         ATTN_ARGS=(actor_rollout_ref.model.use_remove_padding=False
                    +actor_rollout_ref.model.override_config.attn_implementation=sdpa) ;;
  *) echo "ATTN must be auto, flash or sdpa" >&2; exit 1 ;;
esac
if [ "$MAX_TOKENS" -lt $((MAX_PROMPT + MAX_RESPONSE)) ]; then
  echo "MAX_TOKENS=$MAX_TOKENS is below MAX_PROMPT+MAX_RESPONSE=$((MAX_PROMPT + MAX_RESPONSE)): one sequence would not fit" >&2
  exit 1
fi
echo "judgerl: attention=$ATTN tokens/GPU update=$MAX_TOKENS log-prob=$LOGPROB_TOKENS rollout TP=$TP" >&2
REF_OFFLOAD=${REF_OFFLOAD:-0}
GRAD_CKPT=${GRAD_CKPT:-1}
SPEED_ARGS=()
if [ "${BYPASS_OLD_LOGPROB:-0}" = 1 ]; then SPEED_ARGS+=(algorithm.rollout_correction.bypass_mode=True); fi
TRAINER_MODE=${TRAINER_MODE:-sync}
case "$TRAINER_MODE" in
  sync|colocate_async|separate_async) SPEED_ARGS+=(trainer.v1.trainer_mode="judgerl_$TRAINER_MODE") ;;
  *) echo "TRAINER_MODE must be sync, colocate_async or separate_async" >&2; exit 1 ;;
esac
LORA_RANK=${LORA_RANK:-0}
if [ "$LORA_RANK" -gt 0 ]; then LR=${LR:-3e-5}; else LR=${LR:-1e-6}; fi
LORA_ARGS=()
if [ "$LORA_RANK" -gt 0 ]; then
  LORA_ARGS=(actor_rollout_ref.model.lora_rank="$LORA_RANK" actor_rollout_ref.model.lora_alpha="$((2 * LORA_RANK))"
             actor_rollout_ref.model.target_modules=all-linear actor_rollout_ref.rollout.load_format=safetensors)
fi

JUDGE_ARGS=()
if [ -n "${COLOCATED_JUDGE:-}" ]; then
  JUDGE_ARGS=(reward.reward_model.enable=True reward.reward_model.model_path="$COLOCATED_JUDGE"
              reward.reward_model.rollout.name="$ROLLOUT" reward.reward_model.rollout.tensor_model_parallel_size="${JUDGE_TP:-$TP}"
              reward.reward_model.rollout.gpu_memory_utilization=0.4
              judgerl.reward_stage=trainer)
fi

judgerl_train() {
  # $1 = experiment name; remaining args = extra hydra overrides (they win over the defaults below)
  local exp=${EXP:-$1}; shift
  judgerl-train \
    data.train_files="$DATA/train.parquet" data.val_files="$DATA/val.parquet" \
    data.max_prompt_length="$MAX_PROMPT" data.max_response_length="$MAX_RESPONSE" \
    actor_rollout_ref.model.path="$MODEL" \
    actor_rollout_ref.model.enable_gradient_checkpointing="$([ "$GRAD_CKPT" = 1 ] && echo True || echo False)" \
    actor_rollout_ref.actor.optim.lr="$LR" \
    actor_rollout_ref.actor.use_dynamic_bsz=True actor_rollout_ref.actor.ppo_max_token_len_per_gpu="$MAX_TOKENS" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="$LOGPROB_TOKENS" \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="$LOGPROB_TOKENS" \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.use_kl_loss=True actor_rollout_ref.actor.kl_loss_coef=0.01 actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.ref.fsdp_config.param_offload="$([ "$REF_OFFLOAD" = 1 ] && echo True || echo False)" \
    actor_rollout_ref.rollout.name="$ROLLOUT" actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size="$TP" actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.agent.num_workers=8 \
    algorithm.gamma=0.95 algorithm.use_kl_in_reward=False \
    judgerl.env="$ENV" judgerl.estimator=gigpo judgerl.invalid_penalty=0.1 judgerl.mini_batch_rows=256 \
    trainer.n_gpus_per_node="$NGPUS" trainer.nnodes=1 trainer.total_training_steps="$STEPS" trainer.total_epochs=100 \
    trainer.test_freq=5 trainer.save_freq=25 trainer.val_before_train=True trainer.logger="$LOGGER" \
    trainer.project_name=judgerl trainer.experiment_name="$exp" trainer.default_local_dir="$OUT/$exp" \
    "${ATTN_ARGS[@]}" ${SPEED_ARGS[@]+"${SPEED_ARGS[@]}"} \
    ${LORA_ARGS[@]+"${LORA_ARGS[@]}"} ${JUDGE_ARGS[@]+"${JUDGE_ARGS[@]}"} "$@"
}
