# Judge RL

**Agentic reinforcement learning with first-class LLM judges, on stock [verl](https://pypi.org/project/verl/).**

Judge RL trains LLM agents on multi-turn environments (ALFWorld, WebShop, Search, a Python tool,
open-ended instructions, or your own) with per-step advantage estimators such as
[GiGPO](https://arxiv.org/abs/2505.10978), and makes it easy to put an LLM judge anywhere in the
reward: an outcome rubric, per-step process scores, first-error localization, dependency-aware
step credit ([DARS](#dars)), or a program of your own.

It is in the spirit of [verl-agent](https://github.com/langfengQ/verl-agent), with three differences:

* **No fork of verl.** Judge RL plugs into verl 0.9.1's extension points (a registered agent loop and
  a registered V1 trainer), so you get upstream verl's rollout engines, FSDP/Megatron workers and
  updates by upgrading the pin.
* **Judges are infrastructure, not a script.** One async judge client serves every reward program:
  any API model through LiteLLM, any OpenAI-compatible server, any Hugging Face chat/causal LM (served
  for you or loaded in-process), or an engine you register. It queues, rate-limits, retries, fails over,
  validates structured output, caches durably, enforces budgets and circuit breakers, and counts
  every outcome. A judge failure is never silently turned into a reward.
* **Everything is a plug-in.** Environments, task providers, reward programs and advantage estimators
  are selected by name or by `package.module:Name` import path from the command line.

## Contents

- [Install](#install)
- [Quick start](#quick-start)
- [Recipes](#recipes)
- [Evaluate](#evaluate)
- [Choosing a judge](#choosing-a-judge)
- [Reward programs](#reward-programs)
- [Extending](#extending)
- [How it fits into verl](#how-it-fits-into-verl)
- [Verification](#verification)
- [Status and roadmap](#status-and-roadmap)

## Install

For GPU training, use a Linux machine with NVIDIA GPUs and a CUDA-compatible Python
environment. The installation script targets Python 3.10–3.12; its recorded training
combination uses Python 3.12. Core judge/reward development does not require a GPU.
Run the commands below from the repository root.

```bash
git clone https://github.com/JianhuiWei7/DARS.git
cd DARS
bash scripts/install.sh                               # verl 0.9.1 + SGLang rollout + judgerl
# or: pip install -e ".[sglang]"   /   pip install -e ".[vllm]"
ENVS="alfworld" bash scripts/install.sh               # plus environment dependencies
```

`judgerl.judge` and the reward programs alone need only `pip install -e .` (no torch, no verl).
Install FlashAttention for full speed: the recipes detect it and then train on unpadded batches.
Without it they fall back to PyTorch SDPA attention on padded batches, which works everywhere but is
several times slower in the training passes. If no prebuilt flash-attn wheel fits your system (it needs
glibc >= 2.32), build it from source: `pip install flash-attn --no-build-isolation --no-binary flash-attn`
with a C++ compiler >= 9 and a CUDA 12 toolkit (`CUDA_HOME`).

## Quick start

Train a small model on the dependency-free NumberLine environment with GiGPO (a few minutes on one GPU):

```bash
MODEL=Qwen/Qwen2.5-0.5B-Instruct NGPUS=1 bash recipes/numberline.sh
```

Then check a judge configuration before a long run, which sends one real request per backend:

```bash
export DEEPSEEK_API_KEY=...
judgerl-check recipes/rewards/process.yaml --live
```

and train with judge rewards:

```bash
REWARD=recipes/rewards/process.yaml bash recipes/alfworld.sh
```

Every recipe is a thin wrapper around `judgerl-train` (verl's Hydra config plus a `judgerl` node);
anything after the recipe name is passed through as a Hydra override:

```bash
MODEL=Qwen/Qwen2.5-7B-Instruct NGPUS=8 STEPS=300 bash recipes/alfworld.sh trainer.logger='[console,wandb]'
```

The policy model is any Hugging Face causal LM id or local path (`MODEL=...`).

## Recipes

| recipe | environment | default reward | needs |
|---|---|---|---|
| `recipes/numberline.sh` | toy number line | environment | nothing |
| `recipes/alfworld.sh` | ALFWorld (TextWorld) | environment + GiGPO; `REWARD=dars` | `alfworld-download` |
| `recipes/webshop.sh` | WebShop | environment + GiGPO; `REWARD=dars` | WebShop simulator (`WEBSHOP_ROOT`) |
| `recipes/search.sh` | Search-R1 style retrieval QA | environment + GiGPO; `REWARD=dars` | a retrieval server (`RETRIEVER_URL`) |
| `recipes/python_tool.sh` | math with a sandboxed Python tool | exact answer; `REWARD=outcome\|process\|first_error` | nothing (optional docker) |
| `recipes/open_ended.sh` | instruction following with rubrics | LLM judge only | a judge |

ALFWorld, WebShop and Search reproduce verl-agent's environment managers (prompts, memory, action
parsing, validity rules and rewards) and its hyper-parameters (16 tasks × 8 rollouts, history 2,
γ = 0.95, 256-row PPO mini-batches, invalid-action penalty 0.1). See
[docs/environments.md](../docs/environments.md) and
[docs/environments_webshop_search.md](../docs/environments_webshop_search.md).

Common knobs: `MODEL`, `NGPUS`, `TP`, `ROLLOUT=sglang|vllm`, `STEPS`, `LR`, `EXP`, `OUT`, `LOGGER`,
`MAX_PROMPT`/`MAX_RESPONSE`, and `LORA_RANK=32` to train a LoRA adapter instead of the full model; the
speed knobs are below. All are documented at the top of `recipes/common.sh`.

### Speed

The recipes use verl's fast paths by default: FlashAttention with padding removal (when installed),
dynamic batching sized to the model (32768 tokens per GPU for updates below 3B parameters, 16384 up to
10B, 8192 above; twice that for log-prob passes), the reference
model kept on the GPU, and one rollout replica per GPU (`TP=1`). Options that trade something for speed:

| knob | effect | trade-off |
|---|---|---|
| `BYPASS_OLD_LOGPROB=1` | skip the old-log-prob forward pass; use the rollout engine's log-probs (verl's rollout-correction bypass) | small numerical mismatch between inference and training engines |
| `TRAINER_MODE=colocate_async` | generate the next batch while training on the current one (verl's colocated async trainer) | slightly off-policy; each step row records its policy version |
| `TRAINER_MODE=separate_async` | rollout on its own GPUs (`actor_rollout_ref.rollout.nnodes` / `n_gpus_per_node`) | needs extra GPUs |
| `GRAD_CKPT=0` | no activation recomputation in the update | more GPU memory |
| `MAX_TOKENS=...` / `LOGPROB_TOKENS=...` | larger micro-batches | more GPU memory |

`MAX_TOKENS` must hold at least one full step (`MAX_PROMPT + MAX_RESPONSE`); the recipes check this.
The trainer logs `judgerl/prompt_truncated_ratio`: prompts longer than `MAX_PROMPT` are left-truncated
by verl, which cuts the task instruction, so raise `MAX_PROMPT` if it is ever above 0.

## Evaluate

```bash
CKPT=outputs/alfworld_gigpo/global_step_150 N=4 bash scripts/eval.sh alfworld   # a training checkpoint
MODEL=Qwen/Qwen2.5-7B-Instruct bash scripts/eval.sh alfworld                    # any HF model
```

This runs the recipe's validation set only (verl's `val_only`), `N` samples per task, and prints
per-data-source metrics: `val-aux/<env>/success/mean@N` (success rate), `val-core/<env>/reward/mean@N`
(episode score) and `val-aux/<env>/n_steps/mean@N`. Validation during training reports the same
metrics every `trainer.test_freq` steps.

## Choosing a judge

A reward program's `judge:` block configures the judge client ([docs/judge.md](../docs/judge.md)):

```yaml
judge:
  backend: {type: litellm, model: deepseek/deepseek-chat, api_key_env: DEEPSEEK_API_KEY}
  queue: {max_inflight: 32}
  budget_usd: 50
  cache: {path: outputs/judge_cache.sqlite}
  telemetry: {jsonl_path: outputs/judge_telemetry.jsonl}
```

| backend | use it for |
|---|---|
| `litellm` | any hosted model LiteLLM knows (OpenAI, Anthropic, DeepSeek, Gemini, Together, ...) |
| `openai_compat` | your own vLLM / SGLang / TGI server |
| `hf_server` | **any Hugging Face chat/causal LM id**: Judge RL starts and health-checks a vLLM/SGLang server for it |
| `hf_local` | a small generative judge (causal LM) loaded in-process |
| `openai_compat` + `base_url: verl://reward_model` | **a Hugging Face judge on the trainer's own GPUs**, served by verl and woken only while judging (`COLOCATED_JUDGE=<HF id>`) |
| `colocated` | an in-process engine you register |
| `scripted` | tests and dry runs, with fault injection |

A judge on the trainer's GPUs needs no extra hardware: verl serves the model next to the policy, keeps it
asleep during rollout and training, and wakes it after each rollout to judge the batch:

```bash
COLOCATED_JUDGE=Qwen/Qwen3-8B REWARD=recipes/rewards/process_colocated.yaml bash recipes/alfworld.sh
```

(`judgerl.reward_stage=trainer` is set for you; any `reward.reward_model.*` override of verl applies,
e.g. `reward.reward_model.enable_resource_pool=True` to give the judge its own GPUs instead, in which
case episodes are judged during rollout again with `judgerl.reward_stage=rollout`.)

Several backends can form a failover pool. Requests are cached by content and prompt/schema
versions, so re-running an experiment does not pay for the same judgments twice.

## Reward programs

| `type` | what the judge does | reward |
|---|---|---|
| `outcome` | grades the episode against a rubric, score in [0, 1] | episode score (mixed with the environment's) |
| `process` | rates every step −1 / 0 / +1 | step credit, combined with environment rewards |
| `first_error` | names the first wrong step of a failed episode | credit for the steps before it |
| `dars` | annotates which task predicates each step verifies, breaks or depends on | dependency-aware potential-based step credit |
| `judgerl.envs.open_ended:RubricOutcome` | checks each rubric item of a task | episode score |
| `package.module:Class` | yours | yours |

Step credit is combined with the environment's step rewards by `combine: replace | add | redistribute`;
`redistribute` keeps every episode's discounted return equal to the environment's, so shaping can
move credit between steps but cannot change which episodes are better.

Judged rewards are computed as each episode finishes, overlapping the rest of the rollout.
Validation uses environment rewards unless `judgerl.judge_val=true` (rollout-stage judging only).

### Group-level judges

Relative judgments are often more reliable than absolute ones. A group program sees all rollouts of
a task together and runs in the trainer, before advantages are computed:

| `type` | judge calls per group | judged score |
|---|---|---|
| `listwise` | 1: all rollouts in one prompt | a score in [0, 1] per rollout |
| `pairwise` | one per pair (×2 with `swap`, capped by `max_pairs`) | win rate |
| `package.module:Class` | yours (`ascore_group(task, metadata, episodes)`) | yours |

```bash
bash recipes/alfworld.sh judgerl.group_reward_program_file=recipes/rewards/group_pairwise.yaml
```

A failed judgment never becomes a reward: the episode keeps its environment rewards and the failure is
counted. If more than `judgerl.max_reward_failure_ratio` (default 0.5) of a step's episodes lose their
judged reward, the run stops with the error, so a broken judge (key, model name, request parameters)
cannot silently turn a judged run into an environment-reward run.

A group program can be combined with a per-episode reward program; it sees the episode scores the
per-episode program produced. The trainer logs `judgerl/group_program/*` and
`judgerl/reward_program/*_ratio` (share of episodes whose judge call succeeded, failed or was skipped).

### DARS

This directory contains the Judge RL integration for Dependency-Aware Reward Shaping,
not the DARS algorithm package itself. Running `REWARD=dars` requires a separate source
checkout exposing `dars.DARSConfig`, `dars.DARSReward`, `dars.Trajectory` and `dars.Turn`,
along with `configs/alfworld.yaml`, `configs/webshop.yaml` and `configs/search.yaml`.
Those sources and configuration files are not included in this directory.

Once that checkout is available:

```bash
export DARS_ROOT=/path/to/dars-source
python -m pip install -e "$DARS_ROOT"
export DEEPSEEK_API_KEY=your-key
judgerl-check recipes/rewards/dars_alfworld.yaml
REWARD=dars bash recipes/alfworld.sh
```

Annotation requests then go through Judge RL's judge client. This adapter supports
trajectory annotation; prefix annotation is not implemented here. A complete paper
reproduction additionally needs the DARS sources, exact experiment configurations,
data preparation and evaluation settings.

## Extending

All three extension points take an import path, so nothing needs to be registered inside this
repository. Working examples are in [`examples/`](../examples):

```bash
# an environment and its task provider
judgerl-data --env examples.custom_env:GuessWordEnv --tasks-fn examples.custom_env:tasks --train 0 --val 0 --out data/guess
judgerl-train judgerl.env=examples.custom_env:GuessWordEnv data.train_files=data/guess/train.parquet ...

# a reward program (examples/custom_reward.py), referenced from a YAML file
judgerl-train judgerl.reward_program_file=my_reward.yaml ...     # type: examples.custom_reward:Efficiency

# an advantage estimator
judgerl-train judgerl.estimator=examples.custom_estimator:success_weighted_grpo ++judgerl.estimator_kwargs.bonus=0.5 ...
```

* **Environment**: subclass `judgerl.envs.base.Env` (`reset`, `step`, `task_text`, `metadata`).
  `Observation.anchor` is the state id used for GiGPO step groups; `metadata()` carries judge-only
  information (gold answers, rubrics) that the policy never sees. Environments with global state run
  one episode per worker process automatically (`thread_safe = False`). See
  [docs/environments.md](../docs/environments.md).
* **Dataset**: a task is a dict passed to `reset`. Provide a `tasks(split, limit, **kwargs)` function
  or a JSONL file (`judgerl-data --tasks tasks.jsonl`).
* **Reward program**: subclass `judgerl.rewards.RewardProgram` and implement
  `async ascore(task, metadata, rows, success, episode_score) -> RewardRecord`.
* **Estimator**: a function `(StepBatch, **kwargs) -> per-row advantages`.

## How it fits into verl

```
dataset row (task) ──> EnvAgentLoop ── per step: prompt ─> rollout server ─> action ─> env.step
                           │  one training row per step (prompt, response, anchor, reward, ...)
                           └─ episode done ─> reward program ─> judge client ─> step rewards / score
                                                                              (cache, budget, breakers)
JudgeRLTrainerSync ── rows of a batch ─> estimator (gigpo | grpo | rloo | yours) ─> advantages ─> PPO update
```

* `judgerl.backends.verl.agent_loop.EnvAgentLoop` is a verl agent loop that returns one
  `AgentLoopOutput` per environment step; verl's TransferQueue carries the per-step metadata.
* `judgerl.backends.verl.trainer.JudgeRLMixin` adds Judge RL's advantage step to verl's V1 trainers,
  registered as trainer modes `judgerl_sync`, `judgerl_colocate_async` and `judgerl_separate_async`;
  it replaces only the advantage computation. Rollout correction, KL loss, dynamic batching and
  checkpointing are verl's. `judgerl.mini_batch_rows` sets the PPO mini-batch in step rows (as
  verl-agent does), since the number of rows per batch varies with episode lengths.
* Configuration is `judgerl/backends/verl/config/judgerl_trainer.yaml`: verl's `ppo_trainer` plus the
  `judgerl` node.

Judge RL is tested with verl 0.9.1 and SGLang 0.5.8 (vLLM rollout also works through verl).

## Verification

* **GiGPO**: `judgerl.algos.advantages.gigpo` reproduces verl-agent's advantages exactly on 400
  randomized batches (`tests/algos/test_gigpo_parity.py`).
* **Environments**: the ALFWorld, WebShop and Search ports are checked byte-for-byte against the
  upstream managers (prompts, anchors, rewards, validity flags) when `$VERL_AGENT_ROOT` points at a
  verl-agent checkout.
* **End to end**: NumberLine, Qwen2.5-0.5B-Instruct, 30 GiGPO steps: invalid-action rate 0.56 → 0,
  validation reward 6.4 → 10.0. ALFWorld parity runs against verl-agent are in progress.

* **Speed** (ALFWorld, Qwen2.5-1.5B-Instruct, 16 tasks x 8 rollouts, up to 50 steps each, 2 x H100,
  synchronous trainer): about 215 s per GiGPO step with the recipe defaults (FlashAttention, dynamic
  batching), against about 950 s with SDPA on padded batches. Doubling the token budgets from 16384/32768
  to 32768/65536 saved a further ~10% at 36 GB peak memory, hence the size-dependent defaults.
* **Judge served by verl on the training GPU**: NumberLine with a colocated Qwen2.5-1.5B-Instruct judge
  (`reward_stage=trainer`) woke, judged and slept the judge each step (1-9 s once warm).

```bash
python -m pip install -e ".[test]"
python -m pytest -q -ra
```

The default tests use scripted judges and do not need API keys. Optional tests are
skipped when their dependencies are absent: DARS integration needs the separate
`dars` package, upstream parity needs a verl-agent checkout (and torch for GiGPO),
Parquet loading needs pyarrow, and container tests need an available container runtime
and image. Passing the core suite does not verify GPU training or paper reproduction.

## Status and roadmap

`0.1.0.dev0`. Working: the judge client with all backends, the reward programs above, six
environments and a code sandbox, GiGPO/GRPO/RLOO, the verl backend. Planned: vision-language recipes.

## Acknowledgements

Judge RL builds on [verl](https://pypi.org/project/verl/). The ALFWorld, WebShop and Search
environment managers and the GiGPO estimator are ports of [verl-agent](https://github.com/langfengQ/verl-agent)
(Apache-2.0); see [NOTICE](../NOTICE).

```bibtex
@article{feng2025gigpo,
  title   = {Group-in-Group Policy Optimization for LLM Agent Training},
  author  = {Feng, Lang and Xue, Zhenghai and Liu, Tingcong and An, Bo},
  journal = {arXiv preprint arXiv:2505.10978},
  year    = {2025}
}
```

## License

Apache-2.0.
