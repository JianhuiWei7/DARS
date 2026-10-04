# DARS: Dependency-Aware Reward Shaping

Official code repository for **“Dependency-Aware Reward Shaping for Agentic Reinforcement Learning.”**

**Ziyi Chen\*, Yan Zhang\*, Jianhui Wei\*, Daoan Zhang, Zuozhu Liu**  
\* Equal contribution.

[![Paper](https://img.shields.io/badge/Paper-arXiv-B31B1B?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2610.01207)
[![Website](https://img.shields.io/badge/Website-Project_Page-3E5DCC?style=for-the-badge&logo=githubpages&logoColor=white)](https://jianhuiwei7.github.io/DARS/)

DARS assigns step-level credit by tracking **dependency-aware task progress**. It represents task requirements as a dependency graph, tracks which requirements are verified, broken or repaired, and converts changes in graph potential into signed rewards for agentic reinforcement learning.

This repository uses the `judgerl` training infrastructure to connect DARS reward annotations to multi-turn environments and verl. GiGPO is a supported learning backbone and baseline; DARS supplies the dependency-aware reward signal.

> **Implementation status:** this checkout includes the training infrastructure and DARS integration adapter. The standalone `dars` package implementing graph replay and domain-specific annotations, together with its domain configuration files, is not included. DARS training requires that additional source checkout; the current repository alone is not a complete paper reproduction release.

## Contents

- [Method overview](#method-overview)
- [Repository structure](#repository-structure)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Training with DARS](#training-with-dars)
- [Baselines](#baselines)
- [Evaluation](#evaluation)
- [Configuration and documentation](#configuration-and-documentation)
- [Tests](#tests)
- [Citation](#citation)
- [Acknowledgements and license](#acknowledgements-and-license)

## Method overview

An action can make local progress while relying on a prerequisite that is still broken. DARS accounts for these dependencies when assigning credit.

1. **Represent task requirements.** A directed graph encodes predicates and prerequisite relationships.
2. **Annotate the trajectory.** An LLM annotator identifies per-step verification, invalidation and successful repair events.
3. **Replay persistent states.** Each predicate remains unestablished, verified or broken until new evidence changes its state.
4. **Measure dependency-aware progress.** Verified predicates contribute to a graph potential; broken ancestors attenuate the contribution of their descendants.
5. **Assign signed step credit.** Increases in potential earn positive credit, decreases receive negative credit, and unchanged graph states receive zero credit.

For graph potential $\Phi_G(t)$, DARS defines the shaping signal as

$$
\widetilde{r}_t = \rho\\mathrm{clip}\left(\Phi_G(t)-\Phi_G(t-1),-\kappa,\kappa\right)
$$

where $\rho$ scales rewards and $\kappa$ clips each potential change. Graph replay and numerical credit are deterministic conditional on the graph and event annotations. Repeating already established progress does not change the potential. Clipping and learner-specific transformations mean the deployed signal does not carry a general policy-invariance guarantee.

In the paper's GiGPO integration, DARS supplies rewards to the step-level anchor-state channel while retaining the episode-level learning signal. The adapter in this repository delegates the graph computation to the external `dars` package and routes annotation requests through the shared judge client.

## Repository structure

```text
.
├── judgerl/
│   ├── rewards/dars.py       # DARS reward adapter
│   ├── judge/               # Judge backends, validation, caching and request control
│   ├── envs/                # Multi-turn task environments
│   ├── algos/               # GiGPO, GRPO and RLOO estimators
│   ├── backends/verl/       # Training loop and verl configuration
│   └── sandbox/             # Python execution backends
├── recipes/
│   ├── alfworld.sh
│   ├── webshop.sh
│   ├── search.sh
│   ├── numberline.sh        # Training-stack smoke run
│   └── rewards/             # Reward and judge configurations
├── scripts/                 # Installation and evaluation
├── examples/                # Custom environments, rewards and estimators
├── docs/                    # Detailed reference documentation
├── tests/
├── pyproject.toml
├── LICENSE
└── NOTICE
```

All commands below run from the repository root. The installable Python package remains named `judgerl`.

## Installation

For GPU training, use Linux, NVIDIA GPUs and a compatible CUDA environment. The installation script targets Python 3.10–3.12; Python 3.12 is the documented training setup. Core judge and reward development does not require a GPU.

```bash
git clone https://github.com/JianhuiWei7/DARS.git
cd DARS

# Install verl 0.9.1, SGLang 0.5.8 and the local package.
bash scripts/install.sh

# Alternative rollout engine:
# ROLLOUT=vllm bash scripts/install.sh
```

The installer attempts to install FlashAttention. Recipes use SDPA with padded batches when FlashAttention is unavailable. For dependency and performance details, see the [training infrastructure guide](docs/judge_rl.md#install).

For core development and tests only:

```bash
python -m pip install -e ".[test]"
```

## Quick start

Check the training stack using the built-in NumberLine environment:

```bash
MODEL=Qwen/Qwen2.5-0.5B-Instruct NGPUS=1 bash recipes/numberline.sh
```

This uses environment rewards and GiGPO. It needs no external task simulator or judge API, but downloads the policy model and requires the GPU training dependencies. It is a smoke run, not a DARS experiment.

## Training with DARS

### 1. Install the DARS algorithm package

A separate DARS source checkout must provide:

- `dars.DARSConfig`, `dars.DARSReward`, `dars.Trajectory` and `dars.Turn`;
- domain-specific annotation, validation and graph replay code;
- `configs/alfworld.yaml`, `configs/webshop.yaml` and `configs/search.yaml`.

Point `DARS_ROOT` to that source checkout, rather than to this repository unless those files have been added here:

```bash
export DARS_ROOT=/absolute/path/to/dars-source
python -m pip install -e "$DARS_ROOT"
```

The current adapter supports trajectory-level annotation. Prefix annotation is not implemented in this adapter.

### 2. Configure the judge

The supplied DARS recipes use a LiteLLM DeepSeek backend. Set the key in your shell:

```bash
export DEEPSEEK_API_KEY=your-key

# Validate configuration without sending a judge request.
judgerl-check recipes/rewards/dars_alfworld.yaml

# Optional: send a live request to check credentials and backend availability.
judgerl-check recipes/rewards/dars_alfworld.yaml --live
```

Change the `judge` block in the reward YAML to select another backend. API credentials should stay in environment variables. The client also supports OpenAI-compatible endpoints and local or served Hugging Face models; see [judge configuration](docs/judge.md).

### 3. Prepare an environment and train

| Environment | Preparation | DARS reward configuration |
|---|---|---|
| ALFWorld | Install ALFWorld and download its data | [`dars_alfworld.yaml`](recipes/rewards/dars_alfworld.yaml) |
| WebShop | Install simulator, product data and search index; set `WEBSHOP_ROOT` | [`dars_webshop.yaml`](recipes/rewards/dars_webshop.yaml) |
| Search | Prepare QA data and start a compatible retrieval server | [`dars_search.yaml`](recipes/rewards/dars_search.yaml) |

**ALFWorld**

```bash
python -m pip install "alfworld[full]==0.4.2"
alfworld-download
# If using a custom data directory, export ALFWORLD_DATA accordingly.

MODEL=Qwen/Qwen2.5-1.5B-Instruct NGPUS=2 REWARD=dars \
  bash recipes/alfworld.sh
```

**WebShop**

```bash
export WEBSHOP_ROOT=/absolute/path/to/webshop
MODEL=Qwen/Qwen2.5-1.5B-Instruct NGPUS=2 REWARD=dars \
  bash recipes/webshop.sh
```

`WEBSHOP_ROOT` must contain `web_agent_site`. See the [WebShop setup notes](docs/environments_webshop_search.md#webshop-judgerlenvswebshop) for task splits and simulator requirements.

**Search**

```bash
export RETRIEVER_URL=http://127.0.0.1:8000/retrieve
DATA_DIR=/absolute/path/to/searchR1_processed_direct \
  MODEL=Qwen/Qwen2.5-1.5B-Instruct NGPUS=2 REWARD=dars \
  bash recipes/search.sh
```

See the [Search environment documentation](docs/environments_webshop_search.md) for the expected data format and retrieval API. Recipes generate local Parquet training and validation datasets under `data/`.

These are launch examples. Exact paper reproduction additionally requires the released experiment configurations, model and judge versions, data splits, seeds and evaluation protocol; the launch examples alone do not establish those settings.

## Baselines

Run the same environments with their environment rewards and GiGPO:

```bash
REWARD=gigpo bash recipes/alfworld.sh
REWARD=gigpo bash recipes/webshop.sh
REWARD=gigpo bash recipes/search.sh
```

These baseline runs do not need the external DARS package. Each environment still requires its own data and services.

Other judge-based reward programs are available for comparisons:

```bash
REWARD=recipes/rewards/outcome.yaml bash recipes/alfworld.sh
REWARD=recipes/rewards/process.yaml bash recipes/alfworld.sh
REWARD=recipes/rewards/first_error.yaml bash recipes/alfworld.sh
```

Set the credentials required by the selected reward configuration before running a judged baseline.

## Evaluation

Evaluate a training checkpoint using the same base model used during training:

```bash
MODEL=Qwen/Qwen2.5-1.5B-Instruct \
  CKPT=outputs/alfworld_dars/global_step_150 N=4 \
  bash scripts/eval.sh alfworld
```

Or evaluate a Hugging Face policy directly:

```bash
MODEL=Qwen/Qwen2.5-1.5B-Instruct N=4 bash scripts/eval.sh alfworld
```

Replace `alfworld` with `webshop` or `search` to evaluate another environment. The script runs the recipe's validation set through verl's `val_only` mode. `N` controls samples per task; `TEMP` controls sampling temperature. Validation normally uses environment rewards and reports success, episode reward and step-count metrics. The checkpoint path above is an example of the default DARS run naming scheme.

## Configuration and documentation

Common launcher settings include `MODEL`, `NGPUS`, `TP`, `STEPS`, `LR`, `EXP`, `OUT`, `LOGGER`, `MAX_PROMPT` and `MAX_RESPONSE`. Additional arguments are passed to verl as Hydra overrides:

```bash
MODEL=Qwen/Qwen2.5-7B-Instruct NGPUS=8 STEPS=300 REWARD=dars \
  bash recipes/alfworld.sh trainer.logger='[console,wandb]'
```

DARS reward YAML files reference `${DARS_ROOT}/configs/<environment>.yaml`. Their `overrides` mapping can override DARS settings such as `kappa`, `rho` and `lam`; the exact semantics come from the installed DARS package.

- [Training infrastructure and extension guide](docs/judge_rl.md)
- [Judge backends and request configuration](docs/judge.md)
- [Environment interfaces and Python sandbox](docs/environments.md)
- [WebShop and Search setup](docs/environments_webshop_search.md)
- [Shared launcher options](recipes/common.sh)
- [Custom components](examples/)

Generated datasets, outputs, local credentials and build artifacts are excluded by `.gitignore`.

## Tests

```bash
python -m pip install -e ".[test]"
python -m pytest -q -ra
```

Core tests use scripted judges and do not need API keys. Optional tests require additional dependencies: the DARS package for DARS integration, a verl-agent checkout for upstream parity, torch for GiGPO parity, pyarrow for Parquet loading, or a container runtime and local image for container sandbox checks. Set `VERL_AGENT_ROOT` to enable upstream parity checks and inspect skipped tests with `-ra`.

Passing core tests does not verify GPU training or reproduce the paper's results.

## Citation

If you use DARS in your research, please cite:

```bibtex
@misc{chen2026dependencyawarerewardshapingagentic,
      title={Dependency-Aware Reward Shaping for Agentic Reinforcement Learning},
      author={Ziyi Chen and Yan Zhang and Jianhui Wei and Daoan Zhang and Zuozhu Liu},
      year={2026},
      eprint={2610.01207},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2610.01207},
}
```

## Acknowledgements and license

The training infrastructure builds on [verl](https://github.com/volcengine/verl). Environment implementations and the GiGPO estimator include code derived from [verl-agent](https://github.com/langfengQ/verl-agent). We thank their authors for making these implementations available.

The code is distributed under the [Apache License 2.0](LICENSE). See [NOTICE](NOTICE) for upstream attribution.
