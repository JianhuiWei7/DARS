#!/usr/bin/env bash
# Install Judge RL with the verl training backend into the active Python (3.10-3.12) environment.
#
#   bash scripts/install.sh                 # core + verl + SGLang rollout
#   ROLLOUT=vllm bash scripts/install.sh    # vLLM rollout instead of SGLang
#   ENVS="alfworld webshop" bash scripts/install.sh
#
# Tested combination: verl 0.9.1, torch 2.9.1, SGLang 0.5.8, transformers 5.x, ray 2.58, Python 3.12.
set -euo pipefail
ROLLOUT=${ROLLOUT:-sglang}
ENVS=${ENVS:-}
cd "$(dirname "$0")/.."

pip install -U pip
case "$ROLLOUT" in
  sglang) pip install "sglang[all]==0.5.8" ;;
  vllm)   pip install "vllm>=0.11" ;;
  *) echo "ROLLOUT must be sglang or vllm" >&2; exit 1 ;;
esac
pip install "verl==0.9.1" "TransferQueue>=0.1.8"
pip install -e ".[train,hf,test]"
# FlashAttention: recipes use it when importable (padding removal), else fall back to SDPA on padded batches.
pip install flash-attn --no-build-isolation \
  || echo "flash-attn not installed: recipes fall back to SDPA (slower); to build it from source see README > Install"

for e in $ENVS; do
  case "$e" in
    alfworld) pip install "alfworld[full]==0.4.2"; echo "then run: alfworld-download  (and export ALFWORLD_DATA)" ;;
    webshop)  echo "WebShop: install the simulator (web_agent_site) and export WEBSHOP_ROOT; see docs/environments_webshop_search.md" ;;
    search)   echo "Search: start a retrieval server and export RETRIEVER_URL; see docs/environments_webshop_search.md" ;;
    dars)     pip install -e "${DARS_ROOT:?set DARS_ROOT to a DARS checkout}" ;;
    docker)   echo "python_tool can use a docker/podman sandbox; see docs/environments.md" ;;
    *) echo "unknown env $e" >&2 ;;
  esac
done
python -c "import judgerl, verl; print('judgerl', judgerl.__version__ if hasattr(judgerl, '__version__') else 'ok', '| verl', verl.__version__)"
