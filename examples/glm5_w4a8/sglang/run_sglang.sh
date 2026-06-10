#!/bin/bash
# =============================================================================
# GLM-5 W4A8 MTP — SGLang serve 部署 (sglang 0.5.12)
# Architecture: GlmMoeDsaForCausalLM | 256 Experts | MLA | MTP=1
# Max Position: 202752 | Deploy: 128K context
#
# 4 节点: TP=32 PP=1 EP=32 (GLM-5 不支持 PP)
# 单节点: TP=8 PP=1 EP=8
# =============================================================================
set -eo pipefail

set +u
if [[ -f "/usr/local/Ascend/cann/set_env.sh" ]]; then
    source /usr/local/Ascend/cann/set_env.sh
fi
if [[ -f "/usr/local/Ascend/nnal/atb/set_env.sh" ]]; then
    source /usr/local/Ascend/nnal/atb/set_env.sh
fi
set -u

MODEL_PATH="${MODEL_PATH:-/home/jianzhnie/llmtuner/hfhub/models/Eco-Tech/GLM-5-w4a8}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8001}"
TP="${TP:-32}"
PP="${PP:-1}"
EP="${EP:-32}"
CONTEXT_LEN="${CONTEXT_LEN:-131072}"
MAX_RUNNING_REQS="${MAX_RUNNING_REQS:-8}"
MEM_FRACTION="${MEM_FRACTION:-0.85}"

# Multi-node
NNODES="${NNODES:-4}"
NODE_RANK="${NODE_RANK:-0}"
DIST_INIT_ADDR="${DIST_INIT_ADDR:-10.16.201.193:5000}"

# HCCL/NPU
export HCCL_OP_EXPANSION_MODE="${HCCL_OP_EXPANSION_MODE:-AIV}"
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export HCCL_BUFFSIZE=200
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

echo "============================================"
echo "[INFO] GLM-5 W4A8 MTP — SGLang Deployment"
echo "[INFO] TP=$TP PP=$PP EP=$EP PORT=$PORT"
echo "[INFO] CONTEXT_LEN=$CONTEXT_LEN MAX_RUNNING_REQS=$MAX_RUNNING_REQS"
echo "[INFO] NNODES=$NNODES NODE_RANK=$NODE_RANK DIST_INIT_ADDR=$DIST_INIT_ADDR"
echo "============================================"

sglang serve \
    --model-path "$MODEL_PATH" \
    --host "$HOST" \
    --port "$PORT" \
    --served-model-name glm-5 \
    --trust-remote-code \
    --dtype bfloat16 \
    --tensor-parallel-size "$TP" \
    --pipeline-parallel-size "$PP" \
    --expert-parallel-size "$EP" \
    --device npu \
    --quantization modelslim \
    --mem-fraction-static "$MEM_FRACTION" \
    --context-length "$CONTEXT_LEN" \
    --max-running-requests "$MAX_RUNNING_REQS" \
    --max-total-tokens 16384 \
    --chunked-prefill-size 8192 \
    --disable-cuda-graph \
    --tool-call-parser glm47 \
    --nnodes "$NNODES" \
    --node-rank "$NODE_RANK" \
    --dist-init-addr "$DIST_INIT_ADDR" \
    --log-level info \
    "$@"
