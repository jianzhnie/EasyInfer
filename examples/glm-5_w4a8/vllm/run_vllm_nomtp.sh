#!/bin/bash
# =============================================================================
# GLM-5 / GLM-5.1 W4A8 — Agent-optimized vLLM deployment (No MTP)
# =============================================================================
# Architecture: GlmMoeDsaForCausalLM | 256 Experts | MLA
# Max Position: 202752 | Deploy: 202K context (override with MAX_MODEL_LEN)
#
# No-MTP variant — saves memory, suitable for single-node TP=8 deployment.
# Default: TP=8 PP=1 (single-node); multi-node: TP=16 PP=1
#
# Agent Optimization:
#   - Prefix caching ENABLED (critical for Claude Code system prompt reuse)
#   - max-num-seqs=4 (parallel tool calls)
#   - max-num-batched-tokens=16384 (prefill throughput)
#
# Usage:
#   bash run_vllm_nomtp.sh
#   TP=16 MAX_MODEL_LEN=202752 bash run_vllm_nomtp.sh
# =============================================================================
set -euo pipefail

# Load Ascend CANN environment
set +u
if [[ -f "/usr/local/Ascend/cann/set_env.sh" ]]; then
    source /usr/local/Ascend/cann/set_env.sh
fi
if [[ -f "/usr/local/Ascend/nnal/atb/set_env.sh" ]]; then
    source /usr/local/Ascend/nnal/atb/set_env.sh
fi
set -u

# Auto-detect model from MODEL_PATH
readonly MODEL_PATH="${MODEL_PATH:-/home/jianzhnie/llmtuner/hfhub/models/Eco-Tech/GLM-5.1-w4a8}"

if [[ "$MODEL_PATH" == *"GLM-5.1"* ]]; then
    readonly DEFAULT_PORT=8002
    readonly SERVED_NAME="glm-5.1"
    readonly MODEL_LABEL="GLM-5.1"
else
    readonly DEFAULT_PORT=8001
    readonly SERVED_NAME="glm-5"
    readonly MODEL_LABEL="GLM-5"
fi

readonly HOST="${HOST:-0.0.0.0}"
readonly PORT="${PORT:-$DEFAULT_PORT}"
readonly TP="${TP:-8}"
readonly PP="${PP:-1}"
readonly MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
readonly MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
readonly GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.94}"

# NPU environment variables
export HCCL_OP_EXPANSION_MODE="${HCCL_OP_EXPANSION_MODE:-AIV}"
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export HCCL_BUFFSIZE=200
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_ASCEND_BALANCE_SCHEDULING=1
export VLLM_ASCEND_ENABLE_FLASHCOMM1=0
export VLLM_ASCEND_ENABLE_MLAPO=1

echo "============================================"
echo "[INFO] ${MODEL_LABEL} W4A8 — Agent-Optimized Deployment (No MTP)"
echo "[INFO] Model: $MODEL_PATH"
echo "[INFO] TP=$TP PP=$PP PORT=$PORT"
echo "[INFO] MAX_MODEL_LEN=$MAX_MODEL_LEN MAX_NUM_SEQS=$MAX_NUM_SEQS"
echo "[INFO] GPU_MEM_UTIL=$GPU_MEM_UTIL"
echo "[INFO] Prefix Caching: ENABLED"
echo "============================================"

vllm serve "$MODEL_PATH" \
    --host "$HOST" \
    --port "$PORT" \
    --served-model-name "$SERVED_NAME" \
    --trust-remote-code \
    --dtype bfloat16 \
    --tensor-parallel-size "$TP" \
    --pipeline-parallel-size "$PP" \
    --distributed-executor-backend ray \
    --enable-expert-parallel \
    --quantization ascend \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --max-num-batched-tokens 16384 \
    --enable-chunked-prefill \
    --enable-prefix-caching \
    --enforce-eager \
    --enable-auto-tool-choice \
    --tool-call-parser glm47 \
    --seed 1024 \
    "$@"
