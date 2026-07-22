#!/bin/bash
# =============================================================================
# Kimi-K2.6 W4A8 — Agent-optimized vLLM deployment with max context
# =============================================================================
# Architecture: KimiK25ForConditionalGeneration | 384 Experts | MLA | Vision
# Max Position: 262144 | Deploy: 256K context (override with MAX_MODEL_LEN)
# Note: Kimi-K2.6 supports Pipeline Parallelism and multimodal (Vision).
#
# Usage:
#   bash run_vllm.sh
#   TP=8 PP=2 MAX_MODEL_LEN=131072 bash run_vllm.sh
#   TP=16 MAX_MODEL_LEN=131072 bash run_vllm.sh
#
# Reference:
#   https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/index.html
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

# Base configuration
readonly BASE_MODEL_PATH="/home/jianzhnie/llmtuner/hfhub/models/moonshotai"
readonly MODEL_PATH="${MODEL_PATH:-$BASE_MODEL_PATH/Kimi-K2.6-w4a8}"
readonly HOST="${HOST:-0.0.0.0}"
readonly PORT="${PORT:-8003}"
readonly TP="${TP:-8}"
readonly PP="${PP:-2}"
readonly DP="${DP:-1}"
readonly MAX_MODEL_LEN="${MAX_MODEL_LEN:-31744}"
readonly MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
readonly GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.95}"

# NPU environment variables
export HCCL_OP_EXPANSION_MODE=AIV
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export HCCL_BUFFSIZE=800
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TASK_QUEUE_ENABLE=1
export VLLM_USE_MODELSCOPE=False

# Fallback variables for older versions (all overridable from environment)
# NOTE: FLASHCOMM1=0 is mandatory for this W4A8 checkpoint — the FLASHCOMM1
# AOT-compiled graph feeds an empty tensor to QuantMatmul (error 161002
# "QuantMatmul not support to process empty tensor") on v0.22.1/v0.23.0.
export VLLM_ASCEND_ENABLE_MLAPO="${VLLM_ASCEND_ENABLE_MLAPO:-1}"
export VLLM_ASCEND_ENABLE_FLASHCOMM1="${VLLM_ASCEND_ENABLE_FLASHCOMM1:-0}"
export VLLM_ASCEND_BALANCE_SCHEDULING="${VLLM_ASCEND_BALANCE_SCHEDULING:-1}"
readonly ENABLE_EP="${ENABLE_EP:-1}"

# v0.20.2 additional_config format (kept in sync with the env vars above,
# since additional_config takes precedence over env vars on Ascend)
_fc1=false; [[ "$VLLM_ASCEND_ENABLE_FLASHCOMM1" == "1" ]] && _fc1=true
_mlapo=false; [[ "$VLLM_ASCEND_ENABLE_MLAPO" == "1" ]] && _mlapo=true
readonly ADDITIONAL_CONFIG="{\"enable_balance_scheduling\": true, \"enable_flashcomm1\": ${_fc1}, \"enable_mlapo\": ${_mlapo}}"

# Expert parallel flag (set ENABLE_EP=0 to disable)
EP_FLAG=()
[[ "$ENABLE_EP" == "1" ]] && EP_FLAG=(--enable-expert-parallel)

echo "============================================"
echo "[INFO] Kimi-K2.6 W4A8 — Agent-Optimized Deployment"
echo "[INFO] Model: $MODEL_PATH"
echo "[INFO] TP=$TP PP=$PP DP=$DP PORT=$PORT"
echo "[INFO] MAX_MODEL_LEN=$MAX_MODEL_LEN MAX_NUM_SEQS=$MAX_NUM_SEQS"
echo "[INFO] GPU_MEM_UTIL=$GPU_MEM_UTIL"
echo "[INFO] Prefix Caching: ENABLED"
echo "============================================"

vllm serve "$MODEL_PATH" \
    --host "$HOST" \
    --port "$PORT" \
    --served-model-name "kimi-k2.6" \
    --trust-remote-code \
    --dtype bfloat16 \
    --tensor-parallel-size "$TP" \
    --pipeline-parallel-size "$PP" \
    --data-parallel-size "$DP" \
    --distributed-executor-backend ray \
    --quantization ascend \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --max-num-batched-tokens 16384 \
    --enable-chunked-prefill \
    --enable-prefix-caching \
    "${EP_FLAG[@]}" \
    --enable-auto-tool-choice \
    --tool-call-parser kimi_k2 \
    --language-model-only \
    --mm-encoder-tp-mode data \
    --allowed-local-media-path /home/jianzhnie/llmtuner/ \
    --cudagraph-capture-sizes 1 2 4 8 16 32 \
    --additional-config "$ADDITIONAL_CONFIG" \
    --seed 1024 \
    "$@"
