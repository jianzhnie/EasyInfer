#!/bin/bash
# =============================================================================
# MiniMax-M2.5 W8A8 — Direct vllm serve deployment
# =============================================================================
# Architecture: MiniMaxM2ForCausalLM | 256 Experts | MoE | W8A8
# Default: TP=8 PP=1 (single-node A2)
# Note: MTP is configured in the model (num_mtp_modules=3) but not supported
#       by vLLM-Ascend 0.20.2 for MiniMax architecture.
#
# Usage:
#   bash run_vllm.sh
#   TP=8 MAX_MODEL_LEN=65536 bash run_vllm.sh
#   TP=8 PP=2 bash run_vllm.sh
#
# Reference:
#   https://docs.vllm.ai/projects/ascend/zh-cn/releases-v0.20.2rc/tutorials/models/MiniMax-M2.5.html
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
readonly BASE_MODEL_PATH="/home/jianzhnie/llmtuner/hfhub/models/MiniMax"
readonly MODEL_PATH="${MODEL_PATH:-$BASE_MODEL_PATH/MiniMax-M2.5}"
readonly HOST="${HOST:-0.0.0.0}"
readonly PORT="${PORT:-8006}"
readonly TP="${TP:-8}"
readonly PP="${PP:-1}"
readonly MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
readonly MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
readonly GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"

# NPU environment variables
export HCCL_OP_EXPANSION_MODE=AIV
export HCCL_BUFFSIZE=1024
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export TASK_QUEUE_ENABLE=1
export VLLM_ASCEND_ENABLE_FUSED_MC2=1
export VLLM_USE_MODELSCOPE=False

# Fallback variables for older versions
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
export VLLM_ASCEND_BALANCE_SCHEDULING=1

# v0.20.2 additional_config format
readonly ADDITIONAL_CONFIG='{"enable_balance_scheduling": true, "enable_flashcomm1": true}'

echo "============================================"
echo "[INFO] MiniMax-M2.5 W8A8 Deployment"
echo "[INFO] Model: $MODEL_PATH"
echo "[INFO] TP=$TP PP=$PP PORT=$PORT"
echo "[INFO] MAX_MODEL_LEN=$MAX_MODEL_LEN MAX_NUM_SEQS=$MAX_NUM_SEQS"
echo "[INFO] GPU_MEM_UTIL=$GPU_MEM_UTIL"
echo "[INFO] Note: MTP not supported in vLLM-Ascend 0.20.2 for MiniMax"
echo "============================================"

vllm serve "$MODEL_PATH" \
    --host "$HOST" \
    --port "$PORT" \
    --served-model-name "minimax-m2.5" \
    --trust-remote-code \
    --dtype bfloat16 \
    --tensor-parallel-size "$TP" \
    --pipeline-parallel-size "$PP" \
    --distributed-executor-backend ray \
    --quantization ascend \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --max-num-batched-tokens 8192 \
    --enable-chunked-prefill \
    --enable-prefix-caching \
    --enforce-eager \
    --enable-expert-parallel \
    --enable-auto-tool-choice \
    --tool-call-parser minimax_m2 \
    --additional-config "$ADDITIONAL_CONFIG" \
    --seed 1024 \
    "$@"
