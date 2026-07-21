#!/bin/bash
# =============================================================================
# Kimi-K2.7-Code W4A8 — Agent-optimized vLLM deployment with max context
# =============================================================================
# Architecture: KimiK25ForConditionalGeneration | 384 Experts | MLA | Vision
# Deploy: 2-node TP=8 PP=2 (weights ~500G, single A2 node too tight)
# Note: Kimi-K2.x supports Pipeline Parallelism and multimodal (Vision).
#       Code-tuned variant of Kimi-K2.7; deployment mirrors Kimi-K2.6.
#
# Usage:
#   bash run_vllm.sh                       # TP=8 PP=2 (2 nodes via Ray)
#   TP=16 PP=1 bash run_vllm.sh            # 2-node large TP
#   TP=8 PP=2 MAX_MODEL_LEN=131072 bash run_vllm.sh
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
readonly BASE_MODEL_PATH="/home/jianzhnie/llmtuner/hfhub/models/Eco-Tech"
readonly MODEL_PATH="${MODEL_PATH:-$BASE_MODEL_PATH/Kimi-K2.7-Code-w4a8}"
readonly HOST="${HOST:-0.0.0.0}"
readonly PORT="${PORT:-8013}"
readonly TP="${TP:-8}"
readonly PP="${PP:-2}"
readonly DP="${DP:-1}"
readonly MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
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

# Feature toggles (1=on, 0=off), overridable via environment.
# NOTE: FLASHCOMM1=0 is a workaround for "QuantMatmul not support to process
# empty tensor" (aclnnQuantMatmulWeightNz 161002) in profile_run: the flashcomm1
# custom ops (maybe_all_gather_and_maybe_unpad / maybe_chunk_residual) can
# produce empty tensors on some TP ranks, which npu_quant_matmul rejects.
FLASHCOMM1="${FLASHCOMM1:-0}"
MLAPO="${MLAPO:-1}"
BALANCE_SCHEDULING="${BALANCE_SCHEDULING:-1}"

# Fallback variables for older versions
export VLLM_ASCEND_ENABLE_MLAPO="$MLAPO"
export VLLM_ASCEND_ENABLE_FLASHCOMM1="$FLASHCOMM1"
export VLLM_ASCEND_BALANCE_SCHEDULING="$BALANCE_SCHEDULING"

_to_bool() { [[ "$1" == "1" || "$1" == "true" ]] && echo true || echo false; }

# v0.20.2 additional_config format
BS_BOOL="$(_to_bool "$BALANCE_SCHEDULING")"
FC1_BOOL="$(_to_bool "$FLASHCOMM1")"
MLAPO_BOOL="$(_to_bool "$MLAPO")"
readonly ADDITIONAL_CONFIG="{\"enable_balance_scheduling\": ${BS_BOOL}, \"enable_flashcomm1\": ${FC1_BOOL}, \"enable_mlapo\": ${MLAPO_BOOL}}"

echo "============================================"
echo "[INFO] Kimi-K2.7-Code W4A8 — Agent-Optimized Deployment"
echo "[INFO] Model: $MODEL_PATH"
echo "[INFO] TP=$TP PP=$PP DP=$DP PORT=$PORT"
echo "[INFO] MAX_MODEL_LEN=$MAX_MODEL_LEN MAX_NUM_SEQS=$MAX_NUM_SEQS"
echo "[INFO] GPU_MEM_UTIL=$GPU_MEM_UTIL"
echo "[INFO] FLASHCOMM1=$FLASHCOMM1 MLAPO=$MLAPO BALANCE_SCHEDULING=$BALANCE_SCHEDULING"
echo "[INFO] Prefix Caching: ENABLED"
echo "============================================"

vllm serve "$MODEL_PATH" \
    --host "$HOST" \
    --port "$PORT" \
    --served-model-name "kimi-k2.7-code" \
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
    --enable-expert-parallel \
    --enable-auto-tool-choice \
    --tool-call-parser kimi_k2 \
    --language-model-only \
    --mm-encoder-tp-mode data \
    --allowed-local-media-path /home/jianzhnie/llmtuner/ \
    --cudagraph-capture-sizes 1 2 4 8 16 32 \
    --additional-config "$ADDITIONAL_CONFIG" \
    --seed 1024 \
    "$@"
