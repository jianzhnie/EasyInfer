#!/bin/bash
# =============================================================================
# Kimi-K2.5 W4A8 — Direct vllm serve deployment (Multimodal)
# =============================================================================
# Architecture: KimiK25ForConditionalGeneration | 384 Experts | MLA | Vision
# Text Backbone: DeepseekV3ForCausalLM | Max Position: 262144
# Default: TP=8 PP=1 (single-node)
# Note: Multimodal model with Vision Transformer. Use --language-model-only
#       for text-only Agent scenarios to save memory.
#       W4A8 compressed-tensors quantization in text_config.
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
readonly MODEL_PATH="${MODEL_PATH:-$BASE_MODEL_PATH/Kimi-K2.5}"
readonly HOST="${HOST:-0.0.0.0}"
readonly PORT="${PORT:-8005}"
readonly TP="${TP:-8}"
readonly PP="${PP:-1}"
readonly DP="${DP:-1}"
readonly MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
readonly MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
readonly GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.92}"

# NPU environment variables
export HCCL_OP_EXPANSION_MODE=AIV
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export HCCL_BUFFSIZE=800
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TASK_QUEUE_ENABLE=1
export VLLM_USE_MODELSCOPE=False

# Fallback variables for older versions
export VLLM_ASCEND_ENABLE_MLAPO=1
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
export VLLM_ASCEND_BALANCE_SCHEDULING=1

# v0.20.2 additional_config format
readonly ADDITIONAL_CONFIG='{"enable_balance_scheduling": true, "enable_flashcomm1": true, "enable_mlapo": true}'

echo "============================================"
echo "[INFO] Kimi-K2.5 W4A8 — Multimodal Deployment"
echo "[INFO] Model: $MODEL_PATH"
echo "[INFO] TP=$TP PP=$PP DP=$DP PORT=$PORT"
echo "[INFO] MAX_MODEL_LEN=$MAX_MODEL_LEN MAX_NUM_SEQS=$MAX_NUM_SEQS"
echo "[INFO] GPU_MEM_UTIL=$GPU_MEM_UTIL"
echo "[INFO] Prefix Caching: ENABLED"
echo "[INFO] Mode: Multimodal (Vision + Text) — use --language-model-only for text-only"
echo "============================================"

vllm serve "$MODEL_PATH" \
    --host "$HOST" \
    --port "$PORT" \
    --served-model-name "kimi-k2.5" \
    --trust-remote-code \
    --dtype bfloat16 \
    --tensor-parallel-size "$TP" \
    --pipeline-parallel-size "$PP" \
    --data-parallel-size "$DP" \
    --distributed-executor-backend ray \
    --quantization compressed-tensors \
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
    --additional-config "$ADDITIONAL_CONFIG" \
    --seed 1024 \
    "$@"
