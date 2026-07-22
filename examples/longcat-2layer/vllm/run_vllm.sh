#!/bin/bash
# =============================================================================
# LongCat-Flash 2-Layer — vllm serve deployment
# =============================================================================
# Architecture: LongcatFlashForCausalLM | 512 Routed Experts + 256 Zero | MLA
#               2 layers extracted from the original 28-layer model.
# Docker image: quay.io/ascend/vllm-ascend:v0.23.0rc1-a3
#
# Usage:
#   bash run_vllm.sh                                    # stable mode (default)
#   EP=1 TP=2 bash run_vllm.sh                          # EP mode
#   MODEL_PATH=/custom/path TP=8 bash run_vllm.sh       # custom model
#
# Prerequisites:
#   pip install -e /home/jianzhnie/llmtuner/llm/EasyInfer  # once per container
# =============================================================================
set -euo pipefail

# Base configuration
readonly MODEL_PATH="${MODEL_PATH:-/home/jianzhnie/llmtuner/hfhub/models/meituan-longcat/expand/LongCat-Flash-Thinking-2601-2layer}"
readonly HOST="${HOST:-0.0.0.0}"
readonly PORT="${PORT:-8300}"
readonly TP="${TP:-8}"
readonly PP="${PP:-1}"
readonly ENABLE_EP="${EP:-0}"
readonly EXECUTOR="${EXECUTOR:-ray}"
readonly MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
readonly MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
readonly GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
readonly MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
readonly SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-longcat-flash-2layer}"
readonly DTYPE="${DTYPE:-bfloat16}"

# ------------------------------------------------------------------------------
# Ensure EasyInfer plugins are registered
# ------------------------------------------------------------------------------
pip install -e /home/jianzhnie/llmtuner/llm/EasyInfer --quiet 2>/dev/null || true

# ------------------------------------------------------------------------------
# NPU environment variables
# ------------------------------------------------------------------------------
export HCCL_OP_EXPANSION_MODE=AIV
export HCCL_SOCKET_IFNAME="${HCCL_SOCKET_IFNAME:-enp66s0f5}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-enp66s0f5}"
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export HCCL_BUFFSIZE="${HCCL_BUFFSIZE:-2048}"
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_USE_MODELSCOPE=False
export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-1800}"
export HCCL_EXEC_TIMEOUT="${HCCL_EXEC_TIMEOUT:-1800}"

# ------------------------------------------------------------------------------
# Expert Parallel (optional)
# ------------------------------------------------------------------------------
if [[ "$ENABLE_EP" == "1" ]]; then
    export ENABLE_EXPERT_PARALLEL=1
fi

# ------------------------------------------------------------------------------
# Pre-flight checks
# ------------------------------------------------------------------------------
command -v vllm >/dev/null 2>&1 || { echo "[ERROR] vllm not found" >&2; exit 127; }
[[ -d "$MODEL_PATH" ]] || { echo "[ERROR] MODEL_PATH not found: $MODEL_PATH" >&2; exit 2; }

# ------------------------------------------------------------------------------
# Launch vllm serve
# ------------------------------------------------------------------------------

echo "============================================"
echo "[INFO] LongCat-Flash 2-Layer Deployment"
echo "[INFO] Model:   $MODEL_PATH"
echo "[INFO] TP=$TP, PP=$PP, EP=$ENABLE_EP, Backend=$EXECUTOR"
echo "[INFO] Host:    ${HOST}:${PORT}"
echo "[INFO] MaxLen:  $MAX_MODEL_LEN  Seqs: $MAX_NUM_SEQS"
echo "[INFO] MemUtil: $GPU_MEM_UTIL"
echo "============================================"

EP_FLAGS=()
if [[ "$ENABLE_EP" == "1" ]]; then
    EP_FLAGS=(--enable-expert-parallel)
fi

vllm serve "$MODEL_PATH" \
    --host "$HOST" \
    --port "$PORT" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --trust-remote-code \
    --dtype "$DTYPE" \
    --tensor-parallel-size "$TP" \
    --pipeline-parallel-size "$PP" \
    "${EP_FLAGS[@]}" \
    --distributed-executor-backend "$EXECUTOR" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
    --no-enable-prefix-caching \
    --enforce-eager \
    --seed 1024 \
    "$@"
