#!/bin/bash
# =============================================================================
# LongCat-Flash-Chat-depth32 — Direct vllm serve deployment
# =============================================================================
# Architecture: LongcatFlashForCausalLM | 512 Routed Experts + 256 Zero | MLA
# Default: TP=64 PP=1 (multi-node via Ray, 8 nodes × 8 NPU)
# Note: Deep model (32 layers), requires 64 NPUs minimum.
#       Uses --trust-remote-code for custom modeling code.
#       No quantization (bfloat16 native weights).
#
# Usage:
#   bash run_vllm_depth32.sh
#   TP=64 MAX_MODEL_LEN=8192 bash run_vllm_depth32.sh
#
# Reference:
#   https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/index.html
# =============================================================================
set -euo pipefail

# Source Ascend/CANN environment (required inside container)
# Temporarily disable -u (nounset) since CANN set_env.sh references unbound vars
if [[ -f /usr/local/Ascend/cann/set_env.sh ]]; then
    set +u
    # shellcheck disable=SC1091
    source /usr/local/Ascend/cann/set_env.sh
    set -u
fi
if [[ -f /usr/local/Ascend/nnal/atb/set_env.sh ]]; then
    set +u
    # shellcheck disable=SC1091
    source /usr/local/Ascend/nnal/atb/set_env.sh
    set -u
fi

# Base configuration
readonly BASE_MODEL_PATH="/home/jianzhnie/llmtuner/hfhub/models/meituan-longcat/expand"
readonly MODEL_PATH="${MODEL_PATH:-$BASE_MODEL_PATH/LongCat-Flash-Chat-depth32}"
readonly HOST="${HOST:-0.0.0.0}"
readonly PORT="${PORT:-8020}"
readonly TP="${TP:-64}"
readonly PP="${PP:-1}"
readonly DP="${DP:-1}"
readonly MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
readonly MAX_NUM_SEQS="${MAX_NUM_SEQS:-128}"
readonly GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
readonly MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
readonly SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-longcat-flash-depth32}"

# NPU environment variables
export HCCL_OP_EXPANSION_MODE=AIV
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export HCCL_BUFFSIZE=800
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_USE_MODELSCOPE=False

# HCCL multi-node communication
export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-1800}"
export HCCL_EXEC_TIMEOUT="${HCCL_EXEC_TIMEOUT:-1800}"

# Scheduling
export VLLM_ASCEND_BALANCE_SCHEDULING=1

# 前置检查
command -v vllm >/dev/null 2>&1 || { echo "[ERROR] vllm not found" >&2; exit 127; }
[[ -d "$MODEL_PATH" ]] || { echo "[ERROR] MODEL_PATH not found: $MODEL_PATH" >&2; exit 2; }

#=============================================================================
# 启动命令
#=============================================================================

echo "============================================"
echo "[INFO] LongCat-Flash-Chat-depth32 — Deployment"
echo "[INFO] Model: $MODEL_PATH"
echo "[INFO] TP=$TP PP=$PP DP=$DP"
echo "[INFO] Host: ${HOST}:${PORT}"
echo "[INFO] MAX_MODEL_LEN=$MAX_MODEL_LEN MAX_NUM_SEQS=$MAX_NUM_SEQS"
echo "[INFO] GPU_MEM_UTIL=$GPU_MEM_UTIL"
echo "============================================"

vllm serve "$MODEL_PATH" \
    --host "$HOST" \
    --port "$PORT" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --trust-remote-code \
    --dtype bfloat16 \
    --tensor-parallel-size "$TP" \
    --pipeline-parallel-size "$PP" \
    --data-parallel-size "$DP" \
    --distributed-executor-backend ray \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
    --enable-chunked-prefill \
    --no-enable-prefix-caching \
    --enforce-eager \
    --seed 1024 \
    "$@"
