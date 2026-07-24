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
# Verified config (vllm-ascend v0.23.0rc1-a3, 2x Ascend 910C):
#   EP=1 TP=2 EXECUTOR=mp HCCL_BUFFSIZE=2048 GPU_MEM_UTIL=0.75 \
#       bash run_vllm.sh
#   Notes:
#   - EP is required: a single NPU cannot hold 512 experts.
#   - MLA attention kernel only supports block size 128; baked in
#     (override with BLOCK_SIZE=<n>).
#   - Chunked prefill is disabled by default (CHUNKED_PREFILL=1 to enable).
#   - MC2 MoE comm is incompatible with zero-expert weight zeroing
#     (MoeDistributeCombineV2 shape check fails -> collective hang).
#     The EasyInfer plugin overrides the comm method to ALLGATHER via
#     EASYINFER_MOE_COMM=allgather (set automatically when EP=1).
#
# Prerequisites:
#   pip install -e /home/jianzhnie/llmtuner/llm/EasyInfer  # done by the script
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
readonly BASE_MODEL_PATH="${BASE_MODEL_PATH:-/home/jianzhnie/llmtuner/hfhub/models/meituan-longcat}"
readonly MODEL_PATH="${MODEL_PATH:-$BASE_MODEL_PATH/expand/LongCat-Flash-Chat-2layer}"
# readonly MODEL_PATH="${MODEL_PATH:-$BASE_MODEL_PATH/expand/LongCat-Flash-Thinking-2601-2layer}"
readonly HOST="${HOST:-0.0.0.0}"
readonly PORT="${PORT:-8300}"
readonly TP="${TP:-8}"
readonly PP="${PP:-1}"
readonly DP="${DP:-1}"
readonly ENABLE_EP="${EP:-0}"
readonly EXECUTOR="${EXECUTOR:-mp}"
readonly MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
readonly MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
readonly GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
readonly MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
readonly SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-longcat-flash-2layer}"
readonly DTYPE="${DTYPE:-bfloat16}"
# MLA attention kernel only supports block size 128 on this image.
readonly BLOCK_SIZE="${BLOCK_SIZE:-128}"
# Chunked prefill conflicts with EP token dispatch; disable by default.
readonly CHUNKED_PREFILL="${CHUNKED_PREFILL:-0}"

# ------------------------------------------------------------------------------
# Ensure EasyInfer plugins are registered (required for the EP fixes)
# ------------------------------------------------------------------------------
pip install --no-build-isolation --no-deps -e /home/jianzhnie/llmtuner/llm/EasyInfer --quiet 2>/dev/null || true

# ------------------------------------------------------------------------------
# Log file (default: <repo>/logs/vllm_longcat_<timestamp>.log, override with LOG_FILE)
# ------------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs}"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/vllm_longcat_$(date +%Y%m%d_%H%M%S).log}"
readonly LOG_FILE
echo "[INFO] Log file: $LOG_FILE"
exec > >(tee -a "$LOG_FILE") 2>&1

# ------------------------------------------------------------------------------
# NPU environment variables
# ------------------------------------------------------------------------------
# Auto-detect network interface
if [[ -z "${HCCL_SOCKET_IFNAME:-}" ]]; then
    HCCL_SOCKET_IFNAME="$(ip -o -4 route show default | awk '{print $5}' | head -1)"
    HCCL_SOCKET_IFNAME="${HCCL_SOCKET_IFNAME:-enp66s0f5}"
fi
if [[ -z "${GLOO_SOCKET_IFNAME:-}" ]]; then
    GLOO_SOCKET_IFNAME="$HCCL_SOCKET_IFNAME"
fi

export HCCL_OP_EXPANSION_MODE=AIV
export HCCL_SOCKET_IFNAME
export GLOO_SOCKET_IFNAME
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export HCCL_BUFFSIZE="${HCCL_BUFFSIZE:-2048}"
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_USE_MODELSCOPE=False

# HCCL multi-node communication
export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-1800}"
export HCCL_EXEC_TIMEOUT="${HCCL_EXEC_TIMEOUT:-1800}"

# Scheduling
export VLLM_ASCEND_BALANCE_SCHEDULING="${VLLM_ASCEND_BALANCE_SCHEDULING:-1}"
export VLLM_ASCEND_ENABLE_FLASHCOMM1="${VLLM_ASCEND_ENABLE_FLASHCOMM1:-1}"
export VLLM_ASCEND_ENABLE_MLAPO="${VLLM_ASCEND_ENABLE_MLAPO:-1}"

# ------------------------------------------------------------------------------
# Expert Parallel (optional)
# ------------------------------------------------------------------------------
if [[ "$ENABLE_EP" == "1" ]]; then
    export ENABLE_EXPERT_PARALLEL=1
    # MC2 MoE comm breaks with zero-expert weight zeroing; the EasyInfer
    # plugin overrides the comm method to ALLGATHER (see fix_ep_zero_expert.py).
    export EASYINFER_MOE_COMM="${EASYINFER_MOE_COMM:-allgather}"
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
echo "[INFO] Model: $MODEL_PATH"
echo "[INFO] TP=$TP PP=$PP DP=$DP EP=$ENABLE_EP Backend=$EXECUTOR"
echo "[INFO] Host: ${HOST}:${PORT}"
echo "[INFO] MAX_MODEL_LEN=$MAX_MODEL_LEN MAX_NUM_SEQS=$MAX_NUM_SEQS"
echo "[INFO] GPU_MEM_UTIL=$GPU_MEM_UTIL"
echo "============================================"

EP_FLAGS=()
if [[ "$ENABLE_EP" == "1" ]]; then
    EP_FLAGS=(--enable-expert-parallel)
fi

PREFILL_FLAGS=(--enable-chunked-prefill)
if [[ "$CHUNKED_PREFILL" == "0" ]]; then
    PREFILL_FLAGS=(--no-enable-chunked-prefill)
fi

vllm serve "$MODEL_PATH" \
    --host "$HOST" \
    --port "$PORT" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --trust-remote-code \
    --dtype "$DTYPE" \
    --tensor-parallel-size "$TP" \
    --pipeline-parallel-size "$PP" \
    --data-parallel-size "$DP" \
    "${EP_FLAGS[@]}" \
    "${PREFILL_FLAGS[@]}" \
    --block-size "$BLOCK_SIZE" \
    --distributed-executor-backend "$EXECUTOR" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
    --no-enable-prefix-caching \
    --enforce-eager \
    --seed 1024 \
    "$@"
