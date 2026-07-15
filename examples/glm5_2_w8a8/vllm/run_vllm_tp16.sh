#!/bin/bash
# =============================================================================
# GLM-5.2 W8A8 — TP=16 2-Node Deployment (A2, 64GB NPU)
# =============================================================================
# Uses 2 nodes × 8 NPU = 16 TP to fit the 256-expert MoE model on 64GB NPUs.
# No MTP (speculative decoding disabled) to save memory.
# =============================================================================
set -euo pipefail

set +u
if [[ -f "/usr/local/Ascend/cann/set_env.sh" ]]; then
    source /usr/local/Ascend/cann/set_env.sh
fi
if [[ -f "/usr/local/Ascend/nnal/atb/set_env.sh" ]]; then
    source /usr/local/Ascend/nnal/atb/set_env.sh
fi
set -u

readonly CACHE_ROOT="${CACHE_ROOT:-/dev/shm/glm52-cache}"
readonly EXEC_CACHE_ROOT="${EXEC_CACHE_ROOT:-/root/.cache/glm52-cache}"

export PYTHONDONTWRITEBYTECODE=1
export XDG_CACHE_HOME="${CACHE_ROOT}/xdg"
export VLLM_CACHE_ROOT="${CACHE_ROOT}/vllm"
export TMPDIR="${CACHE_ROOT}/tmp"
export TEMP="${TMPDIR}"
export TMP="${TMPDIR}"
export HOME="${CACHE_ROOT}/home"
export ASCEND_PROCESS_LOG_PATH="${CACHE_ROOT}/ascend-log"
export ASCEND_GLOBAL_LOG_PATH="${ASCEND_PROCESS_LOG_PATH}"
export TRITON_CACHE_DIR="${EXEC_CACHE_ROOT}/triton"
export TORCHINDUCTOR_CACHE_DIR="${EXEC_CACHE_ROOT}/torchinductor"

umask 0077
mkdir -p "${XDG_CACHE_HOME}" "${VLLM_CACHE_ROOT}" \
    "${TMPDIR}" "${HOME}" "${ASCEND_PROCESS_LOG_PATH}" \
    "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}"

find "${TRITON_CACHE_DIR}" -mindepth 1 -maxdepth 1 -type d -mtime +0 -exec rm -rf {} + 2>/dev/null || true
find "${TORCHINDUCTOR_CACHE_DIR}" -mindepth 1 -maxdepth 1 -type d -mtime +0 -exec rm -rf {} + 2>/dev/null || true

readonly MODEL_PATH="${MODEL_PATH:-/home/jianzhnie/llmtuner/hfhub/models/ZhipuAI/GLM-5.2-w8a8}"
readonly HOST="${HOST:-0.0.0.0}"
readonly PORT="${PORT:-8007}"
readonly TP="${TP:-16}"
readonly PP="${PP:-1}"
readonly DP="${DP:-1}"
readonly MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
readonly MAX_NUM_SEQS="${MAX_NUM_SEQS:-48}"
readonly MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"
readonly GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.95}"

export HCCL_OP_EXPANSION_MODE=AIV
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export HCCL_BUFFSIZE=200
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_USE_MODELSCOPE=False
export VLLM_ASCEND_BALANCE_SCHEDULING=1
export VLLM_ASCEND_ENABLE_FLASHCOMM1=0
export VLLM_ASCEND_ENABLE_MLAPO=1
export VLLM_USE_V1=1
export ASCEND_LAUNCH_BLOCKING=0
export VLLM_ENGINE_READY_TIMEOUT_S=1800

readonly NIC_NAME="${NIC_NAME:-}"
readonly HCCL_IF_IP="${HCCL_IF_IP:-}"
if [[ -n "$HCCL_IF_IP" ]]; then
    export HCCL_IF_IP
fi
if [[ -n "$NIC_NAME" ]]; then
    export GLOO_SOCKET_IFNAME="$NIC_NAME"
    export TP_SOCKET_IFNAME="$NIC_NAME"
    export HCCL_SOCKET_IFNAME="$NIC_NAME"
fi

readonly COMPILATION_CONFIG='{"cudagraph_mode": "FULL_DECODE_ONLY"}'
readonly ADDITIONAL_CONFIG='{"multistream_overlap_shared_expert": true}'

echo "============================================"
echo "[INFO] GLM-5.2 W8A8 — TP=16 2-Node Deployment"
echo "[INFO] Model:    $MODEL_PATH"
echo "[INFO] TP=$TP  PP=$PP  DP=$DP  PORT=$PORT"
echo "[INFO] MAX_MODEL_LEN=$MAX_MODEL_LEN  MAX_NUM_SEQS=$MAX_NUM_SEQS"
echo "[INFO] GPU_MEM_UTIL=$GPU_MEM_UTIL"
echo "[INFO] MTP: DISABLED (memory saving)"
echo "============================================"

vllm serve "$MODEL_PATH" \
    --host "$HOST" \
    --port "$PORT" \
    --served-model-name "glm-5.2" \
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
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
    --chat-template-content-format string \
    --enable-chunked-prefill \
    --enable-prefix-caching \
    --enable-expert-parallel \
    --additional-config "$ADDITIONAL_CONFIG" \
    --compilation-config "$COMPILATION_CONFIG" \
    --seed 1024 \
    "$@"
