#!/bin/bash
# =============================================================================
# GLM-5.2 W8A8 — Direct vllm serve deployment (official config)
# =============================================================================
# Architecture: GlmMoeDsaForCausalLM | 256 Experts | MLA | MTP=1
# Max Position: 1048576 | Deploy: 32K context (override with MAX_MODEL_LEN)
# Note: GLM-5.2 does not support Pipeline Parallelism; use large TP across nodes.
#
# Hardware:
#   - Atlas 800 A3 (128G x 16): TP=8 DP=2 (single node)
#   - Atlas 800 A2 (64G x 8):   TP=8 DP=1 (single node, low context)
#                                TP=16 (2 nodes, 32K+ context)
#
# Usage:
#   bash run_vllm.sh                      # TP=8 DP=1 single node (A2)
#   DP=2 bash run_vllm.sh                 # TP=8 DP=2 single node (A3, 16 NPUs)
#   TP=16 MAX_MODEL_LEN=131072 bash run_vllm.sh  # 2 nodes (A2, 32K+ context)
#   TP=32 MAX_MODEL_LEN=202752 bash run_vllm.sh  # 4 nodes
#
# Reference:
#   https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/GLM5.2.html
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

# =============================================================================
# Cache directory layout:
#   /dev/shm/... (tmpfs, noexec) → temp files, runtime logs, home
#   /root/.cache/... (overlay, exec) → triton/torchinductor (need .so loading)
# =============================================================================
readonly CACHE_ROOT="${CACHE_ROOT:-/dev/shm/glm52-cache}"
readonly EXEC_CACHE_ROOT="${EXEC_CACHE_ROOT:-/root/.cache/glm52-cache}"

export PYTHONDONTWRITEBYTECODE=1
# Non-executable caches on /dev/shm (tmpfs — avoids overlay writes)
export XDG_CACHE_HOME="${CACHE_ROOT}/xdg"
export VLLM_CACHE_ROOT="${CACHE_ROOT}/vllm"
export TMPDIR="${CACHE_ROOT}/tmp"
export TEMP="${TMPDIR}"
export TMP="${TMPDIR}"
export HOME="${CACHE_ROOT}/home"
export ASCEND_PROCESS_LOG_PATH="${CACHE_ROOT}/ascend-log"
export ASCEND_GLOBAL_LOG_PATH="${ASCEND_PROCESS_LOG_PATH}"
# Executable caches on overlay (triton .so files need +x mount)
export TRITON_CACHE_DIR="${EXEC_CACHE_ROOT}/triton"
export TORCHINDUCTOR_CACHE_DIR="${EXEC_CACHE_ROOT}/torchinductor"

# Restrict cache directories to owner only (shared environment)
umask 0077
mkdir -p "${XDG_CACHE_HOME}" "${VLLM_CACHE_ROOT}" \
    "${TMPDIR}" "${HOME}" "${ASCEND_PROCESS_LOG_PATH}" \
    "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}"

# Remove stale compilation caches to avoid .so mapping errors
find "${TRITON_CACHE_DIR}" -mindepth 1 -maxdepth 1 -type d -mtime +0 -exec rm -rf {} + 2>/dev/null || true
find "${TORCHINDUCTOR_CACHE_DIR}" -mindepth 1 -maxdepth 1 -type d -mtime +0 -exec rm -rf {} + 2>/dev/null || true

# Base configuration
readonly BASE_MODEL_PATH="/home/jianzhnie/llmtuner/hfhub/models/ZhipuAI"
readonly MODEL_PATH="${MODEL_PATH:-$BASE_MODEL_PATH/GLM-5.2-w8a8}"
readonly HOST="${HOST:-0.0.0.0}"
readonly PORT="${PORT:-8007}"
readonly TP="${TP:-8}"
readonly PP="${PP:-1}"
readonly DP="${DP:-1}"
readonly MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
readonly MAX_NUM_SEQS="${MAX_NUM_SEQS:-48}"
readonly MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"
readonly GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.95}"

# NPU environment variables (official docs)
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

# Runtime / debug
export ASCEND_LAUNCH_BLOCKING=0
export VLLM_ENGINE_READY_TIMEOUT_S=1800

# =============================================================================
# Multi-node network interface binding (optional)
# Required for multi-node TP>8 deployments. Set NIC_NAME to your high-speed
# interface (e.g., enp66s0f1). Leave empty for single-node auto-detect.
# =============================================================================
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

# Compilation config (official docs)
readonly COMPILATION_CONFIG='{"cudagraph_mode": "FULL_DECODE_ONLY"}'
readonly ADDITIONAL_CONFIG='{"multistream_overlap_shared_expert": true}'
readonly SPECULATIVE_CONFIG='{"num_speculative_tokens": 3, "method": "deepseek_mtp", "enforce_eager": true}'

echo "============================================"
echo "[INFO] GLM-5.2 W8A8 — vLLM-Ascend Deployment (Official Config)"
echo "[INFO] Model:    $MODEL_PATH"
echo "[INFO] TP=$TP  PP=$PP  DP=$DP  PORT=$PORT"
echo "[INFO] MAX_MODEL_LEN=$MAX_MODEL_LEN  MAX_NUM_SEQS=$MAX_NUM_SEQS"
echo "[INFO] MAX_NUM_BATCHED_TOKENS=$MAX_NUM_BATCHED_TOKENS"
echo "[INFO] GPU_MEM_UTIL=$GPU_MEM_UTIL"
echo "[INFO] MTP: 3 tokens, method=deepseek_mtp"
echo "[INFO] Tool Calling: glm47 parser + glm45 reasoning"
echo "[INFO] Features: chunked-prefill, prefix-caching, eager MTP"
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
    --enable-auto-tool-choice \
    --tool-call-parser glm47 \
    --reasoning-parser glm45 \
    --async-scheduling \
    --speculative-config "$SPECULATIVE_CONFIG" \
    --additional-config "$ADDITIONAL_CONFIG" \
    --compilation-config "$COMPILATION_CONFIG" \
    --seed 1024 \
    "$@"
