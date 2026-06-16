#!/bin/bash
# =============================================================================
# Kimi-K2-Thinking W4A8 — Traditional wrapper deployment
# =============================================================================
# Calls scripts/vllm/vllm_model_server.sh to deploy Kimi-K2-Thinking W4A8.
# Architecture: DeepseekV3ForCausalLM, 384-expert MoE, Thinking model.
# Based on DeepSeek V3 architecture; supports Pipeline Parallelism.
# Text-only (no vision).
#
# Hardware:
#   - Atlas 800 A2 (64G x 8):  single-node W4A8
#   - Atlas 800 A3 (64G x 16): single-node W4A8
#   - Multi-node:              8x8 NPU PP/DP expansion
#
# Usage:
#   ./vllm_server.sh
#   TENSOR_PARALLEL_SIZE=16 MAX_MODEL_LEN=131072 ./vllm_server.sh
#   PIPELINE_PARALLEL_SIZE=8 DATA_PARALLEL_SIZE=8 ./vllm_server.sh
#
# Reference:
#   https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/index.html
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_DIR
readonly VLLM_SCRIPT="${SCRIPT_DIR}/../../../scripts/vllm/vllm_model_server.sh"

if [[ ! -f "$VLLM_SCRIPT" ]]; then
    echo "[ERROR] vLLM startup script not found: $VLLM_SCRIPT" >&2
    exit 1
fi

# ------------------------------------------------------------------------------
# Model path and base configuration
# ------------------------------------------------------------------------------
export MODEL_PATH="${MODEL_PATH:-/home/jianzhnie/llmtuner/hfhub/models/moonshotai/Kimi-K2-Thinking}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-kimi-k2-thinking}"
export HOST="${HOST:-0.0.0.0}"
export PORT="${PORT:-8003}"

# ------------------------------------------------------------------------------
# Huawei NPU environment variables
# ------------------------------------------------------------------------------
export HCCL_OP_EXPANSION_MODE="${HCCL_OP_EXPANSION_MODE:-AIV}"
export OMP_PROC_BIND="${OMP_PROC_BIND:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export HCCL_BUFFSIZE="${HCCL_BUFFSIZE:-800}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"
export VLLM_ASCEND_BALANCE_SCHEDULING="${VLLM_ASCEND_BALANCE_SCHEDULING:-1}"
export TASK_QUEUE_ENABLE="${TASK_QUEUE_ENABLE:-1}"
export VLLM_ASCEND_ENABLE_FLASHCOMM1="${VLLM_ASCEND_ENABLE_FLASHCOMM1:-1}"
export VLLM_ASCEND_ENABLE_MLAPO="${VLLM_ASCEND_ENABLE_MLAPO:-1}"

# ------------------------------------------------------------------------------
# Parallel configuration (Kimi-K2-Thinking MoE, 384 experts, DeepSeek V3 arch)
# ------------------------------------------------------------------------------
export TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-8}"
export PIPELINE_PARALLEL_SIZE="${PIPELINE_PARALLEL_SIZE:-1}"
export ENABLE_EXPERT_PARALLEL="${ENABLE_EXPERT_PARALLEL:-1}"
export DATA_PARALLEL_SIZE="${DATA_PARALLEL_SIZE:-1}"

# ------------------------------------------------------------------------------
# Quantization and memory configuration (W4A8 compressed-tensors)
# ------------------------------------------------------------------------------
export DTYPE="${DTYPE:-bfloat16}"
export QUANTIZATION="${QUANTIZATION:-compressed-tensors}"
export LOAD_FORMAT="${LOAD_FORMAT:-auto}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.92}"
export SWAP_SPACE="${SWAP_SPACE:-32}"

# ------------------------------------------------------------------------------
# Sequence scheduling (W4A8, 262K native context)
# ------------------------------------------------------------------------------
if [[ -z "${MAX_MODEL_LEN:-}" ]]; then
    if [[ "${TENSOR_PARALLEL_SIZE:-8}" -ge 16 ]]; then
        export MAX_MODEL_LEN=131072
    else
        export MAX_MODEL_LEN=32768
    fi
fi
if [[ -z "${MAX_NUM_SEQS:-}" ]]; then
    export MAX_NUM_SEQS=16
fi
export ENABLE_CHUNKED_PREFILL="${ENABLE_CHUNKED_PREFILL:-1}"
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-16384}"
export MAX_TOKENS_PER_SEQUENCE="${MAX_TOKENS_PER_SEQUENCE:-40000}"
export CHAT_TEMPLATE_CONTENT_FORMAT="${CHAT_TEMPLATE_CONTENT_FORMAT:-string}"

# ------------------------------------------------------------------------------
# Acceleration features
# ------------------------------------------------------------------------------
export PREFIX_CACHING="${PREFIX_CACHING:-1}"
export ENFORCE_EAGER="${ENFORCE_EAGER:-1}"

# Kimi-K2-Thinking has no MTP (same as K2.6).
# export SPECULATIVE_METHOD="deepseek_mtp"
# export SPECULATIVE_NUM_TOKENS=3

# ------------------------------------------------------------------------------
# NPU compilation optimization
# ------------------------------------------------------------------------------
export CUDAGRAPH_MODE="${CUDAGRAPH_MODE:-FULL_DECODE_ONLY}"
export ENABLE_NPUGRAPH_EX="${ENABLE_NPUGRAPH_EX:-true}"
export FUSE_MULS_ADD="${FUSE_MULS_ADD:-true}"
export MULTISTREAM_OVERLAP_SHARED_EXPERT="${MULTISTREAM_OVERLAP_SHARED_EXPERT:-true}"

# ------------------------------------------------------------------------------
# Tool calling
# ------------------------------------------------------------------------------
export ENABLE_TOOL_CALLING="${ENABLE_TOOL_CALLING:-1}"
export TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-kimi_k2}"

# ------------------------------------------------------------------------------
# Monitoring and logging
# ------------------------------------------------------------------------------
export ENABLE_METRICS="${ENABLE_METRICS:-1}"
export LOG_LEVEL="${LOG_LEVEL:-info}"
export MAX_RETRIES="${MAX_RETRIES:-3}"
export RETRY_DELAY="${RETRY_DELAY:-10}"

# ------------------------------------------------------------------------------
# Startup arguments
# ------------------------------------------------------------------------------
EXTRA_ARGS=(
    --seed 1024
    --trust-remote-code
)

if [[ "$QUANTIZATION" == "ascend" ]]; then
    EXTRA_ARGS+=(
        --additional-config "{\"fuse_muls_add\": $FUSE_MULS_ADD, \"multistream_overlap_shared_expert\": $MULTISTREAM_OVERLAP_SHARED_EXPERT, \"ascend_compilation_config\": {\"enable_npugraph_ex\": $ENABLE_NPUGRAPH_EX}}"
        --compilation-config "{\"cudagraph_mode\": \"$CUDAGRAPH_MODE\"}"
    )
fi

# ------------------------------------------------------------------------------
# Startup banner
# ------------------------------------------------------------------------------
echo "[INFO] Starting Kimi-K2-Thinking W4A8 server"
echo "[INFO] Model:     ${MODEL_PATH}"
echo "[INFO] Hardware:  TP=$TENSOR_PARALLEL_SIZE, PP=$PIPELINE_PARALLEL_SIZE, DP=$DATA_PARALLEL_SIZE"
echo "[INFO] Quant:     W4A8 (ascend), dtype=$DTYPE"
echo "[INFO] Memory:    max_len=$MAX_MODEL_LEN, max_seqs=$MAX_NUM_SEQS, gpu_util=$GPU_MEMORY_UTILIZATION"
echo "[INFO] Features:  MoE (384 experts), Thinking/Reasoning (text-only)"
echo "[INFO] HCCL:      OP_EXPANSION_MODE=$HCCL_OP_EXPANSION_MODE, BUFFSIZE=${HCCL_BUFFSIZE}MB, TASK_QUEUE=$TASK_QUEUE_ENABLE"

exec bash "$VLLM_SCRIPT" "${EXTRA_ARGS[@]}" "$@"
