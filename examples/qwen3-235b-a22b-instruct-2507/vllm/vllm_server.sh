#!/bin/bash
# =============================================================================
# Qwen3-235B-A22B — Traditional wrapper deployment
# =============================================================================
# Calls scripts/vllm/vllm_model_server.sh to deploy Qwen3-235B-A22B.
# Architecture: Qwen3MoeForCausalLM, 128-expert MoE, BF16.
# Massive MoE model: 235B total, 94 layers, 128 experts (8 activated/token).
#
# Hardware:
#   - Single-node A2 (64G x 8):  Requires W4A8 quantization
#   - Multi-node (2-4 nodes):    BF16 full precision
#
# Usage:
#   ./vllm_server.sh
#   TENSOR_PARALLEL_SIZE=16 MAX_MODEL_LEN=65536 ./vllm_server.sh
#   QUANTIZATION=none TENSOR_PARALLEL_SIZE=16 ./vllm_server.sh
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
export MODEL_PATH="${MODEL_PATH:-/home/jianzhnie/llmtuner/hfhub/models/Qwen/Qwen3-235B-A22B-Instruct-2507}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-235b-a22b}"
export HOST="${HOST:-0.0.0.0}"
export PORT="${PORT:-8006}"

# ------------------------------------------------------------------------------
# Huawei NPU environment variables
# ------------------------------------------------------------------------------
export HCCL_OP_EXPANSION_MODE="${HCCL_OP_EXPANSION_MODE:-AIV}"
export OMP_PROC_BIND="${OMP_PROC_BIND:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export HCCL_BUFFSIZE="${HCCL_BUFFSIZE:-800}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"
export VLLM_ASCEND_BALANCE_SCHEDULING="${VLLM_ASCEND_BALANCE_SCHEDULING:-1}"
export VLLM_ASCEND_ENABLE_FLASHCOMM1="${VLLM_ASCEND_ENABLE_FLASHCOMM1:-1}"

# ------------------------------------------------------------------------------
# Parallel configuration (Qwen3-235B MoE, 128 experts)
# ------------------------------------------------------------------------------
export TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-8}"
export PIPELINE_PARALLEL_SIZE="${PIPELINE_PARALLEL_SIZE:-1}"
export ENABLE_EXPERT_PARALLEL="${ENABLE_EXPERT_PARALLEL:-1}"
export DATA_PARALLEL_SIZE="${DATA_PARALLEL_SIZE:-1}"

# ------------------------------------------------------------------------------
# Quantization and memory configuration (BF16 + W4A8 via ascend)
# ------------------------------------------------------------------------------
export DTYPE="${DTYPE:-bfloat16}"
export QUANTIZATION="${QUANTIZATION:-ascend}"
export LOAD_FORMAT="${LOAD_FORMAT:-auto}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
export SWAP_SPACE="${SWAP_SPACE:-32}"

# ------------------------------------------------------------------------------
# Sequence scheduling (BF16/W4A8, 262K native context)
# ------------------------------------------------------------------------------
if [[ -z "${MAX_MODEL_LEN:-}" ]]; then
    if [[ "${TENSOR_PARALLEL_SIZE:-8}" -ge 16 ]]; then
        export MAX_MODEL_LEN=65536
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

# No MTP for Qwen3-235B.

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
export TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-hermes}"

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
echo "[INFO] Starting Qwen3-235B-A22B server"
echo "[INFO] Model:     ${MODEL_PATH}"
echo "[INFO] Hardware:  TP=$TENSOR_PARALLEL_SIZE, PP=$PIPELINE_PARALLEL_SIZE, DP=$DATA_PARALLEL_SIZE"
echo "[INFO] Quant:     ${QUANTIZATION}, dtype=$DTYPE"
echo "[INFO] Memory:    max_len=$MAX_MODEL_LEN, max_seqs=$MAX_NUM_SEQS, gpu_util=$GPU_MEMORY_UTILIZATION"
echo "[INFO] Features:  MoE (128 experts, 8 activated/token)"
echo "[INFO] HCCL:      OP_EXPANSION_MODE=$HCCL_OP_EXPANSION_MODE, BUFFSIZE=${HCCL_BUFFSIZE}MB"

exec bash "$VLLM_SCRIPT" "${EXTRA_ARGS[@]}" "$@"
