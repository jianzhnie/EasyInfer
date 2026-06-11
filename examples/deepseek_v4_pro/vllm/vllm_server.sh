#!/bin/bash
# DeepSeek-V4-Pro W4A8 — vllm_model_server.sh wrapper
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLLM_SCRIPT="${SCRIPT_DIR}/../../scripts/vllm/vllm_model_server.sh"

if [[ ! -f "$VLLM_SCRIPT" ]]; then
    echo "[ERROR] vLLM startup script not found: $VLLM_SCRIPT" >&2
    exit 1
fi

: "${MODEL_PATH:=/home/jianzhnie/llmtuner/hfhub/models/Eco-Tech/DeepSeek-V4-Pro-w4a8-mtp}"
: "${PORT:=8000}"
: "${SERVED_MODEL_NAME:=deepseek-v4-pro}"

export MODEL_PATH SERVED_MODEL_NAME PORT

# NPU env
export HCCL_OP_EXPANSION_MODE="${HCCL_OP_EXPANSION_MODE:-AIV}"
export OMP_PROC_BIND="${OMP_PROC_BIND:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export HCCL_BUFFSIZE="${HCCL_BUFFSIZE:-400}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"
export VLLM_ASCEND_BALANCE_SCHEDULING="${VLLM_ASCEND_BALANCE_SCHEDULING:-1}"
export VLLM_USE_MODELSCOPE="${VLLM_USE_MODELSCOPE:-False}"

# Parallel
export TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-8}"
export PIPELINE_PARALLEL_SIZE="${PIPELINE_PARALLEL_SIZE:-1}"
export ENABLE_EXPERT_PARALLEL="${ENABLE_EXPERT_PARALLEL:-1}"
export DATA_PARALLEL_SIZE="${DATA_PARALLEL_SIZE:-1}"

# Quantization
export DTYPE="${DTYPE:-bfloat16}"
export QUANTIZATION="${QUANTIZATION:-ascend}"
export LOAD_FORMAT="${LOAD_FORMAT:-auto}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.92}"
export SWAP_SPACE="${SWAP_SPACE:-16}"

# Scheduling
if [[ -z "${MAX_MODEL_LEN:-}" ]]; then
    export MAX_MODEL_LEN=32768
fi
if [[ -z "${MAX_NUM_SEQS:-}" ]]; then
    export MAX_NUM_SEQS=64
fi
export ENABLE_CHUNKED_PREFILL="${ENABLE_CHUNKED_PREFILL:-1}"
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
export CHAT_TEMPLATE_CONTENT_FORMAT="${CHAT_TEMPLATE_CONTENT_FORMAT:-string}"

# Acceleration
export PREFIX_CACHING="${PREFIX_CACHING:-1}"
export ENFORCE_EAGER="${ENFORCE_EAGER:-1}"

# MTP speculative
export SPECULATIVE_METHOD="${SPECULATIVE_METHOD:-mtp}"
export SPECULATIVE_NUM_TOKENS="${SPECULATIVE_NUM_TOKENS:-3}"

# Tool calling
export ENABLE_TOOL_CALLING="${ENABLE_TOOL_CALLING:-1}"
export TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-deepseek_v3}"

EXTRA_ARGS=(
    --seed 1024
    --trust-remote-code
)

if [[ "$SPECULATIVE_METHOD" == "mtp" ]]; then
    EXTRA_ARGS+=(
        --speculative-config "{\"num_speculative_tokens\": $SPECULATIVE_NUM_TOKENS, \"method\": \"$SPECULATIVE_METHOD\"}"
    )
fi

echo "[INFO] Starting DeepSeek-V4-Pro W4A8 server"
echo "[INFO] Model: ${MODEL_PATH}"
echo "[INFO] Hardware: TP=$TENSOR_PARALLEL_SIZE PP=$PIPELINE_PARALLEL_SIZE"

exec bash "$VLLM_SCRIPT" "${EXTRA_ARGS[@]}" "$@"
