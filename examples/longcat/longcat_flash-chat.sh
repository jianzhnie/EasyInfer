#!/bin/bash
#=============================================================================
# LongCat-Flash-Chat 128-card vLLM Deployment Script
#
# 切分方案: TP=4, EP=32, DP=1, PP=1
#   - 总卡数: 4 × 32 × 1 × 1 = 128
#   - 每卡 expert: 768 / 32 = 24
#   - MC2 兼容性: num_local_experts(24) ≤ 典型 decode batch，不触发 fallback
#
# 假设 16 节点，每节点 8 卡 NPU
#=============================================================================
set -euo pipefail

#=============================================================================
# 环境变量
#=============================================================================

# --- 日志 ---
export VLLM_LOGGING_LEVEL=${VLLM_LOGGING_LEVEL:-"INFO"}

# --- 模型路径 ---
# MODEL_PATH="${MODEL_PATH:-/home/jianzhnie/llmtuner/hfhub/models/meituan-longcat/expand/LongCat-Flash-Chat-1024E-512Zero-E-Topk24-v2}"
MODEL_PATH="${MODEL_PATH:-/home/jianzhnie/llmtuner/hfhub/models/meituan-longcat/expand/LongCat-Flash-Chat-combined}"

# --- 并行度 ---
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-64}"         # TP: 单层显存切分

# --- 服务配置 ---
HOST=${HOST:-"0.0.0.0"}
PORT=${PORT:-"8000"}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-"longcat-flash"}

# --- 显存/调度 ---
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-"0.90"}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-"4096"}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-"128"}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-"8192"}


# 前置检查
command -v vllm >/dev/null 2>&1 || { echo "[ERROR] vllm not found" >&2; exit 127; }
[[ -d "$MODEL_PATH" ]] || { echo "[ERROR] MODEL_PATH not found: $MODEL_PATH" >&2; exit 2; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../../scripts/common.sh
source "${SCRIPT_DIR}/../../scripts/common.sh"

#=============================================================================
# 启动命令
#=============================================================================

log_info "============================================"
log_info " LongCat-Flash-Chat vLLM Deployment"
log_info " Model:          ${MODEL_PATH}"
log_info " TP:             ${TENSOR_PARALLEL_SIZE}"
log_info " Host:           ${HOST}:${PORT}"
log_info " Max Model Len:  ${MAX_MODEL_LEN}"
log_info " Max Num Seqs:   ${MAX_NUM_SEQS}"
log_info "============================================"


vllm serve "${MODEL_PATH}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --distributed-executor-backend ray \
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --max-num-seqs "${MAX_NUM_SEQS}" \
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
    --trust-remote-code \
    --no-enable-prefix-caching \
    --enforce-eager
