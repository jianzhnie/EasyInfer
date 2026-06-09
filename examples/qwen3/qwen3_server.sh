#!/bin/bash
# =============================================================================
# Qwen3-32B 部署示例
# =============================================================================
# 调用 vllm_model_server.sh 部署 Qwen3-32B 模型
# Qwen3-32B 是密集型模型 (非 MoE)，单节点即可部署
#
# 硬件要求:
#   - 4x NPU/GPU (BF16, ~64GB 总显存) 或 8x NPU/GPU (充裕)
#   - 2x NPU/GPU (AWQ/GPTQ 量化版本)
#
# 用法:
#   ./qwen3_server.sh                              # 默认配置启动
#   PORT=9000 ./qwen3_server.sh                    # 覆盖端口
#   MAX_MODEL_LEN=65536 ./qwen3_server.sh          # 扩大上下文窗口
#   QUANTIZATION=awq TENSOR_PARALLEL_SIZE=2 ./qwen3_server.sh  # 量化 + 2卡部署
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLLM_SCRIPT="${SCRIPT_DIR}/../scripts/vllm/vllm_model_server.sh"

# 检查启动脚本是否存在
if [[ ! -f "$VLLM_SCRIPT" ]]; then
    echo "[ERROR] vLLM startup script not found: $VLLM_SCRIPT" >&2
    exit 1
fi

# ------------------------------------------------------------------------------
# Qwen3-32B 模型配置
# ------------------------------------------------------------------------------
export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-32B}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-32b}"
export HOST="${HOST:-0.0.0.0}"
export PORT="${PORT:-8000}"

# ------------------------------------------------------------------------------
# 并行配置 (Qwen3-32B 密集型模型，无需 Expert Parallel)
# ------------------------------------------------------------------------------
# 4 卡部署 (推荐，BF16 ~64GB 显存)
export TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-4}"
# 密集型模型无需流水线并行
export PIPELINE_PARALLEL_SIZE="${PIPELINE_PARALLEL_SIZE:-1}"
# 密集型模型禁用专家并行
export ENABLE_EXPERT_PARALLEL="${ENABLE_EXPERT_PARALLEL:-0}"

# ------------------------------------------------------------------------------
# 内存配置
# ------------------------------------------------------------------------------
export DTYPE="${DTYPE:-bfloat16}"
# Qwen3-32B 原生 BF16，无需额外量化；如需量化可设为 awq 或 gptq
# 设为 "none" 而非空字符串，否则主脚本会用默认值 fp8 覆盖
export QUANTIZATION="${QUANTIZATION:-none}"
export LOAD_FORMAT="${LOAD_FORMAT:-auto}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.92}"
# 密集型模型 swap 需求较小
export SWAP_SPACE="${SWAP_SPACE:-16}"

# ------------------------------------------------------------------------------
# 序列调度
# ------------------------------------------------------------------------------
# Qwen3-32B config.json 中 max_position_embeddings=40960，不能超过此值
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-40960}"
# 限制每个序列的最大 tokens (prefill + decode)，给输入留空间，避免 Claude Code 的 32k max_tokens 导致溢出
export MAX_TOKENS_PER_SEQUENCE="${MAX_TOKENS_PER_SEQUENCE:-40000}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
export ENABLE_CHUNKED_PREFILL="${ENABLE_CHUNKED_PREFILL:-1}"
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"

# ------------------------------------------------------------------------------
# 加速特性
# ------------------------------------------------------------------------------
export PREFIX_CACHING="${PREFIX_CACHING:-1}"
# NPU 环境必须设 1 (禁用 CUDA Graph)；NVIDIA GPU 可设 0 启用
export ENFORCE_EAGER="${ENFORCE_EAGER:-1}"
export NUM_SCHEDULER_STEPS="${NUM_SCHEDULER_STEPS:-4}"

# ------------------------------------------------------------------------------
# 工具调用 (Claude Code 集成)
# ------------------------------------------------------------------------------
export ENABLE_TOOL_CALLING="${ENABLE_TOOL_CALLING:-1}"
export TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-hermes}"

# ------------------------------------------------------------------------------
# 监控与日志
# ------------------------------------------------------------------------------
export ENABLE_METRICS="${ENABLE_METRICS:-1}"
export LOG_LEVEL="${LOG_LEVEL:-info}"
export MAX_RETRIES="${MAX_RETRIES:-3}"
export RETRY_DELAY="${RETRY_DELAY:-10}"

# ------------------------------------------------------------------------------
# 启动
# ------------------------------------------------------------------------------
echo "[INFO] Starting Qwen3-32B server (TP=$TENSOR_PARALLEL_SIZE, dtype=$DTYPE, max_len=$MAX_MODEL_LEN)"
exec bash "$VLLM_SCRIPT" "$@"
