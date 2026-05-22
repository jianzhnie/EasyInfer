#!/bin/bash
#
# 容器内快速启动 Qwen3-32B 示例
# 用法: bash run.sh
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Qwen3-32B 模型配置
export MODEL_PATH="${MODEL_PATH:-/home/jianzhnie/llmtuner/hfhub/models/Qwen/Qwen3-32B}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-32b}"
export TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-8}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"

exec bash "${SCRIPT_DIR}/qwen3_server.sh"
