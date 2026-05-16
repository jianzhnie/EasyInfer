#!/usr/bin/env bash
#
# vLLM 单节点简单测试脚本
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/../set_env.sh"

HFHUB="/llm_workspace_1P/robin/hfhub/models"
MODEL_PATH="${MODEL_PATH:-${HFHUB}/Qwen/Qwen3-32B}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-32B}"

NUM_GPUS="${NUM_GPUS:-8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
PORT="${PORT:-8000}"

vllm serve "$MODEL_PATH" \
    --trust-remote-code \
    --served-model-name "$MODEL_NAME" \
    --tensor-parallel-size "$NUM_GPUS" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs 256 \
    --enable-prefix-caching \
    --enforce-eager \
    --port "$PORT"
