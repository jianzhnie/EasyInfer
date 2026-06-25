#!/bin/bash
#
# Kimi-K2 (PCL) 多节点部署示例 — 64 TP, Ray 后端
# 用法: 环境变量覆盖: MODEL_PATH=/path/to/model bash kimi2_pcl.sh
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../../scripts/common.sh
source "${SCRIPT_DIR}/../../scripts/common.sh"

MODEL_PATH="${MODEL_PATH:-/llm_workspace_1P/robin/hfhub/pcl-kimi2-stage2/kimi2-mcore2hf_step_550_v1}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8080}"
TP_SIZE="${TP_SIZE:-64}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"

# 前置检查
command -v vllm >/dev/null 2>&1 || { log_err "vllm not found"; exit "$E_CMD_NOT_FOUND"; }
[[ -e "$MODEL_PATH" ]] || { log_err "MODEL_PATH not found: $MODEL_PATH"; exit "$E_NOT_FOUND"; }

vllm serve "$MODEL_PATH" \
    --distributed-executor-backend ray \
    --tensor-parallel-size "$TP_SIZE" \
    --enable-expert-parallel \
    --max-model-len "${MAX_MODEL_LEN}" \
    --trust-remote-code \
    --enable-prefix-caching \
    --enforce-eager \
    --host "$HOST" \
    --port "$PORT" \
    --hf-overrides '{"model_type":"kimi_k2_mcore","architectures":["KimiK2MCoreV1ForCausalLM"]}'
