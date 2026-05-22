#!/bin/bash
#
# Kimi-K2 (PCL) 多节点部署示例 — 64 TP, Ray 后端
# 用法: 环境变量覆盖: MODEL_PATH=/path/to/model bash kimi2_pcl.sh
#

MODEL_PATH="${MODEL_PATH:-/llm_workspace_1P/robin/hfhub/pcl-kimi2-stage2/kimi2-mcore2hf_step_550_v1}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8080}"
TP_SIZE="${TP_SIZE:-64}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"

# 前置检查
command -v vllm >/dev/null 2>&1 || { echo "[ERROR] vllm not found" >&2; exit 127; }
[[ -e "$MODEL_PATH" ]] || { echo "[ERROR] MODEL_PATH not found: $MODEL_PATH" >&2; exit 2; }

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
