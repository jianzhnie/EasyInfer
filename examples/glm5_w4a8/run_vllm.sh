#!/bin/bash
# =============================================================================
# GLM-5 W4A8 — 直接 vllm serve 部署
# TP=8, PP=2, Ray backend, 2 nodes
# =============================================================================
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/home/jianzhnie/llmtuner/hfhub/models/Eco-Tech/GLM-5-w4a8}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8001}"
TP="${TP:-8}"
PP="${PP:-2}"

export HCCL_OP_EXPANSION_MODE="${HCCL_OP_EXPANSION_MODE:-AIV}"
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export HCCL_BUFFSIZE=200
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_ASCEND_BALANCE_SCHEDULING=1

echo "[INFO] Starting GLM-5 W4A8"
echo "[INFO] TP=$TP PP=$PP PORT=$PORT"

vllm serve "$MODEL_PATH" \
    --host "$HOST" \
    --port "$PORT" \
    --served-model-name glm-5 \
    --trust-remote-code \
    --dtype bfloat16 \
    --tensor-parallel-size "$TP" \
    --pipeline-parallel-size "$PP" \
    --distributed-executor-backend ray \
    --enable-expert-parallel \
    --quantization ascend \
    --gpu-memory-utilization 0.95 \
    --max-model-len 32768 \
    --max-num-seqs 2 \
    --max-num-batched-tokens 4096 \
    --enable-chunked-prefill \
    --enable-prefix-caching \
    --enforce-eager \
    --num-scheduler-steps 4 \
    --speculative-config "{\"num_speculative_tokens\": 3, \"method\": \"deepseek_mtp\"}" \
    --enable-auto-tool-choice \
    --tool-call-parser glm47 \
    --seed 1024 \
    "$@"
