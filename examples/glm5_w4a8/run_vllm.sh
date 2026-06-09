#!/bin/bash
# =============================================================================
# GLM-5 W4A8 — Agent-Optimized vLLM Deployment with Max Context
# Architecture: GlmMoeDsaForCausalLM | 256 Experts | MLA | MTP=1
# Max Position: 202752 | Deploy: 128K context (override with MAX_MODEL_LEN)
#
# GLM-5 不支持 Pipeline Parallelism (PP)，使用大 TP 跨节点部署
# 默认 TP=16 PP=1 (2节点 × 8 NPU); 单节点: TP=8 PP=1
#
# Agent Optimization:
#   - Prefix caching ENABLED (critical for Claude Code system prompt reuse)
#   - max-num-seqs=8 (parallel tool calls)
#   - max-num-batched-tokens=16384 (prefill throughput)
#   - Chunked prefill for better scheduling
#   - MLA optimizations: VLLM_ASCEND_ENABLE_MLAPO=1
# =============================================================================
set -eo pipefail

# Load Ascend CANN environment (required for libascend_hal.so)
# CANN scripts reference unset vars; disable nounset during source
set +u
if [[ -f "/usr/local/Ascend/cann/set_env.sh" ]]; then
    source /usr/local/Ascend/cann/set_env.sh
fi
if [[ -f "/usr/local/Ascend/nnal/atb/set_env.sh" ]]; then
    source /usr/local/Ascend/nnal/atb/set_env.sh
fi
set -u

MODEL_PATH="${MODEL_PATH:-/home/jianzhnie/llmtuner/hfhub/models/Eco-Tech/GLM-5-w4a8}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8001}"
TP="${TP:-16}"
PP="${PP:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.94}"

# NPU performance optimizations
export HCCL_OP_EXPANSION_MODE="${HCCL_OP_EXPANSION_MODE:-AIV}"
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export HCCL_BUFFSIZE=200
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_ASCEND_BALANCE_SCHEDULING=1
export VLLM_ASCEND_ENABLE_FLASHCOMM1=0
export VLLM_ASCEND_ENABLE_MLAPO=1

echo "============================================"
echo "[INFO] GLM-5 W4A8 — Agent-Optimized Deployment"
echo "[INFO] TP=$TP PP=$PP PORT=$PORT"
echo "[INFO] MAX_MODEL_LEN=$MAX_MODEL_LEN MAX_NUM_SEQS=$MAX_NUM_SEQS"
echo "[INFO] GPU_MEM_UTIL=$GPU_MEM_UTIL"
echo "[INFO] Prefix Caching: ENABLED (agent-optimized)"
echo "============================================"

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
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --max-num-batched-tokens 16384 \
    --enable-chunked-prefill \
    --enable-prefix-caching \
    --enforce-eager \
    --speculative-config "{\"num_speculative_tokens\": 3, \"method\": \"deepseek_mtp\"}" \
    --enable-auto-tool-choice \
    --tool-call-parser glm47 \
    --seed 1024 \
    "$@"
