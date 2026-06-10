#!/bin/bash
# =============================================================================
# Kimi-K2.6 W4A8 — Agent-Optimized vLLM Deployment with Max Context
# Architecture: KimiK25ForConditionalGeneration | 384 Experts | MLA | Vision
# Max Position: 262144 | Deploy: 256K context (override with MAX_MODEL_LEN)
#
# Kimi-K2.6 支持 Pipeline Parallelism (PP)
# 默认 TP=8 PP=2 (2节点 × 8 NPU); 单节点: TP=8 PP=1
#
# Agent Optimization:
#   - Prefix caching ENABLED (no MTP, works well with prefix cache)
#   - max-num-seqs=16 (high concurrency, no MTP overhead)
#   - max-num-batched-tokens=16384 (prefill throughput)
#   - Vision: mm-encoder-tp-mode=data (text-only agent use optimized)
# =============================================================================
set -eo pipefail

# Load Ascend CANN environment
set +u
if [[ -f "/usr/local/Ascend/cann/set_env.sh" ]]; then
    source /usr/local/Ascend/cann/set_env.sh
fi
if [[ -f "/usr/local/Ascend/nnal/atb/set_env.sh" ]]; then
    source /usr/local/Ascend/nnal/atb/set_env.sh
fi
set -u

MODEL_PATH="${MODEL_PATH:-/home/jianzhnie/llmtuner/hfhub/models/Eco-Tech/Kimi-K2.6-w4a8}"
SPECULATIVE_MODEL_PATH="${SPECULATIVE_MODEL_PATH:-$MODEL_PATH/speculative-model}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8003}"
TP="${TP:-8}"
PP="${PP:-2}"
DP="${DP:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.92}"

# Optional configurations for compilation and speculative decoding
DEFAULT_COMPILATION_CONFIG='{"cudagraph_capture_sizes":[4,8,16,32,64,128,256], "cudagraph_mode":"FULL_DECODE_ONLY"}'
DEFAULT_SPECULATIVE_CONFIG='{"model":"lightseekorg/kimi-k2.6-eagle3-mla","method":"eagle3","num_speculative_tokens":3}'

COMPILATION_CONFIG="${COMPILATION_CONFIG:-$DEFAULT_COMPILATION_CONFIG}"
SPECULATIVE_CONFIG="${SPECULATIVE_CONFIG:-$DEFAULT_SPECULATIVE_CONFIG}"
LANGUAGE_MODEL_ONLY="${LANGUAGE_MODEL_ONLY:-true}"

# NPU performance optimizations
export HCCL_OP_EXPANSION_MODE="AIV"
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export TASK_QUEUE_ENABLE=1

export HCCL_BUFFSIZE=800
export VLLM_ASCEND_ENABLE_MLAPO=1
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
export VLLM_ASCEND_BALANCE_SCHEDULING=1

echo "============================================"
echo "[INFO] Kimi-K2.6 W4A8 — Agent-Optimized Deployment"
echo "[INFO] TP=$TP PP=$PP DP=$DP PORT=$PORT"
echo "[INFO] MAX_MODEL_LEN=$MAX_MODEL_LEN MAX_NUM_SEQS=$MAX_NUM_SEQS"
echo "[INFO] GPU_MEM_UTIL=$GPU_MEM_UTIL"
echo "[INFO] COMPILATION_CONFIG=$COMPILATION_CONFIG"
echo "[INFO] SPECULATIVE_CONFIG=$SPECULATIVE_CONFIG"
echo "[INFO] LANGUAGE_MODEL_ONLY=$LANGUAGE_MODEL_ONLY"
echo "[INFO] Prefix Caching: ENABLED (agent-optimized)"
echo "============================================"

vllm serve "$MODEL_PATH" \
    --host "$HOST" \
    --port "$PORT" \
    --served-model-name kimi-k2.6 \
    --trust-remote-code \
    --dtype bfloat16 \
    --tensor-parallel-size "$TP" \
    --pipeline-parallel-size "$PP" \
    --data-parallel-size "$DP" \
    --distributed-executor-backend ray \
    --enable-expert-parallel \
    --quantization ascend \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --max-num-batched-tokens 16384 \
    --enable-chunked-prefill \
    --enable-prefix-caching \
    --allowed-local-media-path \
    ${COMPILATION_CONFIG:+--compilation-config "$COMPILATION_CONFIG"} \
    ${SPECULATIVE_CONFIG:+--speculative-config "$SPECULATIVE_CONFIG"} \
    $([[ "$LANGUAGE_MODEL_ONLY" == "true" ]] && echo "--language-model-only") \
    --mm-encoder-tp-mode data \
    --enable-auto-tool-choice \
    --tool-call-parser kimi_k2 \
    --seed 1024 \
    "$@"
