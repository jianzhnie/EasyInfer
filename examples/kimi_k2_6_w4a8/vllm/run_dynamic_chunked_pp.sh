#!/bin/bash
# =============================================================================
# Kimi-K2.6 W4A8 — Dynamic Chunked Pipeline Parallel
# =============================================================================
# Purpose: Dynamic chunking strategy based on profiling to optimize prefill
#          throughput under Pipeline Parallelism.
# Architecture: KimiK25ForConditionalGeneration | supports PP > 1
#
# Requirements:
#   - pipeline_parallel_size > 1 (PP ≥ 2)
#   - --enable-chunked-prefill must be enabled
#   - Not compatible with enable_balance_scheduling
#
# Usage:
#   TP=8 PP=2 MAX_MODEL_LEN=131072 bash run_dynamic_chunked_pp.sh
#   TP=8 PP=4 MAX_MODEL_LEN=131072 bash run_dynamic_chunked_pp.sh
#
# Reference:
#   https://docs.vllm.ai/projects/ascend/zh-cn/releases-v0.20.2rc/tutorials/features/dynamic_chunked_pipeline_parallel.html
# =============================================================================
set -euo pipefail

# Load Ascend CANN environment
set +u
if [[ -f "/usr/local/Ascend/cann/set_env.sh" ]]; then
    source /usr/local/Ascend/cann/set_env.sh
fi
if [[ -f "/usr/local/Ascend/nnal/atb/set_env.sh" ]]; then
    source /usr/local/Ascend/nnal/atb/set_env.sh
fi
set -u

# Base configuration
readonly BASE_MODEL_PATH="/home/jianzhnie/llmtuner/hfhub/models/Eco-Tech"
readonly MODEL_PATH="${MODEL_PATH:-$BASE_MODEL_PATH/Kimi-K2.6-w4a8}"
readonly HOST="${HOST:-0.0.0.0}"
readonly PORT="${PORT:-8003}"
readonly TP="${TP:-8}"
readonly PP="${PP:-2}"
readonly DP="${DP:-1}"
readonly MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
readonly MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
readonly MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-32768}"

# Dynamic Chunked PP configuration
# smooth_factor: smoothing factor (0 < x ≤ 1.0); larger means more trust in dynamic prediction
# min_chunk: minimum chunk size
# need_timing: enable online calibration
readonly PROFILING_CHUNK_CONFIG="${PROFILING_CHUNK_CONFIG:-{\"enabled\": true, \"smooth_factor\": 1.0, \"min_chunk\": 4096, \"need_timing\": true}}"

# Incompatible with balance_scheduling
export VLLM_ASCEND_BALANCE_SCHEDULING=0

# NPU environment variables
export HCCL_OP_EXPANSION_MODE=AIV
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export HCCL_BUFFSIZE=800
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TASK_QUEUE_ENABLE=1
export VLLM_ASCEND_ENABLE_MLAPO=1
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
export VLLM_USE_MODELSCOPE=False

echo "============================================"
echo "[INFO] Kimi-K2.6 W4A8 — Dynamic Chunked Pipeline Parallel"
echo "[INFO] Model: $MODEL_PATH"
echo "[INFO] TP=$TP PP=$PP DP=$DP PORT=$PORT"
echo "[INFO] MAX_MODEL_LEN=$MAX_MODEL_LEN MAX_NUM_SEQS=$MAX_NUM_SEQS"
echo "[INFO] Profiling Config: $PROFILING_CHUNK_CONFIG"
echo "============================================"

vllm serve "$MODEL_PATH" \
    --host "$HOST" \
    --port "$PORT" \
    --served-model-name "kimi-k2.6" \
    --trust-remote-code \
    --dtype bfloat16 \
    --tensor-parallel-size "$TP" \
    --pipeline-parallel-size "$PP" \
    --data-parallel-size "$DP" \
    --distributed-executor-backend ray \
    --quantization ascend \
    --gpu-memory-utilization 0.90 \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
    --enable-chunked-prefill \
    --enable-prefix-caching \
    --enable-expert-parallel \
    --enable-auto-tool-choice \
    --tool-call-parser kimi_k2 \
    --language-model-only \
    --mm-encoder-tp-mode data \
    --allowed-local-media-path /home/jianzhnie/llmtuner/ \
    --additional-config "{\"profiling_chunk_config\": $PROFILING_CHUNK_CONFIG}" \
    --seed 1024 \
    "$@"
