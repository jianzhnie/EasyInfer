#!/bin/bash
# =============================================================================
# MiniMax-M2.7 W8A8 QuaRot — Dynamic Chunked Pipeline Parallel
# =============================================================================
# Purpose: Dynamic chunking strategy based on profiling to optimize prefill
#          throughput under Pipeline Parallelism.
# Architecture: MiniMaxM2ForCausalLM | 256 Experts | supports PP > 1
#
# Requirements:
#   - pipeline_parallel_size > 1
#   - --enable-chunked-prefill must be enabled
#   - Not compatible with enable_balance_scheduling
#
# Usage:
#   TP=8 PP=2 MAX_MODEL_LEN=65536 bash run_dynamic_chunked_pp.sh
#   TP=4 PP=2 MAX_MODEL_LEN=32768 bash run_dynamic_chunked_pp.sh
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
readonly MODEL_PATH="${MODEL_PATH:-$BASE_MODEL_PATH/MiniMax-M2.7-w8a8-QuaRot}"
readonly HOST="${HOST:-0.0.0.0}"
readonly PORT="${PORT:-8004}"
readonly TP="${TP:-4}"
readonly PP="${PP:-2}"
readonly MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
readonly MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
readonly MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-32768}"

# Dynamic Chunked PP configuration
readonly PROFILING_CHUNK_CONFIG="${PROFILING_CHUNK_CONFIG:-{\"enabled\": true, \"smooth_factor\": 1.0, \"min_chunk\": 4096, \"need_timing\": true}}"

# Incompatible with balance_scheduling
export VLLM_ASCEND_BALANCE_SCHEDULING=0

# NPU environment variables
export HCCL_OP_EXPANSION_MODE=AIV
export HCCL_BUFFSIZE=1024
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export TASK_QUEUE_ENABLE=1
export VLLM_ASCEND_ENABLE_FUSED_MC2=1
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
export VLLM_USE_MODELSCOPE=False

echo "============================================"
echo "[INFO] MiniMax-M2.7 W8A8 — Dynamic Chunked Pipeline Parallel"
echo "[INFO] Model: $MODEL_PATH"
echo "[INFO] TP=$TP PP=$PP PORT=$PORT"
echo "[INFO] MAX_MODEL_LEN=$MAX_MODEL_LEN MAX_NUM_SEQS=$MAX_NUM_SEQS"
echo "[INFO] Profiling Config: $PROFILING_CHUNK_CONFIG"
echo "============================================"

vllm serve "$MODEL_PATH" \
    --host "$HOST" \
    --port "$PORT" \
    --served-model-name "minimax-m2.7" \
    --trust-remote-code \
    --dtype bfloat16 \
    --tensor-parallel-size "$TP" \
    --pipeline-parallel-size "$PP" \
    --distributed-executor-backend ray \
    --quantization ascend \
    --gpu-memory-utilization 0.83 \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
    --enable-chunked-prefill \
    --enable-prefix-caching \
    --enforce-eager \
    --enable-expert-parallel \
    --enable-auto-tool-choice \
    --tool-call-parser minimax_m2 \
    --additional-config "{\"profiling_chunk_config\": $PROFILING_CHUNK_CONFIG}" \
    --seed 1024 \
    "$@"
