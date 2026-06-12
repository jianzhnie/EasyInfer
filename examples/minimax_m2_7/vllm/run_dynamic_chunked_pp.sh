#!/bin/bash
# MiniMax-M2.7 W8A8 QuaRot — 动态分块流水线并行 (Dynamic Chunked Pipeline Parallel)
# 功能: 基于 profiling 的动态分块策略优化 PP 场景下的 prefill 性能
# 架构: MiniMaxM2ForCausalLM | 256 Experts | 支持 PP > 1
# 参考: https://docs.vllm.ai/projects/ascend/zh-cn/releases-v0.20.2rc/tutorials/features/dynamic_chunked_pipeline_parallel.html
#
# 要求:
#   - pipeline_parallel_size > 1
#   - --enable-chunked-prefill 必须启用
#   - 与 enable_balance_scheduling 不兼容
#
# 用法:
#   # A3 单节点 (TP=8 PP=2)
#   TP=8 PP=2 MAX_MODEL_LEN=65536 bash run_dynamic_chunked_pp.sh
#
#   # A2 双节点 (TP=4 PP=2)
#   TP=4 PP=2 MAX_MODEL_LEN=32768 bash run_dynamic_chunked_pp.sh
set -eo pipefail

set +u
if [[ -f "/usr/local/Ascend/cann/set_env.sh" ]]; then
    source /usr/local/Ascend/cann/set_env.sh
fi
if [[ -f "/usr/local/Ascend/nnal/atb/set_env.sh" ]]; then
    source /usr/local/Ascend/nnal/atb/set_env.sh
fi
set -u

BASE_MODEL_PATH="/home/jianzhnie/llmtuner/hfhub/models/Eco-Tech"
MODEL_PATH="${MODEL_PATH:-$BASE_MODEL_PATH/MiniMax-M2.7-w8a8-QuaRot}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8004}"
TP="${TP:-4}"
PP="${PP:-2}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-32768}"

# Dynamic Chunked PP 配置
PROFILING_CHUNK_CONFIG="${PROFILING_CHUNK_CONFIG:-{\"enabled\": true, \"smooth_factor\": 1.0, \"min_chunk\": 4096, \"need_timing\": true}}"

# 与 balance_scheduling 不兼容
export VLLM_ASCEND_BALANCE_SCHEDULING=0

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
echo "[INFO] MAX_MODEL_LEN=$MAX_MODEL_LEN"
echo "[INFO] Profiling Config: $PROFILING_CHUNK_CONFIG"
echo "[INFO] Feature: Dynamic Chunked PP (CPP)"
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
