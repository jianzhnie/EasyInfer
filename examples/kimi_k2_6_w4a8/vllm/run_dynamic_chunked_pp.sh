#!/bin/bash
# Kimi-K2.6 W4A8 — 动态分块流水线并行 (Dynamic Chunked Pipeline Parallel)
# 功能: 基于 profiling 的动态分块策略优化 PP 场景下的 prefill 性能
# 架构: KimiK25ForConditionalGeneration | 支持 PP > 1
# 参考: https://docs.vllm.ai/projects/ascend/zh-cn/releases-v0.20.2rc/tutorials/features/dynamic_chunked_pipeline_parallel.html
#
# 要求:
#   - pipeline_parallel_size > 1 (PP ≥ 2)
#   - --enable-chunked-prefill 必须启用
#   - 与 enable_balance_scheduling 不兼容
#
# 用法:
#   # 单节点 A3 (TP=8 PP=2): 16 卡，PP 拆分 prefill/decode
#   TP=8 PP=2 MAX_MODEL_LEN=131072 bash run_dynamic_chunked_pp.sh
#
#   # 多节点 (TP=8 PP=4)
#   TP=8 PP=4 MAX_MODEL_LEN=131072 bash run_dynamic_chunked_pp.sh
#
# 性能参考 (DeepSeek-V3.1 128K input):
#   - CPP (dynamic chunk): TTFT 22.5s
#   - PP (static chunk):   TTFT 27.0s
#   - 提升: ~17% TTFT 改善
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
MODEL_PATH="${MODEL_PATH:-$BASE_MODEL_PATH/Kimi-K2.6-w4a8}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8003}"
TP="${TP:-8}"
PP="${PP:-2}"
DP="${DP:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-32768}"

# Dynamic Chunked PP 配置
# smooth_factor: 平滑因子 (0 < x ≤ 1.0), 越大越信任动态预测
# min_chunk: 最小 chunk 大小
# need_timing: 启用在线校准
PROFILING_CHUNK_CONFIG="${PROFILING_CHUNK_CONFIG:-{\"enabled\": true, \"smooth_factor\": 1.0, \"min_chunk\": 4096, \"need_timing\": true}}"

# 注意: Dynamic Chunked PP 与 balance_scheduling 不兼容
export VLLM_ASCEND_BALANCE_SCHEDULING=0

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
echo "[INFO] MAX_MODEL_LEN=$MAX_MODEL_LEN"
echo "[INFO] Profiling Config: $PROFILING_CHUNK_CONFIG"
echo "[INFO] Feature: Dynamic Chunked PP (CPP) — prefill TTFT 优化"
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
