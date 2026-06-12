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

# 基础路径配置
BASE_MODEL_PATH="/home/jianzhnie/llmtuner/hfhub/models/Eco-Tech"
MODEL_PATH="${MODEL_PATH:-$BASE_MODEL_PATH/Kimi-K2.6-w4a8}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8003}"
TP="${TP:-8}"
PP="${PP:-1}"
DP="${DP:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.92}"

# 环境变量优化 (v0.20.2: balance_scheduling/flashcomm1/mlapo 已迁移至 --additional-config)
export HCCL_OP_EXPANSION_MODE=AIV
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export HCCL_BUFFSIZE=800
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TASK_QUEUE_ENABLE=1
export VLLM_USE_MODELSCOPE=False
# 兼容旧版本的回退变量
export VLLM_ASCEND_ENABLE_MLAPO=1
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
export VLLM_ASCEND_BALANCE_SCHEDULING=1

# v0.20.2 新格式 additional_config
ADDITIONAL_CONFIG='{"enable_balance_scheduling": true, "enable_flashcomm1": true, "enable_mlapo": true}'

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
    --served-model-name "kimi-k2.6" \
    --trust-remote-code \
    --dtype bfloat16 \
    --tensor-parallel-size "$TP" \
    --pipeline-parallel-size "$PP" \
    --data-parallel-size "$DP" \
    --distributed-executor-backend ray \
    --quantization ascend \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --max-num-batched-tokens 16384 \
    --enable-chunked-prefill \
    --enable-prefix-caching \
    --enable-expert-parallel \
    --enable-auto-tool-choice \
    --tool-call-parser kimi_k2 \
    --language-model-only \
    --mm-encoder-tp-mode data \
    --allowed-local-media-path /home/jianzhnie/llmtuner/ \
    --additional-config "$ADDITIONAL_CONFIG" \
    --seed 1024 \
    "$@"
