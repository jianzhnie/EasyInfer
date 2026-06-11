#!/bin/bash
# =============================================================================
# GLM-5.1 W4A8 — Agent-Optimized vLLM Deployment with Max Context
# Architecture: GlmMoeDsaForCausalLM | 256 Experts | MLA | MTP=1
# Max Position: 202752 | Deploy: 202K context (override with MAX_MODEL_LEN)
#
# GLM-5.1 不支持 Pipeline Parallelism (PP)，使用大 TP 跨节点部署
# 默认 TP=16 PP=1 (2节点 × 8 NPU); 单节点: TP=8 PP=1
#
# 与 GLM-5 W4A8 使用完全相同的配置，仅模型路径和端口不同
#
# 用法:
#   # 2 节点全量部署 (202K)
#   MAX_MODEL_LEN=202752 TP=16 bash run_vllm.sh
#
#   # 单节点
#   TP=8 MAX_MODEL_LEN=32768 bash run_vllm.sh
#
# Agent Optimization:
#   - Prefix caching ENABLED (critical for Claude Code system prompt reuse)
#   - max-num-seqs=8 (parallel tool calls)
#   - max-num-batched-tokens=16384 (prefill throughput)
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

# GLM-5.1 W4A8 默认配置
MODEL_PATH="${MODEL_PATH:-/home/jianzhnie/llmtuner/hfhub/models/Eco-Tech/GLM-5.1-w4a8}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8002}"
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
echo "[INFO] GLM-5.1 W4A8 — Agent-Optimized Deployment"
echo "[INFO] Model: $MODEL_PATH"
echo "[INFO] TP=$TP PP=$PP PORT=$PORT"
echo "[INFO] MAX_MODEL_LEN=$MAX_MODEL_LEN MAX_NUM_SEQS=$MAX_NUM_SEQS"
echo "[INFO] GPU_MEM_UTIL=$GPU_MEM_UTIL"
echo "[INFO] Prefix Caching: ENABLED (agent-optimized)"
echo "[INFO] MTP: enabled (3 tokens, method=mtp)"
echo "[INFO] Note: Same config as GLM-5 W4A8"
echo "============================================"

vllm serve "$MODEL_PATH" \
    --host "$HOST" \
    --port "$PORT" \
    --served-model-name glm-5.1 \
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
    --speculative-config "{\"num_speculative_tokens\": 3, \"method\": \"mtp\"}" \
    --enable-auto-tool-choice \
    --tool-call-parser glm47 \
    --seed 1024 \
    "$@"
