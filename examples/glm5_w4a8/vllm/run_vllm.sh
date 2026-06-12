#!/bin/bash
# =============================================================================
# GLM-5 / GLM-5.1 W4A8 — Agent-Optimized vLLM Deployment with Max Context
# Architecture: GlmMoeDsaForCausalLM | 256 Experts | MLA | MTP=1
# Max Position: 202752 | Deploy: 202K context (override with MAX_MODEL_LEN)
#
# GLM-5/5.1 不支持 Pipeline Parallelism (PP)，使用大 TP 跨节点部署
# 默认 TP=16 PP=1 (2节点 × 8 NPU); 单节点: TP=8 PP=1
#
# 用法:
#   # GLM-5.1 (默认)
#   MAX_MODEL_LEN=202752 bash run_vllm.sh
#
#   # GLM-5
#   MODEL_PATH=/path/to/GLM-5-w4a8 PORT=8001 bash run_vllm.sh
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

# 基础路径配置
BASE_MODEL_PATH="/home/jianzhnie/llmtuner/hfhub/models/Eco-Tech"
MODEL_PATH="${MODEL_PATH:-$BASE_MODEL_PATH/GLM-5-w4a8}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8001}"
TP="${TP:-8}"
PP="${PP:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.94}"

# 环境变量优化 (v0.20.2: balance_scheduling/flashcomm1/mlapo 已迁移至 --additional-config)
export HCCL_OP_EXPANSION_MODE=AIV
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export HCCL_BUFFSIZE=200
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_USE_MODELSCOPE=False
# 兼容旧版本的回退变量
export VLLM_ASCEND_BALANCE_SCHEDULING=1
export VLLM_ASCEND_ENABLE_FLASHCOMM1=0
export VLLM_ASCEND_ENABLE_MLAPO=1

# v0.20.2 新格式 additional_config
ADDITIONAL_CONFIG='{"enable_balance_scheduling": true, "enable_flashcomm1": false, "enable_mlapo": true}'

echo "[INFO] Starting GLM-5 W4A8 at $MODEL_PATH"
echo "[INFO] TP=$TP PP=$PP PORT=$PORT MAX_MODEL_LEN=$MAX_MODEL_LEN MAX_NUM_SEQS=$MAX_NUM_SEQS"

vllm serve "$MODEL_PATH" \
    --host "$HOST" \
    --port "$PORT" \
    --served-model-name "glm-5" \
    --trust-remote-code \
    --dtype bfloat16 \
    --tensor-parallel-size "$TP" \
    --pipeline-parallel-size "$PP" \
    --distributed-executor-backend ray \
    --quantization ascend \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --max-num-batched-tokens 16384 \
    --chat-template-content-format string \
    --enable-chunked-prefill \
    --enable-prefix-caching \
    --enforce-eager \
    --enable-expert-parallel \
    --enable-auto-tool-choice \
    --tool-call-parser glm47 \
    --reasoning-parser glm45 \
    --speculative-config '{"num_speculative_tokens": 3, "method": "mtp"}' \
    --additional-config "$ADDITIONAL_CONFIG" \
    --seed 1024 \
    "$@"
