#!/bin/bash
# =============================================================================
# LongCat-Flash-Chat-1024E-512Zero-Topk24-v2 — Direct vllm serve deployment
# =============================================================================
# Architecture: LongcatFlashForCausalLM | 1024 Routed Experts + 512 Zero | MLA
# Default: TP=64 PP=1 (multi-node via Ray, 8 nodes × 8 NPU)
# Note: Massive MoE model (1024 experts, topk=24), requires 64 NPUs minimum.
#       Uses --trust-remote-code for custom modeling code.
#       No quantization (bfloat16 native weights).
#
# Usage:
#   bash run_vllm.sh
#   TP=64 MAX_MODEL_LEN=8192 bash run_vllm.sh
#
# Reference:
#   https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/index.html
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
readonly BASE_MODEL_PATH="/home/jianzhnie/llmtuner/hfhub/models/meituan-longcat/expand"
readonly MODEL_PATH="${MODEL_PATH:-$BASE_MODEL_PATH/LongCat-Flash-Chat-1024E-512Zero-E-Topk24-v2}"
readonly HOST="${HOST:-0.0.0.0}"
readonly PORT="${PORT:-8010}"
readonly TP="${TP:-64}"
readonly PP="${PP:-1}"
readonly DP="${DP:-1}"
readonly MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
readonly MAX_NUM_SEQS="${MAX_NUM_SEQS:-128}"
readonly GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"

# NPU environment variables
export HCCL_OP_EXPANSION_MODE=AIV
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export HCCL_BUFFSIZE=800
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_USE_MODELSCOPE=False

# Disable MC2 MoE dispatch (local_experts=16 < topk=24 at TP=64)
export VLLM_ASCEND_FUSED_MOE_MC2=0
export VLLM_ASCEND_MOE_ALL_TO_ALL_DISABLE_MC2=1

# HCCL multi-node communication
export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-1800}"
export HCCL_EXEC_TIMEOUT="${HCCL_EXEC_TIMEOUT:-1800}"
export HCCL_SOCKET_IFNAME="${HCCL_SOCKET_IFNAME:-enp66s0f1}"

# Scheduling
export VLLM_ASCEND_BALANCE_SCHEDULING=1

echo "============================================"
echo "[INFO] LongCat-Flash-Chat-1024E-512Zero-Topk24-v2 — Deployment"
echo "[INFO] Model: $MODEL_PATH"
echo "[INFO] TP=$TP PP=$PP DP=$DP PORT=$PORT"
echo "[INFO] MAX_MODEL_LEN=$MAX_MODEL_LEN MAX_NUM_SEQS=$MAX_NUM_SEQS"
echo "[INFO] GPU_MEM_UTIL=$GPU_MEM_UTIL"
echo "============================================"

vllm serve "$MODEL_PATH" \
    --host "$HOST" \
    --port "$PORT" \
    --served-model-name "longcat-flash-chat" \
    --trust-remote-code \
    --dtype bfloat16 \
    --tensor-parallel-size "$TP" \
    --pipeline-parallel-size "$PP" \
    --data-parallel-size "$DP" \
    --distributed-executor-backend ray \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --max-num-batched-tokens 8192 \
    --enable-chunked-prefill \
    --no-enable-prefix-caching \
    --enforce-eager \
    --enable-expert-parallel \
    --seed 1024 \
    "$@"
