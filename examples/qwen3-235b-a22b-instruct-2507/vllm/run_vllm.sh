#!/bin/bash
# =============================================================================
# Qwen3-235B-A22B — Direct vllm serve deployment
# =============================================================================
# Architecture: Qwen3MoeForCausalLM | 128 Experts | MoE | BF16
# Max Position: 262144 | Default: TP=16 PP=1 (multi-node full-precision)
# Note: 128-expert MoE model. BF16 full precision (~438G) needs TP>=16.
#       Add --quantization ascend for W4A8 single-node deployment.
#
# Usage:
#   bash run_vllm.sh
#   TP=16 MAX_MODEL_LEN=131072 bash run_vllm.sh
#   QUANTIZATION=none TP=16 bash run_vllm.sh
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
readonly BASE_MODEL_PATH="/home/jianzhnie/llmtuner/hfhub/models/Qwen"
readonly MODEL_PATH="${MODEL_PATH:-$BASE_MODEL_PATH/Qwen3-235B-A22B-Instruct-2507}"
readonly HOST="${HOST:-0.0.0.0}"
readonly PORT="${PORT:-8006}"
readonly TP="${TP:-16}"
readonly PP="${PP:-1}"
readonly MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
readonly MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
readonly GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.95}"

# NPU environment variables
export HCCL_OP_EXPANSION_MODE=AIV
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export HCCL_BUFFSIZE=800
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_USE_MODELSCOPE=False

# Fallback variables for older versions
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
export VLLM_ASCEND_BALANCE_SCHEDULING=1

# v0.20.2 additional_config format
readonly ADDITIONAL_CONFIG='{"enable_balance_scheduling": true, "enable_flashcomm1": true}'

echo "============================================"
echo "[INFO] Qwen3-235B-A22B — Deployment"
echo "[INFO] Model: $MODEL_PATH"
echo "[INFO] TP=$TP PP=$PP PORT=$PORT"
echo "[INFO] MAX_MODEL_LEN=$MAX_MODEL_LEN MAX_NUM_SEQS=$MAX_NUM_SEQS"
echo "[INFO] GPU_MEM_UTIL=$GPU_MEM_UTIL"
echo "[INFO] Prefix Caching: ENABLED"
echo "[INFO] Note: 128-expert MoE, use multi-node for BF16 full precision"
echo "============================================"

vllm serve "$MODEL_PATH" \
    --host "$HOST" \
    --port "$PORT" \
    --served-model-name "qwen3-235b-a22b" \
    --trust-remote-code \
    --dtype bfloat16 \
    --tensor-parallel-size "$TP" \
    --pipeline-parallel-size "$PP" \
    --distributed-executor-backend ray \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --max-num-batched-tokens 16384 \
    --enable-chunked-prefill \
    --enable-prefix-caching \
    --enforce-eager \
    --enable-expert-parallel \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --additional-config "$ADDITIONAL_CONFIG" \
    --seed 1024 \
    "$@"
