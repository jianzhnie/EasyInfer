#!/bin/bash
# =============================================================================
# GLM-5.2 W8A8 — Long Sequence Context Parallel
# =============================================================================
# Purpose: Break the single-card sequence length limit via Context Parallelism.
# Architecture: GlmMoeDsaForCausalLM | 256 Experts
#
# Constraints:
#   - GLM-5.2 W8A8 DSA CP path is incompatible; requires A3 devices
#   - tp_size must be divisible by dcp_size
#
# Usage:
#   TP=16 DCP=2 MAX_MODEL_LEN=131072 bash run_long_seq_cp.sh
#
# Reference:
#   https://docs.vllm.ai/projects/ascend/zh-cn/releases-v0.20.2rc/tutorials/features/long_sequence_context_parallel_single_node.html
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
readonly MODEL_PATH="${MODEL_PATH:-$BASE_MODEL_PATH/GLM-5.2-w8a8}"
readonly HOST="${HOST:-0.0.0.0}"
readonly PORT="${PORT:-8007}"
readonly TP="${TP:-16}"
readonly PP="${PP:-1}"
readonly PCP_SIZE="${PCP_SIZE:-2}"
readonly DCP_SIZE="${DCP_SIZE:-2}"
readonly MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
readonly MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
readonly MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-131072}"

# Long-sequence specific environment variables
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export HCCL_BUFFSIZE=512
export VLLM_ASCEND_BALANCE_SCHEDULING=0
export HCCL_OP_EXPANSION_MODE=AIV
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_ASCEND_ENABLE_FLASHCOMM1=0
export VLLM_ASCEND_ENABLE_MLAPO=1
export TASK_QUEUE_ENABLE=1
export VLLM_USE_MODELSCOPE=False

echo "============================================"
echo "[INFO] GLM-5.2 W8A8 — Long Sequence Context Parallel"
echo "[INFO] Model: $MODEL_PATH"
echo "[INFO] TP=$TP PP=$PP PCP=$PCP_SIZE DCP=$DCP_SIZE"
echo "[INFO] MAX_MODEL_LEN=$MAX_MODEL_LEN MAX_NUM_SEQS=$MAX_NUM_SEQS"
echo "[WARN] Requires Atlas A3 devices"
echo "[WARN] FLASHCOMM1=0 (DSA CP compatibility)"
echo "============================================"

vllm serve "$MODEL_PATH" \
    --host "$HOST" \
    --port "$PORT" \
    --served-model-name "glm-5.2" \
    --trust-remote-code \
    --dtype bfloat16 \
    --tensor-parallel-size "$TP" \
    --pipeline-parallel-size "$PP" \
    --prefill-context-parallel-size "$PCP_SIZE" \
    --decode-context-parallel-size "$DCP_SIZE" \
    --distributed-executor-backend ray \
    --quantization ascend \
    --gpu-memory-utilization 0.95 \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
    --enable-chunked-prefill \
    --enable-expert-parallel \
    --enable-auto-tool-choice \
    --tool-call-parser glm47 \
    --reasoning-parser glm45 \
    --speculative-config '{"num_speculative_tokens": 3, "method": "mtp"}' \
    --no-enable-prefix-caching \
    --seed 1024 \
    "$@"
