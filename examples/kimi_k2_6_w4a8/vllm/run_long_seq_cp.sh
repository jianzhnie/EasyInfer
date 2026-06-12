#!/bin/bash
# =============================================================================
# Kimi-K2.6 W4A8 — Long Sequence Context Parallel
# =============================================================================
# Purpose: Break the single-card sequence length limit via Context Parallelism.
# Architecture: KimiK25ForConditionalGeneration | MLA | supports CP on A3
#
# Constraints:
#   - tp_size must be divisible by dcp_size
#   - dcp_size ≤ max_dcp_size = tp_size // num_kv_heads
#   - Currently only Atlas A3 devices are supported
#   - Kimi-K2.6: MLA kv_lora_rank=512, q_lora_rank=1536, head_dim=128
#
# Usage:
#   TP=16 DCP=2 MAX_MODEL_LEN=131072 bash run_long_seq_cp.sh
#   TP=8 PP=2 DCP=2 MAX_MODEL_LEN=131072 bash run_long_seq_cp.sh
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
readonly MODEL_PATH="${MODEL_PATH:-$BASE_MODEL_PATH/Kimi-K2.6-w4a8}"
readonly HOST="${HOST:-0.0.0.0}"
readonly PORT="${PORT:-8003}"
readonly TP="${TP:-16}"
readonly PP="${PP:-1}"
readonly DP="${DP:-1}"
readonly PCP_SIZE="${PCP_SIZE:-2}"
readonly DCP_SIZE="${DCP_SIZE:-2}"
readonly MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
readonly MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
readonly MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-131072}"

# Long-sequence specific environment variables
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export HCCL_BUFFSIZE=800
export VLLM_ASCEND_BALANCE_SCHEDULING=0
export HCCL_OP_EXPANSION_MODE=AIV
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_ASCEND_ENABLE_MLAPO=1
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
export TASK_QUEUE_ENABLE=1
export VLLM_USE_MODELSCOPE=False

echo "============================================"
echo "[INFO] Kimi-K2.6 W4A8 — Long Sequence Context Parallel"
echo "[INFO] Model: $MODEL_PATH"
echo "[INFO] TP=$TP PP=$PP DP=$DP PCP=$PCP_SIZE DCP=$DCP_SIZE"
echo "[INFO] MAX_MODEL_LEN=$MAX_MODEL_LEN MAX_NUM_SEQS=$MAX_NUM_SEQS"
echo "[WARN] Requires Atlas A3 devices (A2 does not support CP)"
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
    --tool-call-parser kimi_k2 \
    --language-model-only \
    --mm-encoder-tp-mode data \
    --allowed-local-media-path /home/jianzhnie/llmtuner/ \
    --no-enable-prefix-caching \
    --seed 1024 \
    "$@"
