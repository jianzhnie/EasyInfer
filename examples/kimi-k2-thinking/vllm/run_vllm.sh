#!/bin/bash
# =============================================================================
# Kimi-K2-Thinking — vllm serve deployment (text-only reasoning model)
# =============================================================================
# Architecture: DeepseekV3ForCausalLM | 384 Experts | MLA | Thinking
# Max Position: 262144 | Default: TP=16 PP=1 (single A3 node)
# Note: TP=16 minimum required. W4A8 compressed-tensors quantization.
#       Prefix caching disabled per official recommendation.
#
# Hardware: 1× Atlas 800 A3 (64G × 16)
#
# Usage:
#   bash run_vllm.sh
#   TP=16 MAX_MODEL_LEN=131072 bash run_vllm.sh
#
# Reference:
#   https://docs.vllm.ai/projects/ascend/zh-cn/latest/tutorials/models/Kimi-K2-Thinking.html
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
readonly BASE_MODEL_PATH="/home/jianzhnie/llmtuner/hfhub/models/moonshotai"
readonly MODEL_PATH="${MODEL_PATH:-$BASE_MODEL_PATH/Kimi-K2-Thinking}"
readonly HOST="${HOST:-0.0.0.0}"
readonly PORT="${PORT:-8016}"
readonly TP="${TP:-16}"
readonly PP="${PP:-1}"
readonly DP="${DP:-1}"
readonly MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
readonly MAX_NUM_SEQS="${MAX_NUM_SEQS:-12}"
readonly MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-16384}"
readonly GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"

# NPU environment variables (official docs: HCCL_BUFFSIZE=1024)
export HCCL_OP_EXPANSION_MODE=AIV
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export HCCL_BUFFSIZE=1024
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TASK_QUEUE_ENABLE=1
export VLLM_USE_MODELSCOPE=False
export VLLM_ASCEND_ENABLE_MLAPO=1
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
export VLLM_ASCEND_BALANCE_SCHEDULING=1

# Compilation config (official docs)
readonly COMPILATION_CONFIG='{"cudagraph_mode": "FULL_DECODE_ONLY"}'
readonly ADDITIONAL_CONFIG='{"enable_balance_scheduling": true, "enable_flashcomm1": true, "enable_mlapo": true}'

echo "============================================"
echo "[INFO] Kimi-K2-Thinking — vLLM-Ascend Deployment"
echo "[INFO] Model: $MODEL_PATH"
echo "[INFO] TP=$TP PP=$PP DP=$DP PORT=$PORT"
echo "[INFO] MAX_MODEL_LEN=$MAX_MODEL_LEN MAX_NUM_SEQS=$MAX_NUM_SEQS"
echo "[INFO] MAX_NUM_BATCHED_TOKENS=$MAX_NUM_BATCHED_TOKENS"
echo "[INFO] GPU_MEM_UTIL=$GPU_MEM_UTIL"
echo "[INFO] Prefix Caching: DISABLED (official recommendation)"
echo "[INFO] Mode: Text-only (thinking/reasoning)"
echo "============================================"

vllm serve "$MODEL_PATH" \
    --host "$HOST" \
    --port "$PORT" \
    --served-model-name "kimi-k2-thinking" \
    --trust-remote-code \
    --dtype bfloat16 \
    --tensor-parallel-size "$TP" \
    --pipeline-parallel-size "$PP" \
    --data-parallel-size "$DP" \
    --distributed-executor-backend ray \
    --quantization compressed-tensors \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
    --enable-chunked-prefill \
    --no-enable-prefix-caching \
    --enable-expert-parallel \
    --enable-auto-tool-choice \
    --tool-call-parser kimi_k2 \
    --additional-config "$ADDITIONAL_CONFIG" \
    --compilation-config "$COMPILATION_CONFIG" \
    --seed 1024 \
    "$@"
