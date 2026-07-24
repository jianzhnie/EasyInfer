#!/bin/bash
# =============================================================================
# Kimi-K2.5 W4A8 — vllm serve deployment (Multimodal)
# =============================================================================
# Architecture: KimiK25ForConditionalGeneration | 384 Experts | MLA | Vision
# Max Position: 262144 | Official recommendation: TP=4 DP=4 (A3 single-node)
# Note: Multimodal model with Vision Transformer. Use --language-model-only
#       for text-only Agent scenarios. Eagle3 speculative decoding enabled.
#
# Hardware:
#   - 1× Atlas 800 A3 (64G × 16): TP=4 DP=4 (recommended)
#   - 2× Atlas 800 A2 (64G × 8):  TP=4 DP=4 multi-node DP
#
# Usage:
#   bash run_vllm.sh                             # TP=4 DP=4 single node (A3)
#   DP=2 TP=8 bash run_vllm.sh                   # TP=8 DP=2 alternative
#   TP=16 MAX_MODEL_LEN=131072 bash run_vllm.sh  # Long context (128K)
#
# Reference:
#   https://docs.vllm.ai/projects/ascend/zh-cn/latest/tutorials/models/Kimi-K2.5.html
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
readonly MODEL_PATH="${MODEL_PATH:-$BASE_MODEL_PATH/Kimi-K2.5}"
readonly HOST="${HOST:-0.0.0.0}"
readonly PORT="${PORT:-8017}"
readonly TP="${TP:-4}"
readonly PP="${PP:-1}"
readonly DP="${DP:-4}"
readonly MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
readonly MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
readonly MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-16384}"
readonly GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"

# NPU environment variables (official docs)
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

# Eagle3 draft model for speculative decoding
readonly EAGLE3_MODEL="${EAGLE3_MODEL:-lightseekorg/kimi-k2.5-eagle3}"

# Compilation config (official docs)
readonly COMPILATION_CONFIG='{"cudagraph_mode": "FULL_DECODE_ONLY"}'
readonly ADDITIONAL_CONFIG='{"enable_balance_scheduling": true, "enable_flashcomm1": true, "enable_mlapo": true}'

echo "============================================"
echo "[INFO] Kimi-K2.5 W4A8 — vLLM-Ascend Deployment"
echo "[INFO] Model:    $MODEL_PATH"
echo "[INFO] TP=$TP  PP=$PP  DP=$DP  PORT=$PORT"
echo "[INFO] MAX_MODEL_LEN=$MAX_MODEL_LEN  MAX_NUM_SEQS=$MAX_NUM_SEQS"
echo "[INFO] MAX_NUM_BATCHED_TOKENS=$MAX_NUM_BATCHED_TOKENS"
echo "[INFO] GPU_MEM_UTIL=$GPU_MEM_UTIL"
echo "[INFO] Speculative: Eagle3 ($EAGLE3_MODEL, 3 tokens)"
echo "[INFO] Prefix Caching: DISABLED (official recommendation)"
echo "[INFO] Parser: kimi_k2 (tool + reasoning)"
echo "[INFO] Mode: Multimodal (Vision + Text)"
echo "============================================"

vllm serve "$MODEL_PATH" \
    --host "$HOST" \
    --port "$PORT" \
    --served-model-name "kimi-k2.5" \
    --trust-remote-code \
    --tensor-parallel-size "$TP" \
    --pipeline-parallel-size "$PP" \
    --data-parallel-size "$DP" \
    --quantization ascend \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
    --no-enable-prefix-caching \
    --enable-expert-parallel \
    --enable-auto-tool-choice \
    --tool-call-parser kimi_k2 \
    --reasoning-parser kimi_k2 \
    --mm-encoder-tp-mode data \
    --allowed-local-media-path /home/jianzhnie/llmtuner/ \
    --compilation-config "$COMPILATION_CONFIG" \
    --speculative-config "{\"method\":\"eagle3\",\"model\":\"$EAGLE3_MODEL\",\"num_speculative_tokens\":3}" \
    --additional-config "$ADDITIONAL_CONFIG" \
    --seed 1024 \
    "$@"
