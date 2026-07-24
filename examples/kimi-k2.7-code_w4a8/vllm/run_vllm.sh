#!/bin/bash
# =============================================================================
# Kimi-K2.7-Code W4A8 — vllm serve deployment (Code-tuned, multimodal)
# =============================================================================
# Architecture: KimiK25ForConditionalGeneration | 384 Experts | MLA | Vision
# Deploy: TP=8 PP=2 (weights ~500G, 2-node A2 minimum)
#
# Hardware:
#   - 2× Atlas 800 A2 (64G × 8):  TP=8 PP=2
#   - 1× Atlas 800 A3 (64G × 16): TP=8 PP=1 or TP=16
#
# Usage:
#   bash run_vllm.sh                                    # TP=8 PP=2 (2-node A2)
#   TP=16 PP=1 bash run_vllm.sh                         # 2-node A3 large TP
#   TP=8 PP=2 MAX_MODEL_LEN=131072 bash run_vllm.sh     # Long context
#
# Reference:
#   https://docs.vllm.ai/projects/ascend/zh-cn/latest/tutorials/models/Kimi-K2.6.html
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
readonly MODEL_PATH="${MODEL_PATH:-$BASE_MODEL_PATH/Kimi-K2.7-Code-w4a8}"
readonly HOST="${HOST:-0.0.0.0}"
readonly PORT="${PORT:-8013}"
readonly TP="${TP:-8}"
readonly PP="${PP:-2}"
readonly DP="${DP:-1}"
readonly MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
readonly MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
readonly MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-16384}"
readonly GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.87}"

# DFlash draft model for speculative decoding
readonly DFLASH_MODEL="${DFLASH_MODEL:-z-lab/Kimi-K2.5-DFlash}"

# NPU environment variables (Kimi-K2.x family)
export HCCL_OP_EXPANSION_MODE=AIV
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export HCCL_BUFFSIZE="${HCCL_BUFFSIZE:-600}"
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TASK_QUEUE_ENABLE=1
export VLLM_USE_MODELSCOPE=False
export VLLM_ASCEND_ENABLE_FLASHCOMM1="${VLLM_ASCEND_ENABLE_FLASHCOMM1:-1}"
export VLLM_ASCEND_ENABLE_MLAPO=1
export VLLM_ASCEND_BALANCE_SCHEDULING="${VLLM_ASCEND_BALANCE_SCHEDULING:-0}"

# Multi-node network (set NIC_NAME + HCCL_IF_IP for cross-node)
readonly NIC_NAME="${NIC_NAME:-}"
readonly HCCL_IF_IP="${HCCL_IF_IP:-}"
if [[ -n "$HCCL_IF_IP" ]]; then
    export HCCL_IF_IP
fi
if [[ -n "$NIC_NAME" ]]; then
    export GLOO_SOCKET_IFNAME="$NIC_NAME"
    export TP_SOCKET_IFNAME="$NIC_NAME"
    export HCCL_SOCKET_IFNAME="$NIC_NAME"
fi

# Compilation config
readonly COMPILATION_CONFIG='{"cudagraph_mode": "FULL_DECODE_ONLY"}'
readonly ADDITIONAL_CONFIG='{"enable_balance_scheduling": true}'

echo "============================================"
echo "[INFO] Kimi-K2.7-Code W4A8 — vLLM-Ascend Deployment"
echo "[INFO] Model:    $MODEL_PATH"
echo "[INFO] TP=$TP  PP=$PP  DP=$DP  PORT=$PORT"
echo "[INFO] MAX_MODEL_LEN=$MAX_MODEL_LEN  MAX_NUM_SEQS=$MAX_NUM_SEQS"
echo "[INFO] MAX_NUM_BATCHED_TOKENS=$MAX_NUM_BATCHED_TOKENS"
echo "[INFO] GPU_MEM_UTIL=$GPU_MEM_UTIL"
echo "[INFO] FLASHCOMM1=$VLLM_ASCEND_ENABLE_FLASHCOMM1  BALANCE_SCHEDULING=$VLLM_ASCEND_BALANCE_SCHEDULING"
echo "[INFO] Speculative: DFlash ($DFLASH_MODEL, 15 tokens)"
echo "[INFO] Prefix Caching: DISABLED (official recommendation)"
echo "[INFO] Parser: kimi_k2 (tool + reasoning)"
echo "============================================"

vllm serve "$MODEL_PATH" \
    --host "$HOST" \
    --port "$PORT" \
    --served-model-name "kimi-k2.7-code" \
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
    --async-scheduling \
    --mm-encoder-tp-mode data \
    --allowed-local-media-path /home/jianzhnie/llmtuner/ \
    --compilation-config "$COMPILATION_CONFIG" \
    --speculative-config "{\"method\":\"dflash\",\"model\":\"$DFLASH_MODEL\",\"num_speculative_tokens\":15}" \
    --additional-config "$ADDITIONAL_CONFIG" \
    --seed 42 \
    "$@"
