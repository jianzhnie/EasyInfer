#!/bin/bash
# =============================================================================
# MiniMax-M3 W8A8 — vllm serve deployment
# =============================================================================
# Architecture: MiniMaxM3SparseForConditionalGeneration | MoE | VL
# Default: TP=8 (single-node, weights ~418G → ~52G per NPU)
#
# Hardware:
#   - 1× Atlas 800 A3 (64G × 16): TP=8
#   - 2× Atlas 800 A2 (64G × 8):  TP=16 (2 nodes)
#
# Usage:
#   bash run_vllm.sh                           # TP=8 single-node
#   TP=16 bash run_vllm.sh                     # 2 nodes
#
# Reference:
#   https://docs.vllm.ai/projects/ascend/zh-cn/latest/tutorials/models/MiniMax-M2.html
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
readonly MODEL_PATH="${MODEL_PATH:-$BASE_MODEL_PATH/MiniMax-M3-w8a8}"
readonly HOST="${HOST:-0.0.0.0}"
readonly PORT="${PORT:-8014}"
readonly TP="${TP:-8}"
readonly PP="${PP:-1}"
readonly MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
readonly MAX_NUM_SEQS="${MAX_NUM_SEQS:-48}"
readonly MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-16384}"
readonly GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"

# NPU environment variables (official M2 docs adapted for M3)
export HCCL_OP_EXPANSION_MODE=AIV
export HCCL_BUFFSIZE="${HCCL_BUFFSIZE:-1024}"
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export TASK_QUEUE_ENABLE=1
export VLLM_ASCEND_ENABLE_FUSED_MC2=1
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
export VLLM_ASCEND_BALANCE_SCHEDULING="${VLLM_ASCEND_BALANCE_SCHEDULING:-0}"
export VLLM_USE_MODELSCOPE=False

# Compilation config (official docs)
readonly COMPILATION_CONFIG='{"cudagraph_mode": "FULL_DECODE_ONLY"}'
readonly ADDITIONAL_CONFIG='{"enable_cpu_binding":true,"enable_fused_mc2":true,"enable_flashcomm1":true,"weight_nz_mode":true}'

echo "============================================"
echo "[INFO] MiniMax-M3 W8A8 — vLLM-Ascend Deployment"
echo "[INFO] Model:    $MODEL_PATH"
echo "[INFO] TP=$TP  PP=$PP  DP=$DP  PORT=$PORT"
echo "[INFO] MAX_MODEL_LEN=$MAX_MODEL_LEN  MAX_NUM_SEQS=$MAX_NUM_SEQS"
echo "[INFO] MAX_NUM_BATCHED_TOKENS=$MAX_NUM_BATCHED_TOKENS"
echo "[INFO] GPU_MEM_UTIL=$GPU_MEM_UTIL"
echo "[INFO] BALANCE_SCHEDULING=$VLLM_ASCEND_BALANCE_SCHEDULING"
echo "[INFO] Parser: minimax_m2 (tool + reasoning)"
echo "============================================"

vllm serve "$MODEL_PATH" \
    --host "$HOST" \
    --port "$PORT" \
    --served-model-name "MiniMax-M3" \
    --trust-remote-code \
    --tensor-parallel-size "$TP" \
    --pipeline-parallel-size "$PP" \
    --data-parallel-size "$DP" \
    --quantization ascend \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
    --enable-expert-parallel \
    --enable-auto-tool-choice \
    --tool-call-parser minimax_m2 \
    --reasoning-parser minimax_m2_append_think \
    --async-scheduling \
    --compilation-config "$COMPILATION_CONFIG" \
    --additional-config "$ADDITIONAL_CONFIG" \
    --seed 1024 \
    "$@"
