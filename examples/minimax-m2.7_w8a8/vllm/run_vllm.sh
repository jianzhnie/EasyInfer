#!/bin/bash
# =============================================================================
# MiniMax-M2.7 W8A8 QuaRot — vllm serve deployment
# =============================================================================
# Architecture: MiniMaxM2ForCausalLM | 256 Experts | MoE
# Official: TP=4 DP=4 A3 single-node, Eagle3 speculative decoding
#
# Hardware:
#   - 1× Atlas 800 A3 (64G × 16): TP=4 DP=4
#   - 1× Atlas 800 A2 (64G × 8):  TP=8
#
# Usage:
#   bash run_vllm.sh                                  # TP=4 DP=4 single-node A3
#   TP=8 bash run_vllm.sh                             # TP=8 A2 single-node
#   TP=8 DP=2 bash run_vllm.sh                        # TP=8 DP=2 A3 high-throughput
#   TP=8 DECODE_CP=2 MAX_MODEL_LEN=138000 bash run_vllm.sh  # Long context 128K
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
readonly MODEL_PATH="${MODEL_PATH:-$BASE_MODEL_PATH/MiniMax-M2.7-w8a8-QuaRot}"
readonly HOST="${HOST:-0.0.0.0}"
readonly PORT="${PORT:-8004}"
readonly TP="${TP:-4}"
readonly PP="${PP:-1}"
readonly DP="${DP:-4}"
readonly MAX_MODEL_LEN="${MAX_MODEL_LEN:-40690}"
readonly MAX_NUM_SEQS="${MAX_NUM_SEQS:-48}"
readonly MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-16384}"
readonly GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"

# DSA CP for long context (set DECODE_CP to enable, e.g., DECODE_CP=2)
readonly DECODE_CP="${DECODE_CP:-1}"
readonly PREFILL_CP="${PREFILL_CP:-1}"
CP_ARGS=()
if [[ "$DECODE_CP" -gt 1 ]]; then
    CP_ARGS+=(--decode-context-parallel-size "$DECODE_CP")
    CP_ARGS+=(--prefill-context-parallel-size "$PREFILL_CP")
    CP_ARGS+=(--cp-kv-cache-interleave-size 128)
fi

# Eagle3 draft model for speculative decoding
readonly EAGLE3_MODEL="${EAGLE3_MODEL:-/path/to/Eagle3}"
readonly SPEC_TOKENS="${SPEC_TOKENS:-3}"

# NPU environment variables (official docs)
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
echo "[INFO] MiniMax-M2.7 W8A8 QuaRot — vLLM-Ascend Deployment"
echo "[INFO] Model:    $MODEL_PATH"
echo "[INFO] TP=$TP  PP=$PP  DP=$DP  PORT=$PORT"
echo "[INFO] MAX_MODEL_LEN=$MAX_MODEL_LEN  MAX_NUM_SEQS=$MAX_NUM_SEQS"
echo "[INFO] MAX_NUM_BATCHED_TOKENS=$MAX_NUM_BATCHED_TOKENS"
echo "[INFO] GPU_MEM_UTIL=$GPU_MEM_UTIL"
echo "[INFO] BALANCE_SCHEDULING=$VLLM_ASCEND_BALANCE_SCHEDULING"
echo "[INFO] Speculative: Eagle3 ($EAGLE3_MODEL, ${SPEC_TOKENS} tokens)"
echo "[INFO] DSA CP: prefll_cp=$PREFILL_CP decode_cp=$DECODE_CP"
echo "[INFO] Parser: minimax_m2 (tool + reasoning)"
echo "============================================"

vllm serve "$MODEL_PATH" \
    --host "$HOST" \
    --port "$PORT" \
    --served-model-name "MiniMax-M2.7" \
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
    "${CP_ARGS[@]}" \
    --compilation-config "$COMPILATION_CONFIG" \
    --speculative-config "{\"enforce_eager\":true,\"method\":\"eagle3\",\"model\":\"$EAGLE3_MODEL\",\"num_speculative_tokens\":$SPEC_TOKENS}" \
    --additional-config "$ADDITIONAL_CONFIG" \
    --seed 1024 \
    "$@"
