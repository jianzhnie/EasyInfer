#!/bin/bash
# =============================================================================
# Step-3.7-Flash W8A8 MTP — Direct vllm serve deployment
# =============================================================================
# Architecture: Step3p7ForConditionalGeneration (text: Step3p5ForCausalLM)
# Default: TP=8 PP=1 (single-node, weights ~204G → ~26G per NPU)
# Note: vLLM 0.22.1 registry lacks Step3p7ForConditionalGeneration but does
#       include Step3p5ForCausalLM + Step3p5MTP — the outer VL wrapper may
#       fail to load; deployment result to be verified on-cluster.
#
# Usage:
#   bash run_vllm.sh
#   TP=8 MAX_MODEL_LEN=65536 bash run_vllm.sh
#   ENABLE_MTP=0 bash run_vllm.sh          # disable speculative decoding
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
readonly BASE_MODEL_PATH="/home/jianzhnie/llmtuner/hfhub/models/Eco-Tech"
readonly MODEL_PATH="${MODEL_PATH:-$BASE_MODEL_PATH/Step-3.7-Flash-w8a8-mtp}"
readonly HOST="${HOST:-0.0.0.0}"
readonly PORT="${PORT:-8015}"
readonly TP="${TP:-8}"
readonly PP="${PP:-1}"
readonly MAX_MODEL_LEN="${MAX_MODEL_LEN:-31744}"
readonly MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
readonly GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.95}"
readonly ENABLE_MTP="${ENABLE_MTP:-0}"
# Note: Step-3.7-Flash's config.json has no num_nextn_predict_layers.
# MTP is disabled by default; vLLM 0.22.1 does not support method='mtp' for
# this architecture.

# NPU environment variables
export HCCL_OP_EXPANSION_MODE=AIV
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export HCCL_BUFFSIZE=512
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_USE_MODELSCOPE=False

# Fallback variables for older versions
export VLLM_ASCEND_BALANCE_SCHEDULING=1
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
export VLLM_ASCEND_ENABLE_MLAPO=1
if [[ "$PP" -gt 1 || "$TP" -gt 8 ]]; then
    export VLLM_ASCEND_ENABLE_FUSED_MC2=1
else
    export VLLM_ASCEND_ENABLE_FUSED_MC2=0
fi

readonly COMPILATION_CONFIG='{"cudagraph_mode": "FULL_DECODE_ONLY"}'
readonly ADDITIONAL_CONFIG='{"enable_balance_scheduling": true, "enable_flashcomm1": true}'

# MTP speculative decoding (Step3p5MTP is registered in vLLM 0.22.1)
SPEC_ARGS=()
if [[ "$ENABLE_MTP" == "1" ]]; then
    SPEC_ARGS+=(--speculative-config '{"num_speculative_tokens": 3, "method": "mtp"}')
fi

echo "============================================"
echo "[INFO] Step-3.7-Flash W8A8 MTP — vLLM-Ascend Deployment"
echo "[INFO] Model:    $MODEL_PATH"
echo "[INFO] TP=$TP  PP=$PP  PORT=$PORT"
echo "[INFO] MAX_MODEL_LEN=$MAX_MODEL_LEN  MAX_NUM_SEQS=$MAX_NUM_SEQS"
echo "[INFO] GPU_MEM_UTIL=$GPU_MEM_UTIL  MTP=$ENABLE_MTP"
echo "[INFO] FUSED_MC2=$VLLM_ASCEND_ENABLE_FUSED_MC2"
echo "[INFO] Tool Calling: step3p5 parser"
echo "============================================"

vllm serve "$MODEL_PATH" \
    --host "$HOST" \
    --port "$PORT" \
    --served-model-name "step-3.7-flash" \
    --trust-remote-code \
    --dtype bfloat16 \
    --tensor-parallel-size "$TP" \
    --pipeline-parallel-size "$PP" \
    --distributed-executor-backend ray \
    --quantization ascend \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --max-num-batched-tokens 8192 \
    --chat-template-content-format string \
    --enable-chunked-prefill \
    --enable-prefix-caching \
    --enable-expert-parallel \
    --enable-auto-tool-choice \
    --tool-call-parser step3p5 \
    --language-model-only \
    --async-scheduling \
    "${SPEC_ARGS[@]}" \
    --compilation-config "$COMPILATION_CONFIG" \
    --additional-config "$ADDITIONAL_CONFIG" \
    --seed 1024 \
    "$@"
