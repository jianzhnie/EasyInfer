#!/bin/bash
# =============================================================================
# GLM-5 W4A8 — vllm serve deployment
# =============================================================================
# Architecture: GlmMoeDsaForCausalLM | 256 Experts | MLA | MTP=1
# Max Position: 202752 | GLM-5.1 shares the same config (switch MODEL_PATH)
#
# Note: PP>1 blocks MTP. DSA path is incompatible with FLASHCOMM1.
#
# Usage:
#   bash run_vllm.sh                                     # TP=8 single-node
#   TP=16 MAX_MODEL_LEN=202752 bash run_vllm.sh          # multi-node long ctx
#   MODEL_PATH=/path/to/GLM-5-w4a8 PORT=8001 bash run_vllm.sh
#
# Reference:
#   https://docs.vllm.ai/projects/ascend/zh-cn/latest/tutorials/models/GLM5.2.html
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
readonly MODEL_PATH="${MODEL_PATH:-$BASE_MODEL_PATH/GLM-5-w4a8}"
readonly HOST="${HOST:-0.0.0.0}"
readonly PORT="${PORT:-8001}"
readonly TP="${TP:-8}"
readonly PP="${PP:-1}"
readonly MAX_MODEL_LEN="${MAX_MODEL_LEN:-31744}"
readonly MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
readonly GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.95}"

# NPU environment variables
export HCCL_OP_EXPANSION_MODE=AIV
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export HCCL_BUFFSIZE=200
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_USE_MODELSCOPE=False
# DSA path is incompatible with FLASHCOMM1 — must stay disabled
export VLLM_ASCEND_ENABLE_FLASHCOMM1=0
export VLLM_ASCEND_ENABLE_MLAPO=1
export VLLM_ASCEND_BALANCE_SCHEDULING=1
if [[ "$PP" -gt 1 || "$TP" -gt 8 ]]; then
    export VLLM_ASCEND_ENABLE_FUSED_MC2=1
else
    export VLLM_ASCEND_ENABLE_FUSED_MC2=0
fi

# Compilation config
readonly COMPILATION_CONFIG='{"cudagraph_mode": "FULL_DECODE_ONLY"}'
readonly ADDITIONAL_CONFIG='{"enable_balance_scheduling": true, "enable_flashcomm1": false, "enable_mlapo": true}'

echo "============================================"
echo "[INFO] GLM-5 W4A8 — vLLM-Ascend Deployment"
echo "[INFO] Model:    $MODEL_PATH"
echo "[INFO] TP=$TP  PP=$PP  PORT=$PORT"
echo "[INFO] MAX_MODEL_LEN=$MAX_MODEL_LEN  MAX_NUM_SEQS=$MAX_NUM_SEQS"
echo "[INFO] GPU_MEM_UTIL=$GPU_MEM_UTIL"
echo "[INFO] FLASHCOMM1=0 (DSA CP incompatible)"
echo "[INFO] FUSED_MC2=$VLLM_ASCEND_ENABLE_FUSED_MC2"
echo "[INFO] MTP: ON (3 tokens, deepseek_mtp)"
echo "[INFO] Tool Calling: glm47 parser + glm45 reasoning"
echo "============================================"

vllm serve "$MODEL_PATH" \
    --host "$HOST" \
    --port "$PORT" \
    --served-model-name "glm-5" \
    --trust-remote-code \
    --tensor-parallel-size "$TP" \
    --pipeline-parallel-size "$PP" \
    --distributed-executor-backend ray \
    --quantization ascend \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --max-num-batched-tokens 16384 \
    --chat-template-content-format string \
    --enable-chunked-prefill \
    --enable-prefix-caching \
    --enable-expert-parallel \
    --enable-auto-tool-choice \
    --tool-call-parser glm47 \
    --reasoning-parser glm45 \
    --async-scheduling \
    --compilation-config "$COMPILATION_CONFIG" \
    --speculative-config '{"num_speculative_tokens": 3, "method": "deepseek_mtp"}' \
    --additional-config "$ADDITIONAL_CONFIG" \
    --seed 1024 \
    "$@"
