#!/bin/bash
# =============================================================================
# GLM-5 BF16 — vllm serve deployment (multi-node full-precision)
# =============================================================================
# Architecture: GlmMoeDsaForCausalLM | 256 Experts | MoE | MLA | MTP=1
# Max Position: 202752 | BF16 (~1.4T) needs TP>=32
#
# Note: PP>1 blocks MTP. Add --quantization ascend for W4A8/W8A8 single-node.
#
# Usage:
#   bash run_vllm.sh
#   TP=16 MAX_MODEL_LEN=202752 bash run_vllm.sh
#   MODEL_PATH=/path/to/GLM-5 PORT=8001 bash run_vllm.sh
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
readonly BASE_MODEL_PATH="/home/jianzhnie/llmtuner/hfhub/models/ZhipuAI"
readonly MODEL_PATH="${MODEL_PATH:-$BASE_MODEL_PATH/GLM-5}"
readonly HOST="${HOST:-0.0.0.0}"
readonly PORT="${PORT:-8001}"
readonly TP="${TP:-32}"
readonly PP="${PP:-1}"
readonly MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
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
echo "[INFO] GLM-5 BF16 — vLLM-Ascend Deployment"
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
    --dtype bfloat16 \
    --tensor-parallel-size "$TP" \
    --pipeline-parallel-size "$PP" \
    --distributed-executor-backend ray \
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
