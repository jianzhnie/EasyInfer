#!/bin/bash
# DeepSeek-V4-Pro W4A8 — vLLM Ascend 0.22.1rc1 Deployment
# Architecture: DeepseekV4ForCausalLM | 384 Experts | MoE | MTP=1
# Max Position: 1048576 | Deploy: 4K context (2-node default to fit KV cache)
set -euo pipefail

# CANN environment
set +u
if [[ -f "/usr/local/Ascend/cann/set_env.sh" ]]; then
    source /usr/local/Ascend/cann/set_env.sh
fi
if [[ -f "/usr/local/Ascend/nnal/atb/set_env.sh" ]]; then
    source /usr/local/Ascend/nnal/atb/set_env.sh
fi
set -u

BASE_MODEL_PATH="/home/jianzhnie/llmtuner/hfhub/models/Eco-Tech"
MODEL_PATH="${MODEL_PATH:-$BASE_MODEL_PATH/DeepSeek-V4-Pro-w4a8-mtp}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8005}"
TP="${TP:-8}"
PP="${PP:-2}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.95}"

# NPU optimizations
export HCCL_OP_EXPANSION_MODE="${HCCL_OP_EXPANSION_MODE:-AIV}"
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export HCCL_BUFFSIZE=400
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_ASCEND_BALANCE_SCHEDULING=1
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
export VLLM_ASCEND_ENABLE_MLAPO=1
if [[ "$PP" -gt 1 || "$TP" -gt 8 ]]; then
    export VLLM_ASCEND_ENABLE_FUSED_MC2=1
else
    export VLLM_ASCEND_ENABLE_FUSED_MC2=0
fi
export VLLM_USE_MODELSCOPE=False

# Compilation config
readonly COMPILATION_CONFIG='{"cudagraph_mode": "FULL_DECODE_ONLY"}'

echo "============================================"
echo "[INFO] DeepSeek-V4-Pro W4A8 — vLLM-Ascend Deployment"
echo "[INFO] Model:    $MODEL_PATH"
echo "[INFO] TP=$TP  PP=$PP  PORT=$PORT"
echo "[INFO] MAX_MODEL_LEN=$MAX_MODEL_LEN  MAX_NUM_SEQS=$MAX_NUM_SEQS"
echo "[INFO] GPU_MEM_UTIL=$GPU_MEM_UTIL"
echo "[INFO] FUSED_MC2=$VLLM_ASCEND_ENABLE_FUSED_MC2"
echo "[INFO] MTP: ON (1 token, deepseek_mtp)"
echo "============================================"

vllm serve "$MODEL_PATH" \
    --host "$HOST" \
    --port "$PORT" \
    --served-model-name deepseek-v4-pro \
    --trust-remote-code \
    --dtype bfloat16 \
    --tensor-parallel-size "$TP" \
    --pipeline-parallel-size "$PP" \
    --distributed-executor-backend ray \
    --enable-expert-parallel \
    --quantization ascend \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --max-num-batched-tokens 8192 \
    --enable-chunked-prefill \
    --enable-prefix-caching \
    --enable-auto-tool-choice \
    --tool-call-parser deepseek_v4 \
    --reasoning-parser deepseek_v4 \
    --async-scheduling \
    --compilation-config "$COMPILATION_CONFIG" \
    --speculative-config '{"num_speculative_tokens": 1, "method": "deepseek_mtp", "enforce_eager": true}' \
    --seed 1024 \
    "$@"
