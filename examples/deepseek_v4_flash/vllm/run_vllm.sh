#!/bin/bash
# =============================================================================
# DeepSeek-V4-Flash W8A8 MTP — 直接 vllm serve 部署
# ⚠️ 注意: vLLM-Ascend 0.18.0rc1 不支持 DeepseekV4ForCausalLM 架构。
# 本脚本已将 architectures 改为 DeepseekV32ForCausalLM 作为临时方案，
# 但引擎初始化仍可能失败（模型属性不兼容）。
# 需要升级 vLLM-Ascend 到支持 DeepSeek V4 的版本后才能正常使用。
# 默认 TP=8 PP=1 (单节点)
# =============================================================================
set -eo pipefail

# Load Ascend CANN environment (required for libascend_hal.so)
# CANN scripts reference unset vars; disable nounset during source
set +u
if [[ -f "/usr/local/Ascend/cann/set_env.sh" ]]; then
    source /usr/local/Ascend/cann/set_env.sh
fi
if [[ -f "/usr/local/Ascend/nnal/atb/set_env.sh" ]]; then
    source /usr/local/Ascend/nnal/atb/set_env.sh
fi
set -u

MODEL_PATH="${MODEL_PATH:-/home/jianzhnie/llmtuner/hfhub/models/Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
TP="${TP:-8}"
PP="${PP:-1}"

# HCCL/NPU env
export HCCL_OP_EXPANSION_MODE="${HCCL_OP_EXPANSION_MODE:-AIV}"
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=8
export HCCL_BUFFSIZE=200
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_ASCEND_BALANCE_SCHEDULING=1
export USE_MULTI_GROUPS_KV_CACHE=1
export USE_MULTI_BLOCK_POOL=1
export ACL_OP_INIT_MODE=1
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2

echo "[INFO] Starting DeepSeek-V4-Flash W8A8 MTP"
echo "[INFO] TP=$TP PP=$PP PORT=$PORT"

vllm serve "$MODEL_PATH" \
    --host "$HOST" \
    --port "$PORT" \
    --served-model-name deepseek-v4-flash \
    --trust-remote-code \
    --dtype bfloat16 \
    --tensor-parallel-size "$TP" \
    --pipeline-parallel-size "$PP" \
    --distributed-executor-backend ray \
    --enable-expert-parallel \
    --quantization ascend \
    --gpu-memory-utilization 0.90 \
    --max-model-len 65536 \
    --max-num-seqs 16 \
    --max-num-batched-tokens 8192 \
    --block-size 128 \
    --safetensors-load-strategy 'prefetch' \
    --tokenizer-mode deepseek_v4 \
    --enable-chunked-prefill \
    --enable-prefix-caching \
    --enable-auto-tool-choice \
    --tool-call-parser deepseek_v4 \
    --reasoning-parser deepseek_v4 \
    --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE", "custom_ops":["all"]}' \
    --speculative_config '{"method":"mtp","num_speculative_tokens":1}' \
    --seed 1024 \
    "$@"
