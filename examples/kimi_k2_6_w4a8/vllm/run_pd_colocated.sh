#!/bin/bash
# =============================================================================
# Kimi-K2.6 W4A8 — PD Colocated with Mooncake (multi-instance)
# =============================================================================
# Purpose: Prefill-decode colocation via Mooncake distributed KV Cache.
# Architecture: KimiK25ForConditionalGeneration | 384 Experts | MLA | Vision
#
# Prerequisites:
#   1. Mooncake installed: https://github.com/kvcache-ai/Mooncake
#   2. Mooncake Master started: mooncake_master --port 50088
#   3. mooncake.json configured and MOONCAKE_CONFIG_PATH set
#
# Usage:
#   # Single instance
#   MOONCAKE_CONFIG_PATH=/path/to/mooncake.json bash run_pd_colocated.sh
#
#   # Multi-instance (one instance per node, shared KV Cache)
#   # Node1: MOONCAKE_CONFIG_PATH=/path/to/mooncake.json bash run_pd_colocated.sh
#   # Node2: MOONCAKE_CONFIG_PATH=/path/to/mooncake.json PORT=8103 bash run_pd_colocated.sh
#
# Reference:
#   https://docs.vllm.ai/projects/ascend/zh-cn/releases-v0.20.2rc/tutorials/features/pd_colocated_mooncake_multi_instance.html
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
readonly MODEL_PATH="${MODEL_PATH:-$BASE_MODEL_PATH/Kimi-K2.6-w4a8}"
readonly HOST="${HOST:-0.0.0.0}"
readonly PORT="${PORT:-8003}"
readonly TP="${TP:-8}"
readonly PP="${PP:-1}"
readonly DP="${DP:-1}"

# Mooncake configuration
export MOONCAKE_CONFIG_PATH="${MOONCAKE_CONFIG_PATH:-./mooncake.json}"
export ASCEND_BUFFER_POOL="${ASCEND_BUFFER_POOL:-4:8}"

# NPU environment variables
export HCCL_OP_EXPANSION_MODE=AIV
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export HCCL_BUFFSIZE=800
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TASK_QUEUE_ENABLE=1
export VLLM_ASCEND_ENABLE_MLAPO=1
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
export VLLM_ASCEND_BALANCE_SCHEDULING=1
export VLLM_USE_MODELSCOPE=False

echo "============================================"
echo "[INFO] Kimi-K2.6 W4A8 — PD Colocated (Mooncake)"
echo "[INFO] Model: $MODEL_PATH"
echo "[INFO] TP=$TP PP=$PP DP=$DP PORT=$PORT"
echo "[INFO] Mooncake Config: $MOONCAKE_CONFIG_PATH"
echo "[INFO] KV Role: kv_both (producer + consumer)"
echo "============================================"

vllm serve "$MODEL_PATH" \
    --host "$HOST" \
    --port "$PORT" \
    --served-model-name "kimi-k2.6" \
    --trust-remote-code \
    --dtype bfloat16 \
    --tensor-parallel-size "$TP" \
    --pipeline-parallel-size "$PP" \
    --data-parallel-size "$DP" \
    --distributed-executor-backend ray \
    --quantization ascend \
    --gpu-memory-utilization 0.90 \
    --max-model-len 32768 \
    --max-num-seqs 16 \
    --max-num-batched-tokens 16384 \
    --enable-chunked-prefill \
    --enable-prefix-caching \
    --enable-expert-parallel \
    --enable-auto-tool-choice \
    --tool-call-parser kimi_k2 \
    --language-model-only \
    --mm-encoder-tp-mode data \
    --allowed-local-media-path /home/jianzhnie/llmtuner/ \
    --kv-transfer-config '{
        "kv_connector": "MooncakeConnectorStoreV1",
        "kv_role": "kv_both",
        "kv_connector_extra_config": {
            "use_layerwise": false,
            "mooncake_rpc_port": "0",
            "load_async": true,
            "register_buffer": true
        }
    }' \
    --seed 1024 \
    "$@"
