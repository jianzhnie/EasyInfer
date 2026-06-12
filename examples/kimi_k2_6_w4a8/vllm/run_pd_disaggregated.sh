#!/bin/bash
# =============================================================================
# Kimi-K2.6 W4A8 — PD Disaggregation with Mooncake
# =============================================================================
# Purpose: Separate Prefill and Decode onto different nodes and transfer KV
#          Cache via Mooncake.
# Architecture: KimiK25ForConditionalGeneration | 384 Experts | supports PP/DP
#
# Prerequisites:
#   1. At least 2 nodes with RoCE interconnect
#   2. Mooncake installed and configured
#   3. Mooncake Master started
#
# PD disaggregation architecture (2P1D example):
#   - Prefill nodes (2): TP=8 DP=2, kv_role=kv_producer
#   - Decode node (1):   TP=8, kv_role=kv_consumer
#
# Usage:
#   # Prefill node (node1, engine_id=0)
#   KV_ROLE=kv_producer KV_PORT=30000 ENGINE_ID=0 DATA_PARALLEL_SIZE=2 \
#     DATA_PARALLEL_ADDRESS=<MASTER_IP> bash run_pd_disaggregated.sh
#
#   # Decode node (node3, engine_id=2)
#   KV_ROLE=kv_consumer KV_PORT=30002 ENGINE_ID=2 PORT=8103 \
#     DATA_PARALLEL_SIZE=1 bash run_pd_disaggregated.sh
#
# Constraints:
#   - Heterogeneous P/D nodes are not supported (e.g. A2 prefill + A3 decode)
#   - P_tp > D_tp requires P_tp % D_tp == 0
#   - Each node needs kv_port to kv_port + num_chips port range available
#
# Reference:
#   https://docs.vllm.ai/projects/ascend/zh-cn/releases-v0.20.2rc/tutorials/features/pd_disaggregation_mooncake_multi_node.html
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
readonly KV_ROLE="${KV_ROLE:-kv_producer}"
readonly KV_PORT="${KV_PORT:-30000}"
readonly ENGINE_ID="${ENGINE_ID:-0}"
readonly DATA_PARALLEL_SIZE="${DATA_PARALLEL_SIZE:-2}"
readonly DATA_PARALLEL_ADDRESS="${DATA_PARALLEL_ADDRESS:-}"

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
echo "[INFO] Kimi-K2.6 W4A8 — PD Disaggregation"
echo "[INFO] Model: $MODEL_PATH"
echo "[INFO] TP=$TP PP=$PP DP=$DP PORT=$PORT"
echo "[INFO] KV Role: $KV_ROLE (Engine ID: $ENGINE_ID)"
echo "[INFO] Mooncake Config: $MOONCAKE_CONFIG_PATH"
echo "============================================"

SERVE_ARGS=(
    --host "$HOST" --port "$PORT"
    --served-model-name "kimi-k2.6"
    --trust-remote-code
    --dtype bfloat16
    --tensor-parallel-size "$TP"
    --pipeline-parallel-size "$PP"
    --distributed-executor-backend mp
    --quantization ascend
    --gpu-memory-utilization 0.90
    --max-model-len 32768
    --max-num-seqs 16
    --max-num-batched-tokens 16384
    --enable-chunked-prefill
    --enable-expert-parallel
    --language-model-only
    --mm-encoder-tp-mode data
    --allowed-local-media-path /home/jianzhnie/llmtuner/
    --seed 1024
)

# Data Parallel configuration (PD disaggregation uses mp backend + DP)
if [[ "$DATA_PARALLEL_SIZE" -gt 1 ]]; then
    SERVE_ARGS+=(--data-parallel-size "$DATA_PARALLEL_SIZE")
fi
if [[ -n "$DATA_PARALLEL_ADDRESS" ]]; then
    SERVE_ARGS+=(--data-parallel-address "$DATA_PARALLEL_ADDRESS")
fi

# KV Transfer configuration
SERVE_ARGS+=(
    --kv-transfer-config "{
        \"kv_connector\": \"MooncakeLayerwiseConnector\",
        \"kv_role\": \"$KV_ROLE\",
        \"kv_port\": \"$KV_PORT\",
        \"engine_id\": \"$ENGINE_ID\",
        \"kv_connector_module_path\": \"vllm_ascend.distributed.mooncake_layerwise_connector\",
        \"kv_connector_extra_config\": {
            \"prefill\": {\"dp_size\": $DATA_PARALLEL_SIZE, \"tp_size\": $TP},
            \"decode\": {\"dp_size\": 1, \"tp_size\": $TP}
        }
    }"
)

vllm serve "$MODEL_PATH" "${SERVE_ARGS[@]}" "$@"
