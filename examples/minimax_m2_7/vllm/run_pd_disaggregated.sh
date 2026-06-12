#!/bin/bash
# MiniMax-M2.7 W8A8 QuaRot — 预填充-解码分离部署 (PD Disaggregation)
# 功能: 将 Prefill 和 Decode 分离到不同节点，通过 Mooncake 传输 KV Cache
# 架构: MiniMaxM2ForCausalLM | MoE 256E | 支持 PP
# 参考: https://docs.vllm.ai/projects/ascend/zh-cn/releases-v0.20.2rc/tutorials/features/pd_disaggregation_mooncake_multi_node.html
#
# 前置条件:
#   1. 至少 2 节点，RoCE 网络互通
#   2. Mooncake 已安装并配置
#
# 用法:
#   # Prefill 节点: KV_ROLE=kv_producer KV_PORT=30000 ENGINE_ID=0 bash run_pd_disaggregated.sh
#   # Decode 节点:  KV_ROLE=kv_consumer KV_PORT=30001 ENGINE_ID=1 PORT=8104 bash run_pd_disaggregated.sh
set -eo pipefail

set +u
if [[ -f "/usr/local/Ascend/cann/set_env.sh" ]]; then
    source /usr/local/Ascend/cann/set_env.sh
fi
if [[ -f "/usr/local/Ascend/nnal/atb/set_env.sh" ]]; then
    source /usr/local/Ascend/nnal/atb/set_env.sh
fi
set -u

BASE_MODEL_PATH="/home/jianzhnie/llmtuner/hfhub/models/Eco-Tech"
MODEL_PATH="${MODEL_PATH:-$BASE_MODEL_PATH/MiniMax-M2.7-w8a8-QuaRot}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8004}"
TP="${TP:-4}"
PP="${PP:-1}"
KV_ROLE="${KV_ROLE:-kv_producer}"
KV_PORT="${KV_PORT:-30000}"
ENGINE_ID="${ENGINE_ID:-0}"
DATA_PARALLEL_SIZE="${DATA_PARALLEL_SIZE:-2}"
DATA_PARALLEL_ADDRESS="${DATA_PARALLEL_ADDRESS:-}"

export MOONCAKE_CONFIG_PATH="${MOONCAKE_CONFIG_PATH:-./mooncake.json}"
export ASCEND_BUFFER_POOL="${ASCEND_BUFFER_POOL:-4:8}"
export HCCL_OP_EXPANSION_MODE=AIV
export HCCL_BUFFSIZE=1024
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export TASK_QUEUE_ENABLE=1
export VLLM_ASCEND_ENABLE_FUSED_MC2=1
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
export VLLM_ASCEND_BALANCE_SCHEDULING=1
export VLLM_USE_MODELSCOPE=False

echo "============================================"
echo "[INFO] MiniMax-M2.7 W8A8 — PD Disaggregation"
echo "[INFO] Model: $MODEL_PATH"
echo "[INFO] TP=$TP PP=$PP PORT=$PORT"
echo "[INFO] KV Role: $KV_ROLE (Engine ID: $ENGINE_ID)"
echo "============================================"

SERVE_ARGS=(
    --host "$HOST" --port "$PORT"
    --served-model-name "minimax-m2.7"
    --trust-remote-code
    --dtype bfloat16
    --tensor-parallel-size "$TP"
    --pipeline-parallel-size "$PP"
    --distributed-executor-backend mp
    --quantization ascend
    --gpu-memory-utilization 0.83
    --max-model-len 32768
    --max-num-seqs 16
    --max-num-batched-tokens 8192
    --enable-chunked-prefill
    --enforce-eager
    --enable-expert-parallel
    --seed 1024
)

if [[ "$DATA_PARALLEL_SIZE" -gt 1 ]]; then
    SERVE_ARGS+=(--data-parallel-size "$DATA_PARALLEL_SIZE")
fi
if [[ -n "$DATA_PARALLEL_ADDRESS" ]]; then
    SERVE_ARGS+=(--data-parallel-address "$DATA_PARALLEL_ADDRESS")
fi

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
