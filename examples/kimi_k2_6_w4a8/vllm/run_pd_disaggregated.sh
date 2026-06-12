#!/bin/bash
# Kimi-K2.6 W4A8 — 预填充-解码分离部署 (PD Disaggregation)
# 功能: 将 Prefill 和 Decode 分离到不同节点，通过 Mooncake 传输 KV Cache
# 架构: KimiK25ForConditionalGeneration | MoE 384E | 支持 PP/DP
# 参考: https://docs.vllm.ai/projects/ascend/zh-cn/releases-v0.20.2rc/tutorials/features/pd_disaggregation_mooncake_multi_node.html
#
# 前置条件:
#   1. 至少 2 节点，RoCE 网络互通
#   2. Mooncake 已安装并配置
#   3. Mooncake Master 已启动
#
# PD 分离架构 (2P1D 示例):
#   - Prefill 节点 (2台): TP=8 DP=2, kv_role=kv_producer
#   - Decode 节点 (1台):  TP=8, kv_role=kv_consumer
#
# 用法:
#   # Prefill 节点 (节点1, engine_id=0):
#   KV_ROLE=kv_producer KV_PORT=30000 ENGINE_ID=0 DATA_PARALLEL_SIZE=2 \
#     DATA_PARALLEL_ADDRESS=<MASTER_IP> bash run_pd_disaggregated.sh
#
#   # Decode 节点 (节点3, engine_id=2):
#   KV_ROLE=kv_consumer KV_PORT=30002 ENGINE_ID=2 PORT=8103 \
#     DATA_PARALLEL_SIZE=1 bash run_pd_disaggregated.sh
#
# 限制:
#   - 异构 P/D 节点不支持 (A2 prefill + A3 decode 不行)
#   - P_tp > D_tp 需 P_tp % D_tp == 0
#   - 每节点 kv_port 到 kv_port + num_chips 端口范围需可用
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
MODEL_PATH="${MODEL_PATH:-$BASE_MODEL_PATH/Kimi-K2.6-w4a8}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8003}"
TP="${TP:-8}"
PP="${PP:-1}"
DP="${DP:-1}"
KV_ROLE="${KV_ROLE:-kv_producer}"
KV_PORT="${KV_PORT:-30000}"
ENGINE_ID="${ENGINE_ID:-0}"
DATA_PARALLEL_SIZE="${DATA_PARALLEL_SIZE:-2}"
DATA_PARALLEL_ADDRESS="${DATA_PARALLEL_ADDRESS:-}"

export MOONCAKE_CONFIG_PATH="${MOONCAKE_CONFIG_PATH:-./mooncake.json}"
export ASCEND_BUFFER_POOL="${ASCEND_BUFFER_POOL:-4:8}"
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

# 构建命令参数
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

# Data Parallel 配置 (PD 分离使用 mp backend + DP)
if [[ "$DATA_PARALLEL_SIZE" -gt 1 ]]; then
    SERVE_ARGS+=(--data-parallel-size "$DATA_PARALLEL_SIZE")
fi
if [[ -n "$DATA_PARALLEL_ADDRESS" ]]; then
    SERVE_ARGS+=(--data-parallel-address "$DATA_PARALLEL_ADDRESS")
fi

# KV Transfer 配置
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
