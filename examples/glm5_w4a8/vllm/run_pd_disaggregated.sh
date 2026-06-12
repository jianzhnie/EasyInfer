#!/bin/bash
# GLM-5 W4A8 — 预填充-解码分离部署 (PD Disaggregation)
# 功能: 将 Prefill 和 Decode 分离到不同节点，通过 Mooncake 传输 KV Cache
# 注意: GLM-5 不支持 PP，PD 分离使用 TP 跨节点方式
# 参考: https://docs.vllm.ai/projects/ascend/zh-cn/releases-v0.20.2rc/tutorials/features/pd_disaggregation_mooncake_multi_node.html
#
# 前置条件:
#   1. 至少 2 节点，RoCE 网络互通
#   2. Mooncake 已安装并配置
#   3. Mooncake Master 已启动
#
# 用法:
#   # Prefill 节点 (节点1):
#   KV_ROLE=kv_producer KV_PORT=30000 ENGINE_ID=0 bash run_pd_disaggregated.sh
#
#   # Decode 节点 (节点2):
#   KV_ROLE=kv_consumer KV_PORT=30001 ENGINE_ID=1 PORT=8101 bash run_pd_disaggregated.sh
#
# 限制:
#   - GLM-5 不支持 PP，PD 分离仅支持 TP 缩放
#   - 异构 P/D 节点不支持 (如 A2 prefill + A3 decode)
#   - P_tp > D_tp 时需 P_tp % D_tp == 0
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
MODEL_PATH="${MODEL_PATH:-$BASE_MODEL_PATH/GLM-5-w4a8}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8001}"
TP="${TP:-8}"
PP="${PP:-1}"

# PD 分离角色: kv_producer (Prefill) 或 kv_consumer (Decode)
KV_ROLE="${KV_ROLE:-kv_producer}"
KV_PORT="${KV_PORT:-30000}"
ENGINE_ID="${ENGINE_ID:-0}"

# Mooncake 环境变量
export MOONCAKE_CONFIG_PATH="${MOONCAKE_CONFIG_PATH:-./mooncake.json}"
export ASCEND_BUFFER_POOL="${ASCEND_BUFFER_POOL:-4:8}"

# NPU 优化
export HCCL_OP_EXPANSION_MODE=AIV
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export HCCL_BUFFSIZE=200
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_ASCEND_BALANCE_SCHEDULING=1
export VLLM_ASCEND_ENABLE_FLASHCOMM1=0
export VLLM_ASCEND_ENABLE_MLAPO=1
export VLLM_USE_MODELSCOPE=False

echo "============================================"
echo "[INFO] GLM-5 W4A8 — PD Disaggregation"
echo "[INFO] Model: $MODEL_PATH"
echo "[INFO] TP=$TP PP=$PP PORT=$PORT"
echo "[INFO] KV Role: $KV_ROLE (Engine ID: $ENGINE_ID)"
echo "[INFO] Mooncake Config: $MOONCAKE_CONFIG_PATH"
echo "[WARN] GLM-5 不支持 PP，PD 分离仅支持 TP 跨节点"
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
    --quantization ascend \
    --gpu-memory-utilization 0.92 \
    --max-model-len 32768 \
    --max-num-seqs 8 \
    --max-num-batched-tokens 16384 \
    --enable-chunked-prefill \
    --enable-prefix-caching \
    --enforce-eager \
    --enable-expert-parallel \
    --enable-auto-tool-choice \
    --tool-call-parser glm47 \
    --reasoning-parser glm45 \
    --speculative-config '{"num_speculative_tokens": 3, "method": "mtp"}' \
    --kv-transfer-config "{
        \"kv_connector\": \"MooncakeConnector\",
        \"kv_role\": \"$KV_ROLE\",
        \"kv_port\": \"$KV_PORT\",
        \"engine_id\": \"$ENGINE_ID\",
        \"kv_connector_module_path\": \"vllm_ascend.distributed.mooncake_connector\"
    }" \
    --seed 1024 \
    "$@"
