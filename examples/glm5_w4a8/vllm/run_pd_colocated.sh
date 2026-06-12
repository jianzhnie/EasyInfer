#!/bin/bash
# GLM-5 W4A8 — PD 共置与 Mooncake 多实例部署
# 功能: 基于 Mooncake 分布式 KV Cache 实现预填充-解码共置
# 要求: 多节点，Mooncake 已安装，RoCE 网络互通
# 参考: https://docs.vllm.ai/projects/ascend/zh-cn/releases-v0.20.2rc/tutorials/features/pd_colocated_mooncake_multi_instance.html
#
# 前置条件:
#   1. 安装 Mooncake: https://github.com/kvcache-ai/Mooncake
#   2. 启动 Mooncake Master: mooncake_master --port 50088
#   3. 配置 mooncake.json (自动从环境变量 MOONCAKE_CONFIG_PATH 读取)
#
# 用法:
#   # 单实例 (默认)
#   bash run_pd_colocated.sh
#
#   # 多实例 (每个节点运行为独立实例)
#   # 节点1: MOONCAKE_CONFIG_PATH=/path/to/mooncake.json bash run_pd_colocated.sh
#   # 节点2: MOONCAKE_CONFIG_PATH=/path/to/mooncake.json PORT=8101 bash run_pd_colocated.sh
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

# =============================================================================
# Mooncake 配置
# =============================================================================
# mooncake.json 模板 (需要根据实际节点 IP 修改):
# {
#     "metadata_server": "P2PHANDSHAKE",
#     "protocol": "ascend",
#     "device_name": "",
#     "master_server_address": "<MASTER_IP>:50088",
#     "global_segment_size": 107374182400
# }
# =============================================================================

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
echo "[INFO] GLM-5 W4A8 — PD Colocated (Mooncake)"
echo "[INFO] Model: $MODEL_PATH"
echo "[INFO] TP=$TP PP=$PP PORT=$PORT"
echo "[INFO] Mooncake Config: $MOONCAKE_CONFIG_PATH"
echo "[INFO] KV Role: kv_both (producer + consumer)"
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
    --max-model-len "${MAX_MODEL_LEN:-32768}" \
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
