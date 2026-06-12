#!/bin/bash
# GLM-5.1 W4A8 — PD 共置与 Mooncake 多实例部署
# 功能: 基于 Mooncake 分布式 KV Cache 实现预填充-解码共置
# 配置与 GLM-5 W4A8 完全相同
# 参考: https://docs.vllm.ai/projects/ascend/zh-cn/releases-v0.20.2rc/tutorials/features/pd_colocated_mooncake_multi_instance.html
#
# 前置条件:
#   1. 安装 Mooncake: https://github.com/kvcache-ai/Mooncake
#   2. 启动 Mooncake Master: mooncake_master --port 50088
#   3. 配置 mooncake.json
#
# 用法:
#   MOONCAKE_CONFIG_PATH=/path/to/mooncake.json bash run_pd_colocated.sh
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
MODEL_PATH="${MODEL_PATH:-$BASE_MODEL_PATH/GLM-5.1-w4a8}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8002}"
TP="${TP:-8}"
PP="${PP:-1}"

export MOONCAKE_CONFIG_PATH="${MOONCAKE_CONFIG_PATH:-./mooncake.json}"
export ASCEND_BUFFER_POOL="${ASCEND_BUFFER_POOL:-4:8}"
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
echo "[INFO] GLM-5.1 W4A8 — PD Colocated (Mooncake)"
echo "[INFO] TP=$TP PP=$PP PORT=$PORT"
echo "[INFO] KV Role: kv_both"
echo "============================================"

vllm serve "$MODEL_PATH" \
    --host "$HOST" \
    --port "$PORT" \
    --served-model-name "glm-5.1" \
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
