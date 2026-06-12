#!/bin/bash
# MiniMax-M2.7 W8A8 QuaRot — PD 共置与 Mooncake 多实例部署
# 功能: 基于 Mooncake 分布式 KV Cache 实现预填充-解码共置
# 架构: MiniMaxM2ForCausalLM | 256 Experts | MoE | W8A8
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
MODEL_PATH="${MODEL_PATH:-$BASE_MODEL_PATH/MiniMax-M2.7-w8a8-QuaRot}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8004}"
TP="${TP:-4}"
PP="${PP:-1}"

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
echo "[INFO] MiniMax-M2.7 W8A8 — PD Colocated (Mooncake)"
echo "[INFO] Model: $MODEL_PATH"
echo "[INFO] TP=$TP PP=$PP PORT=$PORT"
echo "[INFO] Mooncake Config: $MOONCAKE_CONFIG_PATH"
echo "[INFO] KV Role: kv_both"
echo "============================================"

vllm serve "$MODEL_PATH" \
    --host "$HOST" \
    --port "$PORT" \
    --served-model-name "minimax-m2.7" \
    --trust-remote-code \
    --dtype bfloat16 \
    --tensor-parallel-size "$TP" \
    --pipeline-parallel-size "$PP" \
    --distributed-executor-backend ray \
    --quantization ascend \
    --gpu-memory-utilization 0.83 \
    --max-model-len 32768 \
    --max-num-seqs 16 \
    --max-num-batched-tokens 8192 \
    --enable-chunked-prefill \
    --enable-prefix-caching \
    --enforce-eager \
    --enable-expert-parallel \
    --enable-auto-tool-choice \
    --tool-call-parser minimax_m2 \
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
