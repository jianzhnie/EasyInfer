#!/bin/bash
# Node 0 (master) — Kimi-K2 多节点部署参考配置
# 用法: 根据实际环境修改 NIC_NAME, LOCAL_IP, NODE0_IP 后执行
#       或通过环境变量覆盖: NIC_NAME=eth0 LOCAL_IP=10.x.x.x bash node_1.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./_node_env.sh
source "${SCRIPT_DIR}/_node_env.sh"

# 加载并导出公共环境变量
LOCAL_IP="${LOCAL_IP:-10.42.28.194}"
NODE0_IP="${NODE0_IP:-10.42.28.194}"
export_env_vars

vllm serve "$MODEL_PATH" \
    --host 0.0.0.0 \
    --port "$VLLM_PORT" \
    --data-parallel-size "$DP_SIZE" \
    --data-parallel-size-local "$DP_SIZE_LOCAL" \
    --data-parallel-address "$NODE0_IP" \
    --data-parallel-rpc-port "$DP_RPC_PORT" \
    --tensor-parallel-size "$TP_SIZE" \
    --seed 1024 \
    --served-model-name "$SERVED_MODEL_NAME" \
    --enable-expert-parallel \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
    --trust-remote-code \
    --no-enable-prefix-caching \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes":[8, 16, 24, 32, 40, 48]}' \
    --additional-config '{"layer_sharding": ["q_b_proj", "o_proj"]}'
