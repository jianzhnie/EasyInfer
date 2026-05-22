#!/bin/bash
# Node 0 (master) — Kimi-K2 多节点部署参考配置
# 用法: 根据实际环境修改 NIC_NAME, LOCAL_IP, NODE0_IP 后执行
#       或通过环境变量覆盖: NIC_NAME=eth0 LOCAL_IP=10.x.x.x bash node_1.sh

set -euo pipefail

# 网络配置 — 请根据实际环境修改
NIC_NAME="${NIC_NAME:-enp66s0f0}"
LOCAL_IP="${LOCAL_IP:-10.42.28.194}"
NODE0_IP="${NODE0_IP:-10.42.28.194}"

# 模型配置
MODEL_PATH="${MODEL_PATH:-/llm_workspace_1P/robin/hfhub/models/moonshotai/Kimi-K2-Base}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-kimi-k2-base}"
VLLM_PORT="${VLLM_PORT:-8077}"

# 环境变量
export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_IF_IP="$LOCAL_IP"
export GLOO_SOCKET_IFNAME="$NIC_NAME"
export TP_SOCKET_IFNAME="$NIC_NAME"
export HCCL_SOCKET_IFNAME="$NIC_NAME"
export OMP_PROC_BIND=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-100}"
export VLLM_USE_V1=1
export HCCL_BUFFSIZE=200
export VLLM_ASCEND_ENABLE_MLAPO=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
export HCCL_CONNECT_TIMEOUT=120
export HCCL_INTRA_PCIE_ENABLE=1
export HCCL_INTRA_ROCE_ENABLE=0

vllm serve "$MODEL_PATH" \
    --host 0.0.0.0 \
    --port "$VLLM_PORT" \
    --data-parallel-size "${DP_SIZE:-2}" \
    --data-parallel-size-local "${DP_SIZE_LOCAL:-1}" \
    --data-parallel-address "$NODE0_IP" \
    --data-parallel-rpc-port "${DP_RPC_PORT:-13389}" \
    --tensor-parallel-size "${TP_SIZE:-8}" \
    --seed 1024 \
    --served-model-name "$SERVED_MODEL_NAME" \
    --enable-expert-parallel \
    --max-num-seqs "${MAX_NUM_SEQS:-16}" \
    --max-model-len "${MAX_MODEL_LEN:-8192}" \
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-4096}" \
    --trust-remote-code \
    --no-enable-prefix-caching \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.92}" \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes":[8, 16, 24, 32, 40, 48]}' \
    --additional-config '{"layer_sharding": ["q_b_proj", "o_proj"]}'
