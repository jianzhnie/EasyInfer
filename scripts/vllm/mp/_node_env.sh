#!/bin/bash
#
# Kimi-K2 多节点部署公共环境变量
# 被 node_1.sh / node_2.sh source 使用
#
# 注意: 本文件被 source 而非直接执行，不设 set -euo pipefail。

# 网络配置
NIC_NAME="${NIC_NAME:-enp66s0f0}"
LOCAL_IP="${LOCAL_IP:-}"
NODE0_IP="${NODE0_IP:-}"

# 模型配置
MODEL_PATH="${MODEL_PATH:-/llm_workspace_1P/robin/hfhub/models/moonshotai/Kimi-K2-Base}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-kimi-k2-base}"
VLLM_PORT="${VLLM_PORT:-8077}"

# 并行配置
TP_SIZE="${TP_SIZE:-8}"
DP_SIZE="${DP_SIZE:-2}"
DP_SIZE_LOCAL="${DP_SIZE_LOCAL:-1}"
DP_START_RANK="${DP_START_RANK:-0}"
DP_RPC_PORT="${DP_RPC_PORT:-13389}"

# 性能配置
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.92}"

# HCCL / NPU 环境变量
export_env_vars() {
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
}
