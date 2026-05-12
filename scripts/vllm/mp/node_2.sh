#!/usr/bin/env bash
# Node 1 (worker) — Kimi-K2 多节点部署参考配置 (headless 模式)
# 用法: 根据实际环境修改 nic_name, local_ip, node0_ip 后 source 或执行

set -euo pipefail

# 通过 ifconfig 获取的网卡名称和 IP
nic_name="enp66s0f0"
local_ip="10.42.28.195"

# node0_ip 必须与 master 节点的 local_ip 一致
node0_ip="10.42.28.194"

export HCCL_OP_EXPANSION_MODE="AIV"

export HCCL_IF_IP="$local_ip"
export GLOO_SOCKET_IFNAME="$nic_name"
export TP_SOCKET_IFNAME="$nic_name"
export HCCL_SOCKET_IFNAME="$nic_name"
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=100
export VLLM_USE_V1=1
export HCCL_BUFFSIZE=200
export VLLM_ASCEND_ENABLE_MLAPO=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
export HCCL_CONNECT_TIMEOUT=120
export HCCL_INTRA_PCIE_ENABLE=1
export HCCL_INTRA_ROCE_ENABLE=0

vllm serve /llm_workspace_1P/robin/hfhub/models/moonshotai/Kimi-K2-Base \
    --host 0.0.0.0 \
    --port 8077 \
    --headless \
    --data-parallel-size 2 \
    --data-parallel-size-local 1 \
    --data-parallel-start-rank 1 \
    --data-parallel-address "$node0_ip" \
    --data-parallel-rpc-port 13389 \
    --tensor-parallel-size 8 \
    --seed 1024 \
    --served-model-name deepseek_v3_2 \
    --enable-expert-parallel \
    --max-num-seqs 16 \
    --max-model-len 8192 \
    --max-num-batched-tokens 4096 \
    --trust-remote-code \
    --no-enable-prefix-caching \
    --gpu-memory-utilization 0.92 \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes":[8, 16, 24, 32, 40, 48]}' \
    --additional-config '{"layer_sharding": ["q_b_proj", "o_proj"]}'
