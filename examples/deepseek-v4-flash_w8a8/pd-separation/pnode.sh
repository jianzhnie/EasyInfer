#!/bin/bash
# ==============================================================================
# pnode.sh — DeepSeek-V4-Flash Prefill 节点模板 (kv_producer)
# ==============================================================================
# 官方 §5.2 A2: 4×PNode, 每个 DP=8 TP=1
# MooncakeHybridConnector, enable-prefix-caching, enforce-eager
#
# 调用方式（由 launch_online_dp.py 自动执行）:
#   pnode.sh <devices> <port> <dp_size> <dp_rank> <dp_addr> <dp_rpc> <tp_size>
#
# 依赖环境变量: LOCAL_IP, NIC_NAME, MODEL_PATH, LOG_DIR, ENGINE_ID
# ==============================================================================
set -euo pipefail

visible_devices="$1"; port="$2"; dp_size="$3"; dp_rank="$4"
dp_address="$5"; dp_rpc_port="$6"; tp_size="$7"

nic_name="${NIC_NAME}"; local_ip="${LOCAL_IP}"
model_path="${MODEL_PATH}"; log_dir="${LOG_DIR}"
engine_id="${ENGINE_ID:-0}"

# ---- A2 PD official: proxy cleanup ----
unset ftp_proxy https_proxy http_proxy 2>/dev/null || true
rm -rf ~/ascend/log 2>/dev/null || true

export LD_PRELOAD="/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:${LD_PRELOAD:-}"

# ---- NPU environment (official §5.2 A2 PNode) ----
export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_IF_IP="$local_ip"
export GLOO_SOCKET_IFNAME="$nic_name"
export TP_SOCKET_IFNAME="$nic_name"
export HCCL_SOCKET_IFNAME="$nic_name"
export VLLM_RPC_TIMEOUT=3600000
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
export HCCL_EXEC_TIMEOUT=204
export HCCL_CONNECT_TIMEOUT=1200
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_BUFFSIZE=1024
export TASK_QUEUE_ENABLE=1
export ASCEND_RT_VISIBLE_DEVICES="$visible_devices"
# NOTE: FLASHCOMM1 not set for A2 — requires tp_size > 1
export TMPDIR=/tmp/ray

mkdir -p "$log_dir"

# ---- vllm serve (official §5.2 A2 PNode parameters) ----
nohup vllm serve "$model_path" \
    --host 0.0.0.0 \
    --port "$port" \
    --data-parallel-size "$dp_size" \
    --data-parallel-rank "$dp_rank" \
    --data-parallel-address "$dp_address" \
    --data-parallel-rpc-port "$dp_rpc_port" \
    --tensor-parallel-size "$tp_size" \
    --enable-expert-parallel \
    --seed 1024 \
    --served-model-name dsv4 \
    --max-model-len 135000 \
    --max-num-batched-tokens 4096 \
    --max-num-seqs 16 \
    --block-size 128 \
    --enforce-eager \
    --async-scheduling \
    --no-disable-hybrid-kv-cache-manager \
    --enable-prefix-caching \
    --trust-remote-code \
    --gpu-memory-utilization 0.90 \
    --quantization ascend \
    --safetensors-load-strategy prefetch \
    --model-loader-extra-config '{"enable_multithread_load": "true", "num_threads": 128}' \
    --tokenizer-mode deepseek_v4 \
    --tool-call-parser deepseek_v4 \
    --enable-auto-tool-choice \
    --reasoning-parser deepseek_v4 \
    --speculative-config '{"num_speculative_tokens": 1, "method": "mtp", "enforce_eager": true}' \
    --additional-config '{"enable_cpu_binding": true, "enable_shared_expert_dp": true}' \
    --kv-transfer-config \
    '{"kv_connector": "MooncakeHybridConnector",
      "kv_role": "kv_producer",
      "kv_port": "'"$((30000 + engine_id))"'",
      "engine_id": "'"$engine_id"'",
      "kv_connector_extra_config": {
          "prefill":  {"dp_size": 8,  "tp_size": 4},
          "decode":   {"dp_size": 4,  "tp_size": 8}
      }}' \
    2>&1 | tee "$log_dir/pnode_${local_ip}_rank${dp_rank}.log" &
