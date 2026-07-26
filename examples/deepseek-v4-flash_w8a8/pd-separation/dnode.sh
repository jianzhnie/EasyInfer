#!/bin/bash
# ==============================================================================
# dnode.sh — DeepSeek-V4-Flash Decode 节点模板 (kv_consumer)
# ==============================================================================
# 官方 §5.2 A2: 4×DNode, 总 DP=32 TP=1, 每节点 DP=8
# MooncakeHybridConnector, no-prefix-caching, recompute_scheduler
#
# 调用方式（由 launch_online_dp.py 自动执行）:
#   dnode.sh <devices> <port> <dp_size> <dp_rank> <dp_addr> <dp_rpc> <tp_size>
#
# 依赖环境变量: LOCAL_IP, NIC_NAME, MODEL_PATH, LOG_DIR
# ==============================================================================
set -euo pipefail

visible_devices="$1"; port="$2"; dp_size="$3"; dp_rank="$4"
dp_address="$5"; dp_rpc_port="$6"; tp_size="$7"

nic_name="${NIC_NAME}"; local_ip="${LOCAL_IP}"
model_path="${MODEL_PATH}"; log_dir="${LOG_DIR}"

export LD_PRELOAD="/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:${LD_PRELOAD:-}"

# ---- NPU environment (official §5.2) ----
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

mkdir -p "$log_dir"

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
    --max-num-batched-tokens 60 \
    --max-num-seqs 30 \
    --block-size 128 \
    --async-scheduling \
    --no-disable-hybrid-kv-cache-manager \
    --no-enable-prefix-caching \
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
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
    --additional-config '{
        "enable_cpu_binding": true,
        "multistream_overlap_shared_expert": true,
        "recompute_scheduler_enable": true,
        "ascend_compilation_config": {"enable_npugraph_ex": true, "enable_static_kernel": false}
    }' \
    --kv-transfer-config \
    '{"kv_connector": "MooncakeHybridConnector",
      "kv_role": "kv_consumer",
      "kv_port": "30400",
      "engine_id": "4",
      "kv_connector_extra_config": {
          "prefill":  {"dp_size": 8,  "tp_size": 1},
          "decode":   {"dp_size": 32, "tp_size": 1}
      }}' \
    2>&1 | tee "$log_dir/dnode_${local_ip}_rank${dp_rank}.log" &
