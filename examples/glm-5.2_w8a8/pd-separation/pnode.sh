#!/bin/bash
# GLM-5.2 W8A8 Prefill 节点 (kv_producer) — 简化版
set -euo pipefail

visible_devices="$1"; port="$2"; dp_size="$3"; dp_rank="$4"
dp_address="$5"; dp_rpc_port="$6"; tp_size="$7"

nic_name="${NIC_NAME}"; local_ip="${LOCAL_IP}"
model_path="${MODEL_PATH}"; log_dir="${LOG_DIR}"
engine_id="${ENGINE_ID:-0}"

unset ftp_proxy https_proxy http_proxy 2>/dev/null || true
rm -rf ~/ascend/log 2>/dev/null || true

export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_IF_IP="$local_ip"
export GLOO_SOCKET_IFNAME="$nic_name"
export TP_SOCKET_IFNAME="$nic_name"
export HCCL_SOCKET_IFNAME="$nic_name"
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_ASCEND_ENABLE_MLAPO=1
export HCCL_BUFFSIZE=1024
export TASK_QUEUE_ENABLE=1
export VLLM_USE_V1=1
export ASCEND_RT_VISIBLE_DEVICES="$visible_devices"
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
export TMPDIR=/tmp/ray

mkdir -p "$log_dir"
log_file="$log_dir/pnode_${local_ip}_rank${dp_rank}.log"

nohup vllm serve "$model_path" \
    --host 0.0.0.0 --port "$port" \
    --data-parallel-size "$dp_size" --data-parallel-rank "$dp_rank" \
    --data-parallel-address "$dp_address" --data-parallel-rpc-port "$dp_rpc_port" \
    --tensor-parallel-size "$tp_size" \
    --enable-expert-parallel --enable-prefix-caching --enable-chunked-prefill \
    --seed 1024 --served-model-name glm-5 --async-scheduling \
    --max-model-len 4096 --max-num-batched-tokens 4096 --max-num-seqs 16 \
    --trust-remote-code --gpu-memory-utilization 0.90 \
    --safetensors-load-strategy prefetch --quantization ascend \
    --enforce-eager --enable-auto-tool-choice \
    --tool-call-parser glm47 --reasoning-parser glm45 \
    --speculative-config '{"num_speculative_tokens": 1, "method": "deepseek_mtp", "enforce_eager": true}' \
    --additional-config '{"enable_cpu_binding": true, "enable_shared_expert_dp": true}' \
    --kv-transfer-config \
    '{"kv_connector": "MultiConnector","kv_role": "kv_producer","kv_load_failure_policy": "recompute",
      "kv_connector_extra_config": {"connectors": [
          {"kv_connector": "MooncakeConnectorV1","kv_role": "kv_producer",
           "kv_port": "'"$((30000 + engine_id))"'",
           "kv_connector_extra_config": {"prefill": {"dp_size": 4, "tp_size": 8},"decode": {"dp_size": 8, "tp_size": 4}}},
          {"kv_connector": "AscendStoreConnector","kv_role": "kv_producer",
           "kv_connector_extra_config": {"lookup_rpc_port": "0","backend": "mooncake"}}
      ]}}' \
    < /dev/null > "$log_file" 2>&1 &
disown
