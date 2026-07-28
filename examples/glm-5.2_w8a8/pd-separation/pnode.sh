#!/bin/bash
# GLM-5.2 W8A8 Prefill 节点 (kv_producer)
# 对齐已验证的 batch_remote_deploy_pd_seg/glm52-deploy-scripts
set -euo pipefail

visible_devices="$1"; port="$2"; dp_size="$3"; dp_rank="$4"
dp_address="$5"; dp_rpc_port="$6"; tp_size="$7"

nic_name="${NIC_NAME}"; local_ip="${LOCAL_IP}"
model_path="${MODEL_PATH}"; log_dir="${LOG_DIR}"

export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_BUFFSIZE=400
export ASCEND_AGGREGATE_ENABLE=1
export ASCEND_TRANSPORT_PRINT=1
export ACL_OP_INIT_MODE=1
export ASCEND_A3_ENABLE=1
export VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT=480
export ASCEND_RT_VISIBLE_DEVICES=$visible_devices
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
export VLLM_ASCEND_ENABLE_FUSED_MC2=1
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib
export VLLM_ENGINE_ITERATION_TIMEOUT_S=3600
export VLLM_ENGINE_READY_TIMEOUT_S=3600

mkdir -p "$log_dir"
LOG_FILE="glm5_pnode_$(date +%Y%m%d_%H%M%S)_rank${dp_rank}.log"

nohup vllm serve "$model_path" \
    --host 0.0.0.0 --port "$port" \
    --data-parallel-size "$dp_size" --data-parallel-rank "$dp_rank" \
    --data-parallel-address "$dp_address" --data-parallel-rpc-port "$dp_rpc_port" \
    --tensor-parallel-size "$tp_size" \
    --enable-expert-parallel --seed 1024 --served-model-name glm-52 \
    --max-model-len 135168 \
    --speculative-config '{"num_speculative_tokens": 3, "method":"deepseek_mtp"}' \
    --additional-config '{"enable_sparse_c8":false,"fuse_muls_add": true, "multistream_overlap_shared_expert": true, "recompute_scheduler_enable": true, "ascend_compilation_config": {"enable_npugraph_ex": true},"enable_dsa_cp": true}' \
    --max-num-batched-tokens 4096 --trust-remote-code --max-num-seqs 64 \
    --async-scheduling --quantization ascend --gpu-memory-utilization 0.95 \
    --enforce-eager --enable-chunked-prefill --enable-prefix-caching \
    --enable-auto-tool-choice --tool-call-parser glm47 --reasoning-parser glm45 \
    --kv-transfer-config \
    '{"kv_connector": "MooncakeConnector",
    "kv_role": "kv_producer",
    "kv_port": "30000",
    "engine_id": "0",
    "kv_connector_module_path": "vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector",
    "kv_connector_extra_config": {
        "use_ascend_direct": true,
        "prefill": {"dp_size": 4, "tp_size": 8},
        "decode":  {"dp_size": 8, "tp_size": 4}
    }}' > "$log_dir/$LOG_FILE" 2>&1 &
disown
