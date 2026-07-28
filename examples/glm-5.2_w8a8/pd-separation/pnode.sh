#!/bin/bash
# GLM-5.2 W8A8 Prefill 节点 (kv_producer) — 1M 上下文
# 对齐官方 1M PD 分离：DP4 TP8 PCP1 DCP8, enforce-eager, MTP 1 token
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
export OMP_NUM_THREADS=20
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_BUFFSIZE=768
export ASCEND_AGGREGATE_ENABLE=1
export ASCEND_TRANSPORT_PRINT=1
export ACL_OP_INIT_MODE=1
export ASCEND_A3_ENABLE=1
export VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT=480
export ASCEND_RT_VISIBLE_DEVICES=$visible_devices
# DCP>1 强制要求 FLASHCOMM1=1（DSA CP requires SP，本镜像实测）
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
export VLLM_ASCEND_ENABLE_NZ=1
export VLLM_ASCEND_ENABLE_FUSED_MC2=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TASK_QUEUE_ENABLE=1
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
    --prefill-context-parallel-size 1 \
    --decode-context-parallel-size 8 \
    --cp-kv-cache-interleave-size 128 \
    --enable-expert-parallel --seed 1024 --served-model-name glm-52 \
    --max-model-len 1024000 \
    --speculative-config '{"num_speculative_tokens": 1, "method":"deepseek_mtp","enforce_eager":true}' \
    --additional-config '{"enable_flashcomm1": true, "enable_dsa_cp": true, "ascend_compilation_config": {"enable_npugraph_ex": true, "enable_static_kernel": false}, "fuse_muls_add": true, "multistream_overlap_shared_expert": true, "enable_mc2_hierarchy_comm": false, "enable_sparse_sfa_c8": true, "enable_sparse_li_c8": true, "enable_cpu_binding": true, "recompute_scheduler_enable": false}' \
    --max-num-batched-tokens 16384 --trust-remote-code --max-num-seqs 8 \
    --async-scheduling --quantization ascend --gpu-memory-utilization 0.85 \
    --enforce-eager --enable-chunked-prefill --enable-prefix-caching \
    --enable-auto-tool-choice --tool-call-parser glm47 --reasoning-parser glm45 \
    --kv-transfer-config \
    '{"kv_connector": "MooncakeConnectorV1",
    "kv_role": "kv_producer",
    "kv_port": "30000",
    "engine_id": "0",
    "kv_connector_extra_config": {
        "use_ascend_direct": true,
        "prefill": {"dp_size": 4, "tp_size": 8},
        "decode":  {"dp_size": 4, "tp_size": 8}
    }}' > "$log_dir/$LOG_FILE" 2>&1 &
disown
