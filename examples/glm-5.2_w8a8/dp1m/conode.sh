#!/bin/bash
# GLM-5.2 W8A8 共部署节点模板 — 1M 上下文
# DP=8 TP=8 PCP1 DCP=8 EP=64 | rank0 带 API (--api-server-count 1)，其余 --headless
# 由 launch_online_dp.py 调用（参数契约见其 docstring）
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
export ACL_OP_INIT_MODE=1
export ASCEND_RT_VISIBLE_DEVICES=$visible_devices
# DCP>1 强制 FLASHCOMM1=1（DSA CP requires SP，本镜像实测）
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
export VLLM_ASCEND_ENABLE_NZ=1
# FUSED_MC2: W8A8 下 aclnnDispatchFFNCombine 崩溃（实测），关闭
export VLLM_ASCEND_ENABLE_FUSED_MC2=0
export VLLM_ASCEND_ENABLE_MLAPO=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TASK_QUEUE_ENABLE=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=3600
export VLLM_ENGINE_READY_TIMEOUT_S=3600

# rank0 提供 API，其余 headless（官方多节点共部署模式）
ROLE_ARGS=()
if [[ "$dp_rank" == "0" ]]; then
    ROLE_ARGS=(--api-server-count 1)
else
    ROLE_ARGS=(--headless)
fi

mkdir -p "$log_dir"
LOG_FILE="glm5_conode_$(date +%Y%m%d_%H%M%S)_rank${dp_rank}.log"

nohup vllm serve "$model_path" \
    --host 0.0.0.0 --port "$port" \
    --data-parallel-size "$dp_size" --data-parallel-start-rank "$dp_rank" \
    --data-parallel-size-local 1 \
    --data-parallel-address "$dp_address" --data-parallel-rpc-port "$dp_rpc_port" \
    --tensor-parallel-size "$tp_size" \
    --prefill-context-parallel-size 1 \
    --decode-context-parallel-size 8 \
    --cp-kv-cache-interleave-size 128 \
    --enable-expert-parallel --seed 1024 --served-model-name glm-5.2 \
    --max-model-len 1024000 \
    --max-num-batched-tokens 16384 --max-num-seqs 8 \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [4, 16, 128]}' \
    --speculative-config '{"num_speculative_tokens": 3, "method": "deepseek_mtp", "enforce_eager": true}' \
    --additional-config '{"enable_flashcomm1": true, "enable_dsa_cp": true, "enable_balance_scheduling": true, "ascend_compilation_config": {"enable_npugraph_ex": true, "enable_static_kernel": false}, "fuse_muls_add": true, "multistream_overlap_shared_expert": true, "enable_mc2_hierarchy_comm": false, "enable_sparse_sfa_c8": true, "enable_sparse_li_c8": true, "enable_cpu_binding": true, "recompute_scheduler_enable": false}' \
    --trust-remote-code --quantization ascend \
    --gpu-memory-utilization 0.85 \
    --enable-chunked-prefill --enable-prefix-caching --async-scheduling \
    --enable-auto-tool-choice --tool-call-parser glm47 --reasoning-parser glm45 \
    --safetensors-load-strategy prefetch \
    "${ROLE_ARGS[@]}" \
    > "$log_dir/$LOG_FILE" 2>&1 &
disown
