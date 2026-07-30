#!/bin/bash
# GLM-5.2 W8A8 共部署节点模板 — 1M 上下文（8/16 节点通用，拓扑由 deploy.conf 决定）
# DP=N TP=8 PCP1 DCP=8 EP=N×8 | rank0 带 API (--api-server-count 1)，其余 --headless
# 由 launch_online_dp.py 调用（参数契约见其 docstring）
#
# 实测调参结论（16 节点 EP=128 三轮验证，2026-07-29）：
#   - util 0.85 上限：0.90 时 chunk prefill 的 MoE dispatch 瞬时 buffer(~2.8G) OOM
#   - batched 8192 上限：EP=128 all2all 元数据随组宽翻倍，16384 chunk dispatch
#     buffer(~5.6G) OOM；8192 与官方 A2 全部配置一致（16384 是官方 A3 128G 卡的值）
#   - max-num-seqs 16：EP=128 权重减半/卡（~6G），KV 池增大，decode 并发提高
#   - HCCL CONNECT/EXEC=600：跨节点 EP all2all；单 rank 故障时 watchdog 600s 收尾
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
# 跨节点 EP all2all 建连/执行超时放宽（561000 transport init timeout 对策）
export HCCL_CONNECT_TIMEOUT=600
export HCCL_EXEC_TIMEOUT=600

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
    --max-num-batched-tokens 8192 --max-num-seqs 16 \
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
