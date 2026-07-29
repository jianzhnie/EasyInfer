#!/bin/bash
# ==============================================================================
# dnode.sh — Decode 节点统一启动模板（所有 DNode 共用此脚本）
# ==============================================================================
# 原始文档为每个 DNode 维护一份 dnode0~3.sh，唯一差异是 local_ip。
# 本脚本将 local_ip 改为从环境变量 LOCAL_IP 读取，
# 由 start_dnode.sh 根据 deploy.conf 自动注入。
#
# 调用方式（由 launch_online_dp.py 自动执行，无需手动调用）：
#   dnode.sh <visible_devices> <port> <dp_size> <dp_rank> <dp_address> <dp_rpc_port> <tp_size>
#
# 依赖的环境变量：
#   LOCAL_IP    — 本机 IP（由 start_dnode.sh 注入）
#   NIC_NAME    — 网卡名（由 start_dnode.sh 注入）
#   MODEL_PATH  — 模型路径（由 start_dnode.sh 注入）
#   LOG_DIR     — 日志目录（由 start_dnode.sh 注入）
# ==============================================================================
set -euo pipefail

# ---- 位置参数 ---------------------------------------------------------------
visible_devices="$1"
port="$2"
dp_size="$3"
dp_rank="$4"
dp_address="$5"
dp_rpc_port="$6"
tp_size="$7"

# ---- 从环境变量读取本机配置 --------------------------------------------------
nic_name="${NIC_NAME}"
local_ip="${LOCAL_IP}"
model_path="${MODEL_PATH}"
log_dir="${LOG_DIR}"

# ---- 环境变量（原始文档完全保留） --------------------------------------------
export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
# A2 实测: 400 报 Get WinSize failed (561002)，需 768
export HCCL_BUFFSIZE=768
export ASCEND_AGGREGATE_ENABLE=1
export ASCEND_TRANSPORT_PRINT=1
export ACL_OP_INIT_MODE=1
export ASCEND_A3_ENABLE=1
# 注意: 不要设 VLLM_VERSION（原始文档为 0.21.0），vllm-ascend 会按它套用旧版
# patch（patch_dp_device_ids），在 vllm 0.23.0rc1 上 AttributeError 崩溃
export TASK_QUEUE_ENABLE=1
export ASCEND_RT_VISIBLE_DEVICES=$visible_devices
# DYNAMIC_EPLB: v0.23.0rc1 下导致 aclnnMoeDistributeDispatchV4 报错 561000（2026-07-29 实测
# 8/8 D rank 在 graph capture 阶段全部崩溃），必须关闭
export DYNAMIC_EPLB=0
# 561000 transport init timeout 对策：放宽 HCCL 建连/执行超时
export HCCL_CONNECT_TIMEOUT=600
export HCCL_EXEC_TIMEOUT=600
# A2 + W8A8 跨节点 EP 实测: FUSED_MC2=1 触发 aclnnDispatchFFNCombine 崩溃，必须关
export VLLM_ASCEND_ENABLE_FUSED_MC2=0
export VLLM_ASCEND_ENABLE_MLAPO=1
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib

export VLLM_ENGINE_ITERATION_TIMEOUT_S=3600
export VLLM_ENGINE_READY_TIMEOUT_S=3600

# ---- 确保日志目录存在 --------------------------------------------------------
mkdir -p "$log_dir"
LOG_FILE="glm5_dnode_$(date +%Y%m%d_%H%M%S)_rank${dp_rank}.log"

# ---- 启动 vLLM serve（Decode / KV Consumer） --------------------------------
# 不用 FULL_DECODE_ONLY：本镜像 kv_consumer + graph capture 触发量化 op 形状错误
# (aclnnAscendQuantV3 161002)，与 1M 方案一致退到 enforce-eager（牺牲 decode 吞吐）
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
    --served-model-name glm-52 \
    --max-model-len 135168 \
    --max-num-batched-tokens 164 \
    --enforce-eager \
    --speculative-config '{"num_speculative_tokens": 3, "method":"deepseek_mtp"}' \
    --additional-config '{"enable_sparse_c8":false,"fuse_muls_add": true, "multistream_overlap_shared_expert": true, "recompute_scheduler_enable": true, "ascend_compilation_config": {"enable_npugraph_ex": true}}' \
    --trust-remote-code \
    --max-num-seqs 48 \
    --gpu-memory-utilization 0.92 \
    --async-scheduling \
    --enable-prefix-caching \
    --quantization ascend \
    --enable-auto-tool-choice \
    --tool-call-parser glm47 \
    --reasoning-parser glm45 \
    --kv-transfer-config \
    '{"kv_connector": "MooncakeConnector",
    "kv_role": "kv_consumer",
    "kv_port": "30100",
    "engine_id": "1",
    "kv_connector_module_path": "vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector",
    "kv_connector_extra_config": {
                "use_ascend_direct": true,
                "prefill": {
                        "dp_size": 4,
                        "tp_size": 8
                },
                "decode": {
                        "dp_size": 8,
                        "tp_size": 4
                }
        }
    }' > $log_dir/${LOG_FILE} 2>&1 &
