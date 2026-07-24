#!/bin/bash
# ==============================================================================
# pnode.sh — Prefill 节点统一启动模板 (aligned with official GLM-5.2 PD docs)
# ==============================================================================
# 由 launch_online_dp.py 自动执行，无需手动调用。
#
# 调用方式:
#   pnode.sh <visible_devices> <port> <dp_size> <dp_rank> <dp_address> <dp_rpc_port> <tp_size>
#
# 依赖环境变量 (由 start_pnode.sh 注入):
#   LOCAL_IP, NIC_NAME, MODEL_PATH, LOG_DIR
#
# Reference:
#   https://docs.vllm.ai/projects/ascend/zh-cn/latest/tutorials/models/GLM5.2.html#prefill-decode
# ==============================================================================
set -euo pipefail

# ---- 位置参数 ---------------------------------------------------------------
visible_devices="$1"; port="$2"; dp_size="$3"; dp_rank="$4"
dp_address="$5"; dp_rpc_port="$6"; tp_size="$7"

# ---- 从环境变量读取本机配置 --------------------------------------------------
nic_name="${NIC_NAME}"
local_ip="${LOCAL_IP}"
model_path="${MODEL_PATH}"
log_dir="${LOG_DIR}"

# ---- 环境变量 (official GLM-5.2 PD docs) ------------------------------------
export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_BUFFSIZE=400
export ACL_OP_INIT_MODE=1
export ASCEND_A3_ENABLE=1
export ASCEND_RT_VISIBLE_DEVICES=$visible_devices
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
export VLLM_ASCEND_ENABLE_FUSED_MC2=1
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/lib"
export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-3600}"

# ---- 确保日志目录存在 --------------------------------------------------------
mkdir -p "$log_dir"
LOG_FILE="${log_dir}/glm5_prefill_$(date +%Y%m%d_%H%M%S)_rank${dp_rank}.log"

# ---- vLLM serve (Prefill / KV Producer, official config) --------------------
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
    --speculative-config '{"num_speculative_tokens": 1, "method": "deepseek_mtp", "enforce_eager": true}' \
    --additional-config '{"recompute_scheduler_enable": false, "multistream_overlap_shared_expert": true, "enable_dsa_cp": true, "enable_sparse_sfa_c8": false, "enable_sparse_li_c8": true, "c8_enable_reshape_optim": false}' \
    --max-num-batched-tokens 8192 \
    --trust-remote-code \
    --max-num-seqs 64 \
    --quantization ascend \
    --gpu-memory-utilization 0.92 \
    --enforce-eager \
    --enable-auto-tool-choice \
    --tool-call-parser glm47 \
    --reasoning-parser glm45 \
    --kv-transfer-config \
    '{"kv_connector": "MooncakeConnectorV1",
    "kv_role": "kv_producer",
    "kv_port": "30000",
    "engine_id": "0",
    "kv_connector_extra_config": {
                "use_ascend_direct": true,
                "prefill": {"dp_size": 4, "tp_size": 8},
                "decode":  {"dp_size": 8, "tp_size": 4}
        }
    }' > "$LOG_FILE" 2>&1 &
