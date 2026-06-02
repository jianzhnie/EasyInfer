#!/bin/bash
#
# set_ray_env.sh — Ray 集群环境变量与配置
#
# 用法:
#   source set_ray_env.sh            # 加载所有环境变量与默认配置
#
# Section 1-4 为固定运行时变量，Section 5 的集群配置均支持外部覆盖

# -----------------------------------------------------------------
# 1. Ray / vLLM 核心
# -----------------------------------------------------------------
export VLLM_USE_V1=1
export RESUME_MODE_ENABLE=1
export ASCEND_GLOBAL_LOG_LEVEL=3
export HCCL_ASYNC_ERROR_HANDLING=0
export HCCL_WHITELIST_DISABLE=1

# -----------------------------------------------------------------
# 2. 网络接口
# -----------------------------------------------------------------

export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-enp66s0f0}"
export HCCL_SOCKET_IFNAME="${HCCL_SOCKET_IFNAME:-enp66s0f0}"

# 自动获取业务网 IP 并设置 VLLM_HOST_IP，确保与 Ray 资源标签一致
if [[ -n "${HCCL_SOCKET_IFNAME:-}" ]]; then
    # 使用 || true 避免在 set -eo pipefail 环境下因为网卡不存在而退出
    _HOST_IP=$(ip -4 addr show "${HCCL_SOCKET_IFNAME}" 2>/dev/null | awk '/inet / {print $2}' | cut -d/ -f1 | head -n 1 || true)
    if [[ -n "$_HOST_IP" ]]; then
        export VLLM_HOST_IP="${VLLM_HOST_IP:-$_HOST_IP}"
    fi
fi

# -----------------------------------------------------------------
# 3. 性能调优
# -----------------------------------------------------------------
export TTP_OT=360
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_CONNECT_TIMEOUT=3600
export HCCL_BUFFSIZE=2000
export TASK_QUEUE_ENABLE=1
export NPU_ASD_ENABLE=0
export STREAMS_PER_DEVICE=32
export HCCL_OP_BASE_FFTS_MODE=TRUE
export HCCL_ALGO="alltoall=level0:NA;level1:pipeline"

# -----------------------------------------------------------------
# 4. NPU / Ascend 设备
# -----------------------------------------------------------------
export NPUS_PER_NODE="${NPUS_PER_NODE:-8}"
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

# -----------------------------------------------------------------
# 5. 集群操作配置（可通过 CLI 参数覆盖）
# -----------------------------------------------------------------
export RAY_PORT="${RAY_PORT:-6379}"
export WAIT_TIME="${WAIT_TIME:-5}"
export VERIFY_TIMEOUT="${VERIFY_TIMEOUT:-120}"
export MAX_SSH_PARALLELISM="${MAX_SSH_PARALLELISM:-10}"
export CONTAINER_NAME="${CONTAINER_NAME:-npuslim-env}"

# 项目根目录（通过 BASH_SOURCE 自动推导，适配不同部署路径）
_RAY_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export NODE_LIST="${NODE_LIST:-${_RAY_SCRIPT_DIR}/../node_list.txt}"
export RAY_ENV_SCRIPT="${RAY_ENV_SCRIPT:-${_RAY_SCRIPT_DIR}/set_ray_env.sh}"

# ------------------------------------------
# Ascend NPU 与底层环境配置
# ------------------------------------------

# 加载 Ascend Toolkit 环境
if [[ -f "/usr/local/Ascend/ascend-toolkit/set_env.sh" ]]; then
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi

# 加载 ATB 环境（如果存在）
if [[ -f "/usr/local/Ascend/nnal/atb/set_env.sh" ]]; then
    source /usr/local/Ascend/nnal/atb/set_env.sh
fi
