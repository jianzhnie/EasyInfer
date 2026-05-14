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
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-enp66s0f5}"
export HCCL_SOCKET_IFNAME="${HCCL_SOCKET_IFNAME:-enp66s0f5}"

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
export CONTAINER_NAME="${CONTAINER_NAME:-vllm-ascend-0.18-env}"
export RAY_PORT="${RAY_PORT:-6379}"
export DASHBOARD_PORT="${DASHBOARD_PORT:-8265}"
export MAX_SSH_PARALLELISM="${MAX_SSH_PARALLELISM:-10}"
export VERIFY_TIMEOUT="${VERIFY_TIMEOUT:-120}"
export WAIT_TIME="${WAIT_TIME:-5}"
export NODE_LIST="${NODE_LIST:-}"

# 容器内 set_ray_env.sh 的路径
export RAY_ENV_SCRIPT="${RAY_ENV_SCRIPT:-/workspace/scripts/cluster/set_ray_env.sh}"
