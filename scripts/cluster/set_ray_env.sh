#!/bin/bash
#
# set_ray_env.sh - Consolidates Ray environment variables and startup logic.
#

# -----------------------------------------------------------------
# 1. Ray Environment Variables
# -----------------------------------------------------------------
export VLLM_USE_V1=1
export RESUME_MODE_ENABLE=1
export ASCEND_GLOBAL_LOG_LEVEL=3
export HCCL_ASYNC_ERROR_HANDLING=0
export HCCL_WHITELIST_DISABLE=1

# Network interface configuration
# Default to enp66s0f5, but can be overridden
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-enp66s0f5}"
export HCCL_SOCKET_IFNAME="${HCCL_SOCKET_IFNAME:-enp66s0f5}"

# Performance & Stability
export TTP_OT=360
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_CONNECT_TIMEOUT=3600
export HCCL_BUFFSIZE=2000
export TASK_QUEUE_ENABLE=1
export NPU_ASD_ENABLE=0
export STREAMS_PER_DEVICE=32
export HCCL_OP_BASE_FFTS_MODE=TRUE
# HCCL algorithm configuration
export HCCL_ALGO="alltoall=level0:NA;level1:pipeline"

# Ray ASCEND RT visible devices
export NPUS_PER_NODE="${NPUS_PER_NODE:-8}"
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

# Cluster Defaults
export CONTAINER_NAME="${CONTAINER_NAME:-vllm-ascend-0.18-env}"
export MAX_SSH_PARALLELISM="${PARALLELISM:-10}"
export NODE_LIST="${NODES_FILE:-/home/jianzhnie/llmtuner/llm/EasyInfer/scripts/node_list.txt}"
export RAY_PORT="${RAY_PORT:-6379}"