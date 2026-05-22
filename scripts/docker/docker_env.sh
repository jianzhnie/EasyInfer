#!/bin/bash
#
# 环境变量配置 — Docker 容器管理模块
#
# 用法:
#   source docker_env.sh             # 在调用脚本中 source
#   VAR=value source docker_env.sh   # 通过环境变量覆盖默认值
#
# 环境变量 (均可外部覆盖):
#   NODES_FILE, SSH_USER_HOST_PREFIX, SSH_OPTS, PARALLELISM
#   IMAGE_NAME, IMAGE_TAR, RUN_CONTAINER_SCRIPT, CONTAINER_NAME
#   NPUS_PER_NODE, MASTER_PORT, DASHBOARD_PORT, WAIT_TIME

# 注意: 本文件被 source 而非直接执行，刻意不加 set -euo pipefail，
#       以免影响调用脚本的 shell 选项。

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ------------------------------------------
# 1. 部署与节点配置
# ------------------------------------------
export NODES_FILE="${NODES_FILE:-/llm_workspace_1P/robin/EasyInfer/scripts/node_list.txt}"
export SSH_USER_HOST_PREFIX="${SSH_USER_HOST_PREFIX:-}"
export SSH_OPTS="${SSH_OPTS:--o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10}"
export PARALLELISM="${PARALLELISM:-8}"

# ------------------------------------------
# 2. 容器与镜像配置
# ------------------------------------------

# export IMAGE_NAME="quay.io/ascend/vllm-ascend:v0.18.0rc1-a3-tranformers.5.5.1"
# export IMAGE_TAR="${IMAGE_TAR:-/llm_workspace_1P/robin/hfhub/docker/image/vllm-ascend.v0.18.0rc1-a3.transformers.5.5.1.tar}"
# export RUN_CONTAINER_SCRIPT="${RUN_CONTAINER_SCRIPT:-${SCRIPT_DIR}/ascend_infer_docker_run.sh}"
# export CONTAINER_NAME="${CONTAINER_NAME:-vllm-ascend-0.18-env}"

export IMAGE_NAME="ascend910c-cann8.5.1-torch2.9.0-vllm0.18.0:latest"
export IMAGE_TAR="${IMAGE_TAR:-/llm_workspace_1P/robin/hfhub/docker/image/ascend910c-cann8.5.1-torch2.9.0-vllm0.18.0.tar}"
export RUN_CONTAINER_SCRIPT="${RUN_CONTAINER_SCRIPT:-${SCRIPT_DIR}/ascend_infer_docker_run.sh}"
export CONTAINER_NAME="${CONTAINER_NAME:-npuslim-env}"