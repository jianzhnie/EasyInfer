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
export SSH_USER_HOST_PREFIX="${SSH_USER_HOST_PREFIX:-}"
export SSH_OPTS="${SSH_OPTS:--o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10}"
export PARALLELISM="${PARALLELISM:-8}"

# ------------------------------------------
# 2. 容器与镜像配置
# ------------------------------------------

# 注意: IMAGE_DIR 默认值依赖 $HOME，请按实际环境覆盖:
export IMAGE_DIR="${IMAGE_DIR:-/home/jianzhnie/llmtuner/hfhub/docker/image}"

# --- vLLM-Ascend 镜像配置（当前生效）---
# export IMAGE_NAME="${IMAGE_NAME:-quay.io/ascend/vllm-ascend:v0.22.1rc1-a3}"
# export IMAGE_TAR="${IMAGE_TAR:-${IMAGE_DIR}/vllm-ascend.v0.22.1rc1-a3.tar}"
# export RUN_CONTAINER_SCRIPT="${RUN_CONTAINER_SCRIPT:-${SCRIPT_DIR}/ascend_infer_docker_run.sh}"
# export CONTAINER_NAME="${CONTAINER_NAME:-vllm-ascend-env}"

# --- SGLang 镜像配置（切换到 SGLang 时取消注释下面 4 行，同时注释上面 vLLM 的 4 行）---
export IMAGE_NAME="${IMAGE_NAME:-swr.cn-southwest-2.myhuaweicloud.com/base_image/dockerhub/lmsysorg/sglang:cann9.0.0-a3-B140}"
export IMAGE_TAR="${IMAGE_TAR:-${IMAGE_DIR}/sglang_cann9.0.0-a3-B140.tar.gz}"
export RUN_CONTAINER_SCRIPT="${RUN_CONTAINER_SCRIPT:-${SCRIPT_DIR}/ascend_infer_docker_run.sh}"
export CONTAINER_NAME="${CONTAINER_NAME:-sglang-ascend-env}"
