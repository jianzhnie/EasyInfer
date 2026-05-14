#!/usr/bin/env bash
#
# 环境变量配置 — Ray 集群部署、vLLM 推理及 Ascend NPU 相关配置
#
# 用法:
#   source set_env.sh                # 在调用脚本中 source
#   VAR=value source set_env.sh      # 通过环境变量覆盖默认值
#
# 环境变量 (均可外部覆盖):
#   NODES_FILE, SSH_USER_HOST_PREFIX, SSH_OPTS, PARALLELISM
#   IMAGE_NAME, IMAGE_TAR, RUN_CONTAINER_SCRIPT, CONTAINER_NAME
#   NPUS_PER_NODE, MASTER_PORT, DASHBOARD_PORT, WAIT_TIME
#
# 依赖:
#   - 本脚本由 scripts/docker/ 和 scripts/cluster/ 下脚本 source
#   - 不依赖任何外部文件

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ------------------------------------------
# 1. 部署与节点配置
# ------------------------------------------
export NODES_FILE="${NODES_FILE:-/home/jianzhnie/llmtuner/llm/EasyInfer/scripts/node_list.txt}"
export SSH_USER_HOST_PREFIX="${SSH_USER_HOST_PREFIX:-}"
export SSH_OPTS="${SSH_OPTS:--o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10}"
export PARALLELISM="${PARALLELISM:-8}"

# ------------------------------------------
# 2. 容器与镜像配置
# ------------------------------------------
# export IMAGE_NAME="quay.io/ascend/vllm-ascend:v0.18.0rc1-a3-tranformers.5.5.1"
# export IMAGE_TAR="${IMAGE_TAR:-/home/jianzhnie/llmtuner/hfhub/docker/vllm-ascend.v0.18.0rc1-a3.transformers.5.5.1.tar}"
# export RUN_CONTAINER_SCRIPT="${RUN_CONTAINER_SCRIPT:-${SCRIPT_DIR}/ascend_infer_docker_run.sh}"
# export CONTAINER_NAME="${CONTAINER_NAME:-vllm-ascend-0.18-env}"

export IMAGE_NAME="quay.io/ascend/vllm-ascend:v0.18.0rc1-a3-tranformers.5.5.1"
export IMAGE_TAR="${IMAGE_TAR:-/home/jianzhnie/llmtuner/hfhub/docker/vllm-ascend.v0.18.0rc1-a3.transformers.5.5.1.tar}"
export RUN_CONTAINER_SCRIPT="${RUN_CONTAINER_SCRIPT:-${SCRIPT_DIR}/ascend_infer_docker_run.sh}"
export CONTAINER_NAME="${CONTAINER_NAME:-vllm-ascend-0.18-env}"