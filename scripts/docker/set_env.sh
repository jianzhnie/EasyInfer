#!/usr/bin/env bash

# ==========================================
# 环境变量配置 (set_env.sh)
# 包含 Ray 集群部署、vLLM 推理及 Ascend NPU 的相关配置
# ==========================================

# ------------------------------------------
# 1. 部署与节点配置
# ------------------------------------------
export NODES_FILE="${NODES_FILE:-/home/jianzhnie/llmtuner/tools/ip_list.txt}"
export SSH_USER_HOST_PREFIX="${SSH_USER_HOST_PREFIX:-}"
export SSH_OPTS="${SSH_OPTS:--o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10}"
export PARALLELISM="${PARALLELISM:-8}"

# ------------------------------------------
# 2. 容器与镜像配置
# ------------------------------------------
# 训练镜像
# export IMAGE_NAME="cis-pengcheng.cmecloud.cn/ascendhub/mindspeed-llm:openeuler22.03-mindspeed-llm-2.3.0-a3-arm"
# export IMAGE_TAR="${IMAGE_TAR:-/llm_workspace_1P/robin/hfhub/docker/image/mindspeed-llm-2.3.0-a3-arm.tar}"
# export RUN_CONTAINER_SCRIPT="${RUN_CONTAINER_SCRIPT:-${SCRIPT_DIR}/ascend_train_docker_run.sh}"
# export CONTAINER_NAME="${CONTAINER_NAME:-mindspeed-llm-env}"

# 推理镜像
# export IMAGE_NAME="ascend910c-cann8.5.1-torch2.9.0-vllm0.18.0:latest"
# export IMAGE_TAR="${IMAGE_TAR:-/llm_workspace_1P/robin/hfhub/docker/image/ascend910c-cann8.5.1-torch2.9.0-vllm0.18.0.tar}"
# export RUN_CONTAINER_SCRIPT="${RUN_CONTAINER_SCRIPT:-${SCRIPT_DIR}/ascend_infer_docker_run.sh}"
# export CONTAINER_NAME="${CONTAINER_NAME:-vllm-ascend-npuslim-env}"

# 推理镜像
export IMAGE_NAME="quay.io/ascend/vllm-ascend:v0.18.0rc1-a3-tranformers.5.5.1"
export IMAGE_TAR="${IMAGE_TAR:-/home/jianzhnie/llmtuner/hfhub/docker/vllm-ascend.v0.18.0rc1-a3.transformers.5.5.1.tar}"
export RUN_CONTAINER_SCRIPT="${RUN_CONTAINER_SCRIPT:-${SCRIPT_DIR}/ascend_infer_docker_run.sh}"
export CONTAINER_NAME="${CONTAINER_NAME:-vllm-ascend-0.18-env}"