#!/bin/bash
#
# Ascend NPU 训练容器启动脚本 (MindSpeed-LLM)
# 用法: bash ascend_train_docker_run.sh
# 通过环境变量覆盖: IMAGE_NAME=... CONTAINER_NAME=... bash ascend_train_docker_run.sh

set -euo pipefail

# Configuration
IMAGE_NAME="${IMAGE_NAME:-swr.cn-south-1.myhuaweicloud.com/ascendhub/mindspeed-llm:26.0.0-a3-openeuler24.03-py3.11-aarch64}"
CONTAINER_NAME="${CONTAINER_NAME:-mindspeed-llm-26-env}"

# Check if container exists
if [[ -n "$(docker ps -aq -f name="^/${CONTAINER_NAME}$")" ]]; then
    echo "Container '${CONTAINER_NAME}' already exists. Removing it..."
    docker rm -f "${CONTAINER_NAME}"
fi

# Run Docker container
docker run -d \
    -u root \
    --name "${CONTAINER_NAME}" \
    --ipc=host \
    --net=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --privileged=true \
    --device=/dev/davinci0 \
    --device=/dev/davinci1 \
    --device=/dev/davinci2 \
    --device=/dev/davinci3 \
    --device=/dev/davinci4 \
    --device=/dev/davinci5 \
    --device=/dev/davinci6 \
    --device=/dev/davinci7 \
    --device=/dev/davinci_manager \
    --device=/dev/devmm_svm \
    --device=/dev/hisi_hdc \
    --shm-size=256g \
    -e HCCL_BUFFSIZE=1024 \
    -e HCCL_BUFFER_FILE_SIZE=1024 \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /usr/local/Ascend/add-ons/:/usr/local/Ascend/add-ons/ \
    -v /usr/local/Ascend/driver/tools/hccn_tool:/usr/local/Ascend/driver/tools/hccn_tool \
    -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v /root/.cache:/root/.cache \
    -v /home/jianzhnie/llmtuner:/home/jianzhnie/llmtuner:rw \
    -v /root/.ssh:/root/.ssh \
    -it "${IMAGE_NAME}" \
    /bin/bash -c "while true; do sleep 1000; done"
