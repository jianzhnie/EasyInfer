#!/bin/bash

# Configuration
IMAGE_NAME="cis-pengcheng.cmecloud.cn/ascendhub/mindspeed-llm:openeuler22.03-mindspeed-llm-2.3.0-a3-arm"
CONTAINER_NAME="mindspeed-llm-env"

# Check if container exists
if [ "$(docker ps -aq -f name=^/${CONTAINER_NAME}$)" ]; then
    echo "Container '${CONTAINER_NAME}' already exists. Removing it..."
    docker rm -f ${CONTAINER_NAME}
fi

# Run Docker container
# Improvements:
# 1. Added --ulimit memlock=-1 and stack=67108864 for Ascend NPU performance
# 2. Added --shm-size just in case IPC isn't sufficient (though ipc=host usually covers it)
# 3. Consolidated mounts where possible (kept originals for safety if symlinks exist)
# 4. Added conflict check above

docker run -d \
    -u root \
    --name ${CONTAINER_NAME} \
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
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /usr/local/Ascend/add-ons/:/usr/local/Ascend/add-ons/ \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/Ascend/driver/tools/hccn_tool:/usr/local/Ascend/driver/tools/hccn_tool \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v /root/.cache:/root/.cache \
    -v /llm_workspace_1P:/llm_workspace_1P:rw \
    -v /root/.ssh:/root/.ssh \
    -it ${IMAGE_NAME} \
    /bin/bash -c "while true; do sleep 1000; done"
