#!/bin/bash
# Run vLLM-Ascend Docker Container
# Usage: bash run_container.sh [CARDS] [--multi-node] [--npuslim[=/path]] [--daemon]
#   CARDS: physical card(s) to use, comma-separated (default: 0)
#          e.g. 0,1 means card 0+1 -> chips 0,1,2,3
#   --multi-node: enable multi-node distributed inference mode (all cards)
#   --npuslim: mount npuslim source and auto install in editable mode
#   --npuslim=/path: mount npuslim source from custom directory
#   --daemon: run in background (detached), survives exit

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Version info
CANN_VER="8.5.1"
TORCH_VER="2.9.0"
VLLM_VER="0.18.0"
CHIP_NAME="910c"

IMAGE_NAME="ascend${CHIP_NAME}-cann${CANN_VER}-torch${TORCH_VER}-vllm${VLLM_VER}"
IMAGE_NAME=$(echo "${IMAGE_NAME}" | tr '[:upper:]' '[:lower:]')

# Parse args
CARDS="0"
MULTI_NODE=false
WITH_NPUSLIM=false
NPUSLIM_SRC_PATH=""
DAEMON=false
CONTAINER_NAME="npuslim-env"


while [[ $# -gt 0 ]]; do
    case $1 in
        --multi-node)
            MULTI_NODE=true
            shift
            ;;
        --npuslim=*)
            WITH_NPUSLIM=true
            NPUSLIM_SRC_PATH="${1#--npuslim=}"
            shift
            ;;
        --npuslim)
            WITH_NPUSLIM=true
            shift
            ;;
        --daemon)
            DAEMON=true
            shift
            ;;
        [0-3])
            CARDS="$1"
            shift
            ;;
        [0-3],[0-3]*)
            CARDS="$1"
            shift
            ;;
        *)
            shift
            ;;
    esac
done

# Expand physical cards into chip ids (1 physical card = 2 consecutive chips)
CHIP_LIST=()
VISIBLE_DEVICES=()
IFS=',' read -ra CARD_ARRAY <<< "$CARDS"
for cid in "${CARD_ARRAY[@]}"; do
    c0=$((cid * 2))
    c1=$((c0 + 1))
    CHIP_LIST+=("$c0" "$c1")
    VISIBLE_DEVICES+=("$c0" "$c1")
done

echo "========================================"
echo "Running vLLM-Ascend Container"
echo "========================================"
echo "Image:      ${IMAGE_NAME}"
echo "NPUSlim:    ${WITH_NPUSLIM}"
echo "Multi-node: ${MULTI_NODE}"
echo "Daemon:     ${DAEMON}"
if [[ "$MULTI_NODE" == false ]]; then
    echo "Cards:      ${CARDS} -> chips ${VISIBLE_DEVICES[*]}"
fi
echo ""

# Check image
if ! docker image inspect "${IMAGE_NAME}" &>/dev/null; then
    echo "ERROR: Image not found: ${IMAGE_NAME}"
    exit 1
fi

# Check if container exists
if [[ -n "$(docker ps -aq -f name="^/${CONTAINER_NAME}$")" ]]; then
    echo "Container '${CONTAINER_NAME}' already exists. Removing it..."
    docker rm -f "${CONTAINER_NAME}"
fi

DOCKER_ARGS=(
    -it --rm
    --shm-size=10g
)

if [[ "$MULTI_NODE" == true ]]; then
    # ========== Multi-node mode ==========
    echo "Multi-node mode: using host network, all NPUs"
    echo ""

    DOCKER_ARGS+=(--net=host)

    # All NPU devices
    for i in {0..7}; do
        DOCKER_ARGS+=("--device=/dev/davinci${i}")
    done
    DOCKER_ARGS+=(
        --device=/dev/davinci_manager
        --device=/dev/devmm_svm
        --device=/dev/hisi_hdc
    )

    # Multi-node required mounts
    DOCKER_ARGS+=(
        -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro
        -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware:ro
        -v /usr/local/dcmi:/usr/local/dcmi:ro
        -v /usr/local/Ascend/driver/tools/hccn_tool:/usr/local/Ascend/driver/tools/hccn_tool:ro
        -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info:ro
        -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi:ro
        -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro
        -v /etc/ascend_install.info:/etc/ascend_install.info:ro
        -v /etc/hccn.conf:/etc/hccn.conf:ro
        -v /var/log/npu:/var/log/npu
    )

    # Detect network interface
    NIC_NAME="${NIC_NAME:-$(ip route | grep default | awk '{print $5}' | head -1)}"
    LOCAL_IP="${LOCAL_IP:-$(ip addr show "$NIC_NAME" | grep 'inet ' | awk '{print $2}' | cut -d/ -f1)}"

    echo "Detected NIC: ${NIC_NAME}"
    echo "Detected IP:  ${LOCAL_IP}"
    echo ""

    # Multi-node communication env vars
    # shellcheck disable=SC2054
    DOCKER_ARGS+=(
        -e HCCL_IF_IP="${LOCAL_IP}"
        -e GLOO_SOCKET_IFNAME="${NIC_NAME}"
        -e TP_SOCKET_IFNAME="${NIC_NAME}"
        -e HCCL_SOCKET_IFNAME="${NIC_NAME}"
        -e ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    )
else
    # ========== Single/multi-card mode ==========
    for chip in "${CHIP_LIST[@]}"; do
        DOCKER_ARGS+=("--device=/dev/davinci${chip}")
    done
    DOCKER_ARGS+=(
        --device=/dev/davinci_manager
        --device=/dev/devmm_svm
        --device=/dev/hisi_hdc
        -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro
        -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware:ro
        -v /usr/local/dcmi:/usr/local/dcmi:ro
        -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi:ro
        -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro
        -v /etc/ascend_install.info:/etc/ascend_install.info:ro
        -v /var/log/npu:/var/log/npu
        -e "ASCEND_RT_VISIBLE_DEVICES=$(IFS=,; echo "${VISIBLE_DEVICES[*]}")"
    )
fi

# Common mounts
DOCKER_ARGS+=(
    --name "${CONTAINER_NAME}" \
    -v ~/.cache/huggingface:/root/.cache/huggingface
    -v ~/.cache/modelscope:/root/.cache/modelscope
    -v /llm_workspace_1P/robin:/llm_workspace_1P/robin
    -v /home/jianzhnie/llmtuner:/home/jianzhnie/llmtuner
    -e HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
)

INSIDE_CMD=""

# NPUSlim source mount + editable install
if [[ "$WITH_NPUSLIM" == true ]]; then
    if [[ -z "$NPUSLIM_SRC_PATH" ]]; then
        echo "ERROR: --npuslim requires a path, e.g. --npuslim=/path/to/npuslim"
        exit 1
    fi

    if [ ! -d "$NPUSLIM_SRC_PATH" ] || [ ! -f "$NPUSLIM_SRC_PATH/pyproject.toml" ]; then
        echo "ERROR: NPUSlim source not found or invalid: $NPUSLIM_SRC_PATH"
        echo "Hint: use --npuslim=/path/to/npuslim"
        exit 1
    fi

    echo "NPUSlim source: ${NPUSLIM_SRC_PATH}"
    echo "  (mounted to /workspace/npuslim, editable install on start)"

    DOCKER_ARGS+=(-v "${NPUSLIM_SRC_PATH}:/workspace/npuslim:rw")
    # Clean stale CMake build artifacts that break setuptools package discovery,
    # 清理 CMake 构建产物 + 跳过算子编译，然后 editable 安装
    INSIDE_CMD="git config --global --add safe.directory '*'; "
    INSIDE_CMD+="NPUSLIM_SKIP_OPS=1 pip install --no-build-isolation --no-deps --root-user-action=ignore -e /workspace/npuslim -v;"
fi

if [[ "$DAEMON" == true ]]; then
    # Daemon mode: rebuild args (-d instead of -it --rm)
    DAEMON_ARGS=(-d)
    for arg in "${DOCKER_ARGS[@]}"; do
        case $arg in -it|--rm) ;; *) DAEMON_ARGS+=("$arg") ;; esac
    done
    CONTAINER_ID=$(docker run "${DAEMON_ARGS[@]}" "${IMAGE_NAME}" \
        /bin/bash -lc "${INSIDE_CMD}sleep infinity")
    echo "Container: ${CONTAINER_ID}"
    echo ""
    echo "To enter:  docker exec -it ${CONTAINER_ID:0:12} bash"
    echo "To stop:   docker stop ${CONTAINER_ID:0:12}"
else
    # Interactive mode
    if [[ -n "$INSIDE_CMD" ]]; then
        exec docker run "${DOCKER_ARGS[@]}" "${IMAGE_NAME}" \
            /bin/bash -lc "${INSIDE_CMD}exec /bin/bash"
    else
        exec docker run "${DOCKER_ARGS[@]}" "${IMAGE_NAME}" /bin/bash
    fi
fi
