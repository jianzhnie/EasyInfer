#!/bin/bash
#=============================================================================
# LongCat-Flash-Lite-depth18 vLLM Docker Deployment
#
# 模型: LongCat-Flash-Lite 深度扩展 14→18 层 (copy_source=3,6,9,12)
# 镜像: quay.io/ascend/vllm-ascend:v0.20.2rc1-a3
# 切分: TP=8 (单节点 8 卡 Ascend 910)
#=============================================================================
set -euo pipefail

#=============================================================================
# 配置
#=============================================================================

# --- 模型路径 ---
MODEL_PATH="${MODEL_PATH:-/home/jianzhnie/llmtuner/hfhub/cache/LongCat-Flash-Lite-depth2}"
MODEL_MOUNT="/models/LongCat-Flash-Lite-depth18"

# --- Docker ---
IMAGE="quay.io/ascend/vllm-ascend:v0.20.2rc1-a3"
CONTAINER_NAME="${CONTAINER_NAME:-vllm-longcat-lite-depth18}"

# --- 并行度 ---
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-8}"

# --- 服务配置 ---
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-longcat-flash-lite-depth18}"

# --- 显存/调度 ---
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"

#=============================================================================
# 前置检查
#=============================================================================

if [[ ! -d "$MODEL_PATH" ]]; then
    echo "[ERROR] Model not found: $MODEL_PATH"
    echo "  Run: TARGET_LAYERS=18 COPY_SOURCE='3,6,9,12' \\"
    echo "       bash scripts/expand_longcat_lite_depth.sh"
    exit 2
fi

if ! docker info >/dev/null 2>&1; then
    echo "[ERROR] Docker daemon not running."
    echo "  systemctl start docker"
    exit 1
fi

#=============================================================================
# 停止已有容器
#=============================================================================

if docker ps -a --format '{{.Names}}' \
    | grep -q "^${CONTAINER_NAME}$"; then
    echo "Stopping existing container: ${CONTAINER_NAME}"
    docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
fi

#=============================================================================
# 启动
#=============================================================================

echo "============================================"
echo " LongCat-Flash-Lite-depth18 vLLM Deploy"
echo "============================================"
echo "  Model:     ${MODEL_PATH}"
echo "  Image:     ${IMAGE}"
echo "  TP:        ${TENSOR_PARALLEL_SIZE}"
echo "  Endpoint:  http://${HOST}:${PORT}"
echo "  Max Len:   ${MAX_MODEL_LEN}"
echo "  Container: ${CONTAINER_NAME}"
echo "============================================"

docker run -d \
    --name "${CONTAINER_NAME}" \
    --network host \
    --privileged \
    --device /dev/davinci0 \
    --device /dev/davinci1 \
    --device /dev/davinci2 \
    --device /dev/davinci3 \
    --device /dev/davinci4 \
    --device /dev/davinci5 \
    --device /dev/davinci6 \
    --device /dev/davinci7 \
    --device /dev/davinci_manager \
    --device /dev/hisi_hdc \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v "${MODEL_PATH}:${MODEL_MOUNT}:ro" \
    -e VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}" \
    -e ASCEND_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" \
    "${IMAGE}" \
    vllm serve "${MODEL_MOUNT}" \
        --host "${HOST}" \
        --port "${PORT}" \
        --served-model-name "${SERVED_MODEL_NAME}" \
        --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
        --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
        --max-model-len "${MAX_MODEL_LEN}" \
        --max-num-seqs "${MAX_NUM_SEQS}" \
        --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
        --trust-remote-code \
        --no-enable-prefix-caching \
        --enforce-eager

echo ""
echo "Container started: ${CONTAINER_NAME}"
echo ""
echo "  Logs:  docker logs -f ${CONTAINER_NAME}"
echo "  Stop:  docker rm -f ${CONTAINER_NAME}"
echo "  Test:  curl http://localhost:${PORT}/v1/models"
