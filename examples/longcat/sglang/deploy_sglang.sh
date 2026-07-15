#!/bin/bash
#=============================================================================
# LongCat-Flash-Chat SGLang 集群部署脚本
#=============================================================================
# 一键完成: 启动容器 → 分发脚本 → 启动 SGLang 服务
#
# 用法:
#   bash deploy_sglang.sh                    # 使用默认配置
#   bash deploy_sglang.sh --file nodes.txt   # 指定节点列表
#   MODEL_PATH=/path/to/model bash deploy_sglang.sh
#
# 环境变量:
#   NODES_FILE, MODEL_PATH, CONTAINER_NAME, IMAGE_NAME, IMAGE_TAR
#=============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EASYINFER_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

# --- 加载共享工具函数 ---
source "${EASYINFER_ROOT}/scripts/common.sh"

#=============================================================================
# 配置与参数解析
#=============================================================================
NODES_FILE="${NODES_FILE:-${EASYINFER_ROOT}/node_list1.txt}"
MODEL_PATH="${MODEL_PATH:-/home/jianzhnie/llmtuner/hfhub/models/meituan-longcat/LongCat-Flash-Chat}"

# --- Docker 配置 ---
CONTAINER_NAME="${CONTAINER_NAME:-sglang-ascend-env}"
IMAGE_NAME="${IMAGE_NAME:-swr.cn-southwest-2.myhuaweicloud.com/base_image/dockerhub/lmsysorg/sglang:cann9.0.0-a3-B140}"
IMAGE_TAR="${IMAGE_TAR:-/home/jianzhnie/llmtuner/hfhub/docker/image/sglang_cann9.0.0-a3-B140.tar.gz}"

# --- SGLang 配置 ---
TP_SIZE="${TP_SIZE:-64}"
SERVER_PORT="${SERVER_PORT:-6677}"
MASTER_PORT="${MASTER_PORT:-5000}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-longcat-flash}"

# --- 脚本路径 ---
DOCKER_ENV_SH="${EASYINFER_ROOT}/scripts/docker/docker_env.sh"
MANAGE_CONTAINERS_SH="${EASYINFER_ROOT}/scripts/docker/manage_docker_containers.sh"
COPY_TO_CONTAINER_SH="${EASYINFER_ROOT}/scripts/docker/copy_file_to_containers.sh"
RUN_SGLANG_SH="${EASYINFER_ROOT}/examples/longcat/sglang/run_sglang.sh"

#=============================================================================
# 帮助信息
#=============================================================================
usage() {
    cat <<'USAGE'
Usage:
  bash deploy_sglang.sh [OPTIONS]

Options:
  -h, --help                显示帮助信息
  -f, --file <FILE>         节点列表文件路径
  -m, --model-path <PATH>   模型路径
  --skip-containers         跳过容器启动步骤
  --skip-copy               跳过文件复制步骤
  --skip-start              跳过服务启动步骤

环境变量:
  NODES_FILE, MODEL_PATH, CONTAINER_NAME, IMAGE_NAME, IMAGE_TAR
  TP_SIZE, SERVER_PORT, MASTER_PORT, SERVED_MODEL_NAME
USAGE
}

#=============================================================================
# 参数解析
#=============================================================================
SKIP_CONTAINERS=false
SKIP_COPY=false
SKIP_START=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        -f|--file)
            NODES_FILE="${2:?错误: $1 需要一个参数}"
            shift 2 ;;
        -m|--model-path)
            MODEL_PATH="${2:?错误: $1 需要一个参数}"
            shift 2 ;;
        --skip-containers) SKIP_CONTAINERS=true; shift ;;
        --skip-copy) SKIP_COPY=true; shift ;;
        --skip-start) SKIP_START=true; shift ;;
        *) log_err "未知参数: $1"; usage; exit 2 ;;
    esac
done

#=============================================================================
# 前置检查
#=============================================================================
[[ -f "$NODES_FILE" ]] || log_fatal "节点列表文件未找到: $NODES_FILE"
[[ -d "$MODEL_PATH" ]] || log_fatal "模型路径不存在: $MODEL_PATH"
[[ -f "$RUN_SGLANG_SH" ]] || log_fatal "SGLang 启动脚本未找到: $RUN_SGLANG_SH"

# --- 读取节点列表 ---
NODES=()
while IFS= read -r line; do
    [[ -n "$line" ]] && NODES+=("$line")
done < <(read_nodes "$NODES_FILE")

NNODES=${#NODES[@]}
[[ "$NNODES" -gt 0 ]] || log_fatal "节点列表为空: $NODES_FILE"

MASTER_NODE="${NODES[0]}"

log_info "============================================"
log_info " SGLang 集群部署"
log_info "============================================"
log_info " 节点列表:       ${NODES_FILE}"
log_info " 节点数:         ${NNODES}"
log_info " 主节点:         ${MASTER_NODE}"
log_info " 模型路径:       ${MODEL_PATH}"
log_info " TP 大小:        ${TP_SIZE}"
log_info " 服务端口:       ${SERVER_PORT}"
log_info " 容器名称:       ${CONTAINER_NAME}"
log_info "============================================"

#=============================================================================
# Step 1: 启动容器
#=============================================================================
if ! $SKIP_CONTAINERS; then
    log_info "【Step 1/3】启动 Docker 容器..."
    
    # 确保 docker_env.sh 配置正确
    if [[ -f "$DOCKER_ENV_SH" ]]; then
        sed -i "s|^export CONTAINER_NAME=.*|export CONTAINER_NAME=\"${CONTAINER_NAME}\"|" "$DOCKER_ENV_SH"
        sed -i "s|^export IMAGE_NAME=.*|export IMAGE_NAME=\"${IMAGE_NAME}\"|" "$DOCKER_ENV_SH"
        sed -i "s|^export IMAGE_TAR=.*|export IMAGE_TAR=\"${IMAGE_TAR}\"|" "$DOCKER_ENV_SH"
    fi
    
    bash "${MANAGE_CONTAINERS_SH}" restart --file "$NODES_FILE"
    
    log_info "容器启动完成，等待 5 秒..."
    sleep 5
else
    log_info "【Step 1/3】跳过容器启动"
fi

#=============================================================================
# Step 2: 分发脚本到容器
#=============================================================================
if ! $SKIP_COPY; then
    log_info "【Step 2/3】分发 SGLang 启动脚本到容器..."
    
    for node in "${NODES[@]}"; do
        log_info "复制到 ${node}..."
        
        temp_file="/tmp/run_sglang.sh.$$.$(date +%s%N)"
        scp ${SSH_OPTS:--o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10} \
            "$RUN_SGLANG_SH" "${node}:${temp_file}" 2>/dev/null || {
            log_err "复制到 ${node} 失败"
            continue
        }
        
        ssh ${SSH_OPTS:--o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10} \
            "$node" "docker cp '${temp_file}' '${CONTAINER_NAME}:/run_sglang.sh' && rm -f '${temp_file}' && docker exec '${CONTAINER_NAME}' chmod +x /run_sglang.sh" || {
            log_err "docker cp 到 ${node} 失败"
            continue
        }
    done
    
    log_info "脚本分发完成"
else
    log_info "【Step 2/3】跳过文件复制"
fi

#=============================================================================
# Step 3: 启动 SGLang 服务
#=============================================================================
if ! $SKIP_START; then
    log_info "【Step 3/3】启动 SGLang 服务..."
    
    # 构建启动命令 (使用 NODES_IPS 环境变量传入节点列表)
    NODES_IPS_STR="${NODES[*]}"
    SGLANG_CMD="NODES_IPS='${NODES_IPS_STR}' MODEL_PATH=${MODEL_PATH} TP_SIZE=${TP_SIZE} SERVER_PORT=${SERVER_PORT} MASTER_PORT=${MASTER_PORT} SERVED_MODEL_NAME=${SERVED_MODEL_NAME} bash /run_sglang.sh"
    
    for node in "${NODES[@]}"; do
        log_info "在 ${node} 上启动 SGLang..."
        
        ssh ${SSH_OPTS:--o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10} \
            "$node" "docker exec -d '${CONTAINER_NAME}' bash -c '${SGLANG_CMD}'" 2>/dev/null || {
            log_err "在 ${node} 上启动 SGLang 失败"
        }
    done
    
    log_info "SGLang 服务启动命令已发送"
    log_info "等待服务就绪..."
    
    # 等待主节点服务就绪
    if command -v nc >/dev/null 2>&1; then
        wait_for_port "$MASTER_NODE" "$SERVER_PORT" 300 5 || {
            log_warn "服务可能尚未完全就绪，请稍后手动检查"
        }
    fi
    
    log_info "============================================"
    log_info " 部署完成!"
    log_info " 主节点: ${MASTER_NODE}:${SERVER_PORT}"
    log_info " 健康检查: http://${MASTER_NODE}:${SERVER_PORT}/health"
    log_info " 模型列表: http://${MASTER_NODE}:${SERVER_PORT}/v1/models"
    log_info "============================================"
else
    log_info "【Step 3/3】跳过服务启动"
fi

log_info "部署流程结束"
