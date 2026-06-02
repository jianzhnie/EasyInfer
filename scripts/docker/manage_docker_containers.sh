#!/bin/bash
# ==========================================
# Docker 容器集群管理脚本
# 用于在集群节点上管理 Docker 容器环境 (start/stop/restart)
# ==========================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 加载共享库（日志、SSH、并发、节点解析）
source "${SCRIPT_DIR}/../common.sh"

# ------------------------------------------
# 帮助信息
# ------------------------------------------
usage() {
    cat <<'USAGE'
Usage:
  bash manage_docker_containers.sh [start|stop|restart] [OPTIONS]

操作:
  start     确保 Docker 可用，加载镜像并启动容器（默认）
  stop      仅停止并清理旧容器，不启动新容器
  restart   停止并清理旧容器，加载镜像并启动新容器

Options:
  -h, --help          显示帮助信息
  -f, --file <FILE>   节点列表文件路径（必传）

环境变量配置: scripts/docker/docker_env.sh
USAGE
}

# ------------------------------------------
# 参数解析与初始化
# ------------------------------------------
ACTION="start"

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        -f|--file)
            NODES_FILE="${2:?错误: $1 需要一个参数}"
            shift 2
            ;;
        start|stop|restart) ACTION="$1"; shift ;;
        *)
            log_err "未知参数: $1"; usage; exit "$E_INVALID_ARG"
            ;;
    esac
done

# 加载环境配置（验证失败则退出）
ENV_FILE="${SCRIPT_DIR}/docker_env.sh"
if [[ ! -f "${ENV_FILE}" ]]; then
    log_err "环境配置文件未找到: ${ENV_FILE}"
    exit 1
fi
# shellcheck source=./docker_env.sh
source "${ENV_FILE}"

# 验证必要环境变量
: "${NODES_FILE:?环境变量 NODES_FILE 未设置}"
: "${CONTAINER_NAME:?环境变量 CONTAINER_NAME 未设置}"
: "${IMAGE_NAME:?环境变量 IMAGE_NAME 未设置}"
: "${IMAGE_TAR:?环境变量 IMAGE_TAR 未设置}"
: "${RUN_CONTAINER_SCRIPT:?环境变量 RUN_CONTAINER_SCRIPT 未设置}"
# ------------------------------------------
# 前置依赖检查
# ------------------------------------------
require_cmds ssh awk xargs

if [[ ! -f "$NODES_FILE" ]]; then
    log_err "节点列表文件未找到: $NODES_FILE"
    exit "$E_NOT_FOUND"
fi

if [[ "$ACTION" != "stop" ]]; then
    if [[ ! -f "$IMAGE_TAR" ]]; then
        log_err "镜像文件未找到: $IMAGE_TAR"
        exit "$E_NOT_FOUND"
    fi
    if [[ ! -f "$RUN_CONTAINER_SCRIPT" ]]; then
        log_err "启动脚本未找到: $RUN_CONTAINER_SCRIPT"
        exit "$E_NOT_FOUND"
    fi
fi

# ------------------------------------------
# 远程执行函数
# 注意: 下列 _remote_ 前缀的函数通过 declare -f 序列化后发送到远端节点执行
# ------------------------------------------

_remote_ensure_docker_running() {
    if ! command -v docker >/dev/null 2>&1; then
        echo "[ERROR] docker command not found" >&2
        return 127
    fi
    if docker info >/dev/null 2>&1; then
        return 0
    fi
    echo "[INFO] Docker service not running, attempting to start..."
    systemctl daemon-reload || true
    systemctl start docker
}

_remote_cleanup_containers() {
    echo "[INFO] Stopping and removing all existing containers..."
    local containers
    containers="$(docker ps -aq 2>/dev/null)"
    [[ -z "$containers" ]] && return 0
    echo "$containers" | xargs docker stop -t 30 2>/dev/null || true
    echo "$containers" | xargs docker kill 2>/dev/null || true
    echo "$containers" | xargs docker rm -f 2>/dev/null || true
}

_remote_load_and_run() {
    local image_name="$1" image_tar="$2" run_container_script="$3" container_name="$4"

    if ! docker image inspect "${image_name}" >/dev/null 2>&1; then
        if [[ ! -f "${image_tar}" ]]; then
            echo "[ERROR] image tar not found: ${image_tar}" >&2
            return 2
        fi
        echo "[INFO] Loading image from ${image_tar}..."
        docker load -i "${image_tar}"
    fi

    if [[ ! -f "${run_container_script}" ]]; then
        echo "[ERROR] run script not found: ${run_container_script}" >&2
        return 2
    fi

    IMAGE_NAME="${image_name}" CONTAINER_NAME="${container_name}" bash "${run_container_script}"

    if docker ps --format '{{.Names}}' | grep -Fx "${container_name}" >/dev/null; then
        echo "[INFO] Container ready: ${container_name}"
    else
        echo "[ERROR] Failed to start container: ${container_name}" >&2
        return 1
    fi
}

_remote_prepare_node() {
    local image_name="$1" image_tar="$2" run_container_script="$3" container_name="$4"
    local action="${5:-start}"

    set -euo pipefail
    _remote_ensure_docker_running

    case "$action" in
        restart|stop) _remote_cleanup_containers ;;
    esac
    if [[ "$action" != "stop" ]]; then
        _remote_load_and_run "${image_name}" "${image_tar}" "${run_container_script}" "${container_name}"
    fi
}

# ------------------------------------------
# 主控: 节点调度
# ------------------------------------------

prepare_node() {
    local node="$1"
    log_info "[${node}] 开始环境准备..."

    # 序列化远程函数并 base64 编码（避免引号嵌套问题）
    local func_code call_code b64
    func_code="$(declare -f _remote_ensure_docker_running _remote_cleanup_containers \
        _remote_load_and_run _remote_prepare_node)"
    printf -v call_code '_remote_prepare_node %q %q %q %q %q' \
        "${IMAGE_NAME}" "${IMAGE_TAR}" "${RUN_CONTAINER_SCRIPT}" "${CONTAINER_NAME}" "${ACTION}"
    b64="$(printf '%s\n%s\n' "${func_code}" "${call_code}" | base64 | tr -d '\n')"

    if ! ssh_run "$node" "echo '${b64}' | base64 -d | bash -l"; then
        log_err "[${node}] 环境准备失败"
        return 1
    fi
    log_info "[${node}] 环境准备完成"
}

# ------------------------------------------
# 主流程入口
# ------------------------------------------

nodes="$(read_nodes "$NODES_FILE")"
if [[ -z "$nodes" ]]; then
    log_err "NODES_FILE 中未找到任何节点信息"
    exit "$E_NOT_FOUND"
fi

log_info "目标节点: $nodes"
log_info "动作模式: ${ACTION}"
log_info "=== 开始准备节点 ==="

failed=0
for node in $nodes; do
    if ! prepare_node "$node"; then
        ((failed++)) || true
    fi
done
[[ "$failed" -gt 0 ]] && log_err "部分节点环境准备失败 (${failed} 个节点)"

log_info "=== 节点准备完成 ==="
