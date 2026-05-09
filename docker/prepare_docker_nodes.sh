#!/usr/bin/env bash
# ==========================================
# Docker 容器准备脚本 (prepare_docker_nodes.sh)
# 用于在集群节点上准备 Docker 容器环境
# ==========================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/set_env.sh"


# ------------------------------------------
# 日志函数
# ------------------------------------------
log_info() { echo "[INFO] $(date '+%Y-%m-%d %H:%M:%S') - $*"; }
log_warn() { echo "[WARN] $(date '+%Y-%m-%d %H:%M:%S') - $*" >&2; }
log_err()  { echo "[ERROR] $(date '+%Y-%m-%d %H:%M:%S') - $*" >&2; }

# ------------------------------------------
# 帮助信息
# ------------------------------------------
usage() {
  cat <<'USAGE'
Usage:
  bash prepare_docker_nodes.sh [start|stop|restart] [OPTIONS]

Description:
  该脚本用于在集群节点上准备 Docker 容器环境。
  环境变量请在同目录下的 set_env.sh 中配置。

Options:
  -h, --help           显示帮助信息
  -a, --action <start|stop|restart>
                       start:   确保 Docker 可用，加载镜像并启动新容器（默认，不停止已有容器）
                       stop:    仅优雅停止并清理旧容器，不启动新容器
                       restart: 确保 Docker 可用，停止并清理旧容器，加载镜像并启动新容器
USAGE
}

# ------------------------------------------
# 参数解析
# ------------------------------------------
ACTION="start"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    -a|--action)
      if [[ -n "${2:-}" && "${2:-}" != -* ]]; then
        ACTION="$2"
        shift 2
      else
        log_err "选项 $1 需要一个参数: start, stop 或 restart"
        usage
        exit 1
      fi
      ;;
    start|stop|restart)
      ACTION="$1"
      shift
      ;;
    *)
      log_err "未知参数: $1"
      usage
      exit 1
      ;;
  esac
done

if [[ "$ACTION" != "start" && "$ACTION" != "stop" && "$ACTION" != "restart" ]]; then
  log_err "动作参数必须是 start, stop 或 restart，当前值: $ACTION"
  usage
  exit 1
fi

# ------------------------------------------
# 引入环境变量
# ------------------------------------------
if [[ -f "${ENV_FILE}" ]]; then
  source "${ENV_FILE}"
else
  log_err "环境配置文件未找到: ${ENV_FILE}"
  exit 1
fi

# 验证必要的环境变量
: "${NODES_FILE:?环境变量 NODES_FILE 未设置}"
: "${CONTAINER_NAME:?环境变量 CONTAINER_NAME 未设置}"
: "${IMAGE_NAME:?环境变量 IMAGE_NAME 未设置}"
: "${IMAGE_TAR:?环境变量 IMAGE_TAR 未设置}"
: "${RUN_CONTAINER_SCRIPT:?环境变量 RUN_CONTAINER_SCRIPT 未设置}"
: "${PARALLELISM:=16}"
: "${SSH_OPTS:=-o StrictHostKeyChecking=no -o ConnectTimeout=10}"

# ------------------------------------------
# 前置依赖检查
# ------------------------------------------
check_dependencies() {
  local deps=("ssh" "scp" "awk" "xargs")
  for cmd in "${deps[@]}"; do
    if ! command -v "$cmd" &>/dev/null; then
      log_err "缺少必要依赖: $cmd"
      exit 1
    fi
  done

  if [[ ! -f "$NODES_FILE" ]]; then
    log_err "节点列表文件未找到: $NODES_FILE"
    exit 2
  fi

  if [[ "$ACTION" == "start" || "$ACTION" == "restart" ]]; then
    if [[ ! -f "$IMAGE_TAR" ]]; then
      log_err "镜像文件未找到: $IMAGE_TAR"
      exit 2
    fi
    if [[ ! -f "$RUN_CONTAINER_SCRIPT" ]]; then
      log_err "启动脚本未找到: $RUN_CONTAINER_SCRIPT"
      exit 2
    fi
  fi
}

# 执行依赖检查
check_dependencies

# ------------------------------------------
# 辅助函数
# ------------------------------------------

# 获取非空节点列表
read_nodes() {
  awk 'NF {print $1}' "$NODES_FILE"
}

# 拼接 SSH 目标地址
ssh_target() {
  local node="$1"
  printf "%s%s" "$SSH_USER_HOST_PREFIX" "$node"
}

# 执行 SSH 命令
ssh_run() {
  local node="$1"
  shift
  ssh ${SSH_OPTS} "$(ssh_target "$node")" "$@"
}

# 并发数控制
limit_jobs() {
  local max="$1"
  while [[ "$(jobs -rp | wc -l | tr -d ' ')" -ge "$max" ]]; do
    wait -n 2>/dev/null || true
  done
}

# ------------------------------------------
# 容器准备远程执行逻辑定义
# 注意: 下列带 _remote 前缀的函数将通过 declare -f 被序列化发送至远程执行
# ------------------------------------------

_remote_prepare_node() {
  local image_name="$1"
  local image_tar="$2"
  local run_container_script="$3"
  local container_name="$4"
  local action="${5:-start}"

  set -euo pipefail

  # 确保 Docker 命令可用
  if ! command -v docker >/dev/null 2>&1; then
    echo "[ERROR] docker command not found" >&2
    exit 127
  fi

  # 确保 Docker 服务已启动（如果不可用则尝试启动）
  if ! docker info >/dev/null 2>&1; then
    echo "[INFO] Docker service not running, attempting to start..."
    systemctl daemon-reload && systemctl start docker || {
      echo "[ERROR] Failed to start Docker service" >&2
      exit 1
    }
  fi

  # 在 restart 或 stop 模式下执行 stop & kill 容器操作
  if [[ "$action" == "restart" || "$action" == "stop" ]]; then
    echo "[INFO] Stopping and removing all existing containers..."
    docker ps -aq 2>/dev/null | xargs -r docker stop 2>/dev/null || true
    docker ps -aq 2>/dev/null | xargs -r docker kill 2>/dev/null || true
    docker ps -aq 2>/dev/null | xargs -r docker rm -f 2>/dev/null || true
  fi

  if [[ "$action" == "start" || "$action" == "restart" ]]; then
    if docker image inspect "${image_name}" >/dev/null 2>&1; then
      : # 镜像已存在
    else
      if [ ! -f "${image_tar}" ]; then
        echo "[ERROR] image tar not found: ${image_tar}" >&2
        exit 2
      fi
      echo "[INFO] Loading image from ${image_tar}..."
      docker load -i "${image_tar}"
    fi

    if [ ! -f "${run_container_script}" ]; then
      echo "[ERROR] run script not found: ${run_container_script}" >&2
      exit 2
    fi

    export IMAGE_NAME="${image_name}"
    export CONTAINER_NAME="${container_name}"
    bash "${run_container_script}"

    if docker ps --format '{{.Names}}' | grep -Fx "${container_name}" >/dev/null; then
      echo "[INFO] Container ready: ${container_name}"
    else
      echo "[ERROR] Failed to start container: ${container_name}" >&2
      exit 1
    fi
  else
    echo "[INFO] Action is 'stop', skipping image load and container start."
  fi
}

# ------------------------------------------
# 主控调度逻辑
# ------------------------------------------

prepare_node() {
  local node="$1"
  log_info "[${node}] 开始环境准备..."

  local func_code call_code
  func_code="$(declare -f _remote_prepare_node)"
  call_code="_remote_prepare_node '${IMAGE_NAME}' '${IMAGE_TAR}' '${RUN_CONTAINER_SCRIPT}' '${CONTAINER_NAME}' '${ACTION}'"

  # prepare_node 在宿主机执行，不需要进容器
  if ! echo "${func_code}; ${call_code}" | ssh_run "$node" bash -lc "bash -s"; then
     log_err "[${node}] 环境准备失败"
     return 1
  fi
  log_info "[${node}] 环境准备完成"
}

# ------------------------------------------
# 主流程入口
# ------------------------------------------

nodes="$(read_nodes)"
if [[ -z "$nodes" ]]; then
  log_err "NODES_FILE 中未找到任何节点信息"
  exit 2
fi

log_info "目标节点列表: $(echo $nodes | tr '\n' ' ')"
log_info "动作模式: ${ACTION}"
log_info "=== 开始准备节点 ==="

for node in $nodes; do
  limit_jobs "$PARALLELISM"
  (prepare_node "$node") &
done
wait

log_info "=== 节点准备完成 ==="
