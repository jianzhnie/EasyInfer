#!/bin/bash
# ==========================================
# NPUSlim 容器集群管理脚本
# 在集群节点上管理 NPUSlim Docker 容器 (start/stop/status/restart)
# ==========================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 加载共享库
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../common.sh"
# shellcheck source=./docker_env.sh
source "${SCRIPT_DIR}/docker_env.sh"

# ------------------------------------------
# 帮助信息
# ------------------------------------------
usage() {
    cat <<'USAGE'
Usage:
  bash manage_npuslim_containers.sh <command> [OPTIONS]

Commands:
  start     启动容器（默认挂载 NPUSlim 源码）
  stop      停止所有节点上的容器
  status    查看所有节点上的容器状态
  restart   停止后重新启动容器

Options:
  --hosts <ip1> [ip2] ...   指定目标节点（最高优先级）
  -f, --file <FILE>         节点列表文件路径
  --npuslim                 挂载 NPUSlim 源码（默认）
  --no-npuslim              不挂载 NPUSlim 源码
  --privileged              以特权模式启动容器

环境变量配置: scripts/docker/docker_env.sh
USAGE
}

# ------------------------------------------
# 参数解析
# ------------------------------------------
CMD="${1:-help}"
shift || true

case "$CMD" in
    start|stop|status|restart) ;;
    *) usage; exit 0 ;;
esac

# 在 resolve_nodes 消费参数前，先提取 --npuslim / --no-npuslim / --privileged
WITH_NPUSLIM=true
PRIVILEGED=true
npushift_args=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --npuslim)     WITH_NPUSLIM=true; shift ;;
        --no-npuslim)  WITH_NPUSLIM=false; shift ;;
        --privileged)  PRIVILEGED=true; shift ;;
        *)             npushift_args+=("$1"); shift ;;
    esac
done
set -- "${npushift_args[@]}"

# 确保必须通过 --hosts 或 -f/--file 指定节点
[[ "$*" =~ (--hosts|-f|--file) ]] || log_fatal "必须指定节点: --hosts <ip> 或 -f <FILE>"

# resolve_nodes 会将结果存入全局数组 RESOLVED_NODES
resolve_nodes "$@"

# ------------------------------------------
# 配置
# ------------------------------------------
RUN_CONTAINER="${SCRIPT_DIR}/run_npuslim_container.sh"
[[ ! -f "$RUN_CONTAINER" ]] && log_fatal "启动脚本未找到: $RUN_CONTAINER"

# NPUSlim 路径（所有节点使用同一路径，通过环境变量 NPUSLIM_PATH 覆盖）
NPUSLIM_PATH="${NPUSLIM_PATH:-/home/jianzhnie/llmtuner/llm/npuslim}"

# ------------------------------------------
# 命令实现
# ------------------------------------------

cmd_start() {
    echo "========================================"
    echo "Starting Containers"
    echo "========================================"
    echo "Hosts:      ${RESOLVED_NODES[*]}"
    echo "NPUSlim:    ${WITH_NPUSLIM}"
    echo "Privileged: ${PRIVILEGED}"
    echo ""

    for host in "${RESOLVED_NODES[@]}"; do
        echo "--- Starting on ${host} ---"
        local npuslim_arg=""
        local privileged_arg=""
        if $WITH_NPUSLIM; then
            npuslim_arg="--npuslim=${NPUSLIM_PATH}"
        fi
        if $PRIVILEGED; then
            privileged_arg="--privileged"
        fi
        # shellcheck disable=SC2086
        ssh_run "$host" "bash ${RUN_CONTAINER} --multi-node --daemon ${privileged_arg} ${npuslim_arg}"
        echo ""
    done

    echo "========================================"
    echo "All containers started."
    echo "========================================"
}

cmd_stop() {
    echo "========================================"
    echo "Stopping Containers"
    echo "========================================"
    echo "Hosts: ${RESOLVED_NODES[*]}"
    echo ""

    for host in "${RESOLVED_NODES[@]}"; do
        echo "--- Stopping on ${host} ---"
        local cid
        # shellcheck disable=SC2086
        cid="$(ssh_run "$host" "docker ps -q --filter ancestor=${IMAGE_NAME} | head -1" 2>/dev/null)" || cid=""
        if [[ -n "$cid" ]]; then
            ssh_run "$host" "docker stop $cid" && echo "  Stopped: ${cid:0:12}"
        else
            echo "  No running container found."
        fi
        echo ""
    done

    echo "========================================"
    echo "All containers stopped."
    echo "========================================"
}

cmd_status() {
    echo "========================================"
    echo "Container Status"
    echo "========================================"

    for host in "${RESOLVED_NODES[@]}"; do
        printf "  %-16s" "$host"
        local cid
        # shellcheck disable=SC2086
        cid="$(ssh_run "$host" "docker ps -q --filter ancestor=${IMAGE_NAME} | head -1" 2>/dev/null)" || cid=""
        if [[ -n "$cid" ]]; then
            local runtime
            runtime="$(ssh_run "$host" "docker inspect --format '{{.State.StartedAt}}' $cid" 2>/dev/null || echo '?')"
            echo "running  ${cid:0:12}  since ${runtime}"
        else
            echo "stopped"
        fi
    done

    echo "========================================"
}

# ------------------------------------------
# 主流程
# ------------------------------------------

case "$CMD" in
    start)   cmd_start "$@" ;;
    stop)    cmd_stop "$@" ;;
    status)  cmd_status "$@" ;;
    restart) cmd_stop "$@"; echo ""; cmd_start "$@" ;;
esac
