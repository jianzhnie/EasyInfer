#!/bin/bash
# ==========================================
# NPUSlim Ray 集群管理脚本
# 在容器内管理 Ray 集群 (start/stop/status)
# ==========================================
#
# 依赖: 目标节点容器已运行 (通过 IMAGE_NAME 查找)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../common.sh"
# shellcheck source=./docker_env.sh
source "${SCRIPT_DIR}/../docker/docker_env.sh"

# ------------------------------------------
# 帮助信息
# ------------------------------------------
usage() {
    cat <<'USAGE'
Usage: bash start_npuslim_ray_cluster.sh <start|stop|status> [OPTIONS]

Options:
  --head <IP>          指定 Head 节点（默认使用节点列表第一个）
  --workers <IP> ...   指定 Worker 节点
  --hosts <IP> ...     指定所有节点
  -f, --file <FILE>    节点列表文件路径
USAGE
    exit "$E_INVALID_ARG"
}

# ------------------------------------------
# 参数解析
# ------------------------------------------
CMD="${1:-help}"
shift || true

case "$CMD" in
    start|stop|status) ;;
    *) usage ;;
esac

# 确保必须通过 --hosts/--workers/--head/-f 之一指定节点
[[ "$*" =~ (--hosts|--workers|--head|-f|--file) ]] || \
    log_fatal "必须指定节点: --hosts <IP>、--head <IP>、--workers <IP> 或 -f <FILE>"

# 提取 --head 和 --workers（resolve_nodes 不处理这两个 flag）
HEAD_IP=""
MANUAL_WORKERS=()
filtered_args=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --head)    HEAD_IP="$2"; shift 2 ;;
        --workers) shift
                   while [[ $# -gt 0 && "$1" != -* ]]; do
                       MANUAL_WORKERS+=("$1"); shift
                   done ;;
        *)         filtered_args+=("$1"); shift ;;
    esac
done

# resolve_nodes 处理 --hosts / -f/--file，结果存入 RESOLVED_NODES
resolve_nodes "${filtered_args[@]}"
set -- "${filtered_args[@]}"

# ------------------------------------------
# 配置
# ------------------------------------------
: "${IMAGE_NAME:?IMAGE_NAME 未设置 (请检查 docker_env.sh)}"
RAY_PORT="${RAY_PORT:-6379}"

# 在远程节点的容器内执行命令
node_exec() {
    local host=$1 container
    shift
    container=$(ssh_run "$host" "docker ps -q --filter ancestor=${IMAGE_NAME} | head -1" 2>/dev/null)
    [[ -n "$container" ]] || { log_err "${host}: 未找到运行中的容器 (image=${IMAGE_NAME})"; return 1; }
    ssh_run "$host" "docker exec ${container} bash -lc $(printf '%q' "$*")"
}

# ------------------------------------------
# 命令实现
# ------------------------------------------

cmd_start() {
    local head_ip workers=()
    if [[ -n "$HEAD_IP" ]]; then
        head_ip="$HEAD_IP"
        [[ ${#MANUAL_WORKERS[@]} -gt 0 ]] && workers=("${MANUAL_WORKERS[@]}") || {
            for n in "${RESOLVED_NODES[@]}"; do [[ "$n" != "$head_ip" ]] && workers+=("$n"); done
        }
    else
        head_ip="${RESOLVED_NODES[0]}"
        workers=("${RESOLVED_NODES[@]:1}")
    fi

    log_info "Head: ${head_ip}:${RAY_PORT}  Workers: ${workers[*]:-none}"

    log_info "--- Starting head on ${head_ip} ---"
    node_exec "$head_ip" "ray start --head --port=${RAY_PORT}"
    sleep 2

    for worker in "${workers[@]}"; do
        log_info "--- Starting worker on ${worker} ---"
        node_exec "$worker" "ray start --address=${head_ip}:${RAY_PORT} --node-ip-address=${worker}"
    done

    node_exec "$head_ip" "ray status"
    log_info "Ray cluster started."
}

cmd_stop() {
    log_info "Stopping Ray on ${RESOLVED_NODES[*]}"
    for host in "${RESOLVED_NODES[@]}"; do
        node_exec "$host" "ray stop" || true
    done
    log_info "Ray cluster stopped."
}

cmd_status() {
    for host in "${RESOLVED_NODES[@]}"; do
        echo "--- ${host} ---"
        node_exec "$host" "ray status" || echo "  Ray not running"
        echo ""
    done
}

# ------------------------------------------
# 主流程
# ------------------------------------------

case "$CMD" in
    start)  cmd_start ;;
    stop)   cmd_stop ;;
    status) cmd_status ;;
esac
