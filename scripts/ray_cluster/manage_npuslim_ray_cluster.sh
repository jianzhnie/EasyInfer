#!/bin/bash
# ==========================================
# NPUSlim Ray 集群管理脚本
# 在容器内管理 Ray 集群 (start/stop/status)
# 节点列表从文件读取，首节点为 Head，其余为 Worker
# ==========================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../common.sh"
# shellcheck source=./docker_env.sh
source "${SCRIPT_DIR}/../docker/docker_env.sh"

RAY_PORT="${RAY_PORT:-6379}"

# 校验容器是否运行
check_container() {
    local host=$1
    ssh_run "$host" "docker ps --format '{{.Names}}' | grep -qx '${CONTAINER_NAME}'" 2>/dev/null
}

# 在远程节点的容器内执行命令（通过 CONTAINER_NAME 指定容器, 与 manage_ray_cluster.sh 保持一致）
node_exec() {
    local host=$1
    shift
    check_container "$host" || { log_err "${host}: 容器 '${CONTAINER_NAME}' 未运行"; return 1; }
    ssh_run "$host" "docker exec ${CONTAINER_NAME} bash -lc $(printf '%q' "$*")"
}

# ------------------------------------------
# 参数解析
# ------------------------------------------
CMD="${1:-help}"
shift || true

case "$CMD" in
    start|stop|status) ;;
    *) echo "Usage: bash start_npuslim_ray_cluster.sh <start|stop|status> -f <FILE>"; exit 0 ;;
esac

# 必须通过 -f/--file 指定节点列表文件
[[ "$*" =~ (-f|--file) ]] || log_fatal "必须指定节点文件: -f <FILE>"
resolve_nodes "$@"
head_ip="${RESOLVED_NODES[0]}"
workers=("${RESOLVED_NODES[@]:1}")

# ------------------------------------------
# 主流程
# ------------------------------------------
case "$CMD" in
    start)
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
        ;;
    stop)
        for host in "${RESOLVED_NODES[@]}"; do
            node_exec "$host" "ray stop" || true
        done
        log_info "Ray cluster stopped."
        ;;
    status)
        for host in "${RESOLVED_NODES[@]}"; do
            echo "--- ${host} ---"
            node_exec "$host" "ray status" || echo "  Ray not running"
            echo ""
        done
        ;;
esac
