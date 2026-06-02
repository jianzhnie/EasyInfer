#!/bin/bash
# Ray Cluster Manager for remote Docker containers
# Usage: bash ray_cluster.sh <command> [options]
#   start   [--head <ip>] [--workers <ip1> <ip2> ...] [--file <node_list>]
#   stop    [--hosts <ip1> <ip2> ...] [--file <node_list>]
#   status  [--hosts <ip1> <ip2> ...] [--file <node_list>]
# Default: read nodes from scripts/node_list.txt (first = head, rest = workers)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 加载共享库
source "${SCRIPT_DIR}/../common.sh"

IMAGE_NAME="${IMAGE_NAME:-ascend910c-cann8.5.1-torch2.9.0-vllm0.18.0}"
SSH_USER="${SSH_USER:-root}"
RAY_PORT="${RAY_PORT:-6379}"

# Find the running container
get_container() {
    docker ps -q --filter ancestor="${IMAGE_NAME}" | head -1
}

# Execute a command inside the container
node_exec() {
    local host="$1"
    shift
    local container
    if is_local_ip "$host"; then
        container=$(get_container)
        if [[ -z "$container" ]]; then
            log_err "本地未找到运行中的容器"
            return 1
        fi
        docker exec "$container" bash -lc "$*"
    else
        # shellcheck disable=SC2029
        container=$(ssh "${SSH_USER}@${host}" "docker ps -q --filter ancestor=${IMAGE_NAME} | head -1" 2>/dev/null)
        if [[ -z "$container" ]]; then
            log_err "${host} 上未找到运行中的容器"
            return 1
        fi
        # shellcheck disable=SC2029
        ssh "${SSH_USER}@${host}" "docker exec ${container} bash -lc $(printf '%q' "$*")" 2>/dev/null
    fi
}

CMD="${1:-help}"
shift || true

case "$CMD" in
    start)
        HEAD_IP=""
        WORKERS=()
        local_nodes_file=""
        while [[ $# -gt 0 ]]; do
            case $1 in
                --head) HEAD_IP="$2"; shift 2 ;;
                --workers) shift
                    while [[ $# -gt 0 && "$1" != -* ]]; do
                        WORKERS+=("$1"); shift
                    done ;;
                --file|-f) local_nodes_file="$2"; shift 2 ;;
                *) shift ;;
            esac
        done

        # 解析节点列表
        [[ -n "${local_nodes_file:-}" ]] && NODES_FILE="$local_nodes_file"
        resolve_nodes

        # 默认：第一个 IP = head，其余 = workers
        if [[ -z "$HEAD_IP" ]]; then
            HEAD_IP="${RESOLVED_NODES[0]}"
            for ((i = 1; i < ${#RESOLVED_NODES[@]}; i++)); do
                WORKERS+=("${RESOLVED_NODES[$i]}")
            done
        fi

        log_info "========================================"
        log_info "Starting Ray Cluster"
        log_info "Head:    ${HEAD_IP}:${RAY_PORT}"
        log_info "Workers: ${WORKERS[*]:-none}"

        log_info "--- Starting head on ${HEAD_IP} ---"
        node_exec "$HEAD_IP" "ray start --head --port=${RAY_PORT}"

        sleep 2

        for worker in "${WORKERS[@]}"; do
            log_info "--- Starting worker on ${worker} ---"
            node_exec "$worker" "ray start --address=${HEAD_IP}:${RAY_PORT} --node-ip-address=${worker}"
        done

        log_info "--- Cluster status ---"
        node_exec "$HEAD_IP" "ray status"
        log_info "========================================"
        log_info "Ray cluster started."
        log_info "========================================"
        ;;

    stop)
        resolve_nodes "$@"

        for host in "${RESOLVED_NODES[@]}"; do
            echo "--- Stopping Ray on ${host} ---"
            node_exec "$host" "ray stop" || true
        done
        ;;

    status)
        resolve_nodes "$@"

        for host in "${RESOLVED_NODES[@]}"; do
            echo "--- Ray status on ${host} ---"
            node_exec "$host" "ray status" || echo "Ray not running"
            echo ""
        done
        ;;

    help|*)
        echo "Usage: bash ray_cluster.sh <command> [options]"
        echo ""
        echo "Commands:"
        echo "  start   [--head <ip>] [--workers <ip1> <ip2> ...] [--file <path>]"
        echo "  stop    [--hosts <ip1> <ip2> ...] [--file <path>]"
        echo "  status  [--hosts <ip1> <ip2> ...] [--file <path>]"
        echo ""
        echo "Default node list: ${NODES_FILE}"
        echo "Environment: NODES_FILE, IMAGE_NAME, SSH_USER, RAY_PORT"
        ;;
esac
