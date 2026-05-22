#!/bin/bash
# Ray Cluster Manager for remote Docker containers
# Usage: bash ray_cluster.sh <command> [options]
#   start   [--head <ip>] [--workers <ip1> <ip2> ...]   Start Ray cluster
#   stop    [--hosts <ip1> <ip2> ...]                   Stop Ray on nodes
#   status  [--hosts <ip1> <ip2> ...]                   Check Ray status
# Default IPs: first is head, rest are workers (see IPS below)

set -e

IMAGE_NAME="ascend910c-cann8.5.1-torch2.9.0-vllm0.18.0"
SSH_USER="root"
RAY_PORT=6379

# Default cluster IPs (first = head, rest = workers)
# for PCL-Kimi2
# IPS=(10.42.0.66 10.42.0.67 10.42.0.68 10.42.0.69 10.42.0.70 10.42.0.71 10.42.0.72 10.42.0.73)
# IPS=(10.42.0.74 10.42.0.75 10.42.0.76 10.42.0.77 10.42.0.78 10.42.0.79 10.42.0.80 10.42.0.81)
IPS=(10.42.1.66 10.42.1.67 10.42.1.68 10.42.1.69 10.42.1.70 10.42.1.71 10.42.1.72 10.42.1.73)
# IPS=(10.42.1.66 10.42.1.67 10.42.1.68 10.42.1.69 10.42.1.70 10.42.1.71 10.42.1.72 10.42.1.73 10.42.1.74 10.42.1.75 10.42.1.76 10.42.1.77 10.42.1.78 10.42.1.79 10.42.1.80 10.42.1.81)
# for Qwen3-235B-A22B
# IPS=(10.42.15.194 10.42.15.195)

# Local IPs (skip SSH for these)
LOCAL_IPS=$(hostname -I 2>/dev/null)

is_local() {
    local ip=$1
    for lip in $LOCAL_IPS; do
        [[ "$ip" == "$lip" ]] && return 0
    done
    return 1
}

# Find the running container
get_container() {
    docker ps -q --filter ancestor="${IMAGE_NAME}" | head -1
}

# Execute a command inside the container
node_exec() {
    local host=$1
    shift
    local container
    if is_local "$host"; then
        container=$(get_container)
        if [ -z "$container" ]; then
            echo "ERROR: No running container found locally"
            return 1
        fi
        docker exec "$container" bash -lc "$*"
    else
        # shellcheck disable=SC2029
        container=$(ssh "${SSH_USER}@${host}" "docker ps -q --filter ancestor=${IMAGE_NAME} | head -1" 2>/dev/null)
        if [ -z "$container" ]; then
            echo "ERROR: No running container found on ${host}"
            return 1
        fi
        # shellcheck disable=SC2029
        ssh "${SSH_USER}@${host}" "docker exec ${container} bash -lc '$*'" 2>/dev/null
    fi
}

CMD=${1:-help}
shift || true

case "$CMD" in
    start)
        HEAD_IP=""
        WORKERS=()
        while [[ $# -gt 0 ]]; do
            case $1 in
                --head) HEAD_IP="$2"; shift 2 ;;
                --workers) shift
                    while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                        WORKERS+=("$1"); shift
                    done ;;
                *) shift ;;
            esac
        done

        # Use defaults: first IP = head, rest = workers
        if [ -z "$HEAD_IP" ]; then
            HEAD_IP="${IPS[0]}"
            for ((i=1; i<${#IPS[@]}; i++)); do
                WORKERS+=("${IPS[$i]}")
            done
        fi

        echo "========================================"
        echo "Starting Ray Cluster"
        echo "========================================"
        echo "Head:    ${HEAD_IP}:${RAY_PORT}"
        echo "Workers: ${WORKERS[*]:-none}"
        echo ""

        echo "--- Starting head on ${HEAD_IP} ---"
        node_exec "$HEAD_IP" "ray start --head --port=${RAY_PORT}"
        echo ""

        sleep 2

        for worker in "${WORKERS[@]}"; do
            echo "--- Starting worker on ${worker} ---"
            node_exec "$worker" "ray start --address=${HEAD_IP}:${RAY_PORT} --node-ip-address=${worker}"
            echo ""
        done

        echo "--- Cluster status ---"
        node_exec "$HEAD_IP" "ray status"
        echo ""
        echo "========================================"
        echo "Ray cluster started."
        echo "========================================"
        ;;

    stop)
        HOSTS=()
        while [[ $# -gt 0 ]]; do
            case $1 in
                --hosts) shift
                    while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                        HOSTS+=("$1"); shift
                    done ;;
                *) shift ;;
            esac
        done

        # Default: all IPs
        [[ ${#HOSTS[@]} -eq 0 ]] && HOSTS=("${IPS[@]}")

        for host in "${HOSTS[@]}"; do
            echo "--- Stopping Ray on ${host} ---"
            node_exec "$host" "ray stop" || true
        done
        ;;

    status)
        HOSTS=()
        while [[ $# -gt 0 ]]; do
            case $1 in
                --hosts) shift
                    while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                        HOSTS+=("$1"); shift
                    done ;;
                *) shift ;;
            esac
        done

        # Default: all IPs
        [[ ${#HOSTS[@]} -eq 0 ]] && HOSTS=("${IPS[@]}")

        for host in "${HOSTS[@]}"; do
            echo "--- Ray status on ${host} ---"
            node_exec "$host" "ray status" || echo "Ray not running"
            echo ""
        done
        ;;

    help|*)
        echo "Usage: bash ray_cluster.sh <command> [options]"
        echo ""
        echo "Commands:"
        echo "  start   [--head <ip>] [--workers <ip1> <ip2> ...]"
        echo "  stop    [--hosts <ip1> <ip2> ...]"
        echo "  status  [--hosts <ip1> <ip2> ...]"
        echo ""
        echo "Default cluster:"
        echo "  Head:    ${IPS[0]}"
        echo "  Workers: ${IPS[*]:1}"
        ;;
esac
