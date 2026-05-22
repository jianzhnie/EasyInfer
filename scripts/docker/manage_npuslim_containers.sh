#!/usr/bin/env bash
# Manage Docker containers across cluster nodes
# Usage: bash manage_containers.sh <start|stop|status|restart> [options]
#   start    [--npuslim] [--no-npuslim] [--hosts <ip1> ...]
#   stop     [--hosts <ip1> ...]
#   status   [--hosts <ip1> ...]
#   restart  (stop + start, same options as start)

set -euo pipefail

SSH_USER="${SSH_USER:-root}"
IMAGE_NAME="${IMAGE_NAME:-ascend910c-cann8.5.1-torch2.9.0-vllm0.18.0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_CONTAINER="${SCRIPT_DIR}/run_npuslim_container.sh"

# Default cluster IPs
# shellcheck disable=SC2034
# DEFAULT_HOSTS=(10.42.0.66 10.42.0.67 10.42.0.68 10.42.0.69 10.42.0.70 10.42.0.71 10.42.0.72 10.42.0.73)
# DEFAULT_HOSTS=(10.42.0.74 10.42.0.75 10.42.0.76 10.42.0.77 10.42.0.78 10.42.0.79 10.42.0.80 10.42.0.81)
DEFAULT_HOSTS=(10.42.1.66 10.42.1.67 10.42.1.68 10.42.1.69 10.42.1.70 10.42.1.71 10.42.1.72 10.42.1.73 10.42.1.74 10.42.1.75 10.42.1.76 10.42.1.77 10.42.1.78 10.42.1.79 10.42.1.80 10.42.1.81)
MASTER_IP="${DEFAULT_HOSTS[0]}"

# for longcat 模型
MASTER_NPUSLIM_PATH="${MASTER_NPUSLIM_PATH:-/llm_workspace_1P/robin/npuslim}"
WORKER_NPUSLIM_PATH="${WORKER_NPUSLIM_PATH:-/llm_workspace_1P/robin/npuslim}"

# NPUSlim paths differ: master uses local project dir, workers use synced dir
# MASTER_NPUSLIM_PATH="${MASTER_NPUSLIM_PATH:-/llm_workspace_1P/robin/npuslim-master}"
# WORKER_NPUSLIM_PATH="${WORKER_NPUSLIM_PATH:-/llm_workspace_1P/robin/npuslim-master}"

# Detect local IPs
LOCAL_IPS=$(hostname -I 2>/dev/null || true)

is_local() {
    local ip="$1" lip
    for lip in $LOCAL_IPS; do
        [[ "$ip" == "$lip" ]] && return 0
    done
    return 1
}

# Get running container ID for the image
get_container() {
    docker ps -q --filter "ancestor=${IMAGE_NAME}" | head -1
}

# Get npuslim path for a given host
npuslim_path_for() {
    local host="$1"
    if [[ "$host" == "$MASTER_IP" ]]; then
        echo "$MASTER_NPUSLIM_PATH"
    else
        echo "$WORKER_NPUSLIM_PATH"
    fi
}

# ─── Remote SSH helpers ──────────────────────────────────────────────────────

# Execute docker command on remote or local node
remote_docker_cmd() {
    local host="$1"; shift
    if is_local "$host"; then
        docker "$@"
    else
        # shellcheck disable=SC2029
        ssh "${SSH_USER}@${host}" "docker $*" 2>/dev/null
    fi
}

# Run a bash command on remote or local node
remote_bash() {
    local host="$1"; shift
    if is_local "$host"; then
        bash -c "$*"
    else
        # shellcheck disable=SC2029
        ssh "${SSH_USER}@${host}" "$@" 2>/dev/null
    fi
}

# ─── Commands ────────────────────────────────────────────────────────────────

cmd_start() {
    local hosts=()
    local with_npuslim=true

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --hosts) shift
                while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                    hosts+=("$1"); shift
                done ;;
            --no-npuslim) with_npuslim=false; shift ;;
            --npuslim) with_npuslim=true; shift ;;
            *) shift ;;
        esac
    done
    [[ ${#hosts[@]} -eq 0 ]] && hosts=("${DEFAULT_HOSTS[@]}")

    echo "========================================"
    echo "Starting Containers"
    echo "========================================"
    echo "Hosts:   ${hosts[*]}"
    echo "NPUSlim: ${with_npuslim}"
    echo ""

    for host in "${hosts[@]}"; do
        echo "--- Starting on ${host} ---"

        local npuslim_arg=""
        if $with_npuslim; then
            local npath
            npath=$(npuslim_path_for "$host")
            npuslim_arg="--npuslim=${npath}"
        fi

        remote_bash "$host" "bash ${RUN_CONTAINER} --multi-node --daemon ${npuslim_arg}"
        echo ""
    done

    echo "========================================"
    echo "All containers started."
    echo "========================================"
}

cmd_stop() {
    local hosts=()

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --hosts) shift
                while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                    hosts+=("$1"); shift
                done ;;
            *) shift ;;
        esac
    done
    [[ ${#hosts[@]} -eq 0 ]] && hosts=("${DEFAULT_HOSTS[@]}")

    echo "========================================"
    echo "Stopping Containers"
    echo "========================================"
    echo "Hosts: ${hosts[*]}"
    echo ""

    for host in "${hosts[@]}"; do
        echo "--- Stopping on ${host} ---"
        local cid
        # shellcheck disable=SC2086
        cid=$(remote_docker_cmd "$host" ps -q --filter ancestor="${IMAGE_NAME}" | head -1) || cid=""
        if [[ -n "$cid" ]]; then
            remote_docker_cmd "$host" stop "$cid" && echo "Stopped: ${cid:0:12}"
        else
            echo "No running container found."
        fi
        echo ""
    done

    echo "========================================"
    echo "All containers stopped."
    echo "========================================"
}

cmd_status() {
    local hosts=()

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --hosts) shift
                while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                    hosts+=("$1"); shift
                done ;;
            *) shift ;;
        esac
    done
    [[ ${#hosts[@]} -eq 0 ]] && hosts=("${DEFAULT_HOSTS[@]}")

    echo "========================================"
    echo "Container Status"
    echo "========================================"

    for host in "${hosts[@]}"; do
        local marker=""
        is_local "$host" && marker=" (local)"
        printf "  %-16s" "${host}${marker}"

        local cid
        # shellcheck disable=SC2086
        cid=$(remote_docker_cmd "$host" ps -q --filter ancestor="${IMAGE_NAME}" | head -1) || cid=""
        if [[ -n "$cid" ]]; then
            local runtime
            runtime=$(remote_docker_cmd "$host" inspect --format '{{.State.StartedAt}}' "$cid" 2>/dev/null || echo "?")
            echo "running  ${cid:0:12}  since ${runtime}"
        else
            echo "stopped"
        fi
    done

    echo "========================================"
}

# ─── Main ────────────────────────────────────────────────────────────────────

CMD="${1:-help}"
shift || true

case "$CMD" in
    start)   cmd_start "$@" ;;
    stop)    cmd_stop "$@" ;;
    status)  cmd_status "$@" ;;
    restart) cmd_stop "$@"; echo ""; cmd_start "$@" ;;
    help|*)
        echo "Usage: bash manage_containers.sh <command> [options]"
        echo ""
        echo "Commands:"
        echo "  start    Start containers on all nodes (default: with npuslim)"
        echo "  stop     Stop containers on all nodes"
        echo "  status   Show container status on all nodes"
        echo "  restart  Stop then start all containers"
        echo ""
        echo "Options:"
        echo "  --hosts <ip1> [ip2] ...   Target specific hosts"
        echo "  --npuslim                 Mount npuslim source (default)"
        echo "  --no-npuslim              Skip npuslim mount"
        echo ""
        echo "Environment variables:"
        echo "  SSH_USER                  SSH user (default: root)"
        echo "  IMAGE_NAME                Docker image name"
        echo "  MASTER_NPUSLIM_PATH       NPUSlim path on master node"
        echo "  WORKER_NPUSLIM_PATH       NPUSlim path on worker nodes"
        echo ""
        echo "Default hosts:"
        echo "  ${DEFAULT_HOSTS[*]}"
        echo ""
        echo "Examples:"
        echo "  bash manage_containers.sh start"
        echo "  bash manage_containers.sh stop --hosts 10.42.15.195 10.42.15.196"
        echo "  bash manage_containers.sh status"
        ;;
esac
