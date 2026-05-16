#!/bin/bash
# Manage Docker containers across cluster nodes
# Usage: bash manage_containers.sh <start|stop|status|restart> [options]
#   start    [--npuslim] [--no-npuslim] [--hosts <ip1> ...]
#   stop     [--hosts <ip1> ...]
#   status   [--hosts <ip1> ...]
#   restart  (stop + start, same options as start)

set -e

SSH_USER="root"
IMAGE_NAME="ascend910c-cann8.5.1-torch2.9.0-vllm0.18.0"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_CONTAINER="${SCRIPT_DIR}/run_npuslim_container.sh"

# Default cluster IPs
DEFAULT_HOSTS=(10.42.0.74 10.42.0.75 10.42.0.76 10.42.0.77 10.42.0.78 10.42.0.79 10.42.0.80 10.42.0.81)
MASTER_IP="${DEFAULT_HOSTS[0]}"

# NPUSlim paths differ: master uses local project dir, workers use synced dir
MASTER_NPUSLIM_PATH="/llm_workspace_1P/robin/npuslim-master"
WORKER_NPUSLIM_PATH="/llm_workspace_1P/robin/npuslim-master"
# Detect local IP
LOCAL_IPS=$(hostname -I 2>/dev/null)

is_local() {
    local ip=$1
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
    local host=$1
    if [[ "$host" == "$MASTER_IP" ]]; then
        echo "$MASTER_NPUSLIM_PATH"
    else
        echo "$WORKER_NPUSLIM_PATH"
    fi
}

# ─── Commands ────────────────────────────────────────────────────────────────

cmd_start() {
    local hosts=()
    local with_npuslim=true

    while [[ $# -gt 0 ]]; do
        case $1 in
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

        if is_local "$host"; then
            # Local: run directly
            bash "${RUN_CONTAINER}" --multi-node --daemon "$npuslim_arg"
        else
            # Remote: SSH and run
            ssh "${SSH_USER}@${host}" "bash ${RUN_CONTAINER} --multi-node --daemon $npuslim_arg"
        fi
        echo ""
    done

    echo "========================================"
    echo "All containers started."
    echo "========================================"
}

cmd_stop() {
    local hosts=()

    while [[ $# -gt 0 ]]; do
        case $1 in
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

        if is_local "$host"; then
            local cid
            cid=$(get_container)
            if [ -n "$cid" ]; then
                docker stop "$cid" && echo "Stopped: ${cid:0:12}"
            else
                echo "No running container found."
            fi
        else
            local cid
            cid=$(ssh "${SSH_USER}@${host}" "docker ps -q --filter ancestor=${IMAGE_NAME} | head -1" 2>/dev/null)
            if [ -n "$cid" ]; then
                ssh "${SSH_USER}@${host}" "docker stop ${cid}" 2>/dev/null && echo "Stopped: ${cid:0:12}"
            else
                echo "No running container found."
            fi
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
        case $1 in
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

        if is_local "$host"; then
            local cid
            cid=$(get_container)
            if [ -n "$cid" ]; then
                local runtime
                runtime=$(docker inspect --format '{{.State.StartedAt}}' "$cid" 2>/dev/null || echo "?")
                echo "running  ${cid:0:12}  since ${runtime}"
            else
                echo "stopped"
            fi
        else
            local result
            result=$(ssh "${SSH_USER}@${host}" \
                "cid=\$(docker ps -q --filter ancestor=${IMAGE_NAME} | head -1); \
                 if [ -n \"\$cid\" ]; then \
                     status=\$(docker inspect --format '{{.State.Status}}' \"\$cid\" 2>/dev/null || echo unknown); \
                     runtime=\$(docker inspect --format '{{.State.StartedAt}}' \"\$cid\" 2>/dev/null || echo '?'); \
                     echo \"running  \${cid:0:12}  since \${runtime}\"; \
                 else echo stopped; fi" 2>/dev/null) || result="unreachable"
            echo "$result"
        fi
    done

    echo "========================================"
}

# ─── Main ────────────────────────────────────────────────────────────────────

CMD=${1:-help}
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
        echo "Default hosts:"
        echo "  ${DEFAULT_HOSTS[*]}"
        echo ""
        echo "Examples:"
        echo "  bash manage_containers.sh start"
        echo "  bash manage_containers.sh stop --hosts 10.42.15.195 10.42.15.196"
        echo "  bash manage_containers.sh status"
        ;;
esac
