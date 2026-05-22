#!/usr/bin/env bash
# Manage Docker containers across cluster nodes
# Usage: bash manage_containers.sh <start|stop|status|restart> [options]
#   start    [--npuslim] [--no-npuslim] [--hosts <ip1> ...] [-f <node_list>]
#   stop     [--hosts <ip1> ...] [-f <node_list>]
#   status   [--hosts <ip1> ...] [-f <node_list>]
#   restart  (stop + start, same options as start)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS_ROOT="${SCRIPT_DIR}/.."

# 加载共享库
# shellcheck disable=SC1091
source "${SCRIPTS_ROOT}/common.sh"

# ------------------------------------------
# 默认配置（可被环境变量或命令行参数覆盖）
# ------------------------------------------
SSH_USER="${SSH_USER:-root}"
IMAGE_NAME="${IMAGE_NAME:-ascend910c-cann8.5.1-torch2.9.0-vllm0.18.0}"
NODES_FILE="${NODES_FILE:-${SCRIPTS_ROOT}/node_list.txt}"
RUN_CONTAINER="${SCRIPT_DIR}/run_npuslim_container.sh"

# NPUSlim paths differ: master uses local project dir, workers use synced dir
MASTER_NPUSLIM_PATH="${MASTER_NPUSLIM_PATH:-/llm_workspace_1P/robin/npuslim-master}"
WORKER_NPUSLIM_PATH="${WORKER_NPUSLIM_PATH:-/llm_workspace_1P/robin/npuslim-master}"

# Detect local IPs
LOCAL_IPS=$(hostname -I 2>/dev/null || true)

# ------------------------------------------
# 节点列表解析
# ------------------------------------------
# 优先级: --hosts (CLI) > --file/-f (CLI) > NODES_FILE (env) > 默认 node_list.txt
# 用法: resolve_hosts [--hosts ip1 ip2 ...] [--file path]
# 输出: 将解析后的节点列表存入 HOSTS 数组
resolve_hosts() {
    HOSTS=()
    local nodes_file=""

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --hosts) shift
                while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                    HOSTS+=("$1"); shift
                done ;;
            --file|-f)
                shift || true
                [[ $# -gt 0 ]] || log_fatal "-f/--file 需要指定文件路径"
                nodes_file="$1"; shift
                ;;
            --npuslim|--no-npuslim) shift ;;
            *) shift ;;
        esac
    done

    # 如果已通过 --hosts 指定，直接返回
    [[ ${#HOSTS[@]} -gt 0 ]] && return 0

    # 否则从节点列表文件读取
    local file="${nodes_file:-$NODES_FILE}"
    if [[ ! -f "$file" ]]; then
        log_fatal "节点列表文件未找到: $file"
    fi

    while IFS= read -r ip; do
        [[ -n "$ip" ]] && HOSTS+=("$ip")
    done < <(read_nodes "$file")

    if [[ ${#HOSTS[@]} -eq 0 ]]; then
        log_fatal "节点列表文件为空: $file"
    fi
}

# ------------------------------------------
# 辅助函数
# ------------------------------------------
is_local() {
    local ip="$1" lip
    for lip in $LOCAL_IPS; do
        [[ "$ip" == "$lip" ]] && return 0
    done
    return 1
}

# Get npuslim path for a given host
npuslim_path_for() {
    local host="$1"
    local master="${HOSTS[0]:-}"
    if [[ "$host" == "$master" ]]; then
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
        # shellcheck disable=SC2029,SC2145
        ssh "${SSH_USER}@${host}" "docker $*" 2>/dev/null
    fi
}

# Run a bash command on remote or local node
remote_bash() {
    local host="$1"; shift
    if is_local "$host"; then
        # shellcheck disable=SC2145
        bash -c "$*"
    else
        # shellcheck disable=SC2029
        ssh "${SSH_USER}@${host}" "$@" 2>/dev/null
    fi
}

# ─── Commands ────────────────────────────────────────────────────────────────

cmd_start() {
    resolve_hosts "$@"
    local with_npuslim=true

    # 解析非 hosts 选项
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --no-npuslim) with_npuslim=false; shift ;;
            --npuslim) with_npuslim=true; shift ;;
            --hosts) shift
                while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do shift; done ;;
            --file|-f) shift || true; shift || true ;;
            *) shift ;;
        esac
    done

    echo "========================================"
    echo "Starting Containers"
    echo "========================================"
    echo "Hosts:   ${HOSTS[*]}"
    echo "NPUSlim: ${with_npuslim}"
    echo ""

    for host in "${HOSTS[@]}"; do
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
    resolve_hosts "$@"

    echo "========================================"
    echo "Stopping Containers"
    echo "========================================"
    echo "Hosts: ${HOSTS[*]}"
    echo ""

    for host in "${HOSTS[@]}"; do
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
    resolve_hosts "$@"

    echo "========================================"
    echo "Container Status"
    echo "========================================"

    for host in "${HOSTS[@]}"; do
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
        echo "  --hosts <ip1> [ip2] ...   Target specific hosts (highest priority)"
        echo "  -f, --file <path>         Node list file (default: NODES_FILE or scripts/node_list.txt)"
        echo "  --npuslim                 Mount npuslim source (default)"
        echo "  --no-npuslim              Skip npuslim mount"
        echo ""
        echo "Node list resolution (priority: high → low):"
        echo "  1. --hosts <ip1> <ip2>    Direct CLI host list"
        echo "  2. -f / --file <path>     CLI-specified node file"
        echo "  3. NODES_FILE             Environment variable"
        echo "  4. scripts/node_list.txt  Default file"
        echo ""
        echo "Environment variables:"
        echo "  SSH_USER                  SSH user (default: root)"
        echo "  IMAGE_NAME                Docker image name"
        echo "  NODES_FILE                Node list file path"
        echo "  MASTER_NPUSLIM_PATH       NPUSlim path on master node"
        echo "  WORKER_NPUSLIM_PATH       NPUSlim path on worker nodes"
        echo ""
        echo "Examples:"
        echo "  bash manage_containers.sh start"
        echo "  bash manage_containers.sh stop --hosts 10.42.15.195 10.42.15.196"
        echo "  bash manage_containers.sh status -f /path/to/node_list.txt"
        echo "  NODES_FILE=/tmp/nodes.txt bash manage_containers.sh start"
        ;;
esac
