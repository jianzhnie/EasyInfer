#!/bin/bash
# ==========================================
# Ray 集群管理脚本
# 启动 / 停止多节点 Ray 集群
# ==========================================
#
# 依赖:
#   - 所有节点配置无密码 SSH
#   - 目标容器已运行
#   - set_ray_env.sh 在容器内可访问 ($RAY_ENV_SCRIPT)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

source "${SCRIPTS_ROOT}/common.sh"
source "${SCRIPT_DIR}/set_ray_env.sh"

# 配置
KILL_TIMEOUT="${KILL_TIMEOUT:-3}"
PARALLELISM="${PARALLELISM:-16}"
RAY_KEYWORDS="raylet|plasma_store|gcs_server|ray::|ray.worker|python.*ray|dashboard_agent|runtime_env_agent"

# ------------------------------------------
# 帮助信息
# ------------------------------------------
usage() {
    cat <<'USAGE'
Usage:
  bash manage_ray_cluster.sh <start|stop> [OPTIONS]

Options:
  -f, --file <FILE>   节点列表文件路径
  --on-host           在宿主机上停止 Ray（不进容器，仅 stop 有效）
  --force             强制模式：立即停止并清理所有残余（仅 stop 有效）
  -y, --yes           跳过确认步骤（仅 stop 有效）
  -h, --help          显示帮助信息

Environment Variables:
  NODE_LIST, KILL_TIMEOUT, PARALLELISM, CONTAINER_NAME

环境变量配置: scripts/ray_cluster/set_ray_env.sh
USAGE
    exit "$E_INVALID_ARG"
}

# ------------------------------------------
# 参数解析
# ------------------------------------------
parse_args() {
    ACTION=""
    ON_HOST=false
    FORCE=false
    SKIP_CONFIRM=false

    while [[ $# -gt 0 ]]; do
        case "$1" in
            start|stop)   ACTION="$1"; shift ;;
            --file|-f)
                [[ -n "${2:-}" && "$2" != -* ]] || { log_err "选项 $1 需要一个参数"; usage; }
                NODE_LIST="$2"; shift 2 ;;
            --on-host)    ON_HOST=true; shift ;;
            --force)      FORCE=true; shift ;;
            -y|--yes)     SKIP_CONFIRM=true; export SKIP_CONFIRM; shift ;;
            --help|-h)    usage ;;
            *)            log_err "未知参数: $1"; usage ;;
        esac
    done

    [[ -n "$ACTION" ]] || { log_err "缺少操作 (start 或 stop)"; usage; }

    if [[ "$ACTION" == "start" ]]; then
        $ON_HOST && { log_err "--on-host 仅适用于 stop 操作"; usage; }
        $FORCE && { log_err "--force 仅适用于 stop 操作"; usage; }
    fi
}

# ------------------------------------------
# 通用工具函数
# ------------------------------------------

load_nodes() {
    [[ -n "${NODE_LIST:-}" && -f "$NODE_LIST" ]] || log_fatal "节点列表文件未找到: ${NODE_LIST:-未设置}"
    mapfile -t NODES < <(read_nodes "$NODE_LIST")
    [[ ${#NODES[@]} -gt 0 ]] || log_fatal "节点列表为空: $NODE_LIST"

    HEAD_NODE="${NODES[0]}"
    WORKERS=("${NODES[@]:1}")
}

print_cluster_info() {
    log_info "============================================="
    log_info "Ray 集群操作: $ACTION"
    log_info "Head:       $HEAD_NODE"
    log_info "Workers:    ${WORKERS[*]:-无}"
    log_info "Container:  $CONTAINER_NAME"
    log_info "Ray Port:   $RAY_PORT"
    log_info "NPUs/Node:  $NPUS_PER_NODE"
    if [[ "$ACTION" == "stop" ]]; then
        log_info "Mode:       $(if $ON_HOST; then echo '宿主机'; else echo '容器内'; fi)"
        log_info "Force:      $FORCE"
    fi
    log_info "============================================="
}

check_container() {
    local node=$1
    ssh_run "$node" "docker ps --format '{{.Names}}' | grep -qx '${CONTAINER_NAME}'" 2>/dev/null
}

# 在远程节点的容器内执行命令（base64 编码避免引号转义）
remote_exec() {
    local node=$1 cmd=$2 b64cmd
    b64cmd=$(printf '%s' "$cmd" | base64 | tr -d '\n')
    ssh_run "$node" "docker exec -i \"${CONTAINER_NAME}\" bash -c \"
        if [ ! -f \\\"${RAY_ENV_SCRIPT}\\\" ]; then
            echo 'Error: RAY_ENV_SCRIPT not found at ${RAY_ENV_SCRIPT} in container' >&2
            exit 1
        fi
        source \\\"${RAY_ENV_SCRIPT}\\\"
        echo '${b64cmd}' | base64 -d | bash\""
}

# 通用并行任务执行器
# 用法: run_on_nodes "max_jobs" "task_func" "${NODES[@]}"
run_on_nodes() {
    local max_jobs="$1" task_func="$2"
    shift 2
    local node
    for node in "$@"; do
        limit_jobs "$max_jobs"
        "$task_func" "$node" &
    done
    wait
}

# ------------------------------------------
# start 相关函数
# ------------------------------------------

start_preflight_check() {
    local node=$1
    if ! check_container "$node"; then
        log_err "容器 '$CONTAINER_NAME' 未在 $node 上运行"
        touch "${_TEMP_DIR}/preflight_fail"
    fi
}

stop_ray_on_node() {
    remote_exec "$1" "ray stop -f 2>/dev/null || true" >/dev/null 2>&1 || true
}

start_head() {
    local head_node=$1
    log_info "正在启动 Ray head 节点: $head_node"
    local cmd
    printf -v cmd 'ray start --head --port=%s --resources='\''{"NPU":%s}'\' "$RAY_PORT" "$NPUS_PER_NODE"
    remote_exec "$head_node" "$cmd"
}

start_worker() {
    local worker_node=$1 head_addr=$2
    log_info "Worker 加入集群: $worker_node"
    local cmd
    printf -v cmd 'ray start --address=%s:%s --node-ip-address=%s --resources='\''{"NPU":%s}'\' \
        "$head_addr" "$RAY_PORT" "$worker_node" "$NPUS_PER_NODE"
    remote_exec "$worker_node" "$cmd"
}

start_worker_or_mark_failed() {
    local worker_node=$1
    if ! start_worker "$worker_node" "$HEAD_NODE"; then
        log_err "Worker 启动失败: $worker_node"
        touch "${_TEMP_DIR}/worker_fail"
    fi
}

cleanup_existing_ray() {
    log_info "[2/5] 清理已有 Ray 进程..."
    local _
    for _ in {1..2}; do
        run_on_nodes "$MAX_SSH_PARALLELISM" stop_ray_on_node "${NODES[@]}"
        sleep 2
    done
    log_info "清理完成."
}

wait_for_cluster_ready() {
    log_info "[5/5] 验证集群状态..."
    local start_time elapsed current_nodes status_output
    start_time=$(date +%s)

    while true; do
        status_output=$(remote_exec "$HEAD_NODE" "ray status" 2>/dev/null || echo "")
        current_nodes=$(echo "$status_output" | grep -cE '^[[:space:]]+[0-9]+[[:space:]]+node_' || true)
        current_nodes="${current_nodes//[$'\n']/}"

        if [[ "$current_nodes" -ge "${#NODES[@]}" ]]; then
            log_info "所有 ${#NODES[@]} 个节点已成功加入集群."
            return 0
        fi

        elapsed=$(( $(date +%s) - start_time ))
        if [[ "$elapsed" -gt "$VERIFY_TIMEOUT" ]]; then
            log_err "验证超时 (${VERIFY_TIMEOUT}s). 当前: $current_nodes/${#NODES[@]}"
            log_warn "请检查节点连通性和容器日志"
            return 1
        fi

        log_info "等待节点加入... $current_nodes/${#NODES[@]} (${elapsed}s)"
        sleep 5
    done
}

print_final_status() {
    echo ""
    log_info "============================================="
    echo ""
    remote_exec "$HEAD_NODE" "ray status" 2>&1 || log_warn "无法获取 Ray 集群状态"
    echo ""
    if [[ -f "${_TEMP_DIR}/worker_fail" ]]; then
        log_warn "Ray 集群部分启动 (有 Worker 失败)"
    else
        log_info "Ray 集群启动完成!"
    fi
    log_info "============================================="
}

# ------------------------------------------
# stop 相关函数
# ------------------------------------------

_remote_stop_ray() {
    local force="$1" kill_timeout="$2" pattern="$3"
    set -euo pipefail

    get_ray_pids() {
        # shellcheck disable=SC2009
        ps aux | grep -E "$pattern" | grep -v grep | awk '{print $2}' | sort -u | tr '\n' ' ' || true
    }

    kill_procs() {
        local sig="$1" pids="$2"
        [[ -n "$pids" ]] || return 0
        # shellcheck disable=SC2086
        for pid in $pids; do kill -"$sig" "$pid" 2>/dev/null || true; done
    }

    if command -v ray >/dev/null 2>&1; then
        [[ "$force" == "true" ]] && ray stop -f --grace-period 0 >/dev/null 2>&1 || ray stop -f >/dev/null 2>&1 || true
    fi

    local pids remaining final
    pids=$(get_ray_pids)
    [[ -n "$pids" ]] || { echo "未找到 Ray 进程"; return 0; }

    echo "找到 Ray 进程: $pids"

    if [[ "$force" == "true" ]]; then
        echo "强制终止..."
        kill_procs 9 "$pids"
    else
        echo "温和终止 (SIGTERM)..."
        kill_procs 15 "$pids"
        sleep "$kill_timeout"

        remaining=$(get_ray_pids)
        if [[ -n "$remaining" ]]; then
            echo "强制终止残余进程 (SIGKILL)..."
            kill_procs 9 "$remaining"
            sleep 0.5
        fi
    fi

    final=$(get_ray_pids)
    if [[ -n "$final" ]]; then
        echo "警告: 仍有残余进程: $final"
        return 1
    fi
    echo "Ray 进程清理完成"
}

stop_ray_node() {
    local node="$1"
    log_info "[${node}] 停止 Ray..."

    local func call
    func=$(declare -f _remote_stop_ray)
    call="_remote_stop_ray '${FORCE}' '${KILL_TIMEOUT}' '${RAY_KEYWORDS}'"

    if $ON_HOST; then
        if echo "$func; $call" | ssh_run "$node" bash -s; then
            log_info "[${node}] 已停止"
        else
            log_err "[${node}] 停止失败"
        fi
    else
        local env_file="${SCRIPT_DIR}/set_ray_env.sh"
        local cmd="[[ -f '${env_file}' ]] && source '${env_file}'; docker exec -i \"${CONTAINER_NAME}\" bash -s"
        if echo "$func; $call" | ssh_run "$node" "$cmd"; then
            log_info "[${node}] 已停止"
        else
            log_err "[${node}] 停止失败"
        fi
    fi
}

# ------------------------------------------
# 主流程分发
# ------------------------------------------

main() {
    parse_args "$@"

    _TEMP_DIR=$(mktemp_dir)
    load_nodes
    print_cluster_info

    case "$ACTION" in
        stop)
            if ! confirm "将停止以上节点的 Ray 集群，是否继续?"; then
                log_info "已取消"
                exit 0
            fi
            log_info "正在停止所有节点上的 Ray 进程..."
            run_on_nodes "$PARALLELISM" stop_ray_node "${NODES[@]}"
            log_info "集群已停止."
            ;;
        start)
            log_info "[1/5] 检查容器状态..."
            run_on_nodes "$MAX_SSH_PARALLELISM" start_preflight_check "${NODES[@]}"
            [[ ! -f "${_TEMP_DIR}/preflight_fail" ]] || log_fatal "前置检查失败，请确认容器状态"

            cleanup_existing_ray

            log_info "[3/5] 启动 Head 节点 $HEAD_NODE..."
            start_head "$HEAD_NODE" || log_fatal "Head 节点启动失败: $HEAD_NODE"
            log_info "等待 ${WAIT_TIME}s 完成初始化..."
            sleep "$WAIT_TIME"

            if [[ ${#WORKERS[@]} -gt 0 ]]; then
                log_info "[4/5] 并行启动 ${#WORKERS[@]} 个 Worker..."
                run_on_nodes "$MAX_SSH_PARALLELISM" start_worker_or_mark_failed "${WORKERS[@]}"
            else
                log_info "[4/5] 单节点模式，跳过 Worker 启动."
            fi

            wait_for_cluster_ready
            print_final_status
            ;;
    esac
}

main "$@"
