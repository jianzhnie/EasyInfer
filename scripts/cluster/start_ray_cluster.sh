#!/usr/bin/env bash
#
# Ray Cluster Launcher
#
# 启动多节点 Ray 集群，支持配置 NPU 资源
# 依赖: 所有节点上安装 Ray 并配置无密码 SSH 登录
#
# 环境变量（均可通过外部覆盖）:
#   NODES_FILE         - 节点列表文件路径（默认: scripts/node_list.txt）
#   PARALLELISM        - 并发控制上限（默认: 16）
#   MASTER_ADDR        - 指定主节点地址（默认: 节点列表第一个）
#   SSH_USER_HOST_PREFIX - SSH 用户前缀，如 "user@"（默认: 空）
#   SSH_OPTS           - 额外的 SSH 选项

set -euo pipefail

# -----------------------------------------------------------------
# 路径配置
# -----------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${SCRIPTS_DIR}/vllm/set_env.sh"
NODE_LIST_FILE="${NODES_FILE:-${SCRIPTS_DIR}/node_list.txt}"
PARALLELISM="${PARALLELISM:-16}"

# 加载共享工具函数
source "${SCRIPTS_DIR}/common.sh"

# -----------------------------------------------------------------
# 远程执行辅助函数
# -----------------------------------------------------------------

# 增强版 ssh，支持 -q 等前置 flag 和 SSH_USER_HOST_PREFIX/SSH_OPTS
ssh_cmd() {
    local flags=()
    while [[ $# -gt 0 && "$1" == -* ]]; do
        flags+=("$1"); shift
    done
    local node="$1"; shift
    ssh "${flags[@]}" -o BatchMode=yes -o StrictHostKeyChecking=accept-new \
        -o ConnectTimeout=5 ${SSH_OPTS:-} \
        "${SSH_USER_HOST_PREFIX:-}$node" "$@"
}

# 在远程节点的容器内执行命令
# 使用 base64 编码避免多层 Shell 引号转义问题
# 注意: 不在本层吞掉 stderr —— 调用方按需处理
remote_exec() {
    local node=$1 cmd=$2
    local b64cmd
    b64cmd=$(printf '%s' "$cmd" | base64 | tr -d '\n')
    ssh_cmd "$node" "source '${ENV_FILE}' 2>/dev/null && \
        docker exec -i '${CONTAINER_NAME}' bash -c \"echo '${b64cmd}' | base64 -d | bash\""
}

# -----------------------------------------------------------------
# Ray 操作函数
# -----------------------------------------------------------------

# 停止单个节点的 Ray 进程（允许失败，不打印错误信息）
stop_ray_on_node() {
    local node=$1
    log_info "[STOP] Stopping Ray on $node"
    remote_exec "$node" "ray stop -f 2>/dev/null || true" 2>/dev/null || true
}

# 停止所有节点的 Ray 进程（并行执行）
stop_all_ray() {
    local nodes=("$@")
    log_info "Stopping Ray on ${#nodes[@]} node(s)..."

    local stopdir="${_TEMP_DIR}/stop"
    mkdir -p "$stopdir"

    for i in "${!nodes[@]}"; do
        limit_jobs "$PARALLELISM"
        (
            stop_ray_on_node "${nodes[$i]}" && echo "OK" > "$stopdir/$i" || echo "FAIL" > "$stopdir/$i"
        ) &
    done

    wait

    local failed=0
    for i in "${!nodes[@]}"; do
        [[ "$(cat "$stopdir/$i" 2>/dev/null)" == "OK" ]] || ((failed++))
    done

    [[ $failed -eq 0 ]] && log_info "All Ray processes stopped" || log_warn "$failed node(s) failed to stop"
}

start_head() {
    local node=$1
    log_info "[HEAD] Starting Ray head on $node"

    # 由 remote_exec 的 base64 编码保护，JSON 无需额外转义
    local cmd
    printf -v cmd \
        'ray start --head --node-ip-address=%s --port=%s --dashboard-host=0.0.0.0 --dashboard-port=%s --num-gpus=%s --resources='"'"'{"NPU":%s}'"'" \
        "$node" "$MASTER_PORT" "$DASHBOARD_PORT" "$NPUS_PER_NODE" "$NPUS_PER_NODE"

    remote_exec "$node" "$cmd"
}

start_worker() {
    local node=$1 master=$2
    log_info "[WORKER] Starting Ray worker on $node"

    local cmd
    printf -v cmd \
        'ray start --address=%s:%s --num-gpus=%s --resources='"'"'{"NPU":%s}'"'" \
        "$master" "$MASTER_PORT" "$NPUS_PER_NODE" "$NPUS_PER_NODE"

    remote_exec "$node" "$cmd"
}

# -----------------------------------------------------------------
# 前置检查
# -----------------------------------------------------------------
check_env() {
    [[ -f "$ENV_FILE" ]] || log_fatal "Environment file not found: $ENV_FILE"
    source "$ENV_FILE"

    # 验证 source 后必备变量已设置，避免后续报错不明
    local required_vars=(
        CONTAINER_NAME MASTER_PORT DASHBOARD_PORT NPUS_PER_NODE
    )
    local missing=()
    for var in "${required_vars[@]}"; do
        [[ -z "${!var:-}" ]] && missing+=("$var")
    done
    [[ ${#missing[@]} -eq 0 ]] || log_fatal "Required variables not set in ${ENV_FILE}: ${missing[*]}"
}

check_node() {
    local node=$1
    ssh_cmd -q "$node" "test -f '$ENV_FILE'" 2>/dev/null || {
        log_err "SSH failed or missing env file on: $node"
        return 1
    }
}

# -----------------------------------------------------------------
# 临时目录清理
# -----------------------------------------------------------------
cleanup() {
    local rc=$?
    [[ -n "${_TEMP_DIR:-}" && -d "$_TEMP_DIR" ]] && rm -rf "$_TEMP_DIR"
    exit $rc
}

# -----------------------------------------------------------------
# 主流程
# -----------------------------------------------------------------
main() {
    trap cleanup EXIT INT TERM
    _TEMP_DIR=$(mktemp -d)
    check_env

    # 解析节点列表
    [[ -f "$NODE_LIST_FILE" ]] || log_fatal "Node list file not found: $NODE_LIST_FILE"
    mapfile -t NODES < <(awk 'NF && !/^#/ {print $1}' "$NODE_LIST_FILE")
    [[ ${#NODES[@]} -gt 0 ]] || log_fatal "No valid hosts in: $NODE_LIST_FILE"

    local master="${MASTER_ADDR:-${NODES[0]}}"
    local num_nodes=${#NODES[@]}

    local workers=()
    for node in "${NODES[@]}"; do
        [[ "$node" != "$master" ]] && workers+=("$node")
    done

    log_info "============================================="
    log_info "Ray Cluster Configuration"
    log_info "============================================="
    log_info "Total nodes: $num_nodes"
    log_info "NPUs per node: $NPUS_PER_NODE"
    log_info "Master: ${BLUE}${master}${NC}:${MASTER_PORT}"
    log_info "Dashboard: http://${master}:${DASHBOARD_PORT}"
    log_info "Workers (${#workers[@]}): ${workers[*]:-None}"
    log_info "============================================="

    log_info "Checking all nodes..."
    local failed=0
    for node in "${NODES[@]}"; do
        check_node "$node" || ((failed++))
    done
    [[ $failed -eq 0 ]] || log_fatal "Pre-checks failed with $failed error(s)"
    log_info "All checks passed"

    # Step 1: 停止所有节点的 Ray 进程（执行两次确保干净清理）
    log_info "============================================="
    stop_all_ray "${NODES[@]}"
    sleep 2
    stop_all_ray "${NODES[@]}"
    sleep 2

    # Step 2: 启动 Head 节点
    log_info "============================================="
    start_head "$master" || log_fatal "Failed to start head node"
    sleep "${WAIT_TIME:-2}"

    # Step 3: 启动 Worker 节点（并行）
    log_info "============================================="
    local failed_nodes=() success_count=1  # head 已启动成功

    if [[ ${#workers[@]} -gt 0 ]]; then
        log_info "Starting ${#workers[@]} worker(s) in parallel..."

        local workdir="${_TEMP_DIR}/workers"
        mkdir -p "$workdir"

        for i in "${!workers[@]}"; do
            limit_jobs "$PARALLELISM"
            (
                start_worker "${workers[$i]}" "$master" && echo "OK" > "$workdir/$i" || echo "FAIL" > "$workdir/$i"
            ) &
        done

        wait

        for i in "${!workers[@]}"; do
            if [[ "$(cat "$workdir/$i" 2>/dev/null)" == "OK" ]]; then
                ((success_count++))
            else
                failed_nodes+=("${workers[$i]}")
            fi
        done
    else
        log_info "Single-node cluster mode"
    fi

    # 结果报告
    echo ""
    echo -e "${BLUE}=============================================${NC}"
    if [[ ${#failed_nodes[@]} -eq 0 ]]; then
        log_info "Ray cluster started successfully!"
    else
        log_warn "Cluster started with ${#failed_nodes[@]} failed worker(s)"
        log_info "Failed nodes: ${failed_nodes[*]}"
    fi
    log_info "Dashboard: http://${master}:${DASHBOARD_PORT}"
    log_info "Running: $success_count / $num_nodes nodes"
    echo -e "${BLUE}=============================================${NC}"

    # 显示状态
    log_info "Ray cluster status:"
    ssh_cmd "$master" "source '${ENV_FILE}' 2>/dev/null && \
        docker exec -i '${CONTAINER_NAME}' \
        bash -c 'ray status'" 2>/dev/null || log_warn "Could not retrieve Ray status"
}

main "$@"
