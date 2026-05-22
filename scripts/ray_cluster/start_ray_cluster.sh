#!/usr/bin/env bash
#
# Ray Cluster Launcher — 启动 / 停止多节点 Ray 集群
#
# 用法:
#   start_ray_cluster.sh start  [options]
#   start_ray_cluster.sh stop   [options]
#   start_ray_cluster.sh status [options]
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

# 强制设置 SSH 选项以确保输出纯净
export SSH_OPTS="-q -o BatchMode=yes -o ConnectTimeout=5 -o StrictHostKeyChecking=no"

# -----------------------------------------------------------------
# 参数解析
# -----------------------------------------------------------------
usage() {
    cat <<EOF
Usage: $0 <start|stop> [--file <node_list>]

所有集群配置（端口、容器名、NPU 数等）请在 set_ray_env.sh 中设置。
可通过环境变量覆盖，例如: RAY_PORT=6380 $0 start
EOF
    exit 1
}

ACTION=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        start|stop) ACTION="$1"; shift ;;
        --file|-f)  NODE_LIST="$2"; shift 2 ;;
        --help|-h)  usage ;;
        *) log_err "未知参数: $1"; usage ;;
    esac
done

[[ -n "${ACTION:-}" ]] || { log_err "缺少操作 (start 或 stop)"; usage; }

# -----------------------------------------------------------------
# 临时目录 & 清理
# -----------------------------------------------------------------
_TEMP_DIR=$(mktemp -d)
cleanup() {
    local rc=$?
    [[ -n "${_TEMP_DIR:-}" && -d "$_TEMP_DIR" ]] && rm -rf "$_TEMP_DIR"
    exit $rc
}
trap cleanup EXIT INT TERM

# -----------------------------------------------------------------
# 辅助函数
# -----------------------------------------------------------------

# 检查容器是否在指定节点运行
check_container() {
    local node=$1
    ssh_run "$node" "docker ps --format '{{.Names}}' | grep -qx '${CONTAINER_NAME}'" 2>/dev/null
}

# 在远程节点的容器内执行命令
# 使用 base64 编码避免多层引号转义问题
remote_exec() {
    local node=$1 cmd=$2
    local b64cmd
    b64cmd=$(printf '%s' "$cmd" | base64 | tr -d '\n')
    ssh_run "$node" "docker exec -i \"${CONTAINER_NAME}\" bash -c \"
        if [ ! -f \\\"${RAY_ENV_SCRIPT}\\\" ]; then
            echo 'Error: RAY_ENV_SCRIPT not found at ${RAY_ENV_SCRIPT} in container' >&2
            exit 1
        fi
        source \\\"${RAY_ENV_SCRIPT}\\\"
        echo '${b64cmd}' | base64 -d | bash\""
}

# 停止单个节点的 Ray（容错，静默）
stop_ray_on_node() {
    local node=$1
    remote_exec "$node" "ray stop -f 2>/dev/null || true" >/dev/null 2>&1 || true
}

# 启动 Ray Head
start_head() {
    local head_node=$1
    log_info "正在启动 Ray head 节点: $head_node"
    local cmd
    printf -v cmd \
        'ray start --head --port=%s --resources='"'"'{"NPU":%s}'"'" \
        "$RAY_PORT" "$NPUS_PER_NODE"
    remote_exec "$head_node" "$cmd"
}



# 启动 Ray Worker
start_worker() {
    local worker_node=$1 head_addr=$2
    log_info "Worker 加入集群: $worker_node"
    local cmd
    printf -v cmd \
        'ray start --address=%s:%s --node-ip-address=%s --resources='"'"'{"NPU":%s}'"'" \
        "$head_addr" "$RAY_PORT" "$worker_node" "$NPUS_PER_NODE"
    remote_exec "$worker_node" "$cmd"
}

# -----------------------------------------------------------------
# 主流程
# -----------------------------------------------------------------

# 读取节点列表
log_info "正在读取节点列表..."
[[ -n "${NODE_LIST:-}" && -f "$NODE_LIST" ]] || log_fatal "节点列表文件未找到: ${NODE_LIST:-未设置}"
mapfile -t NODES < <(read_nodes "$NODE_LIST")
[[ ${#NODES[@]} -gt 0 ]] || log_fatal "节点列表为空: $NODE_LIST"

HEAD_NODE="${NODES[0]}"
WORKERS=("${NODES[@]:1}")

log_info "============================================="
log_info "Ray 集群操作: $ACTION"
log_info "Head:       $HEAD_NODE"
log_info "Workers:    ${WORKERS[*]:-无}"
log_info "Container:  $CONTAINER_NAME"
log_info "Ray Port:   $RAY_PORT"
log_info "NPUs/Node:  $NPUS_PER_NODE"
log_info "============================================="

# --- stop ---
if [[ "$ACTION" == "stop" ]]; then
    log_info "正在停止所有节点上的 Ray 进程..."
    for node in "${NODES[@]}"; do
        limit_jobs "$MAX_SSH_PARALLELISM"
        stop_ray_on_node "$node" &
    done
    wait
    log_info "集群已停止."
    exit 0
fi

# --- start ---

# Step 1: 前置检查（并行）
log_info "[1/5] 检查容器状态..."
for node in "${NODES[@]}"; do
    limit_jobs "$MAX_SSH_PARALLELISM"
    (
        if ! check_container "$node"; then
            log_err "容器 '$CONTAINER_NAME' 未在 $node 上运行"
            touch "${_TEMP_DIR}/preflight_fail"
        fi
    ) &
done
wait
[[ ! -f "${_TEMP_DIR}/preflight_fail" ]] || log_fatal "前置检查失败，请确认容器状态"

# Step 2: 清理已有 Ray 进程（执行两次确保干净）
log_info "[2/5] 清理已有 Ray 进程..."
for _ in 1 2; do
    for node in "${NODES[@]}"; do
        limit_jobs "$MAX_SSH_PARALLELISM"
        stop_ray_on_node "$node" &
    done
    wait
    sleep 2
done
log_info "清理完成."

# Step 3: 启动 Head
log_info "[3/5] 启动 Head 节点 $HEAD_NODE..."
start_head "$HEAD_NODE" || log_fatal "Head 节点启动失败: $HEAD_NODE"
log_info "等待 ${WAIT_TIME}s 完成初始化..."
sleep "$WAIT_TIME"

# Step 4: 并行启动 Workers
if [[ ${#WORKERS[@]} -gt 0 ]]; then
    log_info "[4/5] 并行启动 ${#WORKERS[@]} 个 Worker..."
    for worker_node in "${WORKERS[@]}"; do
        limit_jobs "$MAX_SSH_PARALLELISM"
        (
            if ! start_worker "$worker_node" "$HEAD_NODE"; then
                log_err "Worker 启动失败: $worker_node"
                touch "${_TEMP_DIR}/worker_fail"
            fi
        ) &
    done
    wait
else
    log_info "[4/5] 单节点模式，跳过 Worker 启动."
fi

# Step 5: 验证集群
log_info "[5/5] 验证集群状态..."
start_time=$(date +%s)

while true; do
    status_output=$(remote_exec "$HEAD_NODE" "ray status" 2>/dev/null || echo "")
    # 使用 grep | wc -l 避免 grep -c 在未匹配时返回非零状态导致的 || echo "0" 重复输出问题
    current_nodes=$(echo "$status_output" | grep "node_id" | wc -l | xargs)

    if [[ -n "$current_nodes" && "$current_nodes" -ge "${#NODES[@]}" ]]; then
        log_info "所有 ${#NODES[@]} 个节点已成功加入集群."
        break
    fi

    elapsed=$(( $(date +%s) - start_time ))
    if [[ "$elapsed" -gt "$VERIFY_TIMEOUT" ]]; then
        log_err "验证超时 (${VERIFY_TIMEOUT}s). 当前: $current_nodes/${#NODES[@]}"
        log_warn "请检查节点连通性和容器日志"
        break
    fi

    log_info "等待节点加入... $current_nodes/${#NODES[@]} (${elapsed}s)"
    sleep 5
done

# 显示最终状态
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
