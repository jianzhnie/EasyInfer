#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# shellcheck source=../common.sh
source "${SCRIPTS_ROOT}/common.sh"

# 项目目录
PROJECT_DIR="/root/llmtuner/llm/MindSpeed-RL-master"

# Ray 配置
MASTER_PORT="29500"
DASHBOARD_PORT="8266"
NPUS_PER_NODE=8
WAIT_TIME=2

# ------------------------------------------------------------------------------
# 节点列表解析
# ------------------------------------------------------------------------------
NODES=()

# 优先从命令行参数或环境变量获取节点列表
if [[ ${#NODES[@]} -eq 0 && -n "${NODE_LIST:-}" && -f "$NODE_LIST" ]]; then
    while IFS= read -r line; do
        NODES+=("$line")
    done < <(read_nodes "$NODE_LIST")
fi

# 回退到硬编码节点（向后兼容）
if [[ ${#NODES[@]} -eq 0 ]]; then
    NODES=(
        "10.16.201.201"
        "10.16.201.42"
    )
fi

NUM_NODES=${#NODES[@]}
MASTER_ADDR=${NODES[0]}
WORKERS=("${NODES[@]:1}")

# 验证参数
if [[ -z "$MASTER_ADDR" ]]; then
    log_fatal "MASTER_ADDR is empty"
fi

if [[ $NUM_NODES -eq 0 ]]; then
    log_fatal "No nodes defined"
fi

# 打印集群信息
log_info "============================================="
log_info "Ray Cluster Setup Configuration"
log_info "============================================="
log_info "Project directory: $PROJECT_DIR"
log_info "Number of nodes: $NUM_NODES"
log_info "NPUs per node: $NPUS_PER_NODE"
log_info "Master IP: $MASTER_ADDR"
log_info "Master port: $MASTER_PORT"
log_info "Worker nodes: ${WORKERS[*]:-none}"
log_info "============================================="

# 检查项目目录是否存在
check_project_dir() {
    local node=$1
    # shellcheck disable=SC2029
    if ! ssh "${SSH_USER:-root}@${node}" "[ -d \"$PROJECT_DIR\" ]"; then
        log_err "Project directory $PROJECT_DIR not found on node $node"
        return 1
    fi
    return 0
}

# 启动 Ray 节点函数
start_ray_node() {
    local node=$1
    local is_head=$2
    local cmd

    if $is_head; then
        log_info "[HEAD] Starting Ray head node on $node..."
        cmd="ray start --head --port $MASTER_PORT --node-ip-address $MASTER_ADDR --dashboard-host=0.0.0.0 --dashboard-port=$DASHBOARD_PORT --resources='{\"NPU\": $NPUS_PER_NODE}'"
    else
        log_info "[WORKER] Starting Ray worker node on $node..."
        cmd="ray start --address $MASTER_ADDR:$MASTER_PORT --resources='{\"NPU\": $NPUS_PER_NODE}'"
    fi

    # shellcheck disable=SC2029
    if ! ssh "${SSH_USER:-root}@${node}" "cd $PROJECT_DIR && source set_env.sh && ray stop >/dev/null 2>&1 || true && $cmd"; then
        log_err "Failed to start Ray on node $node"
        return 1
    fi
    return 0
}

# 检查所有节点的项目目录
for node in "${NODES[@]}"; do
    if ! check_project_dir "$node"; then
        exit 1
    fi
done

# 启动头节点
if ! start_ray_node "$MASTER_ADDR" true; then
    exit 1
fi

# 等待头节点完全启动
log_info "Waiting ${WAIT_TIME}s for head node to initialize..."
sleep "$WAIT_TIME"

# 并行启动工作节点
pids=()
for worker in "${WORKERS[@]}"; do
    start_ray_node "$worker" false &
    pids+=($!)
done

# 等待所有工作节点启动完成
failed=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        failed=$((failed + 1))
    fi
done

if [[ $failed -gt 0 ]]; then
    log_warn "$failed worker node(s) failed to start"
fi

echo ""
log_info "============================================="
log_info "Ray cluster setup complete!"
log_info "Dashboard URL: http://$MASTER_ADDR:$DASHBOARD_PORT"
log_info "============================================="
