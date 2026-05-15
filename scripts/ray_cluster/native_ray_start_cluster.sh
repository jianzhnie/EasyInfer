#!/bin/bash
set -e  # 遇到错误立即退出

# 项目目录
PROJECT_DIR="/root/llmtuner/llm/MindSpeed-RL-master"

# 定义节点IP数组
NODES=(
    "10.16.201.201"
    "10.16.201.42"
)

# Ray配置
MASTER_ADDR=${NODES[0]}
MASTER_PORT="29500"
DASHBOARD_PORT="8266"
NPUS_PER_NODE=8
NUM_NODES=${#NODES[@]}
WORKERS=("${NODES[@]:1}")  # 除第一个节点外的所有节点作为worker

# 验证参数
if [[ -z "$MASTER_ADDR" ]]; then
    echo "Error: MASTER_ADDR is empty!" >&2
    exit 1
fi

if [[ $NUM_NODES -eq 0 ]]; then
    echo "Error: No nodes defined!" >&2
    exit 1
fi
# 打印集群信息
echo "============================================="
echo "Ray Cluster Setup Configuration"
echo "============================================="
echo "Project directory: $PROJECT_DIR"
echo "Number of nodes: $NUM_NODES"
echo "NPUs per node: $NPUS_PER_NODE"
echo "Master IP: $MASTER_ADDR"
echo "Master port: $MASTER_PORT"
echo "Dashboard port: $DASHBOARD_PORT"
echo "Worker nodes: ${WORKERS[*]}"
echo "============================================="
echo ""

# 检查项目目录是否存在
check_project_dir() {
    local node=$1
    if ! ssh "$node" "[ -d \"$PROJECT_DIR\" ]"; then
        echo "Error: Project directory $PROJECT_DIR not found on node $node" >&2
        return 1
    fi
    return 0
}

# 启动Ray节点函数
start_ray_node() {
    local node=$1
    local is_head=$2
    local cmd
    
    if $is_head; then
        echo "[HEAD] Starting Ray head node on $node..."
        cmd="ray start --head --port $MASTER_PORT --node-ip-address $MASTER_ADDR --dashboard-host=0.0.0.0 --dashboard-port=$DASHBOARD_PORT --resources='{\"NPU\": $NPUS_PER_NODE}'"
    else
        echo "[WORKER] Starting Ray worker node on $node..."
        cmd="ray start --address $MASTER_ADDR:$MASTER_PORT --resources='{\"NPU\": $NPUS_PER_NODE}'"
    fi
    
    if ! ssh "$node" "cd $PROJECT_DIR && source set_env.sh && ray stop >/dev/null 2>&1 || true && $cmd"; then
        echo "Error: Failed to start Ray on node $node" >&2
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
echo "Waiting 10 seconds for head node to initialize..."
sleep 2

# 并行启动工作节点
pids=()
for worker in "${WORKERS[@]}"; do
    start_ray_node "$worker" false &
    pids+=($!)
done

# 等待所有工作节点启动完成
for pid in "${pids[@]}"; do
    if ! wait $pid; then
        echo "Warning: Some worker nodes failed to start" >&2
    fi
done

echo ""
echo "============================================="
echo "Ray cluster setup complete!"
echo "Dashboard URL: http://$MASTER_ADDR:$DASHBOARD_PORT"
echo "============================================="