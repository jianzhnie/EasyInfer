#!/usr/bin/env bash
#
# Optimized Ray Cluster Launcher
# High-performance and robust deployment of Ray clusters on Docker containers.
#

set -euo pipefail

# -----------------------------------------------------------------
# 1. Configuration & Environment
# -----------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Source common utilities and environment variables
# shellcheck source=../common.sh
source "${SCRIPTS_ROOT}/common.sh"

# Overrideable configurations
NODE_LIST="${NODES_FILE:-/llm_workspace_1P/robin/EasyInfer/scripts/node_list.txt}"
HEAD_NODE_SCRIPT="${HEAD_NODE_SCRIPT:-${SCRIPT_DIR}/ray_head.sh}"
WORKER_NODE_SCRIPT="${WORKER_NODE_SCRIPT:-${SCRIPT_DIR}/ray_node.sh}"
CONTAINER_NAME="${CONTAINER_NAME:-vllm-ascend-0.18-env}"
MAX_SSH_PARALLELISM="${PARALLELISM:-10}"
VERIFY_TIMEOUT=${WAIT_TIME:-60}

# -----------------------------------------------------------------
# 2. Node Preparation
# -----------------------------------------------------------------
if [[ ! -f "$NODE_LIST" ]]; then
    log_fatal "Node list file not found: $NODE_LIST"
fi

NODES=($(read_nodes "$NODE_LIST"))
if [ ${#NODES[@]} -eq 0 ]; then
    log_fatal "No nodes found in $NODE_LIST"
fi

HEAD_NODE=${NODES[0]}
WORKERS=("${NODES[@]:1}")

log_info "============================================="
log_info "Ray Cluster Starting..."
log_info "Head Node:      $HEAD_NODE"
log_info "Worker Nodes:   ${WORKERS[*]:-None}"
log_info "Container:      $CONTAINER_NAME"
log_info "Parallelism:    $MAX_SSH_PARALLELISM"
log_info "============================================="

# -----------------------------------------------------------------
# 3. Helper Function
# -----------------------------------------------------------------
# Executes a command inside the target container on a remote node
remote_exec() {
    local node=$1
    local cmd=$2
    local silent=${3:-false}
    
    # Use ssh_run from common.sh for consistent SSH options
    if [ "$silent" = true ]; then
        ssh_run "$node" "docker exec -i $CONTAINER_NAME bash -c \"$cmd\"" >/dev/null 2>&1
    else
        ssh_run "$node" "docker exec -i $CONTAINER_NAME bash -c \"$cmd\""
    fi
}

# -----------------------------------------------------------------
# 4. Main Process
# -----------------------------------------------------------------

# Step 0: Stop existing Ray processes on all nodes
log_info "Step 0: Cleaning up existing Ray processes..."
for node in "${NODES[@]}"; do
    limit_jobs "$MAX_SSH_PARALLELISM"
    remote_exec "$node" "ray stop -f 2>/dev/null || true" true &
done
wait
log_info "Cleanup completed."

# Step 1: Start Ray Head
log_info "Step 1: Starting Ray Head on $HEAD_NODE..."
if ! remote_exec "$HEAD_NODE" "bash $HEAD_NODE_SCRIPT"; then
    log_fatal "Failed to start Ray Head on $HEAD_NODE"
fi

# Step 2: Start Ray Workers
if [ ${#WORKERS[@]} -gt 0 ]; then
    log_info "Step 2: Starting ${#WORKERS[@]} Ray Workers..."
    for node in "${WORKERS[@]}"; do
        limit_jobs "$MAX_SSH_PARALLELISM"
        (
            log_info "Joining $node to cluster..."
            if ! remote_exec "$node" "bash $WORKER_NODE_SCRIPT $HEAD_NODE:6379"; then
                log_err "Failed to start Ray Worker on $node"
            fi
        ) &
    done
    wait
fi

# Step 3: Verify Cluster
log_info "Step 3: Verifying Ray Cluster status..."
start_time=$(date +%s)
expected_nodes=${#NODES[@]}

while true; do
    # Try to get active node count. We use a simple grep on 'ray status' output.
    # Usually 'ray status' shows a list of nodes.
    current_nodes=$(remote_exec "$HEAD_NODE" "ray status 2>/dev/null | grep -c 'node_id:'" || echo "0")
    
    if [ "$current_nodes" -ge "$expected_nodes" ]; then
        log_info "All $expected_nodes nodes have joined the cluster."
        break
    fi
    
    elapsed=$(( $(date +%s) - start_time ))
    if [ "$elapsed" -gt "$VERIFY_TIMEOUT" ]; then
        log_warn "Timeout waiting for all nodes to join. Current nodes: $current_nodes/$expected_nodes"
        break
    fi
    
    log_info "Waiting for nodes to join ($current_nodes/$expected_nodes)..."
    sleep 5
done

log_info "Final Cluster Status:"
remote_exec "$HEAD_NODE" "ray status"

log_info "============================================="
log_info "Ray Cluster setup process finished!"
log_info "============================================="
