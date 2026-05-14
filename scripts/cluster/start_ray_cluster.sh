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

# Source ray environment variables for defaults
RAY_ENV_SCRIPT="${SCRIPT_DIR}/set_ray_env.sh"
# Source without arguments to avoid triggering startup logic
# shellcheck source=./set_ray_env.sh
if [[ -f "$RAY_ENV_SCRIPT" ]]; then
    source "$RAY_ENV_SCRIPT" ""
fi

# -----------------------------------------------------------------
# 2. Node Preparation
# -----------------------------------------------------------------
if [[ ! -f "$ARG_NODE_LIST" ]]; then
    log_fatal "Node list file not found: $ARG_NODE_LIST"
fi

NODES=($(read_nodes "$ARG_NODE_LIST"))
if [ ${#NODES[@]} -eq 0 ]; then
    log_fatal "No nodes found in $ARG_NODE_LIST"
fi

HEAD_NODE=${NODES[0]}
WORKERS=("${NODES[@]:1}")

log_info "============================================="
log_info "Ray Cluster Action: $ARG_ACTION"
log_info "Head Node:      $HEAD_NODE"
log_info "Worker Nodes:   ${WORKERS[*]:-None}"
log_info "Container:      $ARG_CONTAINER_NAME"
log_info "Parallelism:    $ARG_MAX_SSH_PARALLELISM"
log_info "NPUs per Node:  $ARG_NPUS_PER_NODE"
log_info "Ray Port:       $ARG_RAY_PORT"
log_info "============================================="

# -----------------------------------------------------------------
# 3. Helper Functions
# -----------------------------------------------------------------

# Checks if the target container is running on a remote node
check_container() {
    local node=$1
    if ! ssh_run "$node" "docker ps --format '{{.Names}}' | grep -q '^$ARG_CONTAINER_NAME$'" >/dev/null 2>&1; then
        log_err "Container '$ARG_CONTAINER_NAME' is not running on $node"
        return 1
    fi
    return 0
}

# Executes a command inside the target container on a remote node
remote_exec() {
    local node=$1
    local cmd=$2
    local silent=${3:-false}
    
    local docker_cmd="docker exec -i $ARG_CONTAINER_NAME bash -c \"$cmd\""
    
    if [ "$silent" = true ]; then
        ssh_run "$node" "$docker_cmd" >/dev/null 2>&1
    else
        ssh_run "$node" "$docker_cmd"
    fi
}

# -----------------------------------------------------------------
# 4. Main Process
# -----------------------------------------------------------------

if [[ "$ARG_ACTION" == "stop" ]]; then
    log_info "Stopping Ray cluster on all nodes..."
    for node in "${NODES[@]}"; do
        limit_jobs "$ARG_MAX_SSH_PARALLELISM"
        remote_exec "$node" "bash $RAY_CLUSTER_SCRIPT stop" true || log_warn "Failed to stop Ray on $node" &
    done
    wait
    log_info "Cluster stopped."
    exit 0
fi

# Action is "start"

# Step -1: Pre-flight checks
log_info "Step -1: Performing pre-flight checks..."
for node in "${NODES[@]}"; do
    limit_jobs "$ARG_MAX_SSH_PARALLELISM"
    (
        if ! check_container "$node"; then
            exit 1
        fi
    ) &
done
wait

# Step 0: Stop existing Ray processes on all nodes
log_info "Step 0: Cleaning up existing Ray processes..."
for node in "${NODES[@]}"; do
    limit_jobs "$ARG_MAX_SSH_PARALLELISM"
    remote_exec "$node" "bash $RAY_CLUSTER_SCRIPT stop" true || log_warn "Cleanup failed on $node (non-fatal)" &
done
wait
log_info "Cleanup completed."

# Step 1: Start Ray Head
log_info "Step 1: Starting Ray Head on $HEAD_NODE..."
if ! remote_exec "$HEAD_NODE" "bash $RAY_CLUSTER_SCRIPT head $ARG_RAY_PORT $ARG_DASHBOARD_PORT $ARG_NPUS_PER_NODE $HEAD_NODE"; then
    log_fatal "Failed to start Ray Head on $HEAD_NODE"
fi
log_info "Ray Head started. Waiting 5s for initialization..."
sleep 5

# Step 2: Start Ray Workers
if [ ${#WORKERS[@]} -gt 0 ]; then
    log_info "Step 2: Starting ${#WORKERS[@]} Ray Workers in parallel..."
    for node in "${WORKERS[@]}"; do
        limit_jobs "$ARG_MAX_SSH_PARALLELISM"
        (
            log_info "Joining $node to cluster..."
            if ! remote_exec "$node" "bash $RAY_CLUSTER_SCRIPT worker $HEAD_NODE:$ARG_RAY_PORT $ARG_NPUS_PER_NODE $node"; then
                log_err "Failed to start Ray Worker on $node"
            fi
        ) &
    done
    wait
    log_info "All worker join commands issued."
fi

# Step 3: Verify Cluster
log_info "Step 3: Verifying Ray Cluster status..."
start_time=$(date +%s)
expected_nodes=${#NODES[@]}

while true; do
    status_output=$(remote_exec "$HEAD_NODE" "ray status" true || echo "")
    current_nodes=$(echo "$status_output" | grep -A 20 "Active nodes" | grep -c "node_id" || echo "0")
    
    if [ "$current_nodes" -ge "$expected_nodes" ]; then
        log_info "Success: All $expected_nodes nodes have joined the cluster."
        break
    fi
    
    elapsed=$(( $(date +%s) - start_time ))
    if [ "$elapsed" -gt "$VERIFY_TIMEOUT" ]; then
        log_err "Timeout reached ($VERIFY_TIMEOUT s). Only $current_nodes/$expected_nodes nodes joined."
        log_warn "Please check node connectivity and container logs."
        break
    fi
    
    log_info "Progress: $current_nodes/$expected_nodes nodes joined... (Elapsed: ${elapsed}s)"
    sleep 5
done

log_info "Final Cluster Status Summary:"
remote_exec "$HEAD_NODE" "ray status"

log_info "============================================="
log_info "Ray Cluster setup process finished!"
log_info "Dashboard URL:  http://$HEAD_NODE:$ARG_DASHBOARD_PORT"
log_info "To stop cluster: bash $0 stop --file $ARG_NODE_LIST"
log_info "============================================="
