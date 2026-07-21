#!/bin/bash
# =============================================================================
# Eco-Tech 剩余模型 + DeepSeek-V4-Pro 重试并行部署 (5 个 2 节点 Ray 子集群)
# =============================================================================
# 将 16 节点中的前 10 个节点划分为 5 对，每对独立 Ray 集群，部署一个模型并运行 curl 测试。
#
# Usage:
#   bash scripts/parallel_eco_tech_deploy_remaining.sh
#   LOG_DIR=/tmp/easyinfer_remaining bash scripts/parallel_eco_tech_deploy_remaining.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/docker/docker_env.sh"

export SSH_OPTS="${SSH_OPTS:--o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new}"

readonly NODES_FILE="${NODES_FILE:-$ROOT_DIR/node_list3.txt}"
readonly PAIRS_DIR="$ROOT_DIR/scripts/ray_cluster/nodes/pairs"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs/parallel_deploy_remaining_$(date +%Y%m%d_%H%M%S)}"

mkdir -p "$LOG_DIR"
LOG_DIR="$(cd "$LOG_DIR" && pwd)"
readonly LOG_DIR

# -----------------------------------------------------------------------------
# Task definitions: index maps to scripts/ray_cluster/nodes/pairs/pair_<idx>.txt
# -----------------------------------------------------------------------------
readonly NAMES=(
    deepseek-v4-pro-retry
    glm5.2-w8a8
    kimi-k2.6-w4a8
    minimax-m3
    step-3.7-flash
)

readonly EXAMPLE_DIRS=(
    examples/deepseek_v4_pro
    examples/glm5_2_w8a8
    examples/kimi_k2_6_w4a8
    examples/minimax_m3_w8a8
    examples/step_3_7_flash_w8a8
)

readonly PORTS=(8005 8007 8003 8014 8015)

mapfile -t ALL_NODES < <(awk 'NF && !/^#/{print $1}' "$NODES_FILE")

# -----------------------------------------------------------------------------
# Pre-deployment cleanup: stop any existing Ray / vLLM processes
# -----------------------------------------------------------------------------
cleanup_global() {
    log_info "停止现有 Ray / vLLM 进程 (全部 ${#ALL_NODES[@]} 个节点)..."
    local node
    for node in "${ALL_NODES[@]}"; do
        {
            ssh_run "$node" "docker exec ${CONTAINER_NAME} bash -c 'ray stop || true'" >/dev/null 2>&1 || true
            ssh_run "$node" "docker exec ${CONTAINER_NAME} bash -c 'pkill -9 -f \"vllm serve\" || true; pkill -9 -f \"python.*vllm\" || true'" >/dev/null 2>&1 || true
        } &
    done
    wait
    sleep 5
    log_info "清理完成"
}

# -----------------------------------------------------------------------------
# Start an independent Ray cluster for one node pair
# -----------------------------------------------------------------------------
start_pair_ray() {
    local idx=$1
    local pair_file="$PAIRS_DIR/pair_${idx}.txt"
    local name=${NAMES[$idx]}
    local log="$LOG_DIR/${name}_ray.log"

    log_info "[$name] 启动 pair-${idx} Ray 集群 ..."
    bash "$ROOT_DIR/scripts/ray_cluster/manage_npuslim_ray_cluster.sh" start -f "$pair_file" > "$log" 2>&1
}

# -----------------------------------------------------------------------------
# Deploy one model and run curl tests on its pair cluster
# -----------------------------------------------------------------------------
deploy_and_test() {
    local idx=$1
    local name=${NAMES[$idx]}
    local example=${EXAMPLE_DIRS[$idx]}
    local port=${PORTS[$idx]}
    local pair_file="$PAIRS_DIR/pair_${idx}.txt"
    local head_ip
    head_ip=$(head -1 "$pair_file")

    local vllm_log="$LOG_DIR/${name}_vllm.log"
    local test_log="$LOG_DIR/${name}_test.log"
    local result_file="$LOG_DIR/${name}_result.txt"

    echo "START" > "$result_file"

    # 1) Ray cluster for this pair
    if ! start_pair_ray "$idx"; then
        echo "FAIL_RAY" > "$result_file"
        log_err "[$name] Ray 集群启动失败，详情见 $LOG_DIR/${name}_ray.log"
        return 1
    fi

    # 2) Launch vLLM server (detached inside container)
    log_info "[$name] 在 ${head_ip}:${port} 启动 vLLM serve ..."
    ssh_run "$head_ip" \
        "docker exec -e RAY_ADDRESS=${head_ip}:6379 -e VLLM_HOST_IP=${head_ip} ${CONTAINER_NAME} bash -c 'nohup bash ${ROOT_DIR}/${example}/vllm/run_vllm.sh > ${vllm_log} 2>&1 &'"

    # 3) Wait for HTTP service
    log_info "[$name] 等待服务就绪 (最多 30 min) ..."
    if ! wait_for_port "$head_ip" "$port" 1800; then
        echo "FAIL_SERVICE" > "$result_file"
        log_err "[$name] 服务未在端口 ${port} 就绪"
        return 1
    fi

    # 4) Run functional tests
    log_info "[$name] 运行 curl 测试 ..."
    if ssh_run "$head_ip" \
        "docker exec -e RAY_ADDRESS=${head_ip}:6379 -e VLLM_HOST_IP=${head_ip} ${CONTAINER_NAME} bash -c 'bash ${ROOT_DIR}/${example}/vllm/curl_test.sh > ${test_log} 2>&1'"; then
        echo "PASS" > "$result_file"
        log_info "[$name] 测试通过"
    else
        echo "FAIL_TEST" > "$result_file"
        log_err "[$name] curl 测试失败，详情见 $LOG_DIR/${name}_test.log"
        return 1
    fi
}

# -----------------------------------------------------------------------------
# Summarize results
# -----------------------------------------------------------------------------
summarize() {
    echo ""
    log_info "=========================================="
    log_info "剩余模型并行部署结果汇总"
    log_info "日志目录: $LOG_DIR"
    log_info "=========================================="

    local idx name result status
    for idx in "${!NAMES[@]}"; do
        name=${NAMES[$idx]}
        result_file="$LOG_DIR/${name}_result.txt"
        result=$(cat "$result_file" 2>/dev/null || echo "UNKNOWN")
        case "$result" in
            PASS)        status="${GREEN}PASS${NC}" ;;
            FAIL_RAY)    status="${RED}FAIL_RAY${NC}" ;;
            FAIL_SERVICE)status="${RED}FAIL_SERVICE${NC}" ;;
            FAIL_TEST)   status="${RED}FAIL_TEST${NC}" ;;
            *)           status="${YELLOW}UNKNOWN${NC}" ;;
        esac
        printf "  %-22s %s\n" "$name" "$status"
    done
    echo ""
}

# -----------------------------------------------------------------------------
# Main: cleanup, then launch up to 4 parallel deployments
# -----------------------------------------------------------------------------
main() {
    cleanup_global

    log_info "开始 5 路并行部署 (剩余模型 + DeepSeek-V4-Pro 重试) ..."
    local idx
    for idx in "${!NAMES[@]}"; do
        limit_jobs 5
        deploy_and_test "$idx" &
    done
    wait

    summarize
}

main "$@"
