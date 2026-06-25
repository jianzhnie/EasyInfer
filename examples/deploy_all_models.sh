#!/bin/bash
# =============================================================================
# 并行部署全部 4 个模型 (Ray backend, 每组 2 节点)
# =============================================================================
# 用法: bash deploy_all_models.sh
#
# 节点分配:
#   DeepSeek-V4-Flash: 10.16.201.229(head) + 10.16.201.164
#   GLM-5-w4a8:        10.16.201.40(head)  + 10.16.201.163
#   GLM-5.1-w4a8:      10.16.201.193(head) + 10.16.201.201
#   Kimi-K2.6-w4a8:    10.16.201.153(head) + 10.16.201.124
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# shellcheck source=scripts/common.sh
source "scripts/common.sh"

# =============================================================================
# 配置: 模型名称 -> (节点文件, 端口, 模型名, vllm_server 路径, 日志文件)
# =============================================================================
declare -A MODEL_PORTS
MODEL_PORTS["deepseek_v4"]="8000"
MODEL_PORTS["glm5"]="8001"
MODEL_PORTS["glm5_1"]="8002"
MODEL_PORTS["kimi_k2_6"]="8003"

MODEL_NODES["deepseek_v4"]="nodes_deepseek_v4.txt"
MODEL_NODES["glm5"]="nodes_glm5.txt"
MODEL_NODES["glm5_1"]="nodes_glm5_1.txt"
MODEL_NODES["kimi_k2_6"]="nodes_kimi_k2_6.txt"

MODEL_HEADS["deepseek_v4"]="10.16.201.229"
MODEL_HEADS["glm5"]="10.16.201.40"
MODEL_HEADS["glm5_1"]="10.16.201.193"
MODEL_HEADS["kimi_k2_6"]="10.16.201.153"

MODEL_SCRIPTS["deepseek_v4"]="examples/deepseek_v4_flash/vllm_server.sh"
MODEL_SCRIPTS["glm5"]="examples/glm5_w4a8/vllm_server.sh"
MODEL_SCRIPTS["glm5_1"]="examples/glm5_1_w4a8/vllm_server.sh"
MODEL_SCRIPTS["kimi_k2_6"]="examples/kimi_k2_6_w4a8/vllm_server.sh"

MODEL_TEST_SCRIPTS["deepseek_v4"]="examples/deepseek_v4_flash/curl_test.sh"
MODEL_TEST_SCRIPTS["glm5"]="examples/glm5_w4a8/curl_test.sh"
MODEL_TEST_SCRIPTS["glm5_1"]="examples/glm5_1_w4a8/curl_test.sh"
MODEL_TEST_SCRIPTS["kimi_k2_6"]="examples/kimi_k2_6_w4a8/curl_test.sh"

MODEL_ORDER=("deepseek_v4" "glm5" "glm5_1" "kimi_k2_6")
CONTAINER_NAME="${CONTAINER_NAME:-npuslim-env}"

# =============================================================================
# Step 1: 启动所有 4 个 Ray 集群 (并行)
# =============================================================================
log_info "============================================"
log_info "Step 1: Starting 4 Ray clusters in parallel"
log_info "============================================"

RAY_PIDS=()
for model in "${MODEL_ORDER[@]}"; do
    node_file="${MODEL_NODES[$model]}"
    log_info "Starting Ray cluster for ${model} (nodes: ${node_file})..."
    bash scripts/ray_cluster/start_npuslim_ray_cluster.sh start -f "$node_file" \
        > "/tmp/ray_start_${model}.log" 2>&1 &
    RAY_PIDS+=($!)
    log_info "  Ray start PID: ${RAY_PIDS[-1]} (log: /tmp/ray_start_${model}.log)"
done

log_info "Waiting for all Ray clusters to start..."
FAILED_RAY=0
for i in "${!MODEL_ORDER[@]}"; do
    model="${MODEL_ORDER[$i]}"
    pid="${RAY_PIDS[$i]}"
    if wait "$pid"; then
        log_info "[OK] Ray cluster for ${model} started successfully"
    else
        log_err "[FAIL] Ray cluster for ${model} failed (exit code: $?)"
        FAILED_RAY=1
    fi
done

if [[ $FAILED_RAY -eq 1 ]]; then
    log_err "Some Ray clusters failed to start. Check logs in /tmp/ray_start_*.log"
    log_err "Continuing with deployments for successful clusters..."
fi

# =============================================================================
# Step 2: 等待 Ray 集群就绪
# =============================================================================
log_info ""
log_info "============================================"
log_info "Step 2: Waiting for Ray clusters to be ready"
log_info "============================================"

for model in "${MODEL_ORDER[@]}"; do
    head_ip="${MODEL_HEADS[$model]}"
    log_info "Checking Ray status on ${head_ip} (${model})..."

    # Check if Ray is running on head node
    if ssh_run "$head_ip" "docker exec ${CONTAINER_NAME} ray status" > /tmp/ray_status_${model}.log 2>&1; then
        log_info "[OK] Ray cluster for ${model} is ready (head: ${head_ip})"
    else
        log_warn "[WARN] Ray cluster for ${model} may not be ready. Check /tmp/ray_status_${model}.log"
    fi
done

# =============================================================================
# Step 3: 并行部署所有模型 (后台运行)
# =============================================================================
log_info ""
log_info "============================================"
log_info "Step 3: Deploying all 4 models in parallel"
log_info "============================================"

DEPLOY_PIDS=()
for model in "${MODEL_ORDER[@]}"; do
    head_ip="${MODEL_HEADS[$model]}"
    port="${MODEL_PORTS[$model]}"
    vllm_script="${MODEL_SCRIPTS[$model]}"
    log_file="/tmp/vllm_${model}.log"

    log_info "Deploying ${model} on ${head_ip}:${port}..."
    log_info "  Script: ${vllm_script}"
    log_info "  Log:    ${log_file}"

    # SSH into head node, docker exec, run vllm server in background
    # Use TP=8, PP=2 for 2-node multi-node deployment with Ray backend
    ssh_run "$head_ip" "
        docker exec ${CONTAINER_NAME} bash -c '
            cd /home/jianzhnie/llmtuner/llm/EasyInfer
            export TENSOR_PARALLEL_SIZE=8
            export PIPELINE_PARALLEL_SIZE=2
            export PORT=${port}
            export DISTRIBUTED_EXECUTOR_BACKEND=ray
            nohup bash ${vllm_script} > ${log_file} 2>&1 &
            echo \$!
        '
    " > "/tmp/deploy_pid_${model}.txt" 2>&1 &

    DEPLOY_PIDS+=($!)
    log_info "  Deploy PID: ${DEPLOY_PIDS[-1]}"
done

log_info "Waiting for all deployments to launch..."
for i in "${!MODEL_ORDER[@]}"; do
    model="${MODEL_ORDER[$i]}"
    pid="${DEPLOY_PIDS[$i]}"
    if wait "$pid"; then
        vllm_pid=$(cat "/tmp/deploy_pid_${model}.txt" 2>/dev/null | tail -1 || echo "unknown")
        log_info "[OK] ${model} deployment launched (vLLM PID: ${vllm_pid})"
    else
        log_err "[FAIL] ${model} deployment launch failed"
    fi
done

# =============================================================================
# Step 4: 等待所有服务就绪 (轮询 /v1/models 端点)
# =============================================================================
log_info ""
log_info "============================================"
log_info "Step 4: Waiting for all services to be ready"
log_info "============================================"

MAX_WAIT=600  # 10 minutes max wait
CHECK_INTERVAL=10

for model in "${MODEL_ORDER[@]}"; do
    head_ip="${MODEL_HEADS[$model]}"
    port="${MODEL_PORTS[$model]}"
    log_info "Waiting for ${model} (${head_ip}:${port})..."

    waited=0
    while [[ $waited -lt $MAX_WAIT ]]; do
        if ssh_run "$head_ip" "curl -sf --max-time 5 http://localhost:${port}/v1/models" > /dev/null 2>&1; then
            log_info "[OK] ${model} is ready at ${head_ip}:${port} (waited ${waited}s)"
            break
        fi
        sleep "$CHECK_INTERVAL"
        waited=$((waited + CHECK_INTERVAL))
        if [[ $((waited % 30)) -eq 0 ]]; then
            log_info "  Still waiting for ${model}... (${waited}s elapsed)"
        fi
    done

    if [[ $waited -ge $MAX_WAIT ]]; then
        log_err "[FAIL] ${model} did not become ready within ${MAX_WAIT}s"
    fi
done

# =============================================================================
# Step 5: 运行 curl 测试
# =============================================================================
log_info ""
log_info "============================================"
log_info "Step 5: Running API tests"
log_info "============================================"

ALL_PASSED=true
for model in "${MODEL_ORDER[@]}"; do
    head_ip="${MODEL_HEADS[$model]}"
    port="${MODEL_PORTS[$model]}"
    test_script="${MODEL_TEST_SCRIPTS[$model]}"
    test_log="/tmp/curl_test_${model}.log"

    log_info "Testing ${model} at ${head_ip}:${port}..."
    log_info "  Script: ${test_script}"

    if ssh_run "$head_ip" "
        docker exec ${CONTAINER_NAME} bash -c '
            cd /home/jianzhnie/llmtuner/llm/EasyInfer
            BASE_URL=http://localhost:${port} bash ${test_script}
        '
    " > "$test_log" 2>&1; then
        log_info "[PASS] ${model} tests passed"
        # Show key results
        grep -E '\[PASS\]|\[FAIL\]' "$test_log" | head -10 || true
    else
        log_err "[FAIL] ${model} tests failed"
        log_err "Check ${test_log} for details"
        ALL_PASSED=false
    fi
done

# =============================================================================
# Summary
# =============================================================================
log_info ""
log_info "============================================"
log_info "Deployment Summary"
log_info "============================================"
for model in "${MODEL_ORDER[@]}"; do
    head_ip="${MODEL_HEADS[$model]}"
    port="${MODEL_PORTS[$model]}"
    echo "  ${model}: http://${head_ip}:${port}"
    echo "    Log:     /tmp/vllm_${model}.log"
    echo "    Test:    /tmp/curl_test_${model}.log"
    echo "    Ray:     /tmp/ray_start_${model}.log"
done

if [[ "$ALL_PASSED" == "true" ]]; then
    log_info "All models deployed and tested successfully!"
else
    log_warn "Some models had issues. Check the test logs."
fi
