#!/bin/bash
# =============================================================================
# GLM-5.1 W4A8 — API functional test script
# =============================================================================
# Targets localhost:8002 by default.
# Same test logic as GLM-5 W4A8; only the default port/model differ.
#
# Usage:
#   ./curl_test.sh
#   HOST=10.0.0.1 PORT=9000 ./curl_test.sh
#   MODEL_NAME=my-model ./curl_test.sh
# =============================================================================
set -euo pipefail

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
HOST="${HOST:-localhost}"
PORT="${PORT:-8002}"
MODEL_NAME="${MODEL_NAME:-glm-5.1}"
readonly TIMEOUT=300
readonly WAIT_INTERVAL=5
readonly BASE_URL="http://${HOST}:${PORT}"

# ------------------------------------------------------------------------------
# Logging helpers
# ------------------------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info()    { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error()   { echo -e "${RED}[ERROR]${NC} $1"; }

# ------------------------------------------------------------------------------
# Wait for the service to become ready
# ------------------------------------------------------------------------------
wait_for_service() {
    log_info "等待服务启动: ${BASE_URL} ..."
    local start_time elapsed
    start_time=$(date +%s)

    while true; do
        if curl -s "${BASE_URL}/health" >/dev/null 2>&1 || \
           curl -s "${BASE_URL}/v1/models" >/dev/null 2>&1; then
            log_success "服务已就绪!"
            return 0
        fi

        elapsed=$(( $(date +%s) - start_time ))
        if [[ "$elapsed" -ge "$TIMEOUT" ]]; then
            log_error "等待服务超时 (${TIMEOUT} 秒)!"
            return 1
        fi

        log_info "服务未就绪，等待 ${WAIT_INTERVAL}s... (已等待 ${elapsed}s)"
        sleep "$WAIT_INTERVAL"
    done
}

# ------------------------------------------------------------------------------
# Run a single JSON API test
# Args:
#   $1: test name
#   $2: endpoint URL
#   $3: JSON payload
#   $4: optional extra header
#   $5: optional "quiet" flag
# ------------------------------------------------------------------------------
run_test() {
    local test_name="$1"
    local endpoint="$2"
    local payload="$3"
    local header="${4:-}"
    local quiet="${5:-}"
    local response status=0
    local curl_opts=(-s "$endpoint" -H "Content-Type: application/json")

    [[ -n "$header" ]] && curl_opts+=(-H "$header")
    [[ -n "$payload" ]] && curl_opts+=(-d "$payload")

    echo ""
    log_info "测试: ${test_name}"
    log_info "Endpoint: ${endpoint}"

    if [[ "$quiet" == "quiet" ]]; then
        curl "${curl_opts[@]}" >/dev/null 2>&1 || status=$?
    else
        response=$(curl "${curl_opts[@]}") || status=$?
        echo "$response" | python3 -m json.tool 2>/dev/null || echo "$response"
    fi

    if [[ "$status" -eq 0 ]]; then
        log_success "${test_name} 完成 (退出码: ${status})"
    else
        log_warning "${test_name} 可能存在问题 (退出码: ${status})"
    fi
    return "$status"
}

# ------------------------------------------------------------------------------
# Run a streaming chat completion test
# ------------------------------------------------------------------------------
run_stream_test() {
    local test_name="$1"
    local payload="$2"

    echo ""
    log_info "测试: ${test_name}"
    curl -s "${BASE_URL}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "$payload" 2>&1 | head -10 || true
    log_success "${test_name} 完成"
}

# ------------------------------------------------------------------------------
# Main test sequence
# ------------------------------------------------------------------------------
echo "=========================================="
echo "  GLM-5.1 W4A8 API 功能测试"
echo "  目标地址: ${BASE_URL}"
echo "  模型名称: ${MODEL_NAME}"
echo "=========================================="

wait_for_service || exit 1

# 1. Model list (GET)
run_test "模型列表查询" \
    "${BASE_URL}/v1/models" \
    ""

# 2. Chat completion (English)
run_test "Chat Completion (英文)" \
    "${BASE_URL}/v1/chat/completions" \
    '{"model":"'"$MODEL_NAME"'","messages":[{"role":"user","content":"Hello, who are you?"}],"max_tokens":128,"temperature":0.7}'

# 3. Chat completion (Chinese)
run_test "Chat Completion (中文)" \
    "${BASE_URL}/v1/chat/completions" \
    '{"model":"'"$MODEL_NAME"'","messages":[{"role":"system","content":"You are a helpful assistant."},{"role":"user","content":"你好，请简单介绍一下你自己。"}],"max_tokens":128,"temperature":0.7}'

# 4. Tool calling
run_test "Tool Calling" \
    "${BASE_URL}/v1/chat/completions" \
    '{"model":"'"$MODEL_NAME"'","messages":[{"role":"user","content":"What is the weather like in Beijing?"}],"tools":[{"type":"function","function":{"name":"get_weather","description":"Get the current weather","parameters":{"type":"object","properties":{"city":{"type":"string","description":"The city to get weather for"}},"required":["city"]}}}],"tool_choice":"auto","max_tokens":100}'

# 5. Anthropic Messages API
run_test "Anthropic Messages API" \
    "${BASE_URL}/v1/messages" \
    '{"model":"'"$MODEL_NAME"'","max_tokens":100,"messages":[{"role":"user","content":"Hi there!"}]}' \
    "x-api-key: dummy"

# 6. Streaming chat completion
run_stream_test "流式 Chat Completion" \
    '{"model":"'"$MODEL_NAME"'","messages":[{"role":"user","content":"从1数到5"}],"max_tokens":100,"stream":true}'

echo ""
echo "=========================================="
log_success "GLM-5.1 W4A8 所有测试完成!"
echo "=========================================="
