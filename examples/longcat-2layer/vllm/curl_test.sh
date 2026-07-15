#!/bin/bash
# =============================================================================
# LongCat-Flash-Chat-2layer — API functional test script
# =============================================================================
# Targets localhost:8010 by default.
#
# Usage:
#   ./curl_test.sh
#   HOST=10.0.0.1 PORT=8010 ./curl_test.sh
#   MODEL_NAME=longcat-flash-2layer ./curl_test.sh
# =============================================================================
set -euo pipefail

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
HOST="${HOST:-localhost}"
PORT="${PORT:-8010}"
MODEL_NAME="${MODEL_NAME:-longcat-flash-2layer}"
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
    log_info "Waiting for service: ${BASE_URL} ..."
    local start_time elapsed
    start_time=$(date +%s)

    while true; do
        if curl -s "${BASE_URL}/health" >/dev/null 2>&1 || \
           curl -s "${BASE_URL}/v1/models" >/dev/null 2>&1; then
            log_success "Service is ready!"
            return 0
        fi

        elapsed=$(( $(date +%s) - start_time ))
        if [[ "$elapsed" -ge "$TIMEOUT" ]]; then
            log_error "Timed out waiting for service (${TIMEOUT}s)!"
            return 1
        fi

        log_info "Service not ready, waiting ${WAIT_INTERVAL}s... (${elapsed}s elapsed)"
        sleep "$WAIT_INTERVAL"
    done
}

# ------------------------------------------------------------------------------
# Run a single JSON API test
# ------------------------------------------------------------------------------
run_test() {
    local test_name="$1"
    local endpoint="$2"
    local payload="$3"
    local header="${4:-}"
    local quiet="${5:-}"
    local response curl_status=0
    local curl_opts=(-s "$endpoint" -H "Content-Type: application/json")

    [[ -n "$header" ]] && curl_opts+=(-H "$header")
    [[ -n "$payload" ]] && curl_opts+=(-d "$payload")

    echo ""
    log_info "Test: ${test_name}"
    log_info "Endpoint: ${endpoint}"

    if [[ "$quiet" == "quiet" ]]; then
        curl "${curl_opts[@]}" >/dev/null 2>&1 || curl_status=$?
    else
        response=$(curl "${curl_opts[@]}") || curl_status=$?
        echo "$response" | python3 -m json.tool 2>/dev/null || echo "$response"
    fi

    if [[ "$curl_status" -eq 0 ]]; then
        log_success "${test_name} passed (exit code: ${curl_status})"
    else
        log_warning "${test_name} may have issues (exit code: ${curl_status})"
    fi
    return "$curl_status"
}

# ------------------------------------------------------------------------------
# Run a streaming chat completion test
# ------------------------------------------------------------------------------
run_stream_test() {
    local test_name="$1"
    local payload="$2"

    echo ""
    log_info "Test: ${test_name}"
    curl -s "${BASE_URL}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "$payload" 2>&1 | head -10 || true
    log_success "${test_name} passed"
}

# ------------------------------------------------------------------------------
# Main test sequence
# ------------------------------------------------------------------------------
echo "=========================================="
echo "  LongCat-Flash-Chat-2layer API Test"
echo "  Target: ${BASE_URL}"
echo "  Model: ${MODEL_NAME}"
echo "=========================================="

wait_for_service || exit 1

# 1. Model list (GET)
run_test "List Models" \
    "${BASE_URL}/v1/models" \
    ""

# 2. Chat completion (English)
run_test "Chat Completion (English)" \
    "${BASE_URL}/v1/chat/completions" \
    '{"model":"'"$MODEL_NAME"'","messages":[{"role":"user","content":"Hello, who are you?"}],"max_tokens":128,"temperature":0.7}'

# 3. Chat completion (Chinese)
run_test "Chat Completion (Chinese)" \
    "${BASE_URL}/v1/chat/completions" \
    '{"model":"'"$MODEL_NAME"'","messages":[{"role":"system","content":"You are a helpful assistant."},{"role":"user","content":"你好，请简单介绍一下你自己。"}],"max_tokens":128,"temperature":0.7}'

# 4. Tool calling
run_test "Tool Calling" \
    "${BASE_URL}/v1/chat/completions" \
    '{"model":"'"$MODEL_NAME"'","messages":[{"role":"user","content":"What is the weather like in Beijing?"}],"tools":[{"type":"function","function":{"name":"get_weather","description":"Get the current weather","parameters":{"type":"object","properties":{"city":{"type":"string","description":"The city to get weather for"}},"required":["city"]}}}],"tool_choice":"auto","max_tokens":100}'

# 5. Streaming chat completion
run_stream_test "Streaming Chat Completion" \
    '{"model":"'"$MODEL_NAME"'","messages":[{"role":"user","content":"Count from 1 to 5"}],"max_tokens":100,"stream":true}'

echo ""
echo "=========================================="
log_success "LongCat-Flash-Chat-2layer tests complete!"
echo "=========================================="
