#!/bin/bash
# =============================================================================
# GLM-5.1 vLLM 服务 API 测试脚本
# =============================================================================
#
# 用法:
#   ./curl_test.sh                    # 默认测试 localhost:8002 glm-5.1
#   BASE_URL=http://10.0.0.1:9000 ./curl_test.sh
#   MODEL_NAME=my-model ./curl_test.sh
# =============================================================================

set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:8002}"
MODEL_NAME="${MODEL_NAME:-glm-5.1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../../scripts/common.sh
source "${SCRIPT_DIR}/../../scripts/common.sh"

pass() { log_info "[PASS] $1"; }
fail() { log_err "[FAIL] $1"; }
info() { log_info "[INFO] $1"; }

info "Testing service availability..."
if curl -sf --max-time 5 "${BASE_URL}/v1/models" -o /dev/null; then
    pass "Service is reachable at ${BASE_URL}"
else
    fail "Service is NOT reachable at ${BASE_URL}"
    exit 1
fi

info "Listing available models..."
models=$(curl -sf --max-time 5 "${BASE_URL}/v1/models")
if [[ -n "$models" ]]; then
    echo "$models" | python3 -m json.tool 2>/dev/null || echo "$models"
    pass "Model list retrieved"
else
    fail "Failed to retrieve model list"
fi

info "Testing non-streaming chat completion..."
response=$(curl -sf --max-time 120 "${BASE_URL}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"${MODEL_NAME}\",
    \"messages\": [{\"role\": \"user\", \"content\": \"介绍一下你自己，请简要回答。\"}],
    \"max_tokens\": 200,
    \"stream\": false
  }")

if [[ -n "$response" ]]; then
    content=$(echo "$response" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['choices'][0]['message']['content'])" 2>/dev/null || echo "")
    usage=$(echo "$response" | python3 -c "import sys,json; d=json.load(sys.stdin); u=d.get('usage',{}); print(f\"prompt={u.get('prompt_tokens','?')}, completion={u.get('completion_tokens','?')}, total={u.get('total_tokens','?')}\")" 2>/dev/null || echo "")
    if [[ -n "$content" ]]; then
        pass "Response: ${content}"
        [[ -n "$usage" ]] && info "Tokens: ${usage}"
    else
        fail "Empty response content"
    fi
else
    fail "No response from server"
fi

info "Testing streaming chat completion..."
stream_output=$(curl -sf --max-time 30 "${BASE_URL}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"${MODEL_NAME}\",
    \"messages\": [{\"role\": \"user\", \"content\": \"从1数到5\"}],
    \"max_tokens\": 100,
    \"stream\": true
  }" 2>&1 | head -5)

if [[ -n "$stream_output" ]]; then
    pass "Streaming response received"
else
    fail "Streaming failed"
fi

info "Testing tool calling..."
tool_response=$(curl -sf --max-time 120 "${BASE_URL}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"${MODEL_NAME}\",
    \"messages\": [{\"role\": \"user\", \"content\": \"列出当前目录的文件\"}],
    \"tools\": [{
        \"type\": \"function\",
        \"function\": {
            \"name\": \"list_files\",
            \"description\": \"List files in a directory\",
            \"parameters\": {
                \"type\": \"object\",
                \"properties\": {
                    \"path\": {\"type\": \"string\", \"description\": \"Directory path\"}
                },
                \"required\": [\"path\"]
            }
        }
    }],
    \"max_tokens\": 200,
    \"stream\": false
  }")

if [[ -n "$tool_response" ]]; then
    tool_call=$(echo "$tool_response" | python3 -c "
import sys, json
d = json.load(sys.stdin)
msg = d['choices'][0]['message']
if msg.get('tool_calls'):
    tc = msg['tool_calls'][0]
    print(f'function={tc[\"function\"][\"name\"]}, args={tc[\"function\"][\"arguments\"]}')
else:
    print('no_tool_call')
" 2>/dev/null || echo "parse_error")
    if [[ "$tool_call" != "no_tool_call" && "$tool_call" != "parse_error" ]]; then
        pass "Tool calling works: ${tool_call}"
    else
        info "Tool calling: model did not invoke tool"
    fi
else
    fail "Tool calling test failed - no response"
fi

echo ""
info "All tests completed."
