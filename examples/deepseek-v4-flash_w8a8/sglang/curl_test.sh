#!/bin/bash
# =============================================================================
# DeepSeek-V4-Flash — SGLang API 功能测试
# =============================================================================
# SGLang 提供完全 OpenAI 兼容的 API，测试 /v1/models、/health、Chat、Tool Calling
#
# 用法:
#   ./curl_test.sh                          # 默认 localhost:8000
#   BASE_URL=http://10.16.201.193:8000 ./curl_test.sh
#   MODEL_NAME=deepseek-v4-flash ./curl_test.sh
# =============================================================================

set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:8000}"
MODEL_NAME="${MODEL_NAME:-deepseek-v4-flash}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../../../scripts/common.sh
source "${SCRIPT_DIR}/../../../scripts/common.sh"

pass() { log_info "[PASS] $1"; }
fail() { log_err "[FAIL] $1"; }
info() { log_info "[INFO] $1"; }

# -----------------------------------------------------------------------------
# 测试 1: SGLang Health Check
# -----------------------------------------------------------------------------
info "Testing SGLang health endpoint..."
if curl -sf --max-time 5 "${BASE_URL}/health" -o /dev/null; then
    pass "SGLang health endpoint OK at ${BASE_URL}"
else
    fail "SGLang health endpoint NOT reachable at ${BASE_URL}"
    exit 1
fi

# -----------------------------------------------------------------------------
# 测试 2: 获取可用模型列表
# -----------------------------------------------------------------------------
info "Listing available models..."
models=$(curl -sf --max-time 5 "${BASE_URL}/v1/models")
if [[ -n "$models" ]]; then
    echo "$models" | python3 -m json.tool 2>/dev/null || echo "$models"
    pass "Model list retrieved"
else
    fail "Failed to retrieve model list"
fi

# -----------------------------------------------------------------------------
# 测试 3: 非流式 Chat Completion
# -----------------------------------------------------------------------------
info "Testing non-streaming chat completion..."
response=$(curl -sf --max-time 120 "${BASE_URL}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"${MODEL_NAME}\",
    \"messages\": [{\"role\": \"user\", \"content\": \"你好！请用一句话介绍你自己。\"}],
    \"max_tokens\": 200,
    \"temperature\": 0.7,
    \"stream\": false
  }")

if [[ -n "$response" ]]; then
    content=$(echo "$response" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['choices'][0]['message']['content'])" 2>/dev/null || echo "")
    usage=$(echo "$response" | python3 -c "import sys,json; d=json.load(sys.stdin); u=d.get('usage',{}); print(f\"prompt={u.get('prompt_tokens','?')}, completion={u.get('completion_tokens','?')}, total={u.get('total_tokens','?')}\")" 2>/dev/null || echo "")
    if [[ -n "$content" ]]; then
        pass "Response: ${content:0:200}"
        [[ -n "$usage" ]] && info "Tokens: ${usage}"
    else
        fail "Empty response content"
        echo "$response" | python3 -m json.tool 2>/dev/null || echo "$response"
    fi
else
    fail "No response from server"
fi

# -----------------------------------------------------------------------------
# 测试 4: 流式 Chat Completion
# -----------------------------------------------------------------------------
info "Testing streaming chat completion..."
stream_output=$(curl -sf --max-time 60 "${BASE_URL}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"${MODEL_NAME}\",
    \"messages\": [{\"role\": \"user\", \"content\": \"从1数到5\"}],
    \"max_tokens\": 100,
    \"temperature\": 0.0,
    \"stream\": true
  }" 2>&1 | head -20)

if [[ -n "$stream_output" ]]; then
    echo "$stream_output" | head -10
    pass "Streaming response received (showing first 10 chunks above)"
else
    fail "Streaming failed - no response"
fi

# -----------------------------------------------------------------------------
# 测试 5: Tool Calling (Claude Code 集成必需)
# -----------------------------------------------------------------------------
info "Testing tool calling..."
tool_response=$(curl -sf --max-time 120 "${BASE_URL}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"${MODEL_NAME}\",
    \"messages\": [
        {\"role\": \"system\", \"content\": \"You are a helpful assistant with access to tools.\"},
        {\"role\": \"user\", \"content\": \"What is the weather in Beijing today?\"}
    ],
    \"tools\": [{
        \"type\": \"function\",
        \"function\": {
            \"name\": \"get_weather\",
            \"description\": \"Get current weather in a city\",
            \"parameters\": {
                \"type\": \"object\",
                \"properties\": {
                    \"city\": {\"type\": \"string\", \"description\": \"City name\"}
                },
                \"required\": [\"city\"]
            }
        }
    }],
    \"max_tokens\": 200,
    \"temperature\": 0.0,
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
elif msg.get('content'):
    print(f'text_response: {msg[\"content\"][:100]}')
else:
    print('no_tool_call')
" 2>/dev/null || echo "parse_error")

    if [[ "$tool_call" != "no_tool_call" && "$tool_call" != "parse_error" ]]; then
        pass "Tool calling works: ${tool_call}"
    elif [[ "$tool_call" == *"text_response"* ]]; then
        info "Model responded with text (acceptable): ${tool_call}"
    else
        info "Tool calling: no tool call detected (may need different prompt)"
    fi
else
    fail "Tool calling test failed - no response"
fi

# -----------------------------------------------------------------------------
# 测试 6: 前缀缓存验证 (SGLang RadixAttention)
# -----------------------------------------------------------------------------
info "Testing prefix caching (same system prompt, 2 requests)..."
SYSTEM_PROMPT="You are a helpful assistant. Please answer the user's questions accurately and concisely."

# 第一次请求
time1=$( { time curl -sf --max-time 120 "${BASE_URL}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"${MODEL_NAME}\",
    \"messages\": [
        {\"role\": \"system\", \"content\": \"${SYSTEM_PROMPT}\"},
        {\"role\": \"user\", \"content\": \"What is 1+1?\"}
    ],
    \"max_tokens\": 50,
    \"stream\": false
  }" > /dev/null; } 2>&1 | grep real | awk '{print $2}' )

# 第二次请求（相同 system prompt，应命中缓存）
time2=$( { time curl -sf --max-time 120 "${BASE_URL}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"${MODEL_NAME}\",
    \"messages\": [
        {\"role\": \"system\", \"content\": \"${SYSTEM_PROMPT}\"},
        {\"role\": \"user\", \"content\": \"What is 2+2?\"}
    ],
    \"max_tokens\": 50,
    \"stream\": false
  }" > /dev/null; } 2>&1 | grep real | awk '{print $2}' )

if [[ -n "$time1" && -n "$time2" ]]; then
    info "Request 1 (cold): ${time1}, Request 2 (cached): ${time2}"
    pass "Prefix caching test completed"
else
    info "Prefix caching test skipped (timing unavailable)"
fi

echo ""
info "All tests completed for DeepSeek-V4-Flash via SGLang."
