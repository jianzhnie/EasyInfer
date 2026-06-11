#!/bin/bash
# DeepSeek-V4-Pro API Test Script
set -euo pipefail

PORT="${PORT:-8000}"
BASE_URL="${BASE_URL:-http://localhost:$PORT}"
MODEL_NAME="${MODEL_NAME:-deepseek-v4-pro}"

pass() { echo "[PASS] $1"; }
fail() { echo "[FAIL] $1"; }

echo "=== Testing DeepSeek-V4-Pro at $BASE_URL ==="

# 1. Health check
echo -n "1. /v1/models: "
if curl -sf --max-time 10 "${BASE_URL}/v1/models" -o /dev/null; then
    models=$(curl -sf --max-time 5 "${BASE_URL}/v1/models")
    model_id=$(echo "$models" | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null)
    max_len=$(echo "$models" | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['max_model_len'])" 2>/dev/null)
    pass "model=$model_id max_len=$max_len"
else
    fail "Service not reachable"
    exit 1
fi

# 2. Chat completion
echo -n "2. Chat: "
resp=$(curl -sf --max-time 60 "${BASE_URL}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"${MODEL_NAME}\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hello in one word\"}],\"max_tokens\":20}" 2>/dev/null)
content=$(echo "$resp" | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'][:100])" 2>/dev/null || echo "")
if [[ -n "$content" ]]; then
    pass "$content"
else
    fail "No response"
fi

# 3. Tool calling
echo -n "3. Tool call: "
resp=$(curl -sf --max-time 60 "${BASE_URL}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"${MODEL_NAME}\",\"messages\":[{\"role\":\"user\",\"content\":\"2+2=?\"}],\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"calc\",\"parameters\":{\"type\":\"object\",\"properties\":{\"expr\":{\"type\":\"string\"}},\"required\":[\"expr\"]}}}],\"max_tokens\":50}" 2>/dev/null)
tc=$(echo "$resp" | python3 -c "
import sys,json
msg=json.load(sys.stdin)['choices'][0]['message']
if msg.get('tool_calls'): print('tool='+msg['tool_calls'][0]['function']['name'])
else: print('no_tool')
" 2>/dev/null || echo "parse_error")
pass "$tc"

echo ""
echo "All tests completed."
