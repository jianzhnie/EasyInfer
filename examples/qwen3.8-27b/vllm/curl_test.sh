#!/usr/bin/env bash
# =============================================================================
# Qwen3.8-27B - OpenAI-compatible API functional test wrapper
# =============================================================================
# Usage:
#   bash curl_test.sh
#   ENABLE_VISION=1 bash curl_test.sh       # optional image request
#   HOST=10.0.0.1 PORT=8022 bash curl_test.sh
#   SKIP_REASONING=1 bash curl_test.sh       # skip effort matrix
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_DIR

export PORT="${PORT:-8022}"
export MODEL_NAME="${MODEL_NAME:-${SERVED_MODEL_NAME:-qwen3.8}}"
export ENABLE_VISION="${ENABLE_VISION:-0}"
export SKIP_REASONING="${SKIP_REASONING:-0}"

for name in ENABLE_VISION SKIP_REASONING; do
    [[ "${!name}" == 0 || "${!name}" == 1 ]] || {
        echo "ERROR: $name must be 0 or 1" >&2
        exit 1
    }
done
[[ "$PORT" =~ ^[1-9][0-9]*$ ]] && (( PORT <= 65535 )) || {
    echo "ERROR: PORT must be an integer from 1 to 65535" >&2
    exit 1
}
[[ -n "$MODEL_NAME" ]] || {
    echo "ERROR: MODEL_NAME/SERVED_MODEL_NAME cannot be empty" >&2
    exit 1
}

readonly TEST_LIBRARY="${SCRIPT_DIR}/../../curl_test.sh"
[[ -r "$TEST_LIBRARY" ]] || {
    echo "ERROR: shared test library is missing: $TEST_LIBRARY" >&2
    exit 1
}

# shellcheck source=../../curl_test.sh
source "$TEST_LIBRARY"

# Tool calling and vision are required capabilities when their tests are run,
# so this model-specific wrapper treats malformed responses as failures.
curl_test::tools() {
    skip_if SKIP_TOOLS && { CT_RESULT=SKIP; return 0; }
    log_sec "工具调用"
    local body raw code resp info err
    body=$(ct_build_tools "${PROMPTS[3]}")
    body=$(python3 -c \
        'import json,sys; data=json.load(sys.stdin); data["tool_choice"]="required"; print(json.dumps(data))' \
        <<<"$body")
    raw=$(ct_curl_raw "$body" "${BASE_URL}/v1/chat/completions")
    code="${raw##*$'\n'}"
    resp="${raw%$'\n'*}"
    if [[ -z "$resp" || "$code" != 200 ]]; then
        err=$(ct_json error "$resp")
        log_err "工具调用失败 (HTTP ${code}): ${err:-$resp}"
        return 1
    fi
    info=$(ct_json tool "$resp")
    [[ "$info" == tool=* ]] || {
        log_err "未返回结构化 tool_call: ${info:-解析失败}"
        return 1
    }
    log_ok "工具调用: $info"
}

curl_test::vision() {
    [[ "$ENABLE_VISION" == 1 ]] || { CT_RESULT=SKIP; return 0; }
    skip_if SKIP_VISION && { CT_RESULT=SKIP; return 0; }
    log_sec "多模态 Vision (图片 URL)"
    local raw code resp content
    raw=$(ct_curl_raw "$(ct_build_vision)" "${BASE_URL}/v1/chat/completions")
    code="${raw##*$'\n'}"
    resp="${raw%$'\n'*}"
    [[ "$code" == 200 ]] || {
        log_err "Vision 请求失败 (HTTP $code)"
        return 1
    }
    content=$(ct_json content "$resp")
    [[ -n "$content" && "$content" != None ]] || {
        log_err "Vision 响应为空"
        return 1
    }
    log_ok "Vision 回复: ${content:0:150}"
}

qwen_test::reasoning_effort() {
    [[ "$SKIP_REASONING" == 1 ]] && return 0
    log_sec "reasoning_effort"
    local effort body raw code resp err
    for effort in xhigh medium low; do
        body=$(python3 -c \
            'import json,sys; print(json.dumps({"model":sys.argv[1],"messages":[{"role":"user","content":"Reply with OK."}],"reasoning_effort":sys.argv[2],"max_tokens":32}))' \
            "$MODEL_NAME" "$effort")
        raw=$(ct_curl_raw "$body" "${BASE_URL}/v1/chat/completions")
        code="${raw##*$'\n'}"
        resp="${raw%$'\n'*}"
        if [[ "$code" != 200 ]]; then
            err=$(ct_json error "$resp")
            log_err "reasoning_effort=$effort 失败 (HTTP $code): ${err:-$resp}"
            return 1
        fi
        log_ok "reasoning_effort=$effort -> HTTP 200"
    done
}

if ! curl_test::run "$@"; then
    exit 1
fi
qwen_test::reasoning_effort
