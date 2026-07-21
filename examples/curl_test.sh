#!/bin/bash
# =============================================================================
# EasyInfer LLM API 测试库 — 适用于 vLLM / SGLang 兼容 OpenAI API 的服务
# =============================================================================
# 用法:
#   source examples/curl_test.sh && curl_test::init && curl_test::chat
#   MODEL_NAME=mymodel PORT=8000 bash examples/curl_test.sh
#
# 环境变量:
#   HOST PORT BASE_URL MODEL_NAME CURL_TIMEOUT WAIT_TIMEOUT WAIT_INTERVAL
#   SKIP_WAIT SKIP_HEALTH SKIP_MODELS SKIP_CHAT SKIP_STREAM SKIP_TOOLS SKIP_ANTHROPIC
# =============================================================================

# ---- Config（source 后可修改）--------------------------------------------------
# shellcheck disable=SC2034
SYSTEM_PROMPTS=(
    "You are a helpful assistant."
    "You are a helpful assistant. Please answer concisely and accurately."
    "You are a helpful assistant with access to tools. Use tools when appropriate."
)
PROMPTS=(
    "你好，请用一句话介绍你自己。"                   # 0: 中文自我介绍
    "Hello, who are you? Please answer briefly."      # 1: 英文自我介绍
    "从1数到5"                                        # 2: 流式测试
    "What is the weather in Beijing?"                 # 3: 工具调用
    "Hi there!"                                       # 4: Anthropic API
    "What is 123 * 456? Show your reasoning."         # 5: 数学推理
    "Write a Python function to check if a number is prime."  # 6: 代码生成
)

# ---- Lib setup ---------------------------------------------------------------
CT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly CT_DIR
if [[ -f "${CT_DIR}/../scripts/common.sh" ]]; then
    # shellcheck source=../scripts/common.sh
    source "${CT_DIR}/../scripts/common.sh"
fi

# ---- Colors ------------------------------------------------------------------
if [[ -z "${RED:-}" ]]; then
    readonly RED='\033[0;31m' GREEN='\033[0;32m' YELLOW='\033[1;33m' BLUE='\033[0;34m' NC='\033[0m'
elif [[ -z "${BLUE:-}" ]]; then
    BLUE='\033[0;34m'
fi

# ==============================================================================
# Internal helpers — 全部以下划线开头，不对外暴露
# ==============================================================================

# ---- logging ----------------------------------------------------------------
_ok()   { echo -e "${GREEN}[PASS]${NC} $*"; }
_err()  { echo -e "${RED}[FAIL]${NC} $*" >&2; }
_warn() { echo -e "${YELLOW}[WARN]${NC} $*" >&2; }
_info() { echo -e "${BLUE}[INFO]${NC} $*"; }
_sec()  { echo ""; _info "--- $1 ---"; }

# ---- skip guard: _skip SKIP_CHAT → 检查 ${SKIP_CHAT} 是否为 1 ----------------
_skip() { [[ "${!1:-0}" == "1" ]]; }

# ---- curl wrappers -----------------------------------------------------------
_curl()      { curl -sf --max-time "${CURL_TIMEOUT:-120}" "$@"; }
_curl_post() { _curl -H "Content-Type: application/json" -d "$1" "$2"; }
_curl_raw()  { curl -s -w '\n%{http_code}' --max-time "${CURL_TIMEOUT:-120}" \
                   -H "Content-Type: application/json" -d "$1" "$2" 2>/dev/null; }

# ---- JSON builders -----------------------------------------------------------
_build_chat()   { printf '{"model":"%s","messages":[{"role":"user","content":"%s"}],"max_tokens":%d,"stream":false}' \
                     "${MODEL_NAME}" "$1" "${2:-128}"; }
_build_stream() { printf '{"model":"%s","messages":[{"role":"user","content":"%s"}],"max_tokens":%d,"stream":true}' \
                     "${MODEL_NAME}" "$1" "${2:-100}"; }
_build_tools()  { printf '{"model":"%s","messages":[{"role":"user","content":"%s"}],"max_tokens":100,"tool_choice":"auto","tools":[{"type":"function","function":{"name":"get_weather","description":"Get current weather","parameters":{"type":"object","properties":{"city":{"type":"string","description":"City name"}},"required":["city"]}}}]}' \
                     "${MODEL_NAME}" "$1"; }

# ---- JSON parser（统一入口，底层是 _curl_test_json.py）-------------------------
_json() { python3 "${CT_DIR}/_curl_test_json.py" "$1" <<<"$2" 2>/dev/null || echo ""; }

# ==============================================================================
# Public API (curl_test:: namespace)
# ==============================================================================

curl_test::init() {
    HOST="${HOST:-localhost}"
    PORT="${PORT:-8000}"
    MODEL_NAME="${MODEL_NAME:-default-model}"
    BASE_URL="${BASE_URL:-http://${HOST}:${PORT}}"
    CURL_TIMEOUT="${CURL_TIMEOUT:-120}"
    WAIT_TIMEOUT="${WAIT_TIMEOUT:-300}"
    WAIT_INTERVAL="${WAIT_INTERVAL:-5}"
}

curl_test::wait() {
    _skip SKIP_WAIT && { _info "跳过等待"; return 0; }
    _info "等待服务就绪: ${BASE_URL} ..."
    local start_ts elapsed
    start_ts=$(date +%s)
    while true; do
        _curl "${BASE_URL}/v1/models" -o /dev/null && { _ok "服务已就绪"; return 0; }
        _curl "${BASE_URL}/health"    -o /dev/null && { _ok "服务已就绪"; return 0; }
        elapsed=$(( $(date +%s) - start_ts ))
        (( elapsed >= WAIT_TIMEOUT )) && { _err "等待超时 (${WAIT_TIMEOUT}s)"; return 1; }
        _info "等待中... (${elapsed}s/${WAIT_TIMEOUT}s)"
        sleep "$WAIT_INTERVAL"
    done
}

curl_test::health() {
    _skip SKIP_HEALTH && return 0
    _sec "健康检查"
    local code
    code=$(curl -sf --max-time 5 -o /dev/null -w '%{http_code}' "${BASE_URL}/health" 2>/dev/null || echo "000")
    [[ "$code" == "200" ]] && { _ok "/health → 200"; return 0; }
    code=$(curl -sf --max-time 5 -o /dev/null -w '%{http_code}' "${BASE_URL}/v1/models" 2>/dev/null || echo "000")
    [[ "$code" == "200" ]] && { _ok "/v1/models → 200"; return 0; }
    _err "服务不可达"; return 1
}

curl_test::models() {
    _skip SKIP_MODELS && return 0
    _sec "模型列表"
    local resp
    resp=$(_curl "${BASE_URL}/v1/models") || { _err "无法获取"; return 1; }
    echo "$resp" | python3 -m json.tool 2>/dev/null || echo "$resp"
    _ok "获取成功"
}

curl_test::chat() {
    _skip SKIP_CHAT && return 0
    local prompt="${1:-${PROMPTS[0]}}" max_tokens="${2:-128}"
    _sec "非流式对话"
    _info "Prompt: ${prompt}"
    local resp content usage
    resp=$(_curl_post "$(_build_chat "$prompt" "$max_tokens")" "${BASE_URL}/v1/chat/completions") || {
        _err "请求失败"; return 1
    }
    content=$(_json content "$resp")
    if [[ -n "$content" ]]; then
        _ok "回复: ${content:0:200}"
        usage=$(_json usage "$resp")
        [[ -n "$usage" ]] && _info "Tokens: $usage"
    else
        _err "空回复"
        echo "$resp" | python3 -m json.tool 2>/dev/null || echo "$resp"
        return 1
    fi
}

curl_test::stream() {
    _skip SKIP_STREAM && return 0
    local prompt="${1:-${PROMPTS[2]}}" max_tokens="${2:-100}"
    _sec "流式对话"
    _info "Prompt: ${prompt}"
    local output
    output=$(_curl_post "$(_build_stream "$prompt" "$max_tokens")" "${BASE_URL}/v1/chat/completions" 2>&1 | head -10) || true
    if [[ -n "$output" ]]; then
        echo "$output" | head -5
        _ok "接收成功"
    else
        _err "流式输出为空"; return 1
    fi
}

curl_test::tools() {
    _skip SKIP_TOOLS && return 0
    _sec "工具调用"
    local raw code resp info err
    raw=$(_curl_raw "$(_build_tools "${PROMPTS[3]}")" "${BASE_URL}/v1/chat/completions")
    code="${raw##*$'\n'}"
    resp="${raw%$'\n'*}"
    if [[ -z "$resp" || "$code" != "200" ]]; then
        err=$(_json error "$resp")
        if [[ "$err" == *"tool"* || "$err" == *"tool_choice"* ]]; then
            _warn "服务未启用工具调用 (需 --enable-auto-tool-choice --tool-call-parser)"
        else
            _warn "请求失败 (HTTP ${code}): ${err:-$resp}"
        fi
        return 0
    fi
    info=$(_json tool "$resp")
    case "$info" in
        tool=*) _ok   "工具调用: ${info}" ;;
        text=*) _warn "返回文本: ${info}" ;;
        *)      _warn "解析异常: ${info}" ;;
    esac
}

curl_test::anthropic() {
    _skip SKIP_ANTHROPIC && return 0
    _sec "Anthropic Messages API"
    local resp content
    resp=$(curl -sf --max-time "$CURL_TIMEOUT" "${BASE_URL}/v1/messages" \
        -H "Content-Type: application/json" -H "x-api-key: dummy" \
        -d "{\"model\":\"${MODEL_NAME}\",\"max_tokens\":100,\"messages\":[{\"role\":\"user\",\"content\":\"${PROMPTS[4]}\"}]}" \
        2>/dev/null) || { _warn "不可用（部分服务不支持）"; return 0; }
    content=$(_json anthropic "$resp")
    if [[ -n "$content" ]]; then _ok "Anthropic API: ${content}"; else _warn "响应为空"; fi
    return 0
}

curl_test::banner() {
    printf '\n==========================================\n'
    printf '  EasyInfer API 功能测试\n'
    printf '  目标地址: %s\n' "${BASE_URL}"
    printf '  模型名称: %s\n' "${MODEL_NAME}"
    printf '==========================================\n\n'
}

curl_test::run() {
    curl_test::init
    curl_test::banner
    local failed=0
    curl_test::wait      || { _err "服务未就绪，终止测试"; return 1; }
    curl_test::health    || ((failed++))
    curl_test::models    || ((failed++))
    curl_test::chat      || ((failed++))
    curl_test::stream    || ((failed++))
    curl_test::tools     || ((failed++))
    curl_test::anthropic || true
    echo ""
    echo "=========================================="
    if (( failed > 0 )); then _err "测试完成，${failed} 项失败"; else _ok "所有测试通过"; fi
    echo "=========================================="
    return "$failed"
}

# ==============================================================================
# Direct execution
# ==============================================================================
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    set -euo pipefail
    curl_test::run
fi
