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
#   SKIP_WAIT SKIP_HEALTH SKIP_MODELS SKIP_CHAT SKIP_STREAM SKIP_TOOLS
#   SKIP_ANTHROPIC SKIP_CODE
#   ENABLE_VISION=1  启用多模态图片测试(默认关闭,仅多模态模型开启)
#   VISION_URL       自定义测试图片 URL
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

# ---- Colors（逐变量兜底，兼容 common.sh 只定义了部分颜色的情况）---------------
: "${RED:='\033[0;31m'}"
: "${GREEN:='\033[0;32m'}"
: "${YELLOW:='\033[1;33m'}"
: "${BLUE:='\033[0;34m'}"
: "${NC:='\033[0m'}"

# ==============================================================================
# Internal helpers（ct_ 前缀：curl wrapper / builder / parser；log_ 前缀：日志；skip_if：跳过检测）
# ==============================================================================

# ---- logging ----------------------------------------------------------------
log_ok()   { echo -e "${GREEN}[PASS]${NC} $*"; }
log_err()  { echo -e "${RED}[FAIL]${NC} $*" >&2; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $*" >&2; }
log_info() { echo -e "${BLUE}[INFO]${NC} $*"; }
log_sec()  { echo ""; log_info "--- $1 ---"; }

# ---- skip guard: skip_if SKIP_CHAT → 检查 ${SKIP_CHAT} 是否为 1 ----------------
skip_if() { [[ "${!1:-0}" == "1" ]]; }

# ---- curl wrappers -----------------------------------------------------------
ct_curl()      { curl -sf --max-time "${CURL_TIMEOUT:-120}" "$@"; }
ct_curl_post() { ct_curl -H "Content-Type: application/json" -d "$1" "$2"; }
ct_curl_raw() {
    curl -s -w '\n%{http_code}' --max-time "${CURL_TIMEOUT:-120}" \
        -H "Content-Type: application/json" -d "$1" "$2" 2>/dev/null
}

# ---- JSON builders -----------------------------------------------------------
ct_build_chat() {
    printf '{"model":"%s","messages":[{"role":"user","content":"%s"}],' \
        "${MODEL_NAME}" "$1"
    printf '"max_tokens":%d,"stream":false}' "${2:-128}"
}
ct_build_stream() {
    printf '{"model":"%s","messages":[{"role":"user","content":"%s"}],' \
        "${MODEL_NAME}" "$1"
    printf '"max_tokens":%d,"stream":true}' "${2:-100}"
}
ct_build_tools() {
    local tool='[{"type":"function","function":{"name":"get_weather",'
    tool+='"description":"Get current weather","parameters":{'
    tool+='"type":"object","properties":{"city":{'
    tool+='"type":"string","description":"City name"}},'
    tool+='"required":["city"]}}}]'
    printf '{"model":"%s","messages":[{"role":"user","content":"%s"}],' \
        "${MODEL_NAME}" "$1"
    printf '"max_tokens":100,"tool_choice":"auto","tools":%s}' "$tool"
}
ct_build_vision() {
    local url="${VISION_URL:-https://example.com/test.png}"
    printf '{"model":"%s","messages":[{"role":"user","content":[' "${MODEL_NAME}"
    printf '{"type":"text","text":"Describe this image briefly."},'
    printf '{"type":"image_url","image_url":{"url":"%s"}}]}],' "$url"
    printf '"max_tokens":128}'
}

# ---- JSON helpers -----------------------------------------------------------
ct_json()      { python3 "${CT_DIR}/curl_helper.py" "$1" <<<"$2" 2>/dev/null || echo ""; }
pretty_json() {
    local input
    input=$(cat)
    python3 -m json.tool <<<"$input" 2>/dev/null || echo "$input"
}

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
    skip_if SKIP_WAIT && { log_info "跳过等待"; return 0; }
    log_info "等待服务就绪: ${BASE_URL} ..."
    local start_ts elapsed
    start_ts=$(date +%s)
    while true; do
        if ct_curl "${BASE_URL}/v1/models" -o /dev/null || ct_curl "${BASE_URL}/health" -o /dev/null; then
            log_ok "服务已就绪"; return 0
        fi
        elapsed=$(( $(date +%s) - start_ts ))
        (( elapsed >= WAIT_TIMEOUT )) && { log_err "等待超时 (${WAIT_TIMEOUT}s)"; return 1; }
        log_info "等待中... (${elapsed}s/${WAIT_TIMEOUT}s)"
        sleep "$WAIT_INTERVAL"
    done
}

curl_test::health() {
    skip_if SKIP_HEALTH && return 0
    log_sec "健康检查"
    local code ep
    for ep in "/health" "/v1/models"; do
        code=$(curl -sf --max-time 5 -o /dev/null -w '%{http_code}' "${BASE_URL}${ep}" 2>/dev/null || echo "000")
        [[ "$code" == "200" ]] && { log_ok "${ep} → 200"; return 0; }
    done
    log_err "服务不可达"; return 1
}

curl_test::models() {
    skip_if SKIP_MODELS && return 0
    log_sec "模型列表"
    local resp
    resp=$(ct_curl "${BASE_URL}/v1/models") || { log_err "无法获取"; return 1; }
    echo "$resp" | pretty_json
    log_ok "获取成功"
}

curl_test::chat() {
    skip_if SKIP_CHAT && return 0
    local prompt="${1:-${PROMPTS[0]}}" max_tokens="${2:-128}"
    log_sec "非流式对话"
    log_info "Prompt: ${prompt}"
    local resp content usage
    resp=$(ct_curl_post "$(ct_build_chat "$prompt" "$max_tokens")" "${BASE_URL}/v1/chat/completions") || {
        log_err "请求失败"; return 1
    }
    content=$(ct_json content "$resp")
    if [[ -n "$content" ]]; then
        log_ok "回复: ${content:0:200}"
        usage=$(ct_json usage "$resp")
        [[ -n "$usage" ]] && log_info "Tokens: $usage"
    else
        log_err "空回复"
        echo "$resp" | pretty_json
        return 1
    fi
}

curl_test::stream() {
    skip_if SKIP_STREAM && return 0
    local prompt="${1:-${PROMPTS[2]}}" max_tokens="${2:-100}"
    log_sec "流式对话"
    log_info "Prompt: ${prompt}"
    local output
    output=$(ct_curl_post "$(ct_build_stream "$prompt" "$max_tokens")" \
        "${BASE_URL}/v1/chat/completions" 2>/dev/null | head -10) || true
    if [[ -n "$output" ]]; then
        echo "$output" | head -5
        log_ok "接收成功"
    else
        log_err "流式输出为空"; return 1
    fi
}

curl_test::tools() {
    skip_if SKIP_TOOLS && return 0
    log_sec "工具调用"
    local raw code resp info err
    raw=$(ct_curl_raw "$(ct_build_tools "${PROMPTS[3]}")" "${BASE_URL}/v1/chat/completions")
    code="${raw##*$'\n'}"
    resp="${raw%$'\n'*}"
    if [[ -z "$resp" || "$code" != "200" ]]; then
        err=$(ct_json error "$resp")
        if [[ "$err" == *"tool"* || "$err" == *"tool_choice"* ]]; then
            log_warn "服务未启用工具调用 (需 --enable-auto-tool-choice --tool-call-parser)"
        else
            log_warn "请求失败 (HTTP ${code}): ${err:-$resp}"
        fi
        return 0
    fi
    info=$(ct_json tool "$resp")
    case "$info" in
        tool=*) log_ok   "工具调用: ${info}" ;;
        text=*) log_warn "返回文本: ${info}" ;;
        *)      log_warn "解析异常: ${info}" ;;
    esac
}

curl_test::code() {
    skip_if SKIP_CODE && return 0
    log_sec "代码生成"
    curl_test::chat "${PROMPTS[6]}" 200
}

curl_test::vision() {
    [[ "${ENABLE_VISION:-0}" == "1" ]] || return 0
    skip_if SKIP_VISION && return 0
    log_sec "多模态 Vision (图片 URL)"
    local raw code resp content
    raw=$(ct_curl_raw "$(ct_build_vision)" "${BASE_URL}/v1/chat/completions")
    code="${raw##*$'\n'}"
    resp="${raw%$'\n'*}"
    if [[ "$code" != "200" ]]; then
        log_warn "Vision 请求失败 (HTTP ${code})，模型可能不支持图片输入"
        return 0
    fi
    content=$(ct_json content "$resp")
    if [[ -n "$content" && "$content" != "None" ]]; then
        log_ok "Vision 回复: ${content:0:150}"
    else
        log_warn "Vision 响应为空"
    fi
    return 0
}

curl_test::anthropic() {
    skip_if SKIP_ANTHROPIC && return 0
    log_sec "Anthropic Messages API"
    local resp content body
    body='{"model":"'"${MODEL_NAME}"'","max_tokens":100,'
    body+='"messages":[{"role":"user","content":"'"${PROMPTS[4]}"'"}]}'
    resp=$(curl -sf --max-time "$CURL_TIMEOUT" "${BASE_URL}/v1/messages" \
        -H "Content-Type: application/json" -H "x-api-key: dummy" \
        -d "$body" 2>/dev/null) || { log_warn "不可用（部分服务不支持）"; return 0; }
    content=$(ct_json anthropic "$resp")
    if [[ -n "$content" ]]; then log_ok "Anthropic API: ${content}"; else log_warn "响应为空"; fi
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
    curl_test::wait      || { log_err "服务未就绪，终止测试"; return 1; }
    curl_test::health    || ((failed++))
    curl_test::models    || ((failed++))
    curl_test::chat "${PROMPTS[0]}" || ((failed++))   # 中文
    curl_test::chat "${PROMPTS[1]}" || ((failed++))   # 英文
    curl_test::code      || ((failed++))
    curl_test::stream    || ((failed++))
    curl_test::tools     || ((failed++))
    curl_test::vision    || ((failed++))
    curl_test::anthropic
    echo ""
    echo "=========================================="
    if (( failed > 0 )); then log_err "测试完成，${failed} 项失败"; else log_ok "所有测试通过"; fi
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
