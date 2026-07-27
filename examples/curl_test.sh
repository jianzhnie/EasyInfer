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
#   SKIP_ANTHROPIC SKIP_CODE SKIP_LONG_CONTEXT
#   VERBOSE=1        打印完整 JSON 响应 (如模型列表)
#   ENABLE_VISION=1  启用多模态图片测试(默认关闭,仅多模态模型开启)
#   VISION_URL       自定义测试图片 URL
#   ENABLE_LONG_CONTEXT=1  启用长上下文大海捞针矩阵(默认关闭,需长上下文部署)
#   LONG_CONTEXT_CASES     用例选择, 如 "1,3,8" (默认全部 8 个)
#   TARGET_TOKENS MIN_ACCEPT_TOKENS LONG_CONTEXT_MAX_TOKENS
#   LONG_CONTEXT_TIMEOUT   大海捞针旋钮
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
# 非 TTY (管道/重定向) 或 NO_COLOR 时关闭颜色, 避免转义序列变成乱码。
# 注意: common.sh 里的颜色变量是 readonly, 不能清空重赋值, 因此日志函数
# 一律使用 C_* 副本。
if [[ ! -t 1 || -n "${NO_COLOR:-}" ]]; then
    C_RED='' C_GREEN='' C_YELLOW='' C_BLUE='' C_NC=''
else
    C_RED="$RED" C_GREEN="$GREEN" C_YELLOW="$YELLOW" C_BLUE="$BLUE" C_NC="$NC"
fi

# ==============================================================================
# Internal helpers（ct_ 前缀：curl wrapper / builder / parser；log_ 前缀：日志；skip_if：跳过检测）
# ==============================================================================

# ---- logging ----------------------------------------------------------------
# 用 printf '%s' 输出内容, 避免 echo -e 把回复里的反斜杠序列 (如 \times)
# 误解析为转义字符; 颜色码走 C_* 副本 (见上方说明)
log_ok()   { printf "${C_GREEN}[PASS]${C_NC} %s\n" "$*"; }
log_err()  { printf "${C_RED}[FAIL]${C_NC} %s\n" "$*" >&2; }
log_warn() { printf "${C_YELLOW}[WARN]${C_NC} %s\n" "$*" >&2; }
log_info() { printf "${C_BLUE}[INFO]${C_NC} %s\n" "$*"; }
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
# 构造逻辑在 curl_helper.py 的 build 命令, bash 侧只做薄调用;
# 用 python3 json.dumps 构造, 避免 prompt 含引号/反斜杠时 JSON 注入
ct_json_build() {  # $1=chat|stream|tools|vision  $2=prompt  $3=max_tokens
    python3 "${CT_DIR}/curl_helper.py" build "$1" "${MODEL_NAME}" "$2" "${3:-128}" \
        "${VISION_URL:-https://example.com/test.png}"
}
ct_build_chat()   { ct_json_build chat   "$1" "${2:-128}"; }
ct_build_stream() { ct_json_build stream "$1" "${2:-100}"; }
ct_build_tools()  { ct_json_build tools  "$1" 100; }
ct_build_vision() { ct_json_build vision "" 128; }

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
    local resp summary
    resp=$(ct_curl "${BASE_URL}/v1/models") || { log_err "无法获取"; return 1; }
    summary=$(ct_json models "$resp")
    if [[ -n "$summary" ]]; then
        echo "$summary" | while IFS= read -r line; do log_ok "$line"; done
    fi
    # VERBOSE=1 时打印完整 JSON
    [[ "${VERBOSE:-0}" == "1" ]] && echo "$resp" | pretty_json
    [[ -z "$summary" ]] && { log_err "模型列表为空"; return 1; }
    return 0
}

curl_test::chat() {
    skip_if SKIP_CHAT && return 0
    local prompt="${1:-${PROMPTS[0]}}" max_tokens="${2:-128}"
    log_sec "非流式对话"
    log_info "Prompt: ${prompt}"
    local resp content usage start_ts elapsed
    start_ts=$(date +%s)
    resp=$(ct_curl_post "$(ct_build_chat "$prompt" "$max_tokens")" "${BASE_URL}/v1/chat/completions") || {
        log_err "请求失败"; return 1
    }
    elapsed=$(( $(date +%s) - start_ts ))
    content=$(ct_json content "$resp")
    if [[ -n "$content" ]]; then
        # 单行展示, 截断 200 字符, 避免长回复刷屏
        local one_line="${content//$'\n'/ }"
        log_ok "回复: ${one_line:0:200}"
        usage=$(ct_json usage "$resp")
        [[ -n "$usage" ]] && log_info "Tokens: ${usage} 耗时: ${elapsed}s"
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
    local output info
    output=$(ct_curl_post "$(ct_build_stream "$prompt" "$max_tokens")" \
        "${BASE_URL}/v1/chat/completions" 2>/dev/null) || true
    if [[ -z "$output" ]]; then
        log_err "流式输出为空"; return 1
    fi
    # 校验: 拼接收到的 delta 文本, 统计 chunk 数, 确认 [DONE] 收尾
    info=$(ct_json stream "$output")
    case "$info" in
        ok=*)
            log_ok "接收成功: ${info}"
            ;;
        *)
            log_err "流式响应异常: ${info:-无内容}"
            echo "$output" | head -5
            return 1
            ;;
    esac
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
        CT_RESULT="WARN"
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
        text=*) log_warn "返回文本: ${info}"; CT_RESULT="WARN" ;;
        *)      log_warn "解析异常: ${info}"; CT_RESULT="WARN" ;;
    esac
}

curl_test::code() {
    skip_if SKIP_CODE && return 0
    log_sec "代码生成"
    curl_test::chat "${PROMPTS[6]}" 200
}

curl_test::vision() {
    [[ "${ENABLE_VISION:-0}" == "1" ]] || { CT_RESULT="SKIP"; return 0; }
    skip_if SKIP_VISION && { CT_RESULT="SKIP"; return 0; }
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
        -d "$body" 2>/dev/null) || { log_warn "不可用（部分服务不支持）"; CT_RESULT="WARN"; return 0; }
    content=$(ct_json anthropic "$resp")
    if [[ -n "$content" ]]; then log_ok "Anthropic API: ${content}"; else log_warn "响应为空"; CT_RESULT="WARN"; fi
    return 0
}

# ==============================================================================
# 长上下文大海捞针矩阵 (ENABLE_LONG_CONTEXT=1 启用)
# ==============================================================================
# 用例 (LONG_CONTEXT_CASES 选择, 默认全部):
#   1 单针@开头 32K   2 单针@中间 32K   3 单针@结尾 32K
#   4 单针@中间 64K   5 三针检索 64K    6 中文填充 32K
#   7 多轮对话 32K    8 大海捞针 TARGET_TOKENS (默认 130K)
# 旋钮: TARGET_TOKENS MIN_ACCEPT_TOKENS LONG_CONTEXT_MAX_TOKENS
#       LONG_CONTEXT_TIMEOUT LONG_CONTEXT_CASES MAGIC_NUMBERS SKIP_LONG_CONTEXT
# ==============================================================================

CT_LC_M1=7391842 CT_LC_M2=5820371 CT_LC_M3=2468109

ct_lc_tokenize() {  # $1=prompt file -> stdout: token count (失败输出空)
    python3 "${CT_DIR}/curl_helper.py" build tokenize "${MODEL_NAME}" - 0 \
        < "$1" > "${CT_LC_DIR}/tok_req.json"
    curl -s -m 120 "${BASE_URL}/tokenize" -H "Content-Type: application/json" \
        -d @"${CT_LC_DIR}/tok_req.json" \
        | python3 "${CT_DIR}/curl_helper.py" count 2>/dev/null
}

ct_lc_gen_body() {  # $1=lines $2=lang $3=pos $4=multi $5=outfile
    local filler="the quick brown fox jumps over the lazy dog and runs through the forest"
    [[ "$2" == "zh" ]] && filler="秋日的阳光洒在宁静的湖面上，微风拂过岸边芦苇，泛起层层涟漪。"
    {
        [[ "$3" == "start" ]] && \
            echo "IMPORTANT FACT: The first magic number is ${CT_LC_M1}."
        awk -v n=$(($1/2)) -v s="$filler" 'BEGIN{for(i=0;i<n;i++)print "Record "i": "s}'
        [[ "$3" == "middle" ]] && \
            echo "IMPORTANT FACT: The first magic number is ${CT_LC_M1}."
        [[ "$4" == "1" ]] && \
            echo "IMPORTANT FACT: The second magic number is ${CT_LC_M2}."
        awk -v n=$(($1-$1/2)) -v s="$filler" 'BEGIN{for(i=0;i<n;i++)print "Record "i": "s}'
        [[ "$3" == "end" ]] && \
            echo "IMPORTANT FACT: The first magic number is ${CT_LC_M1}."
        [[ "$4" == "1" ]] && \
            echo "IMPORTANT FACT: The third magic number is ${CT_LC_M3}."
    } > "$5"
}

ct_lc_calibrate() {  # $1=target $2=lang $3=pos $4=multi -> 写 body.txt, stdout: count
    local lines=$(($1/15)) count i per diff
    for i in 1 2 3 4 5; do
        ct_lc_gen_body "$lines" "$2" "$3" "$4" "${CT_LC_DIR}/body.txt"
        count=$(ct_lc_tokenize "${CT_LC_DIR}/body.txt")
        [[ -z "$count" ]] && return 1
        diff=$(($1 - count)); (( diff < 0 )) && diff=$((-diff))
        (( diff <= 300 )) && break
        per=$(( (count - 100) / lines )); (( per < 1 )) && per=15
        lines=$((lines + ($1 - count) / per))
    done
    echo "$count"
}

ct_lc_ask() {  # $1=question $2=magics_csv $3=min_tokens $4=kind(chat|multiturn)
    # -> stdout "0|detail"|"1|detail"; body 来自 ${CT_LC_DIR}/body.txt
    local max_out="${LONG_CONTEXT_MAX_TOKENS:-64}"
    if [[ "$4" == "multiturn" ]]; then
        cp "${CT_LC_DIR}/body.txt" "${CT_LC_DIR}/prompt.txt"
    else
        cat "${CT_LC_DIR}/body.txt" > "${CT_LC_DIR}/prompt.txt"
        echo "" >> "${CT_LC_DIR}/prompt.txt"
        echo "$1" >> "${CT_LC_DIR}/prompt.txt"
    fi
    python3 "${CT_DIR}/curl_helper.py" build "$4" "${MODEL_NAME}" - "$max_out" \
        < "${CT_LC_DIR}/prompt.txt" > "${CT_LC_DIR}/chat_req.json"
    curl -s -m "${LONG_CONTEXT_TIMEOUT:-1800}" "${BASE_URL}/v1/chat/completions" \
        -H "Content-Type: application/json" -d @"${CT_LC_DIR}/chat_req.json" \
        > "${CT_LC_DIR}/resp.json"
    python3 "${CT_DIR}/curl_helper.py" needle "$2" "$3" \
        < "${CT_LC_DIR}/resp.json"
}

ct_lc_case() {  # $1=name $2=target $3=pos $4=lang $5=multi $6=question $7=magics
    local count detail
    log_info "用例: $1 (target=$2)"
    count=$(ct_lc_calibrate "$2" "$4" "$3" "$5") || {
        log_err "$1: tokenize 校准失败"; return 1; }
    detail=$(ct_lc_ask "$6" "$7" $(($2 - 2000)) chat)
    if [[ "${detail%%|*}" == "0" ]]; then
        log_ok "$1: ${detail#*|}"
    else
        log_err "$1: ${detail#*|}"; return 1
    fi
}

ct_lc_case_multiturn() {  # $1=name $2=target
    local count detail
    log_info "用例: $1 (target=$2)"
    count=$(ct_lc_calibrate "$2" en start 0) || {
        log_err "$1: tokenize 校准失败"; return 1; }
    detail=$(ct_lc_ask "" "${CT_LC_M1}" $(($2 - 2000)) multiturn)
    if [[ "${detail%%|*}" == "0" ]]; then
        log_ok "$1: ${detail#*|}"
    else
        log_err "$1: ${detail#*|}"; return 1
    fi
}

curl_test::long_context() {
    [[ "${ENABLE_LONG_CONTEXT:-0}" == "1" ]] || { CT_RESULT="SKIP"; return 0; }
    skip_if SKIP_LONG_CONTEXT && { CT_RESULT="SKIP"; return 0; }
    log_sec "长上下文大海捞针矩阵"
    CT_LC_DIR=$(mktemp -d)
    local cases="${LONG_CONTEXT_CASES:-1,2,3,4,5,6,7,8}"
    local target="${TARGET_TOKENS:-130000}"
    local min_accept="${MIN_ACCEPT_TOKENS:-100000}"
    local failed=0 count detail

    _lc_in() { [[ ",${cases}," == *",$1,"* ]]; }

    _lc_in 1 && { ct_lc_case "单针@开头 32K" 32000 start en 0 \
        "What is the first magic number? Answer with the number only." "${CT_LC_M1}" || failed=$((failed+1)); }
    _lc_in 2 && { ct_lc_case "单针@中间 32K" 32000 middle en 0 \
        "What is the first magic number? Answer with the number only." "${CT_LC_M1}" || failed=$((failed+1)); }
    _lc_in 3 && { ct_lc_case "单针@结尾 32K" 32000 end en 0 \
        "What is the first magic number? Answer with the number only." "${CT_LC_M1}" || failed=$((failed+1)); }
    _lc_in 4 && { ct_lc_case "单针@中间 64K" 64000 middle en 0 \
        "What is the first magic number? Answer with the number only." "${CT_LC_M1}" || failed=$((failed+1)); }
    _lc_in 5 && { ct_lc_case "三针检索 64K" 64000 middle en 1 \
        "What are the first, second and third magic numbers? List them in order, numbers only." \
        "${CT_LC_M1},${CT_LC_M2},${CT_LC_M3}" || failed=$((failed+1)); }
    _lc_in 6 && { ct_lc_case "中文填充 32K" 32000 middle zh 0 \
        "文中提到的魔法数字是什么？只回答数字。" "${CT_LC_M1}" || failed=$((failed+1)); }
    _lc_in 7 && { ct_lc_case_multiturn "多轮对话 32K" 32000 || failed=$((failed+1)); }
    if _lc_in 8; then
        log_info "用例: 大海捞针 ${target}"
        count=$(ct_lc_calibrate "$target" en start 0) || {
            log_err "大海捞针: tokenize 校准失败"; failed=$((failed+1)); count=""; }
        if [[ -n "$count" ]]; then
            if [[ "$count" -gt 131000 ]]; then
                log_err "大海捞针: prompt tokens ($count) 超过安全上限 131000"
                failed=$((failed+1))
            else
                detail=$(ct_lc_ask \
                    "What is the first magic number? Answer with the number only." \
                    "${CT_LC_M1}" "$min_accept" chat)
                if [[ "${detail%%|*}" == "0" ]]; then
                    log_ok "大海捞针 ${target}: ${detail#*|}"
                else
                    log_err "大海捞针 ${target}: ${detail#*|}"; failed=$((failed+1))
                fi
            fi
        fi
    fi

    rm -rf "$CT_LC_DIR"
    (( failed > 0 )) && { log_err "长上下文矩阵: ${failed} 个用例失败"; return 1; }
    log_ok "长上下文矩阵全部通过"
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
    declare -a CT_SUMMARY=()
    # 包装: 记录每个用例结果, 结尾打印汇总表
    # 用例函数可设 CT_RESULT=SKIP|WARN 覆盖默认的 PASS 标记
    _case() {  # $1=名称 $2...=命令
        local name="$1"; shift
        CT_RESULT="PASS"
        if "$@"; then CT_SUMMARY+=("${CT_RESULT}|${name}"); return 0;
        else CT_SUMMARY+=("FAIL|${name}"); return 1; fi
    }
    curl_test::wait      || { log_err "服务未就绪，终止测试"; return 1; }
    _case "健康检查"   curl_test::health              || failed=$((failed + 1))
    _case "模型列表"   curl_test::models              || failed=$((failed + 1))
    _case "中文对话"   curl_test::chat "${PROMPTS[0]}" || failed=$((failed + 1))
    _case "英文对话"   curl_test::chat "${PROMPTS[1]}" || failed=$((failed + 1))
    _case "数学推理"   curl_test::chat "${PROMPTS[5]}" || failed=$((failed + 1))
    _case "代码生成"   curl_test::code                || failed=$((failed + 1))
    _case "流式对话"   curl_test::stream              || failed=$((failed + 1))
    _case "工具调用"   curl_test::tools               || failed=$((failed + 1))
    _case "多模态"     curl_test::vision              || failed=$((failed + 1))
    _case "Anthropic"  curl_test::anthropic            || true
    _case "长上下文矩阵" curl_test::long_context      || failed=$((failed + 1))
    echo ""
    echo "================== 测试汇总 =================="
    local item st name
    for item in "${CT_SUMMARY[@]}"; do
        st="${item%%|*}"; name="${item#*|}"
        case "$st" in
            PASS) printf "${C_GREEN}%-6s${C_NC} %s\n" "$st" "$name" ;;
            FAIL) printf "${C_RED}%-6s${C_NC} %s\n" "$st" "$name" ;;
            WARN) printf "${C_YELLOW}%-6s${C_NC} %s\n" "$st" "$name" ;;
            *)    printf "${C_BLUE}%-6s${C_NC} %s\n" "$st" "$name" ;;
        esac
    done
    echo "=============================================="
    if (( failed > 0 )); then log_err "测试完成，${failed} 项失败"; else log_ok "所有测试通过 (${#CT_SUMMARY[@]} 项)"; fi
    return "$failed"
}

# ==============================================================================
# Direct execution
# ==============================================================================
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    set -euo pipefail
    curl_test::run
fi
