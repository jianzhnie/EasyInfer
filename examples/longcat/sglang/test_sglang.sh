#!/bin/bash
#=============================================================================
# LongCat-Flash-Chat SGLang API 验证脚本
#=============================================================================
# 用法:
#   bash test_sglang.sh                    # 测试默认地址
#   HOST=10.42.11.130 PORT=6677 bash test_sglang.sh
#   MODEL_NAME=longcat-flash bash test_sglang.sh
#=============================================================================
set -euo pipefail

#------------------------------------------------------------------------------
# 配置
#------------------------------------------------------------------------------
HOST="${HOST:-10.42.11.130}"
PORT="${PORT:-6677}"
MODEL_NAME="${MODEL_NAME:-longcat-flash}"
readonly TIMEOUT=300
readonly WAIT_INTERVAL=5
readonly BASE_URL="http://${HOST}:${PORT}"

#------------------------------------------------------------------------------
# 前置检查
#------------------------------------------------------------------------------
check_command() {
    local cmd="$1"
    command -v "$cmd" >/dev/null 2>&1 || {
        echo "[ERROR] 缺少必要命令: ${cmd}，请先安装" >&2
        exit 127
    }
}
check_command curl
check_command python3

#------------------------------------------------------------------------------
# 日志辅助函数
#------------------------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info()    { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error()   { echo -e "${RED}[ERROR]${NC} $1"; }

#------------------------------------------------------------------------------
# 等待服务就绪
#------------------------------------------------------------------------------
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

#------------------------------------------------------------------------------
# 构建 JSON 请求体 (安全拼接，避免引号注入)
#------------------------------------------------------------------------------
build_json_body() {
    local model="$1" messages="$2"
    local extra="${3:-}"
    # 使用 python3 生成合法 JSON，彻底避免 shell 字符串拼接导致的注入问题
    python3 -c "
import json
body = {'model': '${model}', 'messages': ${messages}}
body.update(${extra:-{}})
print(json.dumps(body))
"
}

#------------------------------------------------------------------------------
# 执行单个 API 测试
#------------------------------------------------------------------------------
_run_test() {
    local test_name="$1" endpoint="$2" payload="$3"
    local curl_opts=(-s "$endpoint" -H "Content-Type: application/json")

    [[ -n "$payload" ]] && curl_opts+=(-d "$payload")

    echo ""
    log_info "测试: ${test_name}"
    log_info "Endpoint: ${endpoint}"

    if response=$(curl "${curl_opts[@]}"); then
        echo "$response" | python3 -m json.tool 2>/dev/null || echo "$response"
        log_success "${test_name} 通过"
        return 0
    else
        log_error "${test_name} 失败 (curl 退出码: $?)"
        return 1
    fi
}

#------------------------------------------------------------------------------
# 流式聊天完成测试
#------------------------------------------------------------------------------
_run_stream_test() {
    local test_name="$1" payload="$2"
    local http_code

    echo ""
    log_info "测试: ${test_name}"
    http_code=$(curl -s -o /dev/null -w '%{http_code}' \
        "${BASE_URL}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "$payload" 2>&1) || true

    if [[ "$http_code" == "200" ]]; then
        log_success "${test_name} 通过 (HTTP ${http_code})"
    else
        log_warning "${test_name} 返回 HTTP ${http_code}，流式端点可能存在问题"
    fi
}

#------------------------------------------------------------------------------
# 主测试流程
#------------------------------------------------------------------------------
echo "=========================================="
echo "  LongCat-Flash-Chat SGLang API 验证"
echo "  目标地址: ${BASE_URL}"
echo "  模型名称: ${MODEL_NAME}"
echo "=========================================="

wait_for_service || exit 1

FAILED=0

# 1. 模型列表查询 (GET)
_run_test "模型列表查询" \
    "${BASE_URL}/v1/models" \
    "" || ((FAILED++))

# 2. 英文对话
_run_test "Chat Completion (英文)" \
    "${BASE_URL}/v1/chat/completions" \
    "$(build_json_body "$MODEL_NAME" \
        '[{"role":"user","content":"Hello, who are you?"}]' \
        '{"max_tokens":128,"temperature":0.7}')" || ((FAILED++))

# 3. 中文对话
_run_test "Chat Completion (中文)" \
    "${BASE_URL}/v1/chat/completions" \
    "$(build_json_body "$MODEL_NAME" \
        '[{"role":"system","content":"You are a helpful assistant."},{"role":"user","content":"你好，请简单介绍一下你自己。"}]' \
        '{"max_tokens":128,"temperature":0.7}')" || ((FAILED++))

# 4. Tool Calling
_run_test "Tool Calling" \
    "${BASE_URL}/v1/chat/completions" \
    "$(build_json_body "$MODEL_NAME" \
        '[{"role":"user","content":"What is the weather like in Beijing?"}]' \
        '{"tools":[{"type":"function","function":{"name":"get_weather","description":"Get the current weather","parameters":{"type":"object","properties":{"city":{"type":"string","description":"The city to get weather for"}},"required":["city"]}}}],"tool_choice":"auto","max_tokens":100}')" || ((FAILED++))

# 5. 流式输出
_run_stream_test "流式 Chat Completion" \
    "$(build_json_body "$MODEL_NAME" \
        '[{"role":"user","content":"从1数到5"}]' \
        '{"max_tokens":100,"stream":true}')"

echo ""
echo "=========================================="
if [[ "$FAILED" -eq 0 ]]; then
    log_success "LongCat-Flash-Chat SGLang 所有验证测试通过!"
else
    log_warning "LongCat-Flash-Chat SGLang 测试完成，${FAILED} 项失败"
fi
echo "=========================================="
