#!/bin/bash
# GLM-5 W4A8 — API 功能测试脚本
# 默认使用 localhost:8001
#
# 用法:
#   ./curl_test.sh                               # 默认测试 localhost:8001
#   BASE_URL=http://10.0.0.1:9000 ./curl_test.sh  # 指定地址
#   MODEL_NAME=my-model ./curl_test.sh             # 指定模型名

# ================ 配置区域 ================
HOST="${HOST:-localhost}"
PORT="${PORT:-8001}"
MODEL_NAME="${MODEL_NAME:-glm-5}"
TIMEOUT=300
WAIT_INTERVAL=5

# ================ 颜色输出 ================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info()    { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error()   { echo -e "${RED}[ERROR]${NC} $1"; }

# ================ 工具函数 ================
wait_for_service() {
    log_info "等待服务启动: http://$HOST:$PORT ..."
    local start_time
    start_time=$(date +%s)

    while true; do
        if curl -s "http://$HOST:$PORT/health" > /dev/null 2>&1 || \
           curl -s "http://$HOST:$PORT/v1/models" > /dev/null 2>&1; then
            log_success "服务已就绪!"
            return 0
        fi

        local elapsed
        elapsed=$(( $(date +%s) - start_time ))
        if [[ $elapsed -ge $TIMEOUT ]]; then
            log_error "等待服务超时 ($TIMEOUT 秒)!"
            return 1
        fi

        log_info "服务未就绪，等待 ${WAIT_INTERVAL}s... (已等待 ${elapsed}s)"
        sleep "$WAIT_INTERVAL"
    done
}

run_test() {
    local test_name="$1"
    local curl_cmd="$2"
    local expected="${3:-}"

    echo ""
    log_info "测试: $test_name"
    log_info "执行命令: $curl_cmd"

    if [[ "$expected" == "quiet" ]]; then
        eval "$curl_cmd" > /dev/null 2>&1
    else
        eval "$curl_cmd" | python3 -m json.tool 2>/dev/null || eval "$curl_cmd"
    fi

    local exit_code=$?
    if [[ $exit_code -eq 0 ]]; then
        log_success "$test_name 完成 (退出码: $exit_code)"
    else
        log_warning "$test_name 可能存在问题 (退出码: $exit_code)"
    fi
    return $exit_code
}

# ================ 主程序 ================
echo "=========================================="
echo "  GLM-5 W4A8 API 功能测试"
echo "  目标地址: http://$HOST:$PORT"
echo "  模型名称: $MODEL_NAME"
echo "=========================================="

# 0. 等待服务就绪
wait_for_service || exit 1

# 1. 健康检查 / 模型列表
run_test "模型列表查询" \
    "curl -s http://$HOST:$PORT/v1/models"

# 2. 简单 Chat Completion (英文)
run_test "Chat Completion (英文)" \
    "curl -s http://$HOST:$PORT/v1/chat/completions \
      -H \"Content-Type: application/json\" \
      -d '{
        \"model\": \"$MODEL_NAME\",
        \"messages\": [
          {\"role\": \"user\", \"content\": \"Hello, who are you?\"}
        ],
        \"max_tokens\": 128,
        \"temperature\": 0.7
      }'"

# 3. 简单 Chat Completion (中文)
run_test "Chat Completion (中文)" \
    "curl -s http://$HOST:$PORT/v1/chat/completions \
      -H \"Content-Type: application/json\" \
      -d '{
        \"model\": \"$MODEL_NAME\",
        \"messages\": [
          {\"role\": \"system\", \"content\": \"You are a helpful assistant.\"},
          {\"role\": \"user\", \"content\": \"你好，请简单介绍一下你自己。\"}
        ],
        \"max_tokens\": 128,
        \"temperature\": 0.7
      }'"

# 4. Tool Calling 测试
run_test "Tool Calling" \
    "curl -s http://$HOST:$PORT/v1/chat/completions \
      -H \"Content-Type: application/json\" \
      -d '{
        \"model\": \"$MODEL_NAME\",
        \"messages\": [
          {\"role\": \"user\", \"content\": \"What is the weather like in Beijing?\"}
        ],
        \"tools\": [
          {
            \"type\": \"function\",
            \"function\": {
              \"name\": \"get_weather\",
              \"description\": \"Get the current weather\",
              \"parameters\": {
                \"type\": \"object\",
                \"properties\": {
                  \"city\": {
                    \"type\": \"string\",
                    \"description\": \"The city to get weather for\"
                  }
                },
                \"required\": [\"city\"]
              }
            }
          }
        ],
        \"tool_choice\": \"auto\",
        \"max_tokens\": 100
      }'"

# 5. Anthropic Messages API
run_test "Anthropic Messages API" \
    "curl -s http://$HOST:$PORT/v1/messages \
      -H \"Content-Type: application/json\" \
      -H \"x-api-key: dummy\" \
      -d '{
        \"model\": \"$MODEL_NAME\",
        \"max_tokens\": 100,
        \"messages\": [
          {\"role\": \"user\", \"content\": \"Hi there!\"}
        ]
      }'"

# 6. 流式 Chat Completion
run_test "流式 Chat Completion" \
    "curl -s http://$HOST:$PORT/v1/chat/completions \
      -H \"Content-Type: application/json\" \
      -d '{
        \"model\": \"$MODEL_NAME\",
        \"messages\": [
          {\"role\": \"user\", \"content\": \"从1数到5\"}
        ],
        \"max_tokens\": 100,
        \"stream\": true
      }' 2>&1 | head -10" "quiet"

echo ""
echo "=========================================="
log_success "GLM-5 W4A8 所有测试完成!"
echo "=========================================="
