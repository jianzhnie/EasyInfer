#!/bin/bash
#
# GLM 部署示例公共函数
# 被 glm5-1_quant_server.sh / glm5_full_server.sh source 使用
#
# 注意: 本文件被 source 而非直接执行，不设 set -euo pipefail。

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../scripts/common.sh
source "${SCRIPT_DIR}/../scripts/common.sh"

# ------------------------------------------------------------------------------
# 前置检查
# ------------------------------------------------------------------------------
check_prereqs() {
    local model_path="$1"

    if ! command -v vllm >/dev/null 2>&1; then
        log_err "vllm command not found. Are you inside the Docker container?"
        exit "$E_CMD_NOT_FOUND"
    fi

    if [[ ! -d "$model_path" ]]; then
        log_err "Model path not found: $model_path"
        exit "$E_NOT_FOUND"
    fi

    if [[ ! -f "$model_path/config.json" ]]; then
        log_err "config.json not found in: $model_path"
        exit "$E_NOT_FOUND"
    fi
}

# ------------------------------------------------------------------------------
# 等待 vLLM 服务就绪
# ------------------------------------------------------------------------------
wait_for_server() {
    local host="$1"
    local port="$2"
    local vllm_pid="$3"
    local max_wait="${4:-600}"
    local url="http://${host}:${port}/health"
    local elapsed=0
    local interval=5

    log_info "Waiting for server to become ready..."

    while (( elapsed < max_wait )); do
        if kill -0 "$vllm_pid" 2>/dev/null; then
            if curl -sf "$url" >/dev/null 2>&1; then
                log_info "================================================================================="
                log_info "  vLLM server is READY"
                log_info "================================================================================="
                log_info "  Health check:  http://${host}:${port}/health"
                log_info "  API endpoint:  http://${host}:${port}/v1"
                log_info "  Models list:   http://${host}:${port}/v1/models"
                log_info "================================================================================="
                return 0
            fi
        else
            log_err "vLLM process died unexpectedly (PID $vllm_pid)"
            return 1
        fi

        sleep "$interval"
        elapsed=$((elapsed + interval))
        printf "."
    done

    printf "\n"
    log_err "Server did not become ready within ${max_wait}s"
    return 1
}

# ------------------------------------------------------------------------------
# 服务就绪后的 Claude Code 配置输出
# ------------------------------------------------------------------------------
print_claude_config() {
    local host_ip="$1"
    local port="$2"
    local model_name="$3"

    log_info "================================================================================="
    log_info "  vLLM server is READY"
    log_info "================================================================================="
    log_info "  Health check:  http://${host_ip}:${port}/health"
    log_info "  API endpoint:  http://${host_ip}:${port}/v1"
    log_info "  Models list:   http://${host_ip}:${port}/v1/models"
    log_info ""
    log_info "  --- Claude Code 配置 ---"
    log_info ""
    log_info "  方式一: 写入 ~/.claude/settings.json"
    log_info "  {"
    log_info "    \"env\": {"
    log_info "      \"ANTHROPIC_BASE_URL\": \"http://${host_ip}:${port}/v1\","
    log_info "      \"ANTHROPIC_API_KEY\": \"dummy\","
    log_info "      \"ANTHROPIC_AUTH_TOKEN\": \"dummy\","
    log_info "      \"ANTHROPIC_DEFAULT_SONNET_MODEL\": \"${model_name}\","
    log_info "      \"ANTHROPIC_DEFAULT_HAIKU_MODEL\": \"${model_name}\","
    log_info "      \"ANTHROPIC_DEFAULT_OPUS_MODEL\": \"${model_name}\""
    log_info "    }"
    log_info "  }"
    log_info ""
    log_info "  方式二: 命令行直接使用"
    log_info "  ANTHROPIC_BASE_URL=http://${host_ip}:${port}/v1 \\"
    log_info "  ANTHROPIC_API_KEY=dummy \\"
    log_info "  ANTHROPIC_AUTH_TOKEN=dummy \\"
    log_info "  ANTHROPIC_DEFAULT_SONNET_MODEL=${model_name} \\"
    log_info "  ANTHROPIC_DEFAULT_HAIKU_MODEL=${model_name} \\"
    log_info "  ANTHROPIC_DEFAULT_OPUS_MODEL=${model_name} \\"
    log_info "  claude"
    log_info ""
    log_info "================================================================================="
}
