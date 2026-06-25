#!/bin/bash
#
# _common.sh — GLM-5 系列示例脚本的共享辅助函数
# 由 glm5_full_server.sh / glm5-1_quant_server.sh 等 source 使用
#

_COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 载入仓库级公共库 (提供 log_info, log_err 等)
# shellcheck source=../../scripts/common.sh
source "${_COMMON_DIR}/../../scripts/common.sh"

# ------------------------------------------------------------------------------
# check_prereqs — 前置检查 (vllm 命令、模型路径)
# 用法: check_prereqs <model_path>
# ------------------------------------------------------------------------------
check_prereqs() {
    local model_path="${1:?用法: check_prereqs <model_path>}"

    if ! command -v vllm &>/dev/null; then
        log_err "vllm 命令未找到，请确认已安装 vLLM 并激活虚拟环境"
        exit "${E_CMD_NOT_FOUND:-127}"
    fi

    if [[ ! -d "$model_path" ]]; then
        log_err "模型路径不存在: $model_path"
        exit "${E_NOT_FOUND:-3}"
    fi

    if [[ ! -f "$model_path/config.json" ]]; then
        log_warn "模型目录缺少 config.json: $model_path (可能不完整)"
    fi

    log_info "前置检查通过: vllm=$(command -v vllm), model=$model_path"
}

# ------------------------------------------------------------------------------
# wait_for_server — 等待 vLLM HTTP 服务就绪 (带 PID 监控)
# 用法: wait_for_server <host> <port> <pid> [timeout_sec]
# ------------------------------------------------------------------------------
wait_for_server() {
    local host="${1:?用法: wait_for_server <host> <port> <pid> [timeout]}"
    local port="${2:?}"
    local pid="${3:?}"
    local max_wait="${4:-600}"
    local url="http://${host}:${port}/health"
    local elapsed=0 interval=5

    log_info "等待服务就绪 (PID=$pid, timeout=${max_wait}s)..."
    while (( elapsed < max_wait )); do
        if ! kill -0 "$pid" 2>/dev/null; then
            log_err "vLLM 进程 (PID=$pid) 已退出"
            return 1
        fi
        if curl -sf "$url" >/dev/null 2>&1; then
            log_info "服务就绪! http://${host}:${port} (${elapsed}s)"
            return 0
        fi
        sleep "$interval"
        elapsed=$((elapsed + interval))
    done
    log_err "服务未在 ${max_wait}s 内就绪"
    return 1
}

# ------------------------------------------------------------------------------
# print_claude_config — 输出 Claude Code 集成配置
# 用法: print_claude_config <host_ip> <port> <model_name>
# ------------------------------------------------------------------------------
print_claude_config() {
    local host_ip="${1:?用法: print_claude_config <host_ip> <port> <model_name>}"
    local port="${2:?}"
    local model_name="${3:?}"

    cat <<EOF

================================================================================
  Claude Code 集成配置
================================================================================
  在客户端设置以下环境变量:

  export ANTHROPIC_BASE_URL=http://${host_ip}:${port}/v1
  export ANTHROPIC_API_KEY=dummy
  export ANTHROPIC_AUTH_TOKEN=dummy
  export ANTHROPIC_DEFAULT_SONNET_MODEL=${model_name}
  export ANTHROPIC_DEFAULT_HAIKU_MODEL=${model_name}
  export ANTHROPIC_DEFAULT_OPUS_MODEL=${model_name}
================================================================================
EOF
}
