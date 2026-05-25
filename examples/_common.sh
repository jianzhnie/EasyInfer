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

# wait_for_server and print_server_ready are now provided by common.sh
# print_claude_config is kept here as it is specific to the GLM examples
