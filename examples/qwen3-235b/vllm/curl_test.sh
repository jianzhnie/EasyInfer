#!/bin/bash
# =============================================================================
# qwen3-235b-a22b — API 功能测试（薄封装，复用 examples/curl_test.sh 通用测试库）
# =============================================================================
# 测试项: health / models / 中英文 chat / 代码生成 / 流式 / 工具调用 / Anthropic API
# 多模态模型可设 ENABLE_VISION=1 开启图片测试。
# Usage:
#   bash curl_test.sh
#   HOST=10.0.0.1 PORT=9000 bash curl_test.sh
#   SKIP_TOOLS=1 SKIP_CODE=1 bash curl_test.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_DIR

export PORT="${PORT:-8006}"
export MODEL_NAME="${MODEL_NAME:-qwen3-235b-a22b}"
# export ENABLE_VISION=1  # 多模态模型取消注释
exec bash "${SCRIPT_DIR}/../../curl_test.sh" "$@"
