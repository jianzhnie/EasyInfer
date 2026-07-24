#!/bin/bash
# =============================================================================
# kimi-k2.6 — API 功能测试（薄封装，复用 examples/curl_test.sh 通用测试库）
# =============================================================================
# 测试项: health / models / 中英文 chat / 代码生成 / 流式 / 工具调用 / Anthropic API
# 多模态 Vision 测试默认开启(ENABLE_VISION=1)。
# Usage:
#   bash curl_test.sh
#   HOST=10.0.0.1 PORT=9000 bash curl_test.sh
#   SKIP_TOOLS=1 SKIP_CODE=1 bash curl_test.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_DIR

export PORT="${PORT:-8003}"
export MODEL_NAME="${MODEL_NAME:-kimi-k2.6}"
export ENABLE_VISION="${ENABLE_VISION:-1}"
exec bash "${SCRIPT_DIR}/../../curl_test.sh" "$@"
