#!/bin/bash
# =============================================================================
# longcat-flash — API 功能测试（薄封装，复用 examples/curl_test.sh 通用测试库）
# =============================================================================
# 测试项: health / models / 中英文 chat / 数学推理 / 代码生成 / 流式 / 工具调用 / Anthropic API
# 多模态模型可设 ENABLE_VISION=1 开启图片测试。
# 长上下文部署 (run_vllm_long-context.sh) 可设 ENABLE_LONG_CONTEXT=1 开启
# 大海捞针实测 (默认 ~130K tokens, 用 TARGET_TOKENS 调节)。
# Usage:
#   bash curl_test.sh
#   HOST=10.0.0.1 PORT=9000 bash curl_test.sh
#   SKIP_TOOLS=1 SKIP_CODE=1 bash curl_test.sh
#   ENABLE_LONG_CONTEXT=1 bash curl_test.sh
#   ENABLE_LONG_CONTEXT=1 TARGET_TOKENS=65536 bash curl_test.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_DIR

export PORT="${PORT:-8200}"
export MODEL_NAME="${MODEL_NAME:-longcat-flash}"
# export ENABLE_VISION=1  # 多模态模型取消注释
exec bash "${SCRIPT_DIR}/../../curl_test.sh" "$@"
