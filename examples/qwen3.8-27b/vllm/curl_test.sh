#!/bin/bash
# =============================================================================
# Qwen3.8-27B - OpenAI-compatible API functional test wrapper
# =============================================================================
# Usage:
#   bash curl_test.sh
#   ENABLE_VISION=1 bash curl_test.sh       # optional image request
#   HOST=10.0.0.1 PORT=8022 bash curl_test.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_DIR

export PORT="${PORT:-8022}"
export MODEL_NAME="${MODEL_NAME:-qwen3.8}"
export ENABLE_VISION="${ENABLE_VISION:-0}"

exec bash "${SCRIPT_DIR}/../../curl_test.sh" "$@"
