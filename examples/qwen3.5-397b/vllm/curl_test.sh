#!/bin/bash
# =============================================================================
# qwen3.5-397b — API test (thin wrapper reusing examples/curl_test.sh)
# =============================================================================
# Tests: health / models / chat / code / streaming / tool calling / Anthropic API
# Usage:
#   bash curl_test.sh
#   HOST=10.0.0.1 PORT=9000 bash curl_test.sh
#   SKIP_TOOLS=1 SKIP_CODE=1 bash curl_test.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_DIR

export PORT="${PORT:-8019}"
export MODEL_NAME="${MODEL_NAME:-qwen3.5}"
exec bash "${SCRIPT_DIR}/../../curl_test.sh" "$@"
