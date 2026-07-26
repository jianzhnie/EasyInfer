#!/bin/bash
# =============================================================================
# qwen3.6-35b — API test (thin wrapper reusing examples/curl_test.sh)
# =============================================================================
# Usage:
#   bash curl_test.sh
#   HOST=10.0.0.1 PORT=9000 bash curl_test.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_DIR

export PORT="${PORT:-8020}"
export MODEL_NAME="${MODEL_NAME:-qwen3.6}"
exec bash "${SCRIPT_DIR}/../../curl_test.sh" "$@"
