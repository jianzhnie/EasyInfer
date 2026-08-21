#!/bin/bash
# =============================================================================
# qwen3-dense / qwen3-8b — API functional test wrapper
# =============================================================================
# Usage:
#   bash curl_test.sh
#   HOST=10.0.0.1 PORT=8021 bash curl_test.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_DIR

export PORT="${PORT:-8021}"
export MODEL_NAME="${MODEL_NAME:-qwen3}"
export SKIP_VISION="${SKIP_VISION:-1}"

exec bash "${SCRIPT_DIR}/../../curl_test.sh" "$@"
