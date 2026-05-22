#!/bin/bash
#
# ray_node.sh — 启动 Ray Worker 节点
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/set_ray_env.sh"

HEAD_ADDR="${1:-127.0.0.1}"

echo "Starting Ray worker node, connecting to ${HEAD_ADDR}:${RAY_PORT}..."
ray start \
    --address="${HEAD_ADDR}:${RAY_PORT}" \
    --resources="{\"NPU\": ${NPUS_PER_NODE:-8}}"
