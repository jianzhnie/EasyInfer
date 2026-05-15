#!/bin/bash
# ray_head.sh - 启动 Ray Head 节点

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/set_ray_env.sh"

echo "Starting Ray head node..."
ray start --head \
    --port="${RAY_PORT:-6379}" \
    --resources="{\"NPU\": ${NPUS_PER_NODE:-8}}"
