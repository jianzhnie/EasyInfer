#!/bin/bash
# =============================================================================
# LongCat-Flash-Chat — 一键启动脚本
# =============================================================================
# 直接在 docker 容器内启动 LongCat 服务，自动调用 run_vllm_long-context.sh。
#
# Usage:
#   # 容器内直接启动 (前台)
#   bash examples/longcat/longcat_server.sh
#
#   # 容器内后台启动 (nohup, 推荐)
#   nohup bash examples/longcat/longcat_server.sh > longcat_flash-chat4.log 2>&1 &
#
#   # 从远程节点启动 (SSH + docker exec)
#   NODE=10.16.201.229 bash examples/longcat/longcat_server.sh --remote
#
#   环境变量覆盖:
#   RAY_ADDRESS=<head-ip>:6379 PP=4 TP=32 MAX_MODEL_LEN=131072 \
#     bash examples/longcat/longcat_server.sh
#
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# 默认并行配置
export PP="${PP:-2}"
export TP="${TP:-32}"
export EP="${EP:-1}"

export MODEL_PATH="${MODEL_PATH:-/home/jianzhnie/llmtuner/hfhub/models/meituan-longcat/LongCat-Flash-Chat}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-longcat-flash}"
export PORT="${PORT:-8010}"
export RAY_ADDRESS="${RAY_ADDRESS:-}"

# ---------------------------------------------------------------------------
# --remote: 从远程节点 SSH + docker exec
# ---------------------------------------------------------------------------
if [[ "${1:-}" == "--remote" ]]; then
    NODE="${NODE:-10.16.201.229}"
    CONTAINER="${CONTAINER:-vllm-ascend-env}"

    echo "[INFO] 远程启动: ssh ${NODE} -> docker exec ${CONTAINER}"
    RAY_ADDRESS="${RAY_ADDRESS:-${NODE}:6379}"
    echo "[INFO] PP=$PP TP=$TP EP=$EP RAY_ADDRESS=$RAY_ADDRESS"
    echo ""

    ssh "$NODE" \
        "docker exec ${CONTAINER} bash -c 'cd ${REPO_ROOT} && RAY_ADDRESS=${RAY_ADDRESS} PP=${PP} TP=${TP} EP=${EP} bash examples/longcat/longcat_server.sh'"

    exit 0
fi

# ---------------------------------------------------------------------------
# 本地/容器内启动
# ---------------------------------------------------------------------------
if ! command -v vllm >/dev/null 2>&1; then
    echo "[ERROR] vllm 不在 PATH 中，请在 docker 容器内运行此脚本。" >&2
    echo "[INFO] 从远程节点运行: NODE=<ip> bash examples/longcat/longcat_server.sh --remote" >&2
    exit 1
fi

cd "$REPO_ROOT"

echo "[INFO] 容器内启动 LongCat-Flash-Chat"
echo "[INFO] PP=$PP TP=$TP EP=$EP RAY_ADDRESS=${RAY_ADDRESS:-<unset>}"
echo ""

exec bash examples/longcat/vllm/run_vllm_long-context.sh
