#!/bin/bash
# ==============================================================================
# check_status.sh — 检查所有 PNode / DNode 的健康状态
# ==============================================================================
# 用法:
#   ./check_status.sh            检查所有节点（PNode + DNode）
#   ./check_status.sh pnode      只检查 PNode
#   ./check_status.sh dnode      只检查 DNode
#
# 检查方式: 对每个节点发 GET /v1/models 请求，判断是否返回 200。
# 需要 curl 和网络连通。
# ==============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/deploy.conf"

role="${1:-all}"
timeout=10

check_node() {
    local name="$1"
    local ip="$2"
    local port="$3"
    local url="http://${ip}:${port}/v1/models"

    printf "  %-12s %-16s :%-6s  " "$name" "$ip" "$port"

    response=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout "$timeout" "$url" 2>/dev/null) || response="000"

    if [ "$response" = "200" ]; then
        echo "OK (200)"
    elif [ "$response" = "000" ]; then
        echo "UNREACHABLE (连接超时/拒绝)"
    else
        echo "FAIL (HTTP $response)"
    fi
}

echo "============================================================"
echo "  GLM-5.2 部署状态检查  $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
echo ""

case "$role" in
    pnode|all)
        echo "[PNode]  (期望: 4 个节点全部 OK)"
        echo "  -----------------------------------------------"
        for i in "${!PNODE_IPS[@]}"; do
            check_node "PNode$i" "${PNODE_IPS[$i]}" "$P_VLLM_START_PORT"
        done
        echo ""
        ;;
esac

case "$role" in
    dnode|all)
        echo "[DNode]  (期望: 4 个节点全部 OK)"
        echo "  -----------------------------------------------"
        for i in "${!DNODE_IPS[@]}"; do
            check_node "DNode$i" "${DNODE_IPS[$i]}" "$D_VLLM_START_PORT"
        done
        echo ""
        ;;
esac

echo "============================================================"
echo "  提示: 只有 rank-0 的 vLLM 实例在 vllm-start-port 上监听。"
echo "  DNode 的第二个实例在 vllm-start-port+1 上监听。"
echo "  如需检查所有实例，可手动 curl 各端口。"
echo "============================================================"
