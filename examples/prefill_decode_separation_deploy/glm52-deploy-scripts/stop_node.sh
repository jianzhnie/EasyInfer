#!/bin/bash
# ==============================================================================
# stop_node.sh — 停止当前节点上的 vLLM 进程
# ==============================================================================
# 用法:
#   ./stop_node.sh              停止当前节点上所有 vllm 进程
#   ./stop_node.sh pnode        只停止 PNode 进程（监听 P_VLLM_START_PORT 起的端口）
#   ./stop_node.sh dnode        只停止 DNode 进程（监听 D_VLLM_START_PORT 起的端口）
#
# 停止顺序: 先发 SIGTERM（优雅退出），等待 5 秒，若仍在则 SIGKILL。
# ==============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/deploy.conf"

role="${1:-all}"

stop_pids() {
    local pids=("$@")
    if [ ${#pids[@]} -eq 0 ]; then
        echo "  没有找到匹配的进程"
        return
    fi
    echo "  发送 SIGTERM 到 PID: ${pids[*]}"
    kill -TERM "${pids[@]}" 2>/dev/null || true
    sleep 5
    local still_alive=()
    for pid in "${pids[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            still_alive+=("$pid")
        fi
    done
    if [ ${#still_alive[@]} -gt 0 ]; then
        echo "  仍在运行，发送 SIGKILL 到 PID: ${still_alive[*]}"
        kill -KILL "${still_alive[@]}" 2>/dev/null || true
    fi
    echo "  完成"
}

echo "============================================================"
echo "  停止 vLLM 进程 (role=${role})"
echo "============================================================"

case "$role" in
    pnode)
        echo "[PNode] 查找监听端口 ${P_VLLM_START_PORT}+ 的 vllm 进程..."
        mapfile -t pids < <(pgrep -f "vllm serve.*--port $P_VLLM_START_PORT" || true)
        # 也匹配 start_port+1 等递增端口
        for i in $(seq 1 $((P_DP_SIZE_LOCAL - 1))); do
            mapfile -t -O "${#pids[@]}" pids < <(pgrep -f "vllm serve.*--port $((P_VLLM_START_PORT + i))" || true)
        done
        stop_pids "${pids[@]}"
        ;;
    dnode)
        echo "[DNode] 查找监听端口 ${D_VLLM_START_PORT}+ 的 vllm 进程..."
        mapfile -t pids < <(pgrep -f "vllm serve.*--port $D_VLLM_START_PORT" || true)
        for i in $(seq 1 $((D_DP_SIZE_LOCAL - 1))); do
            mapfile -t -O "${#pids[@]}" pids < <(pgrep -f "vllm serve.*--port $((D_VLLM_START_PORT + i))" || true)
        done
        stop_pids "${pids[@]}"
        ;;
    all)
        echo "[All] 查找所有 vllm serve 进程..."
        mapfile -t pids < <(pgrep -f "vllm serve" || true)
        stop_pids "${pids[@]}"
        ;;
    *)
        echo "错误: 参数必须是 pnode / dnode / all"
        echo "用法: $0 [pnode|dnode|all]"
        exit 1
        ;;
esac

echo ""
echo "  剩余 vllm 进程:"
remaining=$(pgrep -f "vllm serve" || true)
if [ -z "$remaining" ]; then
    echo "    (无)"
else
    ps -fp $(echo "$remaining" | tr '\n' ' ')
fi
