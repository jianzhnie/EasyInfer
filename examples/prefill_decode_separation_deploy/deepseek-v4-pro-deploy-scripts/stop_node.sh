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
# 最后清理僵尸进程。
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

cleanup_zombies() {
    local pattern="$1"
    local zombies
    zombies=$(ps aux | grep -E "$pattern" | awk '/<defunct>/{print $2}' || true)
    if [ -n "$zombies" ]; then
        echo "  清理僵尸进程 PID: $(echo "$zombies" | tr '\n' ' ')"
        # 收割僵尸——向其父进程发送 SIGCHLD 或直接 wait 不可行，
        # 最彻底的方式是 kill -SIGCHLD 通知父进程回收
        for zp in $zombies; do
            ppid=$(ps -o ppid= -p "$zp" 2>/dev/null | tr -d ' ')
            [ -n "$ppid" ] && kill -CHLD "$ppid" 2>/dev/null || true
        done
        sleep 1
        # 二次检查：还有残留僵尸则 kill 父进程
        remaining_zombies=$(ps aux | grep -E "$pattern" | awk '/<defunct>/{print $2}' || true)
        if [ -n "$remaining_zombies" ]; then
            for zp in $remaining_zombies; do
                ppid=$(ps -o ppid= -p "$zp" 2>/dev/null | tr -d ' ')
                if [ -n "$ppid" ] && [ "$ppid" != "0" ]; then
                    echo "  父进程 $ppid 未回收僵尸 $zp，终止父进程"
                    kill -TERM "$ppid" 2>/dev/null || true
                fi
            done
        fi
    fi
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
        echo ""
        echo "  检查僵尸进程..."
        cleanup_zombies "vllm"
        ;;
    dnode)
        echo "[DNode] 查找监听端口 ${D_VLLM_START_PORT}+ 的 vllm 进程..."
        mapfile -t pids < <(pgrep -f "vllm serve.*--port $D_VLLM_START_PORT" || true)
        for i in $(seq 1 $((D_DP_SIZE_LOCAL - 1))); do
            mapfile -t -O "${#pids[@]}" pids < <(pgrep -f "vllm serve.*--port $((D_VLLM_START_PORT + i))" || true)
        done
        stop_pids "${pids[@]}"
        echo ""
        echo "  检查僵尸进程..."
        cleanup_zombies "vllm"
        ;;
    all)
        echo "[All] 查找所有 vllm serve 进程..."
        mapfile -t pids < <(pgrep -f "vllm serve" || true)
        stop_pids "${pids[@]}"
        echo ""
        echo "  检查僵尸进程..."
        cleanup_zombies "vllm"
        ;;
    *)
        echo "错误: 参数必须是 pnode / dnode / all"
        echo "用法: $0 [pnode|dnode|all]"
        exit 1
        ;;
esac

echo ""
echo "  清理 vLLM 相关端口 (宿主机)..."
# vLLM 使用 --net=host，容器内外网络共享
# 清理已知的内部通信端口（ZMQ RPC 等）
for port in 12321; do
    if command -v fuser &>/dev/null; then
        fuser -k "${port}/tcp" 2>/dev/null && echo "  端口 ${port}: 已释放" || true
    elif command -v ss &>/dev/null; then
        pids=$(ss -tlnp "sport = :${port}" 2>/dev/null | grep -oP 'pid=\K\d+' || true)
        if [ -n "$pids" ]; then
            echo "  端口 ${port}: 被 PID $pids 占用，正在终止..."
            kill -9 $pids 2>/dev/null || true
        fi
    fi
done

echo ""
echo "  最终检查 vllm 进程:"
remaining=$(ps aux | grep -E '[v]llm' || true)
if [ -z "$remaining" ]; then
    echo "    (无)"
else
    ps -fp $(echo "$remaining" | awk '{print $2}' | tr '\n' ' ')
fi
