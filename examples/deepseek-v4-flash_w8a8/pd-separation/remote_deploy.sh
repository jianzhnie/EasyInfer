#!/bin/bash
# ==============================================================================
# DeepSeek-V4-Flash PD — 远程一键部署 (SSH 自动化)
# ==============================================================================
# 在控制节点执行，自动 SSH 到所有 PNode/DNode 完成部署。
#
# Usage:
#   bash remote_deploy.sh deploy      一键全流程部署
#   bash remote_deploy.sh stop         停止所有节点
#   bash remote_deploy.sh status       检查所有节点状态
#   bash remote_deploy.sh restart      一键重启
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONF="${SCRIPT_DIR}/deploy.conf"

if [[ ! -f "$CONF" ]]; then
    echo "ERROR: $CONF not found"
    exit 1
fi
source "$CONF"

# ---- Helpers ----
GREEN='\033[0;32m'; RED='\033[0;31m'; YELLOW='\033[1;33m'; NC='\033[0m'

_ssh_opts() {
    local opts="-o StrictHostKeyChecking=no -o ConnectTimeout=${SSH_CONNECT_TIMEOUT} -p ${SSH_PORT}"
    [[ -n "${SSH_KEY:-}" ]] && opts="$opts -i ${SSH_KEY}"
    echo "$opts"
}

_ssh_cmd() {
    local ip="$1"; shift
    # shellcheck disable=SC2046
    ssh $(_ssh_opts) "${SSH_USER}@${ip}" "$@"
}

http_get() {
    local ip="$1"; local port="$2"
    curl -s -o /dev/null -w "%{http_code}" --max-time 5 "http://${ip}:${port}/health" 2>/dev/null || echo "000"
}

# ==============================================================================
# _launch — SSH 到目标节点启动 launch_online_dp.py
# ==============================================================================
_launch_node() {
    local role="$1"  # pnode or dnode
    local index="$2"
    local ip="$3"
    local dp_size dp_size_local tp_size dp_address dp_rpc_port vllm_port dp_rank_start

    if [[ "$role" == "pnode" ]]; then
        dp_size="$P_DP_SIZE"
        tp_size="$P_TP_SIZE"
        dp_size_local="$P_DP_SIZE_LOCAL"
        dp_address="$P_DP_ADDRESS"
        dp_rpc_port="$P_DP_RPC_PORT"
        vllm_port="$P_VLLM_START_PORT"
        dp_rank_start="$index"
    else
        dp_size="$D_DP_SIZE"
        tp_size="$D_TP_SIZE"
        dp_size_local="$D_DP_SIZE_LOCAL"
        dp_address="$D_DP_ADDRESS"
        dp_rpc_port="$D_DP_RPC_PORT"
        vllm_port="$D_VLLM_START_PORT"
        dp_rank_start="${DNODE_RANK_STARTS[$index]}"
    fi

    echo -n "  ${role^^} [$index] $ip (dp_rank=$dp_rank_start) ... "

    _ssh_cmd "$ip" "
        export LOCAL_IP='$ip'
        export NIC_NAME='$NIC_NAME'
        export MODEL_PATH='$MODEL_PATH'
        export LOG_DIR='$LOG_DIR'
        cd '$REMOTE_SCRIPT_DIR'
        python3 launch_online_dp.py \
            --script "${role}.sh" \
            --dp-size '$dp_size' \
            --tp-size '$tp_size' \
            --dp-size-local '$dp_size_local' \
            --dp-rank-start '$dp_rank_start' \
            --dp-address '$dp_address' \
            --dp-rpc-port '$dp_rpc_port' \
            --vllm-start-port '$vllm_port'
    " > /dev/null 2>&1 &

    echo -e "${GREEN}launched${NC}"
}

# ==============================================================================
# _health_check — 等待角色节点全部就绪
# ==============================================================================
_health_check() {
    local role="$1"
    local -n ips="${role^^}_IPS"
    local var_name="${role^^}_VLLM_START_PORT"
    local port="${!var_name}"
    local timeout="${HEALTH_CHECK_TIMEOUT}"
    local interval="${HEALTH_CHECK_INTERVAL}"
    local elapsed=0

    echo "  等待 ${role^^} 就绪..."
    while [[ $elapsed -lt $timeout ]]; do
        local ok=0
        for ip in "${ips[@]}"; do
            [[ $(http_get "$ip" "$port") == "200" ]] && ((ok++))
        done
        if [[ $ok -eq ${#ips[@]} ]]; then
            echo -e "  ${GREEN}${role^^} 全部就绪 ($ok/$ok)${NC}"
            return 0
        fi
        sleep "$interval"
        elapsed=$((elapsed + interval))
        echo "  ${ok}/${#ips[@]} ready, elapsed=${elapsed}s ..."
    done
    echo -e "  ${RED}${role^^} 超时 ($ok/${#ips[@]})${NC}"
}

# ==============================================================================
# deploy
# ==============================================================================
cmd_deploy() {
    echo "============================================"
    echo " DeepSeek-V4-Flash PD 远程部署"
    echo " PNodes: ${#PNODE_IPS[@]} | DNodes: ${#DNODE_IPS[@]}"
    echo "============================================"

    if [[ "${CLEAN_BEFORE_DEPLOY:-false}" == "true" ]]; then
        echo "[1/3] 清理已有进程..."
        cmd_stop
    fi

    echo "[2/3] 启动 Prefill 节点..."
    for i in "${!PNODE_IPS[@]}"; do
        _launch_node "pnode" "$i" "${PNODE_IPS[$i]}"
    done
    wait
    _health_check "pnode"

    echo "[3/3] 启动 Decode 节点..."
    for i in "${!DNODE_IPS[@]}"; do
        _launch_node "dnode" "$i" "${DNODE_IPS[$i]}"
    done
    wait
    _health_check "dnode"

    echo ""
    echo "============================================"
    echo " 部署完成！bash remote_deploy.sh status"
    echo "============================================"
}

# ==============================================================================
# stop
# ==============================================================================
cmd_stop() {
    echo "停止所有 vLLM 进程..."
    for ip in "${PNODE_IPS[@]}" "${DNODE_IPS[@]}"; do
        echo -n "  $ip ... "
        _ssh_cmd "$ip" "pkill -f 'vllm serve' || true" 2>/dev/null \
            && echo -e "${GREEN}stopped${NC}" \
            || echo -e "${YELLOW}not running${NC}"
    done
    sleep 2
    echo "完成"
}

# ==============================================================================
# status
# ==============================================================================
cmd_status() {
    local port="$P_VLLM_START_PORT"

    echo "=== PNode ==="
    for i in "${!PNODE_IPS[@]}"; do
        local ip="${PNODE_IPS[$i]}"; local code; code=$(http_get "$ip" "$port")
        if [[ "$code" == "200" ]]; then
            echo -e "  [$i] $ip:$port ${GREEN}OK${NC}"
        else
            echo -e "  [$i] $ip:$port ${RED}DOWN ($code)${NC}"
        fi
    done

    echo "=== DNode ==="
    for i in "${!DNODE_IPS[@]}"; do
        local ip="${DNODE_IPS[$i]}"; local code; code=$(http_get "$ip" "$port")
        if [[ "$code" == "200" ]]; then
            echo -e "  [$i] $ip:$port ${GREEN}OK${NC}"
        else
            echo -e "  [$i] $ip:$port ${RED}DOWN ($code)${NC}"
        fi
    done
}

# ==============================================================================
# restart
# ==============================================================================
cmd_restart() {
    cmd_stop
    sleep 3
    cmd_deploy
}

# ==============================================================================
# Dispatch
# ==============================================================================
CMD="${1:-deploy}"
case "$CMD" in
    deploy|restart|stop|status) "cmd_${CMD}" ;;
    *) echo "Usage: $0 {deploy|stop|status|restart}"; exit 1 ;;
esac
