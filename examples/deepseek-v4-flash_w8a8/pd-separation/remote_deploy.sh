#!/bin/bash
# ==============================================================================
# DeepSeek-V4-Flash PD — 远程一键部署 (SSH 自动化)
# ==============================================================================
# 官方 §5.2 A2: 4×PNode + 4×DNode | DP=8/32 TP=1 | MooncakeHybridConnector
#
# Usage:
#   bash remote_deploy.sh deploy       一键全流程
#   bash remote_deploy.sh stop          停止所有节点
#   bash remote_deploy.sh status        检查所有节点状态
#   bash remote_deploy.sh restart       一键重启
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
    _ssh_cmd "$ip" \
        "docker exec ${CONTAINER_NAME} bash -c 'curl -s -o /dev/null -w \"%{http_code}\" --max-time 5 http://localhost:${port}/v1/models'" \
        2>/dev/null || echo "000"
}

# ==============================================================================
# _launch_node — SSH 到目标节点, 通过 docker exec -d 启动 launch_online_dp.py
# ==============================================================================
_launch_node() {
    local role="$1"  # "p" or "d"
    local index="$2"
    local ip="$3"

    # Resolve variable names dynamically: P_DP_SIZE, D_DP_SIZE etc.
    local -n dp_size="${role^^}_DP_SIZE"
    local -n tp_size="${role^^}_TP_SIZE"
    local -n dp_size_local="${role^^}_DP_SIZE_LOCAL"
    local -n dp_rpc_port="${role^^}_DP_RPC_PORT"
    local -n vllm_port="${role^^}_VLLM_START_PORT"

    local dp_rank_start engine_id dp_addr
    if [[ "$role" == "p" ]]; then
        dp_rank_start=$((index * dp_size_local))  # DP=8, 每节点2实例: 0,2,4,6
        engine_id="$index"
        dp_addr="$P_DP_ADDRESS"
    else
        local -n rank_starts="D_RANK_STARTS"
        dp_rank_start="${rank_starts[$index]}"
        engine_id=$((4 + index))
        dp_addr="$D_DP_ADDRESS"
    fi

    echo -n "  ${role^^}Node [$index] $ip (dp_rank=$dp_rank_start eng=$engine_id) ... "

    # 通过 SSH heredoc 在远程节点写脚本并执行, 避免多层引号转义
    _ssh_cmd "$ip" bash -s -- "$ip" "$NIC_NAME" "$MODEL_PATH" "$LOG_DIR" "$engine_id" \
        "$REMOTE_SCRIPT_DIR" "${role}node.sh" "$dp_size" "$tp_size" "$dp_size_local" \
        "$dp_rank_start" "$dp_addr" "$dp_rpc_port" "$vllm_port" << 'REMOTE_SCRIPT'
        local_ip="$1"; nic="$2"; model="$3"; logdir="$4"; engid="$5"
        script_dir="$6"; node_script="$7"; dps="$8"; tps="$9"; dpl="${10}"
        rank="${11}"; dpaddr="${12}"; rpc="${13}"; vport="${14}"
        docker exec -d vllm-ascend-env bash -l -c "
export LOCAL_IP='$local_ip' NIC_NAME='$nic' MODEL_PATH='$model' LOG_DIR='$logdir' ENGINE_ID='$engid'
cd '$script_dir'
nohup python3 launch_online_dp.py \
    --script '$node_script' \
    --dp-size '$dps' --tp-size '$tps' --dp-size-local '$dpl' --dp-rank-start '$rank' \
    --dp-address '$dpaddr' --dp-rpc-port '$rpc' --vllm-start-port '$vport' \
    < /dev/null > '$logdir/pd_launch_${local_ip}.log' 2>&1 &
"
REMOTE_SCRIPT

    echo -e "${GREEN}launched${NC}"
}

# ==============================================================================
# _health_check — 等待角色节点全部就绪
# ==============================================================================
_health_check() {
    local role="$1"  # "p" or "d"
    local -n ips="${role^^}_IPS"
    local -n port="${role^^}_VLLM_START_PORT"
    local timeout="${HEALTH_CHECK_TIMEOUT}"
    local interval="${HEALTH_CHECK_INTERVAL}"
    local elapsed=0

    echo "  等待 ${role^^}Node 就绪..."
    while [[ $elapsed -lt $timeout ]]; do
        local ok=0
        for ip in "${ips[@]}"; do
            [[ $(http_get "$ip" "$port") == "200" ]] && ((ok++))
        done
        if [[ $ok -eq ${#ips[@]} ]]; then
            echo -e "  ${GREEN}${role^^}Node 全部就绪 ($ok/${#ips[@]})${NC}"
            return 0
        fi
        sleep "$interval"
        elapsed=$((elapsed + interval))
        [[ $((elapsed % 60)) -eq 0 ]] && echo "  ${ok}/${#ips[@]} ready, ${elapsed}s ..."
    done
    echo -e "  ${RED}${role^^}Node 超时${NC}"
}

# ==============================================================================
# deploy
# ==============================================================================
cmd_deploy() {
    echo "============================================"
    echo " DeepSeek-V4-Flash PD 部署 (A2: 4P+4D)"
    echo " PNode: ${P_IPS[*]}"
    echo " DNode: ${D_IPS[*]}"
    echo " P: DP=$P_DP_SIZE TP=$P_TP_SIZE | D: DP=$D_DP_SIZE TP=$D_TP_SIZE"
    echo "============================================"

    if [[ "${CLEAN_BEFORE_DEPLOY:-false}" == "true" ]]; then
        echo "[1/4] 清理已有进程..."
        cmd_stop
    fi

    # 并行启动所有节点 (P + D 同时)
    echo "[2/4] 启动所有节点 (P+D 并行)..."
    for i in "${!P_IPS[@]}"; do
        _launch_node "p" "$i" "${P_IPS[$i]}"
    done
    for i in "${!D_IPS[@]}"; do
        _launch_node "d" "$i" "${D_IPS[$i]}"
    done

    # 健康检查 (非阻塞, 仅报告)
    echo "[3/4] PNode 健康检查..."
    _health_check "p"
    echo "[4/4] DNode 健康检查..."
    _health_check "d"

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
    for ip in "${P_IPS[@]}" "${D_IPS[@]}"; do
        echo -n "  $ip ... "
        _ssh_cmd "$ip" "docker exec ${CONTAINER_NAME} bash -c 'pkill -f \"vllm serve\" || true'" 2>/dev/null \
            && echo -e "${GREEN}stopped${NC}" \
            || echo -e "${YELLOW}not running${NC}"
    done
    sleep 2
    echo "完成"
}

# 彻底清理 (含容器重启, 清除僵尸进程)
cmd_clean() {
    echo "彻底清理所有节点 (重启容器)..."
    for ip in "${P_IPS[@]}" "${D_IPS[@]}"; do
        echo -n "  $ip ... "
        _ssh_cmd "$ip" "docker restart ${CONTAINER_NAME}" 2>/dev/null \
            && echo -e "${GREEN}restarted${NC}" \
            || echo -e "${RED}failed${NC}"
    done
    sleep 15
    echo "完成"
}

# ==============================================================================
# status
# ==============================================================================
cmd_status() {
    echo "=== PNode (Prefill / kv_producer) ==="
    for i in "${!P_IPS[@]}"; do
        local ip="${P_IPS[$i]}"
        for port in $(seq "$P_VLLM_START_PORT" $((P_VLLM_START_PORT + P_DP_SIZE_LOCAL - 1))); do
            local code; code=$(http_get "$ip" "$port")
            if [[ "$code" == "200" ]]; then
                echo -e "  [$i] $ip:$port ${GREEN}OK${NC}"
            else
                echo -e "  [$i] $ip:$port ${RED}DOWN${NC}"
            fi
        done
    done

    echo "=== DNode (Decode / kv_consumer) ==="
    for i in "${!D_IPS[@]}"; do
        local ip="${D_IPS[$i]}"
        for port in $(seq "$D_VLLM_START_PORT" $((D_VLLM_START_PORT + D_DP_SIZE_LOCAL - 1))); do
            local code; code=$(http_get "$ip" "$port")
            if [[ "$code" == "200" ]]; then
                echo -e "  [$i] $ip:$port ${GREEN}OK${NC}"
            else
                echo -e "  [$i] $ip:$port ${RED}DOWN${NC}"
            fi
        done
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
    deploy|restart|stop|status|clean) "cmd_${CMD}" ;;
    *) echo "Usage: $0 {deploy|stop|status|restart|clean}"; exit 1 ;;
esac
