#!/bin/bash
# ==============================================================================
# GLM-5.2 W8A8 共部署（1M 上下文，8/16 节点通用）— 远程一键部署
# ==============================================================================
# 拓扑: N 节点 × 8 卡 | DP=N TP=8 PCP1 DCP=8 EP=N×8 | rank0 API, 其余 headless
# 节点与端口配置见 deploy.conf（当前: nodes/node_list0.txt, 10.42.11.194-209）
#
# Usage:
#   bash remote_deploy.sh deploy    一键部署（全部节点启动 + master 健康检查）
#   bash remote_deploy.sh status    master API + 各节点进程状态
#   bash remote_deploy.sh stop      停止所有节点 vLLM 进程
#   bash remote_deploy.sh restart   stop + deploy
#   bash remote_deploy.sh clean     彻底清理（重启容器）
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONF="${SCRIPT_DIR}/deploy.conf"

GREEN='\033[0;32m'; RED='\033[0;31m'; YELLOW='\033[1;33m'; NC='\033[0m'

# ------------------------------------------------------------------------------
# 配置加载与校验
# ------------------------------------------------------------------------------
load_conf() {
    [[ -f "$CONF" ]] || { echo -e "${RED}ERROR: $CONF 不存在${NC}"; exit 1; }
    # shellcheck source=deploy.conf
    source "$CONF"

    local rc=0
    local expect=$((DP_SIZE_LOCAL * ${#NODES[@]}))
    [[ "$DP_SIZE" -eq "$expect" ]] \
        || { echo -e "${RED}CONF: DP_SIZE=$DP_SIZE != DP_SIZE_LOCAL($DP_SIZE_LOCAL) × 节点数(${#NODES[@]})${NC}"; rc=1; }
    [[ "${NODES[0]}" == "$DP_ADDRESS" ]] \
        || echo -e "${YELLOW}WARN: DP_ADDRESS 建议设为 NODES[0]（DP master）${NC}"
    for f in conode.sh launch_online_dp.py; do
        [[ -f "${SCRIPT_DIR}/$f" ]] || { echo -e "${RED}ERROR: ${SCRIPT_DIR}/$f 缺失${NC}"; rc=1; }
    done
    [[ $rc -eq 0 ]] || exit 1
}

# ------------------------------------------------------------------------------
# SSH / HTTP 工具
# ------------------------------------------------------------------------------
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
    local ip="$1" port="$2"
    _ssh_cmd "$ip" \
        "docker exec ${CONTAINER_NAME} bash -c 'curl -s -o /dev/null -w \"%{http_code}\" --max-time 5 http://localhost:${port}/v1/models'" \
        2>/dev/null || echo "000"
}

# ------------------------------------------------------------------------------
# 节点启动
#   传给远程 heredoc 的位置参数映射:
#     $1 local_ip  $2 nic  $3 model_path  $4 log_dir  $5 script_dir
#     $6 dp_size  $7 tp_size  $8 dp_rank  $9 dp_address  ${10} dp_rpc_port  ${11} vllm_port
# ------------------------------------------------------------------------------
_launch_node() {
    local index="$1" ip="$2"
    echo -n "  Node [$index] $ip (dp_rank=$index) ... "
    _ssh_cmd "$ip" bash -s -- \
        "$ip" "$NIC_NAME" "$MODEL_PATH" "$LOG_DIR" "$REMOTE_SCRIPT_DIR" \
        "$DP_SIZE" "$TP_SIZE" "$index" "$DP_ADDRESS" "$DP_RPC_PORT" "$VLLM_PORT" \
        << 'REMOTE_SCRIPT'
        local_ip="$1"; nic="$2"; model="$3"; logdir="$4"; script_dir="$5"
        dps="$6"; tps="$7"; rank="$8"; dpaddr="$9"; rpc="${10}"; vport="${11}"
        docker exec -d vllm-ascend-env bash -l -c "
export LOCAL_IP='$local_ip' NIC_NAME='$nic' MODEL_PATH='$model' LOG_DIR='$logdir'
cd '$script_dir'
nohup python3 launch_online_dp.py \
    --script ./conode.sh \
    --dp-size '$dps' --tp-size '$tps' --dp-size-local 1 --dp-rank-start '$rank' \
    --dp-address '$dpaddr' --dp-rpc-port '$rpc' --vllm-start-port '$vport' \
    < /dev/null > '$logdir/dp1m_launch_${local_ip}.log' 2>&1 &
"
REMOTE_SCRIPT
    echo -e "${GREEN}launched${NC}"
}

# ------------------------------------------------------------------------------
# 健康检查: 只有 DP master (rank0) 提供 API
# ------------------------------------------------------------------------------
_health_check() {
    local elapsed=0
    echo "  等待 master ($DP_ADDRESS:$VLLM_PORT) 就绪 (最长 ${HEALTH_CHECK_TIMEOUT}s)..."
    while [[ $elapsed -lt $HEALTH_CHECK_TIMEOUT ]]; do
        if [[ $(http_get "$DP_ADDRESS" "$VLLM_PORT") == "200" ]]; then
            echo -e "  ${GREEN}master API 就绪${NC}"
            return 0
        fi
        sleep "$HEALTH_CHECK_INTERVAL"
        elapsed=$((elapsed + HEALTH_CHECK_INTERVAL))
        [[ $((elapsed % 120)) -eq 0 ]] && echo "  not ready, ${elapsed}s ..."
    done
    echo -e "  ${RED}master 就绪超时, 日志: ${LOG_DIR}/glm5_conode16_*.log${NC}"
    return 1
}

# ------------------------------------------------------------------------------
# 子命令
# ------------------------------------------------------------------------------
cmd_deploy() {
    echo "============================================"
    echo " GLM-5.2 W8A8 共部署 (1M, A2: ${#NODES[@]}节点 DP=$DP_SIZE)"
    echo " Nodes: ${NODES[*]}"
    echo " DP=$DP_SIZE TP=$TP_SIZE DCP=8 EP=$((DP_SIZE * TP_SIZE)) | API: $DP_ADDRESS:$VLLM_PORT"
    echo "============================================"

    if [[ "${CLEAN_BEFORE_DEPLOY:-false}" == "true" ]]; then
        echo "[1/3] 清理已有进程..."
        cmd_stop
    fi

    echo "[2/3] 启动所有节点..."
    local i
    for i in "${!NODES[@]}"; do _launch_node "$i" "${NODES[$i]}"; done

    echo "[3/3] 健康检查..."
    if _health_check; then
        echo "============================================"
        echo " 部署完成！入口: http://$DP_ADDRESS:$VLLM_PORT"
        echo "============================================"
    else
        echo -e "${RED}master 未就绪, 请排查日志后重试${NC}"
        return 1
    fi
}

cmd_stop() {
    echo "停止所有 vLLM 进程..."
    local ip
    for ip in "${NODES[@]}"; do
        echo -n "  $ip ... "
        # [v] 技巧避免 pkill 匹配到外层 bash -c 自身；EngineCore 进程名不含 "vllm serve"，需单独杀
        _ssh_cmd "$ip" "docker exec ${CONTAINER_NAME} bash -c 'pkill -f \"[v]llm serve\"; sleep 2; pkill -9 -f \"[v]llm serve\"; pkill -9 -f \"VLLM::[E]ngineCore\"; true'" 2>/dev/null \
            && echo -e "${GREEN}stopped${NC}" \
            || echo -e "${YELLOW}not running${NC}"
    done
    sleep 2
    echo "完成"
}

cmd_clean() {
    echo "彻底清理所有节点 (重启容器)..."
    local ip
    for ip in "${NODES[@]}"; do
        echo -n "  $ip ... "
        _ssh_cmd "$ip" "docker restart ${CONTAINER_NAME}" 2>/dev/null \
            && echo -e "${GREEN}restarted${NC}" \
            || echo -e "${RED}failed${NC}"
    done
    sleep 15
    echo "完成"
}

cmd_status() {
    local code
    code=$(http_get "$DP_ADDRESS" "$VLLM_PORT")
    echo "=== master API ($DP_ADDRESS:$VLLM_PORT) ==="
    [[ "$code" == "200" ]] && echo -e "  ${GREEN}OK${NC}" || echo -e "  ${RED}DOWN${NC}"

    echo "=== 各节点 vllm 进程 ==="
    local i ip cnt
    for i in "${!NODES[@]}"; do
        ip="${NODES[$i]}"
        cnt=$(_ssh_cmd "$ip" "docker exec ${CONTAINER_NAME} bash -c 'ps aux | grep \"[v]llm serve\" | wc -l'" 2>/dev/null || echo 0)
        [[ "$cnt" -ge 1 ]] \
            && echo -e "  [$i] $ip ${GREEN}running ($cnt)${NC}" \
            || echo -e "  [$i] $ip ${RED}no process${NC}"
    done
}

cmd_restart() {
    cmd_stop
    sleep 3
    cmd_deploy
}

# ------------------------------------------------------------------------------
# 入口
# ------------------------------------------------------------------------------
load_conf
CMD="${1:-deploy}"
case "$CMD" in
    deploy|restart|stop|status|clean) "cmd_${CMD}" ;;
    *) echo "Usage: $0 {deploy|stop|status|restart|clean}"; exit 1 ;;
esac
