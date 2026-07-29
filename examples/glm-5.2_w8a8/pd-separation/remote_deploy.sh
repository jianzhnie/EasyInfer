#!/bin/bash
# ==============================================================================
# GLM-5.2 W8A8 PD 分离（1M 上下文）— 远程一键部署
# ==============================================================================
# 拓扑: 4×PNode + 4×DNode | P/D 均 DP=4 TP=8 PCP1 DCP=8 | MooncakeConnectorV1
#
# 文件分工:
#   deploy.conf         节点/拓扑/端口 —— 唯一需要按集群修改的文件
#   pnode.sh / dnode.sh P/D 节点 vllm 启动模板（由 launch_online_dp.py 调用）
#   launch_online_dp.py 每节点 DP 实例启动器（本脚本通过 SSH 远程触发）
#
# Usage:
#   bash remote_deploy.sh deploy       一键部署（P+D 并行启动 + 健康检查）
#   bash remote_deploy.sh status       节点与代理状态
#   bash remote_deploy.sh stop         停止所有节点 vLLM 进程
#   bash remote_deploy.sh restart      stop + deploy
#   bash remote_deploy.sh clean        彻底清理（重启容器，清除僵尸进程）
#   bash remote_deploy.sh proxy        在 P0 容器内启动请求转发代理 (:8123)
#   bash remote_deploy.sh proxy-stop   停止代理
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONF="${SCRIPT_DIR}/deploy.conf"
PROXY_SCRIPT="${SCRIPT_DIR}/../../prefill_decode_separation_deploy/load_balance_proxy_server_example.py"
PROXY_PORT="${PROXY_PORT:-8000}"
PROXY_LOG="${SCRIPT_DIR}/proxy.log"

GREEN='\033[0;32m'; RED='\033[0;31m'; YELLOW='\033[1;33m'; NC='\033[0m'

# ------------------------------------------------------------------------------
# 配置加载与校验（启动前 fail-fast，避免把错误拓扑推到 8 个节点）
# ------------------------------------------------------------------------------
load_conf() {
    [[ -f "$CONF" ]] || { echo -e "${RED}ERROR: $CONF 不存在${NC}"; exit 1; }
    # shellcheck source=deploy.conf
    source "$CONF"

    local rc=0
    local p_expect=$((P_DP_SIZE_LOCAL * ${#P_IPS[@]}))
    local d_expect=$((D_DP_SIZE_LOCAL * ${#D_IPS[@]}))
    [[ "$P_DP_SIZE" -eq "$p_expect" ]] \
        || { echo -e "${RED}CONF: P_DP_SIZE=$P_DP_SIZE != P_DP_SIZE_LOCAL($P_DP_SIZE_LOCAL) × P 节点数(${#P_IPS[@]})${NC}"; rc=1; }
    [[ "$D_DP_SIZE" -eq "$d_expect" ]] \
        || { echo -e "${RED}CONF: D_DP_SIZE=$D_DP_SIZE != D_DP_SIZE_LOCAL($D_DP_SIZE_LOCAL) × D 节点数(${#D_IPS[@]})${NC}"; rc=1; }
    [[ "${#D_RANK_STARTS[@]}" -eq "${#D_IPS[@]}" ]] \
        || { echo -e "${RED}CONF: D_RANK_STARTS 数量(${#D_RANK_STARTS[@]}) != D 节点数(${#D_IPS[@]})${NC}"; rc=1; }
    [[ ${#P_IPS[@]} -gt 0 && "${P_IPS[0]}" == "$P_DP_ADDRESS" ]] \
        || echo -e "${YELLOW}WARN: P_DP_ADDRESS 建议设为 P_IPS[0]（DP master）${NC}"
    [[ ${#D_IPS[@]} -gt 0 && "${D_IPS[0]}" == "$D_DP_ADDRESS" ]] \
        || echo -e "${YELLOW}WARN: D_DP_ADDRESS 建议设为 D_IPS[0]（DP master）${NC}"
    for f in pnode.sh dnode.sh launch_online_dp.py; do
        [[ -f "${SCRIPT_DIR}/$f" ]] || { echo -e "${RED}ERROR: ${SCRIPT_DIR}/$f 缺失${NC}"; rc=1; }
    done
    [[ $rc -eq 0 ]] || exit 1
}

# ------------------------------------------------------------------------------
# SSH / HTTP 工具
# ------------------------------------------------------------------------------
ssh_opts() {
    local opts="-o StrictHostKeyChecking=no -o ConnectTimeout=${SSH_CONNECT_TIMEOUT} -p ${SSH_PORT}"
    [[ -n "${SSH_KEY:-}" ]] && opts="$opts -i ${SSH_KEY}"
    echo "$opts"
}

ssh_run() {
    local ip="$1"; shift
    # shellcheck disable=SC2046
    ssh $(ssh_opts) "${SSH_USER}@${ip}" "$@"
}

# 查询某节点容器内 vllm API 状态码（000 = 不可达）
http_get() {
    local ip="$1" port="$2"
    ssh_run "$ip" \
        "docker exec ${CONTAINER_NAME} bash -c 'curl -s -o /dev/null -w \"%{http_code}\" --max-time 5 http://localhost:${port}/v1/models'" \
        2>/dev/null || echo "000"
}

# ------------------------------------------------------------------------------
# 节点启动
#   传给远程 heredoc 的位置参数映射:
#     $1 local_ip  $2 nic  $3 model_path  $4 log_dir  $5 script_dir
#     $6 node_script(pnode.sh|dnode.sh)
#     $7 dp_size  $8 tp_size  $9 dp_size_local  ${10} dp_rank_start
#     ${11} dp_address  ${12} dp_rpc_port  ${13} vllm_start_port
# ------------------------------------------------------------------------------
_launch_node() {
    local role="$1" index="$2" ip="$3"   # role: "p" | "d"

    # 按角色解析 conf 变量（P_DP_SIZE / D_DP_SIZE ...）
    local -n _dp_size="${role^^}_DP_SIZE"
    local -n _tp_size="${role^^}_TP_SIZE"
    local -n _dp_local="${role^^}_DP_SIZE_LOCAL"
    local -n _rpc_port="${role^^}_DP_RPC_PORT"
    local -n _vllm_port="${role^^}_VLLM_START_PORT"

    local dp_rank_start dp_address
    if [[ "$role" == "p" ]]; then
        dp_rank_start=$((index * _dp_local))          # local=1 时: 0,1,2,3
        dp_address="$P_DP_ADDRESS"
    else
        dp_rank_start="${D_RANK_STARTS[$index]}"
        dp_address="$D_DP_ADDRESS"
    fi

    echo -n "  ${role^^}Node [$index] $ip (dp_rank=$dp_rank_start) ... "
    ssh_run "$ip" bash -s -- \
        "$ip" "$NIC_NAME" "$MODEL_PATH" "$LOG_DIR" "$REMOTE_SCRIPT_DIR" "${role}node.sh" \
        "$_dp_size" "$_tp_size" "$_dp_local" "$dp_rank_start" "$dp_address" "$_rpc_port" "$_vllm_port" \
        << 'REMOTE_SCRIPT'
        local_ip="$1"; nic="$2"; model="$3"; logdir="$4"
        script_dir="$5"; node_script="$6"
        dps="$7"; tps="$8"; dpl="$9"; rank="${10}"; dpaddr="${11}"; rpc="${12}"; vport="${13}"
        docker exec -d vllm-ascend-env bash -l -c "
export LOCAL_IP='$local_ip' NIC_NAME='$nic' MODEL_PATH='$model' LOG_DIR='$logdir'
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

# ------------------------------------------------------------------------------
# 健康检查: 全部就绪返回 0, 超时返回 1
# ------------------------------------------------------------------------------
health_check() {
    local role="$1"
    local -n _ips="${role^^}_IPS"
    local -n _port="${role^^}_VLLM_START_PORT"
    local elapsed=0 ok=0

    echo "  等待 ${role^^}Node 就绪 (最长 ${HEALTH_CHECK_TIMEOUT}s)..."
    while [[ $elapsed -lt $HEALTH_CHECK_TIMEOUT ]]; do
        ok=0
        for ip in "${_ips[@]}"; do
            [[ $(http_get "$ip" "$_port") == "200" ]] && ((ok++))
        done
        if [[ $ok -eq ${#_ips[@]} ]]; then
            echo -e "  ${GREEN}${role^^}Node 全部就绪 ($ok/${#_ips[@]})${NC}"
            return 0
        fi
        sleep "$HEALTH_CHECK_INTERVAL"
        elapsed=$((elapsed + HEALTH_CHECK_INTERVAL))
        [[ $((elapsed % 120)) -eq 0 ]] && echo "  ${ok}/${#_ips[@]} ready, ${elapsed}s ..."
    done
    echo -e "  ${RED}${role^^}Node 就绪超时 ($ok/${#_ips[@]}), 日志: ${LOG_DIR}/glm5_${role}node_*.log${NC}"
    return 1
}

# ------------------------------------------------------------------------------
# 子命令
# ------------------------------------------------------------------------------
cmd_deploy() {
    echo "============================================"
    echo " GLM-5.2 W8A8 PD 部署 (1M, A2: 4P+4D)"
    echo " PNode: ${P_IPS[*]}"
    echo " DNode: ${D_IPS[*]}"
    echo " P: DP=$P_DP_SIZE TP=$P_TP_SIZE DCP=8 | D: DP=$D_DP_SIZE TP=$D_TP_SIZE DCP=8"
    echo "============================================"

    if [[ "${CLEAN_BEFORE_DEPLOY:-false}" == "true" ]]; then
        echo "[1/4] 清理已有进程..."
        cmd_stop
    fi

    echo "[2/4] 启动所有节点 (P+D 并行)..."
    local i
    for i in "${!P_IPS[@]}"; do _launch_node "p" "$i" "${P_IPS[$i]}"; done
    for i in "${!D_IPS[@]}"; do _launch_node "d" "$i" "${D_IPS[$i]}"; done

    local rc=0
    echo "[3/4] PNode 健康检查..."
    health_check "p" || rc=1
    echo "[4/4] DNode 健康检查..."
    health_check "d" || rc=1

    echo ""
    if [[ $rc -ne 0 ]]; then
        echo -e "${RED}部分节点未就绪, 请排查日志后重试${NC}"
        return 1
    fi
    echo "============================================"
    echo " 部署完成！bash remote_deploy.sh proxy 启动请求转发"
    echo "============================================"
}

cmd_stop() {
    echo "停止所有 vLLM 进程..."
    local ip
    for ip in "${P_IPS[@]}" "${D_IPS[@]}"; do
        echo -n "  $ip ... "
        ssh_run "$ip" "docker exec ${CONTAINER_NAME} bash -c 'pkill -f \"vllm serve\" || true'" 2>/dev/null \
            && echo -e "${GREEN}stopped${NC}" \
            || echo -e "${YELLOW}not running${NC}"
    done
    sleep 2
    echo "完成"
}

cmd_clean() {
    echo "彻底清理所有节点 (重启容器)..."
    local ip
    for ip in "${P_IPS[@]}" "${D_IPS[@]}"; do
        echo -n "  $ip ... "
        ssh_run "$ip" "docker restart ${CONTAINER_NAME}" 2>/dev/null \
            && echo -e "${GREEN}restarted${NC}" \
            || echo -e "${RED}failed${NC}"
    done
    sleep 15
    echo "完成"
}

status_role() {
    local role="$1" label="$2"
    local -n _ips="${role^^}_IPS"
    local -n _port="${role^^}_VLLM_START_PORT"
    local -n _local="${role^^}_DP_SIZE_LOCAL"
    local i ip port code
    echo "=== ${label} ==="
    for i in "${!_ips[@]}"; do
        ip="${_ips[$i]}"
        for port in $(seq "$_port" $((_port + _local - 1))); do
            code=$(http_get "$ip" "$port")
            [[ "$code" == "200" ]] \
                && echo -e "  [$i] $ip:$port ${GREEN}OK${NC}" \
                || echo -e "  [$i] $ip:$port ${RED}DOWN${NC}"
        done
    done
}

cmd_status() {
    status_role "p" "PNode (Prefill / kv_producer)"
    status_role "d" "DNode (Decode / kv_consumer)"

    echo "=== Proxy (${P_IPS[0]}:${PROXY_PORT}) ==="
    local code
    code=$(http_get "${P_IPS[0]}" "$PROXY_PORT")
    if [[ "$code" == "200" ]]; then
        echo -e "  ${GREEN}running${NC} (http=$code)"
    else
        echo -e "  ${YELLOW}not running${NC} (bash remote_deploy.sh proxy 启动)"
    fi
}

cmd_restart() {
    cmd_stop
    sleep 3
    cmd_deploy
}

# 在 P_IPS[0] 容器内启动请求转发代理（注册全部 P/D 端点，代理层负载均衡）。
# 注：不跑在本机——本机 python 缺 fastapi，且容器内可直接访问 P/D 节点。
cmd_proxy() {
    if ssh_run "${P_IPS[0]}" "docker exec ${CONTAINER_NAME} bash -c 'pgrep -f load_balance_[p]roxy_server_example >/dev/null'" 2>/dev/null; then
        echo -e "${YELLOW}代理已在运行 (${P_IPS[0]}:${PROXY_PORT})${NC}"
        return 0
    fi

    local p_ports=() d_ports=() _
    for _ in "${P_IPS[@]}"; do p_ports+=("$P_VLLM_START_PORT"); done
    for _ in "${D_IPS[@]}"; do d_ports+=("$D_VLLM_START_PORT"); done

    ssh_run "${P_IPS[0]}" bash -s -- \
        "$PROXY_PORT" "${P_VLLM_START_PORT}" "$D_VLLM_START_PORT" \
        "${P_IPS[*]}" "${D_IPS[*]}" "${p_ports[*]}" "${d_ports[*]}" "$LOG_DIR" << 'REMOTE_SCRIPT'
        port="$1"; pport="$2"; dport="$3"
        read -ra p_ips <<< "$4"; read -ra d_ips <<< "$5"
        read -ra p_ports <<< "$6"; read -ra d_ports <<< "$7"; logdir="$8"
        docker exec -d vllm-ascend-env bash -c "
unset http_proxy https_proxy
nohup python3 /home/jianzhnie/llmtuner/llm/EasyInfer/examples/prefill_decode_separation_deploy/load_balance_proxy_server_example.py \
    --host 0.0.0.0 --port $port \
    --prefiller-hosts ${p_ips[*]} --prefiller-ports ${p_ports[*]} \
    --decoder-hosts ${d_ips[*]} --decoder-ports ${d_ports[*]} \
    > '$logdir/pd_proxy.log' 2>&1 &
"
REMOTE_SCRIPT
    echo -e "${GREEN}代理已启动: ${P_IPS[0]}:${PROXY_PORT}${NC} (P×${#P_IPS[@]} :$P_VLLM_START_PORT, D×${#D_IPS[@]} :$D_VLLM_START_PORT)"
    echo "  日志: $LOG_DIR/pd_proxy.log"
}

cmd_proxy-stop() {
    ssh_run "${P_IPS[0]}" "docker exec ${CONTAINER_NAME} bash -c 'pkill -f load_balance_[p]roxy_server_example'" \
        && echo -e "${GREEN}代理已停止${NC}" \
        || echo -e "${YELLOW}代理未运行${NC}"
}

# ------------------------------------------------------------------------------
# 入口
# ------------------------------------------------------------------------------
load_conf
CMD="${1:-deploy}"
case "$CMD" in
    deploy|restart|stop|status|clean|proxy|proxy-stop) "cmd_${CMD}" ;;
    *) echo "Usage: $0 {deploy|stop|status|restart|clean|proxy|proxy-stop}"; exit 1 ;;
esac
