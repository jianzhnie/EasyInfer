#!/bin/bash
# ==============================================================================
# GLM-5.2 W8A8 PD 分离 (135K) — 远程部署管理
# ==============================================================================
# 策略: batch_remote_deploy_pd_seg/glm52-deploy-scripts（官方 GLM5.2 指南 4/5 章）
#   P: 10.42.11.202-205 :9081  TP=8 DP=4 (每节点 1 rank, enforce-eager, MTP 1)
#   D: 10.42.11.206-209 :9900/9901  TP=4 DP=8 (每节点 2 rank, enforce-eager, MTP 3)
#   KV: MooncakeConnector (kv_p2p, use_ascend_direct)  代理: 202:8123 (D 双端点注册)
# 2026-07-29 修正: D 侧 DYNAMIC_EPLB=0 (561000 崩溃根因) + HCCL 超时 600 +
#   enforce-eager (kv_consumer+FULL_DECODE_ONLY capture 触发 aclnnAscendQuantV3 161002)
#
# Usage:
#   bash remote_deploy.sh deploy    P 全部就绪后再起 D，最后健康检查
#   bash remote_deploy.sh status    P/D 各节点 /v1/models 状态
#   bash remote_deploy.sh stop      停止所有节点 vLLM 进程
#   bash remote_deploy.sh clean     彻底清理（重启容器，回收驱动级显存残留）
#   bash remote_deploy.sh proxy     启动 P/D 负载均衡代理 (202:8123)
#   bash remote_deploy.sh proxy-stop
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONF="${SCRIPT_DIR}/deploy.conf"
# shellcheck source=deploy.conf
source "$CONF"

CONTAINER_NAME=vllm-ascend-env
PROXY_PORT=8123
PROXY_SCRIPT=/home/jianzhnie/llmtuner/llm/EasyInfer/examples/prefill_decode_separation_deploy/load_balance_proxy_server_example.py
HEALTH_TIMEOUT=1800

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'

ssh_run() { ssh -o BatchMode=yes -o ConnectTimeout="${SSH_CONNECT_TIMEOUT:-10}" "$@"; }

wait_ready() {  # $1=ip $2=port $3=label
    local ip="$1" port="$2" label="$3" elapsed=0
    while true; do
        local code
        # 注意: 必须 || true —— set -e + pipefail 下 curl 000 (exit 7) 会让后台子 shell 直接退出
        code=$(ssh_run "$ip" "docker exec $CONTAINER_NAME bash -c 'curl -s -o /dev/null -w %{http_code} --max-time 5 http://localhost:$port/v1/models'" 2>/dev/null | tail -1) || true
        [[ "$code" == "200" ]] && { echo -e "  ${GREEN}$label ($ip:$port) 就绪 (${elapsed}s)${NC}"; return 0; }
        (( elapsed >= HEALTH_TIMEOUT )) && { echo -e "  ${RED}$label ($ip:$port) 等待超时${NC}"; return 1; }
        sleep 15; (( elapsed += 15 ))
    done
}

launch_role() {  # $1=start_script $2=node_index $3=ip
    ssh_run "$3" "docker exec -d $CONTAINER_NAME bash -c 'cd $SCRIPT_DIR && bash $1 $2 > /tmp/$(basename "$1" .sh)_$2.log 2>&1'"
    echo "  $3 ($1 $2) launched"
}

cmd_deploy() {
    echo "============================================"
    echo " GLM-5.2 W8A8 PD 分离 (135K, 4P+4D)"
    echo " P: ${PNODE_IPS[*]}  D: ${DNODE_IPS[*]}"
    echo "============================================"
    echo "[1/4] 启动 PNode (rank0 优先)..."
    launch_role start_pnode.sh 0 "${PNODE_IPS[0]}"
    for i in 1 2 3; do launch_role start_pnode.sh "$i" "${PNODE_IPS[$i]}"; done

    echo "[2/4] 等待 PNode 就绪..."
    for i in 0 1 2 3; do wait_ready "${PNODE_IPS[$i]}" "$P_VLLM_START_PORT" "PNode$i" & done
    wait

    echo "[3/4] 启动 DNode..."
    for i in 0 1 2 3; do launch_role start_dnode.sh "$i" "${DNODE_IPS[$i]}"; done

    echo "[4/4] 等待 DNode 就绪..."
    for i in 0 1 2 3; do
        for p in $(seq "$D_VLLM_START_PORT" $((D_VLLM_START_PORT + D_DP_SIZE_LOCAL - 1))); do
            wait_ready "${DNODE_IPS[$i]}" "$p" "DNode$i:$p" &
        done
    done
    wait

    echo -e "${GREEN}部署完成！${NC}下一步: bash remote_deploy.sh proxy"
}

cmd_status() {
    for i in 0 1 2 3; do
        printf "  PNode%s %-15s :%s " "$i" "${PNODE_IPS[$i]}" "$P_VLLM_START_PORT"
        ssh_run "${PNODE_IPS[$i]}" "docker exec $CONTAINER_NAME bash -c 'curl -s -o /dev/null -w %{http_code} --max-time 5 http://localhost:$P_VLLM_START_PORT/v1/models'" 2>/dev/null | tail -1 || true; echo
    done
    for i in 0 1 2 3; do
        for p in $(seq "$D_VLLM_START_PORT" $((D_VLLM_START_PORT + D_DP_SIZE_LOCAL - 1))); do
            printf "  DNode%s %-15s :%s " "$i" "${DNODE_IPS[$i]}" "$p"
            ssh_run "${DNODE_IPS[$i]}" "docker exec $CONTAINER_NAME bash -c 'curl -s -o /dev/null -w %{http_code} --max-time 5 http://localhost:$p/v1/models'" 2>/dev/null | tail -1 || true; echo
        done
    done
}

cmd_stop() {
    echo "停止所有节点 vLLM 进程..."
    for ip in "${PNODE_IPS[@]}" "${DNODE_IPS[@]}"; do
        printf "  %s ... " "$ip"
        # [v] 技巧避免 pkill 匹配到外层 bash -c 自身；EngineCore 进程名不含 "vllm serve"，需单独杀
        ssh_run "$ip" "docker exec $CONTAINER_NAME bash -c 'pkill -f \"[v]llm serve\"; sleep 2; pkill -9 -f \"[v]llm serve\"; pkill -9 -f \"VLLM::[E]ngineCore\"; true'" \
            && echo -e "${GREEN}stopped${NC}"
    done
    echo "完成"
}

cmd_clean() {
    echo "重启所有节点容器（彻底回收 NPU 显存）..."
    for ip in "${PNODE_IPS[@]}" "${DNODE_IPS[@]}"; do
        printf "  %s ... " "$ip"
        ssh_run "$ip" "docker restart $CONTAINER_NAME >/dev/null" && echo -e "${GREEN}restarted${NC}"
    done
    echo "完成"
}

cmd_proxy() {
    if ssh_run "${PNODE_IPS[0]}" "docker exec $CONTAINER_NAME bash -c 'pgrep -f load_balance_[p]roxy_server_example >/dev/null'" 2>/dev/null; then
        echo -e "${YELLOW}代理已在运行 (${PNODE_IPS[0]}:${PROXY_PORT})${NC}"; return 0
    fi
    local p_ports=("$P_VLLM_START_PORT" "$P_VLLM_START_PORT" "$P_VLLM_START_PORT" "$P_VLLM_START_PORT")
    # 每个 DNode 有 D_DP_SIZE_LOCAL 个实例（9900/9901），全部注册到代理（官方 8×A2 布局）
    local d_hosts=() d_ports=()
    for ip in "${DNODE_IPS[@]}"; do
        for p in $(seq "$D_VLLM_START_PORT" $((D_VLLM_START_PORT + D_DP_SIZE_LOCAL - 1))); do
            d_hosts+=("$ip"); d_ports+=("$p")
        done
    done
    ssh_run "${PNODE_IPS[0]}" "docker exec -d $CONTAINER_NAME bash -c '
        unset http_proxy https_proxy
        nohup python3 $PROXY_SCRIPT \
            --host 0.0.0.0 --port $PROXY_PORT \
            --prefiller-hosts ${PNODE_IPS[*]} --prefiller-ports ${p_ports[*]} \
            --decoder-hosts ${d_hosts[*]} --decoder-ports ${d_ports[*]} \
            > $LOG_DIR/pd135k_proxy.log 2>&1 &'"
    echo -e "${GREEN}代理已启动: ${PNODE_IPS[0]}:${PROXY_PORT}${NC}  日志: $LOG_DIR/pd135k_proxy.log"
}

cmd_proxy-stop() {
    ssh_run "${PNODE_IPS[0]}" "docker exec $CONTAINER_NAME bash -c 'pkill -f load_balance_[p]roxy_server_example'" \
        && echo -e "${GREEN}代理已停止${NC}" || echo -e "${YELLOW}代理未运行${NC}"
}

CMD="${1:-deploy}"
case "$CMD" in
    deploy|stop|status|clean|proxy|proxy-stop) "cmd_${CMD}" ;;
    *) echo "Usage: $0 {deploy|stop|status|clean|proxy|proxy-stop}"; exit 1 ;;
esac
