#!/bin/bash
# ==============================================================================
# manage_nodes.sh — PD 分离节点生命周期管理
# ==============================================================================
# 在控制节点运行，通过 SSH → docker exec 管理 NPU 节点上的 Prefill/Decode 进程。
#
# 依赖:
#   - 所有节点配置无密码 SSH
#   - Docker 容器已运行（通过 manage_docker_containers.sh 管理）
#   - deploy.conf 在共享存储上可见
#
# 用法:
#   bash manage_nodes.sh start [-r pnode|dnode] [-f node_list.txt]
#   bash manage_nodes.sh stop  [-r pnode|dnode] [-f node_list.txt]
#   bash manage_nodes.sh clean [-f node_list.txt]
#   bash manage_nodes.sh status [-f node_list.txt]
#
# 环境变量:
#   DEPLOY_DIR        deploy-scripts 目录（必需）
#   CONTAINER_NAME    Docker 容器名（默认 vllm-ascend-env）
#   PARALLELISM       并行度（默认 4）
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common.sh"

# ---- 默认值 ------------------------------------------------------------------
DEPLOY_DIR="${DEPLOY_DIR:-}"
CONTAINER_NAME="${CONTAINER_NAME:-vllm-ascend-env}"
PARALLELISM="${PARALLELISM:-4}"
SSH_TIMEOUT="${SSH_TIMEOUT:-600}"
ROLE="${ROLE:-all}"   # pnode, dnode, all
ACTION=""
NODES_FILE="${NODES_FILE:-}"

usage() {
    cat <<'EOF'
Usage:
  bash manage_nodes.sh start  [-r pnode|dnode|all] [-f nodes.txt] [-d deploy_dir]
  bash manage_nodes.sh stop   [-r pnode|dnode|all] [-f nodes.txt] [-d deploy_dir]
  bash manage_nodes.sh clean  [-f nodes.txt] [-d deploy_dir]
  bash manage_nodes.sh status [-f nodes.txt] [-d deploy_dir]

Options:
  -r, --role   ROLE    节点角色: pnode, dnode, all（默认 all）
  -f, --file   FILE    节点列表文件（可选，默认读 deploy.conf）
  -d, --deploy DIR     deploy-scripts 目录路径
  -h, --help           显示帮助

Environment:
  DEPLOY_DIR, CONTAINER_NAME, PARALLELISM, SSH_TIMEOUT
EOF
    exit 1
}

# ---- 参数解析 ----------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        start|stop|clean|status) ACTION="$1"; shift ;;
        -r|--role) ROLE="${2:?}"; shift 2 ;;
        -f|--file) NODES_FILE="${2:?}"; shift 2 ;;
        -d|--deploy) DEPLOY_DIR="${2:?}"; shift 2 ;;
        -h|--help) usage ;;
        *) echo "未知参数: $1"; usage ;;
    esac
done

[[ -n "$ACTION" ]] || { echo "错误: 需要指定动作 (start|stop|clean|status)"; usage; }
[[ -n "$DEPLOY_DIR" ]] || { echo "错误: DEPLOY_DIR 未设置，使用 -d 指定"; exit 1; }
[[ -f "$DEPLOY_DIR/deploy.conf" ]] || { echo "错误: deploy.conf 未找到: $DEPLOY_DIR/deploy.conf"; exit 1; }

source "$DEPLOY_DIR/deploy.conf"

# ---- 解析节点列表 ------------------------------------------------------------
_get_ips() {
    case "${ROLE:-all}" in
        pnode) echo "${PNODE_IPS[@]:-}" ;;
        dnode) echo "${DNODE_IPS[@]:-}" ;;
        all)   echo "${PNODE_IPS[@]:-} ${DNODE_IPS[@]:-}" ;;
    esac
}

_docker_exec() {
    printf "docker exec %s bash -c '%s'" "$CONTAINER_NAME" "$1"
}

# ---- 远程函数（序列化到节点，在宿主机执行 docker exec）------------------------
_remote_start_node() {
    local role="$1" index="$2" deploy_dir="$3" container="$4"
    local cmd
    cmd="cd $deploy_dir && nohup bash start_${role}.sh $index > /tmp/${role}_${index}.log 2>&1 &"
    docker exec "$container" bash -c "$cmd" 2>/dev/null && echo "STARTED" || echo "FAILED"
}

_remote_stop_node() {
    local role="${1:-all}" deploy_dir="$2" container="$3"
    docker exec "$container" bash -c "cd $deploy_dir && bash stop_node.sh $role 2>/dev/null || true"
    echo "STOPPED"
}

_remote_check_vllm() {
    local container="$1"
    if docker exec "$container" pgrep -f "vllm serve" >/dev/null 2>&1; then
        echo "RUNNING"
    else
        echo "FREE"
    fi
}

_remote_check_model() {
    local path="$1" container="$2"
    if docker exec "$container" test -d "$path" 2>/dev/null; then
        echo "EXISTS"
    else
        echo "MISSING"
    fi
}

# ---- 并行执行节点操作 --------------------------------------------------------
_run_on_nodes() {
    local action="$1"; shift
    local ips=($(_get_ips))
    [[ ${#ips[@]} -gt 0 ]] || { echo "[ERROR] 无节点"; return 1; }

    local func_code call_code b64 idx=0
    local tmpdir
    tmpdir=$(mktemp -d "/tmp/manage_nodes.XXXXXX")
    trap 'rm -rf "$tmpdir"' EXIT

    for ip in "${ips[@]}"; do
        case "$action" in
            start)
                local role
                if [[ " ${PNODE_IPS[*]:-} " == *" $ip "* ]]; then role="pnode"
                elif [[ " ${DNODE_IPS[*]:-} " == *" $ip "* ]]; then role="dnode"
                else continue; fi
                func_code="$(declare -f _remote_start_node)"
                printf -v call_code '_remote_start_node %q %q %q %q' "$role" "$idx" "$DEPLOY_DIR" "$CONTAINER_NAME"
                ;;
            stop|clean)
                func_code="$(declare -f _remote_stop_node)"
                printf -v call_code '_remote_stop_node %q %q %q' "all" "$DEPLOY_DIR" "$CONTAINER_NAME"
                ;;
        esac

        b64="$(printf '%s\n%s\n' "${func_code}" "${call_code}" | base64 | tr -d '\n')"
        ssh_run_timeout "$SSH_TIMEOUT" "$ip" "echo '$b64' | base64 -d | bash" > "$tmpdir/${ip}.out" 2>&1 &
        limit_jobs "$PARALLELISM"
        idx=$((idx + 1))
    done
    wait

    # 汇总结果
    local ok=0 fail=0
    for ip in "${ips[@]}"; do
        local outfile="$tmpdir/${ip}.out"
        if [[ -f "$outfile" ]] && grep -q "STARTED\|STOPPED" "$outfile" 2>/dev/null; then
            echo "[OK] $ip: $(head -1 "$outfile")"
            ((ok++)) || true
        else
            echo "[FAIL] $ip: $(cat "$outfile" 2>/dev/null || echo 'no output')"
            ((fail++)) || true
        fi
    done
    echo "--- 结果: 成功=$ok 失败=$fail ---"
    return $fail
}

# ---- start 动作 --------------------------------------------------------------
cmd_start() {
    log_info "启动节点 (role=$ROLE, dir=$DEPLOY_DIR)"
    _run_on_nodes start

    # 健康检查
    log_info "等待健康检查..."
    local port_key label ips
    for role in pnode dnode; do
        [[ "$ROLE" == "all" || "$ROLE" == "$role" ]] || continue
        label="${role^}Node"
        port_key="${role^^}_VLLM_START_PORT"
        local port="${!port_key:-7100}"
        local ip_var="${role^^}_IPS[@]"
        local role_ips=("${!ip_var}")

        for ip in "${role_ips[@]}"; do
            log_info "等待 $label ($ip:$port)..."
            wait_for_server "$ip" "$port" 600 || log_err "$label ($ip:$port) 未就绪"
        done
    done
    log_info "节点启动完成"
}

# ---- stop/clean 动作 ---------------------------------------------------------
cmd_stop() {
    log_info "停止节点 (role=$ROLE, dir=$DEPLOY_DIR)"
    _run_on_nodes stop
    log_info "节点停止完成"
}

cmd_clean() {
    log_info "清理所有节点进程"
    ROLE=all _run_on_nodes stop
    log_info "清理完成"
}

# ---- status 动作 -------------------------------------------------------------
cmd_status() {
    local check_script="$DEPLOY_DIR/check_status.sh"
    if [[ -f "$check_script" ]]; then
        bash "$check_script"
    else
        log_err "check_status.sh 未找到: $check_script"
        return 1
    fi
}

# ---- 主入口 ------------------------------------------------------------------
case "$ACTION" in
    start) cmd_start ;;
    stop)  cmd_stop ;;
    clean) cmd_clean ;;
    status) cmd_status ;;
esac
