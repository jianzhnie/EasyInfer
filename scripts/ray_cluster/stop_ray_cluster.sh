#!/bin/bash
#
# Ray 集群停止脚本 — 停止并清理所有节点上的 Ray 进程
#
# 用法:
#   ./stop_ray_cluster.sh [OPTIONS]
#   KILL_TIMEOUT=5 ./stop_ray_cluster.sh  # 通过环境变量覆盖
#
# 环境变量 (均可外部覆盖):
#   NODES_FILE    - 节点列表文件 (默认: set_env.sh 中配置)
#   KILL_TIMEOUT  - SIGTERM 后等待超时秒数 (默认: 3)
#   PARALLELISM   - 并发节点数 (默认: 16)
#
# 依赖:
#   - source common.sh, vllm/set_env.sh
#   - 远程节点需要: ray, docker

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/set_ray_env.sh"
# shellcheck source=./set_ray_env.sh
[[ -f "${ENV_FILE}" ]] && source "${ENV_FILE}"

# 加载共享工具函数
source "${SCRIPT_DIR}/../common.sh"

# 配置
KILL_TIMEOUT="${KILL_TIMEOUT:-3}"
PARALLELISM="${PARALLELISM:-16}"
RAY_KEYWORDS="raylet|plasma_store|gcs_server|ray::|ray.worker|python.*ray|dashboard_agent|runtime_env_agent"

# 帮助
usage() {
    cat <<'EOF'
Usage: bash stop_ray_cluster.sh [OPTIONS]

Options:
  --on-host    在宿主机上停止 Ray（不进容器）
  -f, --force  强制模式：立即停止并清理所有残余
  -y, --yes    跳过确认步骤
  -h, --help   显示帮助信息
EOF
}

# 参数解析
ON_HOST=false FORCE=false SKIP_CONFIRM=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --on-host) ON_HOST=true; shift ;;
        -f|--force) FORCE=true; shift ;;
        -y|--yes) SKIP_CONFIRM=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) log_err "未知选项: $1"; usage >&2; exit 2 ;;
    esac
done

_remote_stop_ray() {
    local force="$1" kill_timeout="$2" pattern="$3"
    set -euo pipefail

    get_ray_pids() {
        # shellcheck disable=SC2009
        ps aux | grep -E "$pattern" | grep -v grep | awk '{print $2}' | sort -u | tr '\n' ' ' || true
    }

    kill_procs() {
        local sig="$1" pids="$2"
        [[ -n "$pids" ]] || return 0
        # shellcheck disable=SC2086
        for pid in $pids; do kill -"$sig" "$pid" 2>/dev/null || true; done
    }

    # 步骤1: 尝试 ray stop
    if command -v ray >/dev/null 2>&1; then
        [[ "$force" == "true" ]] && ray stop -f --grace-period 0 >/dev/null 2>&1 || ray stop -f >/dev/null 2>&1 || true
    fi

    # 步骤2: 终止 Ray 进程
    local pids
    pids=$(get_ray_pids)
    [[ -n "$pids" ]] || { echo "未找到 Ray 进程"; return 0; }

    echo "找到 Ray 进程: $pids"

    if [[ "$force" == "true" ]]; then
        echo "强制终止..."
        kill_procs 9 "$pids"
    else
        echo "温和终止 (SIGTERM)..."
        kill_procs 15 "$pids"
        sleep "$kill_timeout"

        local remaining
        remaining=$(get_ray_pids)
        if [[ -n "$remaining" ]]; then
            echo "强制终止残余进程 (SIGKILL)..."
            kill_procs 9 "$remaining"
            sleep 0.5
        fi
    fi

    # 最终检查
    local final
    final=$(get_ray_pids)
    if [[ -n "$final" ]]; then
        echo "警告: 仍有残余进程: $final"
        return 1
    fi
    echo "Ray 进程清理完成"
}

# 停止单个节点
stop_ray_node() {
    local node="$1"
    log_info "[${node}] 停止 Ray..."

    local func call
    func=$(declare -f _remote_stop_ray)
    call="_remote_stop_ray '${FORCE}' '${KILL_TIMEOUT}' '${RAY_KEYWORDS}'"

    if $ON_HOST; then
        if echo "$func; $call" | ssh_run "$node" bash -s; then
            log_info "[${node}] 已停止"
        else
            log_err "[${node}] 停止失败"
        fi
    else
        # 加载环境文件后再执行 docker exec
        local cmd="[[ -f '${ENV_FILE}' ]] && source '${ENV_FILE}'; docker exec -i \"\${CONTAINER_NAME:-vllm-ascend-env-a3}\" bash -s"
        if echo "$func; $call" | ssh_run "$node" "$cmd"; then
            log_info "[${node}] 已停止"
        else
            log_err "[${node}] 停止失败"
        fi
    fi
}

# 主流程
: "${NODE_LIST:?NODE_LIST 未设置，请检查 set_ray_env.sh 是否正确加载}"
nodes=$(read_nodes "$NODE_LIST")
[[ -n "$nodes" ]] || { log_err "未找到节点信息"; exit 2; }

# 确认
if ! $SKIP_CONFIRM; then
    echo "================================"
    echo "将停止以下节点的 Ray 集群:"
    echo "  $nodes"
    echo "  模式: $( [[ "$ON_HOST" == "true" ]] && echo '宿主机' || echo '容器内' )"
    echo "  强制: $( [[ "$FORCE" == "true" ]] && echo '是' || echo '否' )"
    echo "================================"
    read -r -p "输入 'yes' 继续: " confirm
    [[ "$confirm" == "yes" ]] || { log_info "已取消"; exit 0; }
fi

log_info "开始停止 Ray 集群..."

for node in $nodes; do
    limit_jobs "$PARALLELISM"
    (stop_ray_node "$node") &
done
wait

log_info "Ray 集群停止完成"
