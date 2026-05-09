#!/usr/bin/env bash
# Ray 集群停止脚本

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[[ -f "${SCRIPT_DIR}/set_env.sh" ]] && source "${SCRIPT_DIR}/set_env.sh"

# 配置
KILL_TIMEOUT="${KILL_TIMEOUT:-3}"
PARALLELISM="${PARALLELISM:-16}"
RAY_KEYWORDS="raylet|plasma_store|gcs_server|ray::|ray.worker|python.*ray|dashboard_agent|runtime_env_agent"

# 日志
log() { echo "[$(date '+%H:%M:%S')] $*"; }
log_err() { echo "[$(date '+%H:%M:%S')] ERROR: $*" >&2; }

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

# 辅助函数
read_nodes() {
  [[ -f "${NODES_FILE:-}" ]] || { log_err "节点列表文件未找到: ${NODES_FILE:-}"; exit 2; }
  awk 'NF && !/^#/ {print $1}' "$NODES_FILE"
}

ssh_target() { printf "%s%s" "${SSH_USER_HOST_PREFIX:-}" "$1"; }

ssh_run() {
  local node="$1"; shift
  ssh ${SSH_OPTS:-} "$(ssh_target "$node")" "$@"
}

limit_jobs() {
  while [[ "$(jobs -rp | wc -l)" -ge "$1" ]]; do
    wait -n 2>/dev/null || sleep 0.1
  done
}

# 远程执行函数
_remote_stop_ray() {
  local force="$1" kill_timeout="$2"
  set -euo pipefail

  get_ray_pids() {
    ps aux | grep -E "$3" | grep -v grep | awk '{print $2}' | sort -u | tr '\n' ' ' || true
  }

  kill_procs() {
    local sig="$1" pids="$2"
    [[ -n "$pids" ]] || return 0
    for pid in $pids; do kill -"$sig" "$pid" 2>/dev/null || true; done
  }

  # 步骤1: 尝试 ray stop
  if command -v ray >/dev/null 2>&1; then
    [[ "$force" == "true" ]] && ray stop -f --grace-period 0 >/dev/null 2>&1 || ray stop -f >/dev/null 2>&1 || true
  fi

  # 步骤2: 终止 Ray 进程
  local pids=$(get_ray_pids)
  [[ -n "$pids" ]] || { echo "未找到 Ray 进程"; return 0; }

  echo "找到 Ray 进程: $pids"

  if [[ "$force" == "true" ]]; then
    echo "强制终止..."
    kill_procs 9 "$pids"
  else
    echo "温和终止 (SIGTERM)..."
    kill_procs 15 "$pids"
    sleep "$kill_timeout"

    local remaining=$(get_ray_pids)
    if [[ -n "$remaining" ]]; then
      echo "强制终止残余进程 (SIGKILL)..."
      kill_procs 9 "$remaining"
      sleep 0.5
    fi
  fi

  # 最终检查
  local final=$(get_ray_pids)
  if [[ -n "$final" ]]; then
    echo "警告: 仍有残余进程: $final"
    return 1
  fi
  echo "Ray 进程清理完成"
}

# 停止单个节点
stop_ray_node() {
  local node="$1"
  log "[${node}] 停止 Ray..."

  local func=$(declare -f _remote_stop_ray)
  local call="_remote_stop_ray '${FORCE}' '${KILL_TIMEOUT}' '${RAY_KEYWORDS}'"

  if $ON_HOST; then
    echo "$func; $call" | ssh_run "$node" bash -lc "bash -s" 2>/dev/null \
      && log "[${node}] 已停止" || log_err "[${node}] 停止失败"
  else
    local cmd="cd '${SCRIPT_DIR}' && source set_env.sh && docker exec -i '\${CONTAINER_NAME:-vllm-ascend-env-a3}' bash -s"
    echo "$func; $call" | ssh_run "$node" "$cmd" 2>/dev/null \
      && log "[${node}] 已停止" || log_err "[${node}] 停止失败"
  fi
}

# 主流程
nodes=$(read_nodes)
[[ -n "$nodes" ]] || { log_err "未找到节点信息"; exit 2; }

# 确认
if ! $SKIP_CONFIRM; then
  echo "================================"
  echo "将停止以下节点的 Ray 集群:"
  echo "  $(echo $nodes | tr '\n' ' ')"
  echo "  模式: $(${ON_HOST} && echo '宿主机' || echo '容器内')"
  echo "  强制: $(${FORCE} && echo '是' || echo '否')"
  echo "================================"
  read -p "输入 'yes' 继续: " confirm
  [[ "$confirm" == "yes" ]] || { log "已取消"; exit 0; }
fi

log "开始停止 Ray 集群..."

for node in $nodes; do
  limit_jobs "$PARALLELISM"
  (stop_ray_node "$node") &
done
wait

log "Ray 集群停止完成"
