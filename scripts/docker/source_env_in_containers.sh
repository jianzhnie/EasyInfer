#!/usr/bin/env bash
#
# 批量进入所有节点容器并 source set_env.sh
#
# 用法:
#   ./source_env_in_containers.sh
#
# 环境变量 (均可外部覆盖):
#   NODES_FILE         - 节点列表文件 (默认: set_env.sh 中配置的值)
#   CONTAINER_NAME     - Docker 容器名 (默认: vllm-ascend-env-a3)
#   PARALLELISM        - 并发数 (默认: 8)
#
# 依赖:
#   - source common.sh, set_env.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/set_env.sh"

# 加载共享工具函数
source "${SCRIPT_DIR}/../common.sh"

# ------------------------------------------
# 引入环境变量（操作机）
# ------------------------------------------
if [[ -f "${ENV_FILE}" ]]; then
    # shellcheck source=/dev/null
    source "${ENV_FILE}"
else
    log_warn "本地环境配置文件未找到: ${ENV_FILE}"
fi

# 默认值（set_env.sh 未加载时）
NODES_FILE="${NODES_FILE:-${SCRIPT_DIR}/../node_list.txt}"
SSH_USER_HOST_PREFIX="${SSH_USER_HOST_PREFIX:-}"
SSH_OPTS="${SSH_OPTS:--o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10}"
CONTAINER_NAME="${CONTAINER_NAME:-vllm-ascend-env-a3}"
PARALLELISM="${PARALLELISM:-8}"

# ------------------------------------------
# 帮助信息
# ------------------------------------------
usage() {
    cat <<'USAGE'
Usage:
  bash source_env_in_containers.sh [OPTIONS]

Description:
  读取节点列表中的所有节点，SSH 登录后在对应 Docker 容器内 source set_env.sh。
  要求宿主机和容器内 set_env.sh 路径一致。

Options:
  -h, --help       显示帮助信息
USAGE
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    usage
    exit 0
fi

if [[ ! -f "${NODES_FILE}" ]]; then
    log_err "节点列表文件未找到: ${NODES_FILE}"
    exit 1
fi

# ------------------------------------------
# 对单个节点：在容器内 source set_env.sh
# ------------------------------------------
source_env_for_node() {
    local node="$1"

    log_info "[${node}] 正在容器 ${CONTAINER_NAME} 内 source ${ENV_FILE} ..."

    if ssh_run "$node" "docker exec '${CONTAINER_NAME}' bash -c 'source ${ENV_FILE} && echo \"[INFO] Environment sourced successfully in container ${CONTAINER_NAME}\"'" ; then
        log_info "[${node}] 完成"
    else
        log_err "[${node}] 失败"
        return 1
    fi
}

# ------------------------------------------
# 主流程入口
# ------------------------------------------
nodes=$(read_nodes "$NODES_FILE")
if [[ -z "$nodes" ]]; then
    log_err "NODES_FILE 中未找到任何节点信息"
    exit 2
fi

log_info "目标容器: ${CONTAINER_NAME}"
log_info "环境文件: ${ENV_FILE}"
log_info "=== 开始批量 source 环境变量 ==="

for node in $nodes; do
    limit_jobs "${PARALLELISM}"
    (source_env_for_node "${node}") &
done
wait

log_info "=== 全部处理完成 ==="
