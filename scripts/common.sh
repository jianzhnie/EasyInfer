#!/usr/bin/env bash
#
# 共享工具函数库 — 供 scripts/ 下各子目录的脚本 source 使用
#
# 注意: 本文件被 source 而非直接执行，刻意不加 set -euo pipefail，
#       以免影响调用脚本的 shell 选项。所有函数内部自行处理错误。

# 颜色常量
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly NC='\033[0m'

# ------------------------------------------------------------------------------
# 日志函数
# ------------------------------------------------------------------------------
_timestamp() { date '+%Y-%m-%d %H:%M:%S'; }
log_info()  { printf "${GREEN}[INFO]${NC}  %s - %s\n" "$(_timestamp)" "$*"; }
log_warn()  { printf "${YELLOW}[WARN]${NC}  %s - %s\n" "$(_timestamp)" "$*" >&2; }
log_err()   { printf "${RED}[ERROR]${NC} %s - %s\n" "$(_timestamp)" "$*" >&2; }
log_fatal() { printf "${RED}[FATAL]${NC} %s - %s\n" "$(_timestamp)" "$*" >&2; exit 1; }

# ------------------------------------------------------------------------------
# 节点列表读取
# ------------------------------------------------------------------------------
read_nodes() {
    local nodes_file="${1:?用法: read_nodes <nodes_file>}"
    if [[ ! -f "$nodes_file" ]]; then
        log_err "节点列表文件未找到: $nodes_file"
        return 1
    fi
    awk 'NF && !/^#/ {print $1}' "$nodes_file"
}

# ------------------------------------------------------------------------------
# SSH 辅助函数
# ------------------------------------------------------------------------------
ssh_target() {
    printf "%s%s" "${SSH_USER_HOST_PREFIX:-}" "$1"
}

# SSH_OPTS 通过词分割传递多个选项（如 "-o BatchMode=yes -o ConnectTimeout=10"）。
# 这是有意设计的简单约定，不改为数组以保持向后兼容。
# shellcheck disable=SC2029
ssh_run() {
    local node="$1"; shift
    # shellcheck disable=SC2086
    ssh ${SSH_OPTS:-} "$(ssh_target "$node")" "$@"
}

# ------------------------------------------------------------------------------
# 并发控制
# ------------------------------------------------------------------------------
limit_jobs() {
    local max="${1:?用法: limit_jobs <max>}"
    while [[ "$(jobs -rp 2>/dev/null | wc -l)" -ge "$max" ]]; do
        wait -n 2>/dev/null || sleep 0.1
    done
}

# ------------------------------------------------------------------------------
# 网络工具：获取业务网卡 IP
# ------------------------------------------------------------------------------
get_node_ip() {
    local interface="${1:?用法: get_node_ip <interface>}"
    local ip=""
    if command -v ip >/dev/null 2>&1; then
        ip=$(ip -4 addr show "${interface}" 2>/dev/null | awk '/inet / {print $2}' | cut -d/ -f1 | head -n 1)
    elif command -v ifconfig >/dev/null 2>&1; then
        ip=$(ifconfig "${interface}" 2>/dev/null | awk '/inet / {print $2}' | head -n 1)
    fi
    printf "%s" "$ip"
}

# ------------------------------------------------------------------------------
# scripts 目录路径 (基于 common.sh 的位置)
# ------------------------------------------------------------------------------
SCRIPTS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC2034
readonly SCRIPTS_ROOT
