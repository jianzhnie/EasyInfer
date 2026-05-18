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
readonly CYAN='\033[0;36m'
readonly NC='\033[0m'

# ------------------------------------------------------------------------------
# 日志函数
# ------------------------------------------------------------------------------
_timestamp() { date '+%Y-%m-%d %H:%M:%S'; }

_log() {
    local color="$1" level="$2"; shift 2
    printf "${color}[%-5s]${NC} %s - %s\n" "$level" "$(_timestamp)" "$*"
}

log_info()  { _log "$GREEN"  "INFO"  "$@"; }
log_warn()  { _log "$YELLOW" "WARN"  "$*" >&2; }
log_err()   { _log "$RED"    "ERROR" "$*" >&2; }
log_fatal() { _log "$RED"    "FATAL" "$*" >&2; exit 1; }
log_debug() { [[ "${DEBUG:-0}" == "1" ]] && _log "$CYAN" "DEBUG" "$*" >&2; }

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

# 带超时的 SSH 命令，超时返回 124
ssh_run_timeout() {
    local timeout_sec="${1:?用法: ssh_run_timeout <timeout> <node> <cmd...>}"; shift
    local node="$1"; shift
    if command -v timeout >/dev/null 2>&1; then
        # shellcheck disable=SC2086
        timeout "$timeout_sec" ssh ${SSH_OPTS:-} "$(ssh_target "$node")" "$@" 2>&1
    else
        # 无 timeout 命令时直接执行
        # shellcheck disable=SC2086,SC2029
        ssh ${SSH_OPTS:-} "$(ssh_target "$node")" "$@" 2>&1
    fi
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

# 等待所有后台任务完成，收集失败数
# 用法: wait_jobs → 返回失败任务数
wait_jobs() {
    local failed=0
    while true; do
        if wait -n 2>/dev/null; then
            :
        else
            # wait -n 返回非零：该任务失败，已从 job table 中收割
            ((failed++)) || true
            # 继续等待剩余任务，直到没有更多后台任务
            if ! jobs -p 2>/dev/null | grep -q .; then
                break
            fi
        fi
    done
    echo "$failed"
}

# ------------------------------------------------------------------------------
# 前置检查
# ------------------------------------------------------------------------------
# 检查命令是否存在，不存在则 log_fatal
require_cmd() {
    local cmd="$1"
    if ! command -v "$cmd" >/dev/null 2>&1; then
        log_fatal "缺少必要命令: $cmd"
    fi
}

# 批量检查命令
require_cmds() {
    local cmd
    for cmd in "$@"; do
        require_cmd "$cmd"
    done
}

# ------------------------------------------------------------------------------
# 等待服务就绪
# ------------------------------------------------------------------------------
# 等待 TCP 端口可达
# 用法: wait_for_port <host> <port> [timeout_sec] [interval_sec]
wait_for_port() {
    local host="${1:?用法: wait_for_port <host> <port> [timeout] [interval]}"
    local port="${2:?用法: wait_for_port <host> <port> [timeout] [interval]}"
    local timeout="${3:-120}"
    local interval="${4:-5}"
    local start elapsed=0
    start=$(date +%s)

    while true; do
        if command -v nc >/dev/null 2>&1; then
            nc -z -w 2 "$host" "$port" 2>/dev/null && return 0
        elif command -v timeout >/dev/null 2>&1; then
            timeout 2 bash -c "echo >/dev/tcp/$host/$port" 2>/dev/null && return 0
        elif [[ -e /dev/tcp ]]; then
            (echo >/dev/tcp/"$host"/"$port") 2>/dev/null && return 0
        fi
        elapsed=$(( $(date +%s) - start ))
        [[ "$elapsed" -ge "$timeout" ]] && return 1
        sleep "$interval"
    done
}

# ------------------------------------------------------------------------------
# 用户确认
# ------------------------------------------------------------------------------
# 用法: confirm "确认操作?" [default_yes|default_no]
# 返回 0 表示用户确认, 1 表示取消
confirm() {
    local msg="${1:?用法: confirm <message> [default]}"
    local default="${2:-default_no}"
    local prompt
    if [[ "$default" == "default_yes" ]]; then
        prompt="$msg [Y/n] "
    else
        prompt="$msg [y/N] "
    fi
    read -r -p "$prompt" answer 2>/dev/null || answer=""
    case "$answer" in
        y|Y|yes|YES) return 0 ;;
        n|N|no|NO)   return 1 ;;
        "")          [[ "$default" == "default_yes" ]] && return 0 || return 1 ;;
        *)           return 1 ;;
    esac
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

# 自动探测默认网卡
get_default_nic() {
    local nic=""
    if command -v ip >/dev/null 2>&1; then
        nic=$(ip route 2>/dev/null | awk '/default/ {print $5; exit}')
    fi
    printf "%s" "${nic:-}"
}

# ------------------------------------------------------------------------------
# 临时文件管理
# ------------------------------------------------------------------------------
# 创建临时目录并注册清理 trap
# 用法: mktemp_dir → 输出路径; 全局变量 _TEMP_DIR 可用于 cleanup
mktemp_dir() {
    _TEMP_DIR=$(mktemp -d "${TMPDIR:-/tmp}/easyinfer.XXXXXX")
    # shellcheck disable=SC2329
    _tempdir_cleanup() { # invoked via trap below
        local rc=$?
        [[ -n "${_TEMP_DIR:-}" && -d "$_TEMP_DIR" ]] && rm -rf "$_TEMP_DIR"
        exit "$rc"
    }
    trap _tempdir_cleanup EXIT INT TERM
    echo "$_TEMP_DIR"
}

# ------------------------------------------------------------------------------
# scripts 目录路径 (基于 common.sh 的位置)
# ------------------------------------------------------------------------------
SCRIPTS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC2034
readonly SCRIPTS_ROOT
