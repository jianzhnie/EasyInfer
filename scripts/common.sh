#!/bin/bash
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
log_warn()  { _log "$YELLOW" "WARN"  "$@" >&2; }
log_err()   { _log "$RED"    "ERROR" "$@" >&2; }
log_fatal() { _log "$RED"    "FATAL" "$@" >&2; exit 1; }
log_debug() { [[ "${DEBUG:-0}" == "1" ]] && _log "$CYAN" "DEBUG" "$@" >&2; }

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
    elif command -v perl >/dev/null 2>&1; then
        # Fallback: perl alarm-based timeout
        # shellcheck disable=SC2086
        perl -e '
            use strict; use warnings;
            my $timeout = shift @ARGV; my @cmd = @ARGV;
            eval { local $SIG{ALRM} = sub { die "TIMEOUT\n" }; alarm $timeout; system(@cmd); alarm 0; };
            if ($@ eq "TIMEOUT\n") { print STDERR "[ERROR] Command timed out after ${timeout}s\n"; exit 124; }
            exit $? >> 8;
        ' "$timeout_sec" ssh ${SSH_OPTS:-} "$(ssh_target "$node")" "$@" 2>&1
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
        sleep 0.5
    done
}

# 等待所有后台任务完成，收集失败数
# 用法: wait_jobs → 返回失败任务数
wait_jobs() {
    local failed=0 pid
    # 兼容 bash 4.2+: 逐个 wait PID 而非 wait -n
    for pid in $(jobs -rp 2>/dev/null); do
        wait "$pid" 2>/dev/null || ((failed++)) || true
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
# 参数解析：统一处理 --file/-f 节点列表参数
# ------------------------------------------------------------------------------
# 用法: NODE_LIST=$(parse_nodes_file_arg "$@")
# 从脚本参数中提取 --file/-f 值，未提供则返回 ${NODES_FILE:-scripts/node_list.txt}
parse_nodes_file_arg() {
    local nodes_file="${NODES_FILE:-scripts/node_list.txt}"
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --file|-f)
                if [[ -n "${2:-}" && "$2" != -* ]]; then
                    nodes_file="$2"
                    shift 2
                else
                    log_fatal "选项 $1 需要一个参数: 节点列表文件路径"
                fi
                ;;
            *) shift ;;
        esac
    done
    printf '%s\n' "$nodes_file"
}

# ------------------------------------------------------------------------------
# 统一节点解析: CLI --hosts > --file/-f > NODES_FILE 环境变量 > 默认文件
# 用法: resolve_nodes "$@" → 将结果存入全局数组 RESOLVED_NODES
# ------------------------------------------------------------------------------
resolve_nodes() {
    RESOLVED_NODES=()
    local nodes_file=""

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --hosts) shift
                while [[ $# -gt 0 && "$1" != -* ]]; do
                    RESOLVED_NODES+=("$1"); shift
                done ;;
            --file|-f)
                [[ -n "${2:-}" && "$2" != -* ]] || log_fatal "选项 $1 需要一个参数: 节点列表文件路径"
                nodes_file="$2"; shift 2 ;;
            *) shift ;;
        esac
    done

    # 如果已通过 --hosts 指定，直接返回
    [[ ${#RESOLVED_NODES[@]} -gt 0 ]] && return 0

    # 从文件读取
    local file="${nodes_file:-${NODES_FILE:-scripts/node_list.txt}}"
    [[ -f "$file" ]] || log_fatal "节点列表文件未找到: $file"

    while IFS= read -r line; do
        [[ -n "$line" ]] && RESOLVED_NODES+=("$line")
    done < <(read_nodes "$file")

    [[ ${#RESOLVED_NODES[@]} -gt 0 ]] || log_fatal "节点列表为空: $file"
}

# ------------------------------------------------------------------------------
# 等待 vLLM HTTP 服务就绪
# ------------------------------------------------------------------------------
wait_for_server() {
    local host="${1:?用法: wait_for_server <host> <port> [timeout_sec]}"
    local port="${2:?用法: wait_for_server <host> <port> [timeout_sec]}"
    local max_wait="${3:-600}"
    local url="http://${host}:${port}/health"
    local elapsed=0 interval=5

    log_info "Waiting for server to become ready..."
    while (( elapsed < max_wait )); do
        if curl -sf "$url" >/dev/null 2>&1; then
            log_info "================================================================================="
            log_info "  vLLM server is READY"
            log_info "================================================================================="
            log_info "  Health check:  http://${host}:${port}/health"
            log_info "  API endpoint:  http://${host}:${port}/v1"
            log_info "  Models list:   http://${host}:${port}/v1/models"
            log_info "================================================================================="
            return 0
        fi
        sleep "$interval"
        elapsed=$((elapsed + interval))
        printf "."
    done
    printf "\n"
    log_err "Server did not become ready within ${max_wait}s"
    return 1
}

# ------------------------------------------------------------------------------
# 打印服务就绪后的 Claude Code 配置输出
# ------------------------------------------------------------------------------
print_server_ready() {
    local host_ip="${1:?用法: print_server_ready <host> <port> [model_name]}"
    local port="${2:?用法: print_server_ready <host> <port> [model_name]}"
    local model_name="${3:-}"

    log_info "================================================================================="
    log_info "  vLLM server is READY"
    log_info "================================================================================="
    log_info "  Health check:  http://${host_ip}:${port}/health"
    log_info "  API endpoint:  http://${host_ip}:${port}/v1"
    log_info "  Models list:   http://${host_ip}:${port}/v1/models"
    if [[ -n "$model_name" ]]; then
        log_info ""
        log_info "  --- Claude Code 配置 ---"
        log_info ""
        log_info "  方式一: 写入 ~/.claude/settings.json"
        log_info "  {"
        log_info "    \"env\": {"
        log_info "      \"ANTHROPIC_BASE_URL\": \"http://${host_ip}:${port}/v1\","
        log_info "      \"ANTHROPIC_API_KEY\": \"dummy\","
        log_info "      \"ANTHROPIC_AUTH_TOKEN\": \"dummy\","
        log_info "      \"ANTHROPIC_DEFAULT_SONNET_MODEL\": \"${model_name}\","
        log_info "      \"ANTHROPIC_DEFAULT_HAIKU_MODEL\": \"${model_name}\","
        log_info "      \"ANTHROPIC_DEFAULT_OPUS_MODEL\": \"${model_name}\""
        log_info "    }"
        log_info "  }"
        log_info ""
        log_info "  方式二: 命令行直接使用"
        log_info "  ANTHROPIC_BASE_URL=http://${host_ip}:${port}/v1 \\"
        log_info "  ANTHROPIC_API_KEY=dummy \\"
        log_info "  ANTHROPIC_AUTH_TOKEN=dummy \\"
        log_info "  ANTHROPIC_DEFAULT_SONNET_MODEL=${model_name} \\"
        log_info "  ANTHROPIC_DEFAULT_HAIKU_MODEL=${model_name} \\"
        log_info "  ANTHROPIC_DEFAULT_OPUS_MODEL=${model_name} \\"
        log_info "  claude"
    fi
    log_info ""
    log_info "================================================================================="
}

# ------------------------------------------------------------------------------
# 要求环境变量已设置
# ------------------------------------------------------------------------------
require_env() {
    local var="$1"
    local desc="${2:-$var}"
    if [[ -z "${!var:-}" ]]; then
        log_fatal "环境变量 ${var} (${desc}) 未设置"
    fi
}

# ------------------------------------------------------------------------------
# 用户确认
# ------------------------------------------------------------------------------
# 用法: confirm "确认操作?" [default_yes|default_no]
# 返回 0 表示用户确认, 1 表示取消
# 若 SKIP_CONFIRM=true，自动跳过交互返回 0
confirm() {
    local msg="${1:?用法: confirm <message> [default]}"
    local default="${2:-default_no}"
    [[ "${SKIP_CONFIRM:-false}" == "true" ]] && return 0
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
# 错误码常量 (供外部脚本使用)
# ------------------------------------------------------------------------------
# shellcheck disable=SC2034
readonly E_OK=0
# shellcheck disable=SC2034
readonly E_GENERAL=1
# shellcheck disable=SC2034
readonly E_INVALID_ARG=2
# shellcheck disable=SC2034
readonly E_NOT_FOUND=3
# shellcheck disable=SC2034
readonly E_TIMEOUT=124
# shellcheck disable=SC2034
readonly E_CMD_NOT_FOUND=127

# ------------------------------------------------------------------------------
# 网络工具：获取节点网卡 IP
# ------------------------------------------------------------------------------
# 获取指定节点上指定网卡的 IP 地址
# 用法: get_node_ip [node] <interface>
#   node      - 目标节点（空字符串表示本地）
#   interface - 网卡名称
get_node_ip() {
    local node="${1:-}"
    local interface="${2:-}"
    local ip=""

    # 兼容旧版单参数调用: get_node_ip <interface>
    if [[ -z "$interface" && -n "$node" ]]; then
        interface="$node"
        node=""
    fi

    [[ -n "$interface" ]] || { printf '\n'; return; }

    local cmd=""
    if command -v ip >/dev/null 2>&1; then
        cmd="ip -4 addr show ${interface} 2>/dev/null | awk '/inet / {print \$2}' | cut -d/ -f1 | head -n 1"
    elif command -v ifconfig >/dev/null 2>&1; then
        cmd="ifconfig ${interface} 2>/dev/null | awk '/inet / {print \$2}' | head -n 1"
    else
        printf '\n'
        return
    fi

    if [[ -z "$node" || "$node" == "$(hostname -s 2>/dev/null)" || "$node" == "$(hostname 2>/dev/null)" ]]; then
        ip=$(eval "$cmd")
    else
        # shellcheck disable=SC2086,SC2029
        ip=$(ssh ${SSH_OPTS:-} "$(ssh_target "$node")" "$cmd" 2>/dev/null)
    fi
    printf '%s\n' "${ip:-}"
}

# 自动探测默认网卡
get_default_nic() {
    local nic=""
    if command -v ip >/dev/null 2>&1; then
        nic=$(ip route 2>/dev/null | awk '/default/ {print $5; exit}')
    fi
    printf "%s" "${nic:-}"
}

# 获取本机指定网卡的 IP
# 用法: get_local_ip <interface>
get_local_ip() {
    local interface="${1:?用法: get_local_ip <interface>}"
    get_node_ip "" "$interface"
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
# vLLM 参数探测工具
# ------------------------------------------------------------------------------

# 获取 vllm serve --help 输出
vllm_help() {
    vllm serve --help 2>/dev/null || true
}

# 检查 help_text 中是否包含指定 flag
# 用法: has_flag <help_text> <flag>
has_flag() {
    local help_text="$1" flag="$2"
    [[ "$help_text" == *"$flag"* ]]
}

# 从 help_text 中选择支持的 flag（优先使用 preferred，不存在则回退到 fallback）
# 用法: choose_flag <help_text> <preferred> <fallback>
choose_flag() {
    local help_text="$1" preferred="$2" fallback="$3"
    if [[ -n "$preferred" && "$help_text" == *"$preferred"* ]]; then
        printf '%s' "$preferred"
        return 0
    fi
    if [[ -n "$fallback" && "$help_text" == *"$fallback"* ]]; then
        printf '%s' "$fallback"
        return 0
    fi
    printf '%s' "$preferred"
}

# ------------------------------------------------------------------------------
# scripts 目录路径 (基于 common.sh 的位置)
# ------------------------------------------------------------------------------
SCRIPTS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC2034
readonly SCRIPTS_ROOT

# ------------------------------------------------------------------------------
# 判断 IP 是否为本机
# ------------------------------------------------------------------------------
is_local_ip() {
    local ip="$1"
    local lip
    for lip in $(hostname -I 2>/dev/null || true); do
        [[ "$ip" == "$lip" ]] && return 0
    done
    [[ "$ip" == "$(hostname -s 2>/dev/null)" || "$ip" == "$(hostname 2>/dev/null)" ]] && return 0
    return 1
}
