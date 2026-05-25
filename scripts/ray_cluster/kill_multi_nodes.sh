#!/bin/bash
# ==============================================================================
# 多节点进程清理脚本 (kill_multi_nodes.sh)
#
# 该脚本通过 SSH 并发连接到多个节点，根据关键字终止指定的进程。
# 脚本会首先尝试温和地终止进程（SIGTERM），超时后若进程仍存活，则强制终止（SIGKILL）。
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${SCRIPT_DIR}/set_ray_env.sh"

# 加载共享工具函数
source "${SCRIPTS_DIR}/common.sh"

# ------------------------------------------
# 引入环境变量
# ------------------------------------------
if [[ -f "${ENV_FILE}" ]]; then
    # shellcheck source=/dev/null
    source "${ENV_FILE}"
fi

# ------------------------------------------
# 默认配置（可被环境变量或命令行参数覆盖）
# ------------------------------------------
# 优先使用 NODES_FILE，其次是 set_ray_env.sh 中的 NODE_LIST，最后是默认路径
NODE_LIST_FILE="${NODES_FILE:-${NODE_LIST:-${SCRIPTS_DIR}/node_list.txt}}"
MAX_JOBS="${MAX_JOBS:-${MAX_SSH_PARALLELISM:-16}}"
SSH_TIMEOUT="${SSH_TIMEOUT:-10}"
KILL_TIMEOUT="${KILL_TIMEOUT:-3}"

# 定义要 kill 的关键词（支持正则，可通过环境变量扩展）
DEFAULT_KEYWORDS=("llmtuner" "mindspeed" "ray" "vllm" "verl" "raylet" "plasma_store" "gcs_server" "dashboard_agent" "runtime_env_agent")
if [[ -n "${EXTRA_KEYWORDS:-}" ]]; then
    IFS=',' read -ra EXTRA_KEYWORDS_ARRAY <<< "$EXTRA_KEYWORDS"
    KEYWORDS=("${DEFAULT_KEYWORDS[@]}" "${EXTRA_KEYWORDS_ARRAY[@]}")
else
    KEYWORDS=("${DEFAULT_KEYWORDS[@]}")
fi

# SSH 选项
SSH_OPTS="${SSH_OPTS:-}"
SSH_USER_HOST_PREFIX="${SSH_USER_HOST_PREFIX:-}"

# ------------------------------------------
# 全局状态跟踪
# ------------------------------------------
declare -a FAILED_NODES=()
declare -a TIMEOUT_NODES=()

# ------------------------------------------
# 帮助信息
# ------------------------------------------
usage() {
    cat <<EOF
Usage: $0 [OPTIONS] [node_list_file]

Options:
  -y, --yes              跳过确认步骤，直接执行
  -n, --dry-run          仅显示会终止的进程，不实际执行
  -k, --keywords LIST    自定义关键词列表（逗号分隔），替换默认列表
                         默认: $(IFS=','; echo "${DEFAULT_KEYWORDS[*]}")
  -t, --timeout SEC      终止进程超时时间，秒 (默认: $KILL_TIMEOUT)
  -j, --jobs NUM         最大并发任务数 (默认: $MAX_JOBS)
  --ssh-timeout SEC      SSH 连接超时时间，秒 (默认: $SSH_TIMEOUT)
  -q, --quiet            静默模式，减少输出
  -h, --help             显示此帮助信息

Environment Variables:
  NODES_FILE, SSH_OPTS, SSH_USER_HOST_PREFIX, EXTRA_KEYWORDS
  MAX_JOBS, KILL_TIMEOUT, SSH_TIMEOUT

Examples:
  $0                          # 使用默认配置
  $0 /path/to/nodes.txt       # 指定节点列表文件
  $0 -y                       # 跳过确认
  $0 -n                       # 干运行模式
  $0 -k "myapp,worker"        # 自定义关键词
  $0 -y -k "ray" -t 5         # 强制模式，只杀 ray 进程，超时 5 秒
EOF
}

# ------------------------------------------
# 信号处理
# ------------------------------------------
# shellcheck disable=SC2329
cleanup_jobs() {
    log_warn "接收到中断信号，正在清理后台作业..."
    local job
    for job in $(jobs -p); do
        kill "$job" 2>/dev/null || true
    done
    wait 2>/dev/null || true
    exit 130
}
trap cleanup_jobs INT TERM

# ------------------------------------------
# SSH 辅助
# ------------------------------------------
ssh_run_with_timeout() {
    local node="$1"
    shift
    local exit_code=0

    if command -v timeout >/dev/null 2>&1; then
        # shellcheck disable=SC2086
        timeout "$SSH_TIMEOUT" ssh ${SSH_OPTS} "$(ssh_target "$node")" "$@" || exit_code=$?
    else
        # shellcheck disable=SC2086
        perl -e '
            use strict; use warnings;
            my $timeout = shift @ARGV; my @cmd = @ARGV;
            eval { local $SIG{ALRM} = sub { die "TIMEOUT\n" }; alarm $timeout; system(@cmd); alarm 0; };
            if ($@ eq "TIMEOUT\n") { print STDERR "[ERROR] Command timed out after ${timeout}s\n"; exit 124; }
            exit $? >> 8;
        ' "$SSH_TIMEOUT" ssh ${SSH_OPTS} "$(ssh_target "$node")" "$@" || exit_code=$?
    fi

    return $exit_code
}

# ------------------------------------------
# 正则转义
# ------------------------------------------
escape_regex() {
    local s="$1"
    s="${s//\\/\\\\}"
    s="${s//./\\.}"
    s="${s//\*/\\*}"
    s="${s//+/\\+}"
    s="${s//\?/\\?}"
    s="${s//^/\\^}"
    s="${s//\$/\\$}"
    s="${s\//\(/\\(}"
    s="${s\//\)/\\)}"
    s="${s//\[/\\[}"
    s="${s//\]/\\]}"
    s="${s//\{/\\{}"
    s="${s//\}/\\}}"
    s="${s//|/\\|}"
    printf '%s' "$s"
}

# ------------------------------------------
# 参数解析
# ------------------------------------------
parse_args() {
    SKIP_CONFIRM=false
    DRY_RUN=false
    QUIET=false

    while [[ $# -gt 0 ]]; do
        case $1 in
            -y|--yes)      SKIP_CONFIRM=true; shift ;;
            -n|--dry-run)  DRY_RUN=true; shift ;;
            -q|--quiet)    QUIET=true; shift ;;
            -k|--keywords)
                [[ -n "${2:-}" && "$2" != -* ]] || { log_err "选项 $1 需要一个参数"; exit 1; }
                IFS=',' read -ra KEYWORDS <<< "$2"
                shift 2
                ;;
            -t|--timeout)
                [[ -n "${2:-}" && "$2" != -* ]] || { log_err "选项 $1 需要一个参数"; exit 1; }
                [[ "$2" =~ ^[0-9]+$ ]] || { log_err "超时时间必须是正整数"; exit 1; }
                KILL_TIMEOUT="$2"; shift 2
                ;;
            -j|--jobs)
                [[ -n "${2:-}" && "$2" != -* ]] || { log_err "选项 $1 需要一个参数"; exit 1; }
                [[ "$2" =~ ^[0-9]+$ && "$2" -ge 1 ]] || { log_err "并发数必须是正整数"; exit 1; }
                MAX_JOBS="$2"; shift 2
                ;;
            --ssh-timeout)
                [[ -n "${2:-}" && "$2" != -* ]] || { log_err "选项 $1 需要一个参数"; exit 1; }
                [[ "$2" =~ ^[0-9]+$ ]] || { log_err "SSH 超时时间必须是正整数"; exit 1; }
                SSH_TIMEOUT="$2"; shift 2
                ;;
            -h|--help)     usage; exit 0 ;;
            -*)            log_err "未知选项: $1"; usage >&2; exit 1 ;;
            *)            NODE_LIST_FILE="$1"; shift ;;
        esac
    done
}

# ------------------------------------------
# 远程脚本生成
# ------------------------------------------
_build_kill_pattern() {
    local escaped_keywords=()
    local kw
    for kw in "${KEYWORDS[@]}"; do
        escaped_keywords+=("$(escape_regex "$kw")")
    done
    IFS='|'; echo "${escaped_keywords[*]}"
}

_gen_kill_remote_script() {
    local pattern="$1" kill_timeout="$2" dry_run="$3"
    local script
    read -r -d '' script << 'REMOTE_SCRIPT'
        set -euo pipefail
        PATTERN="__PATTERN__"
        KILL_TIMEOUT="__KILL_TIMEOUT__"
        DRY_RUN="__DRY_RUN__"

        get_matching_pids() {
            ps aux | grep -E "$PATTERN" | grep -v grep | \
                grep -v -E '(vscode-server|code-server|sshd:|/bin/sh -c|extension|/agent/|ssh.*:)' | \
                awk '{print $2}' | sort -u | tr '\n' ' ' || true
        }

        get_process_info() {
            local pids="$1"
            # shellcheck disable=SC2086
            ps -p $pids -o pid,ppid,user,%cpu,%mem,etime,args 2>/dev/null || true
        }

        all_pids=$(get_matching_pids)
        if [ -z "$all_pids" ] || [ "$all_pids" = " " ]; then
            echo "STATUS:NO_PROCESSES"
            exit 0
        fi

        echo "STATUS:FOUND"
        echo "PIDS:$all_pids"
        echo "PROCESS_INFO:"
        get_process_info "$all_pids"

        if [ "$DRY_RUN" = "true" ]; then
            echo "ACTION:SKIP_DRY_RUN"
            exit 0
        fi

        echo "ACTION:SIGTERM"
        kill -15 $all_pids 2>/dev/null || true
        sleep "$KILL_TIMEOUT"

        remaining=""
        for pid in $all_pids; do
            if kill -0 "$pid" 2>/dev/null; then
                remaining="$remaining $pid"
            fi
        done
        remaining="${remaining# }"

        if [ -z "$remaining" ]; then
            echo "STATUS:TERMINATED"
            exit 0
        fi

        echo "ACTION:SIGKILL:$remaining"
        kill -9 $remaining 2>/dev/null || true
        sleep 1

        still_alive=""
        for pid in $remaining; do
            if kill -0 "$pid" 2>/dev/null; then
                still_alive="$still_alive $pid"
            fi
        done
        still_alive="${still_alive# }"

        if [ -n "$still_alive" ]; then
            echo "STATUS:FAILED:$still_alive"
            exit 1
        fi
        echo "STATUS:KILLED"
REMOTE_SCRIPT
    script="${script//__PATTERN__/$pattern}"
    script="${script//__KILL_TIMEOUT__/$kill_timeout}"
    script="${script//__DRY_RUN__/$dry_run}"
    printf '%s' "$script"
}

# ------------------------------------------
# 结果解析与日志
# ------------------------------------------
_parse_kill_status() {
    local output="$1" exit_code="$2"
    if [[ "$output" == *"STATUS:NO_PROCESSES"* ]]; then
        echo "no_processes"
    elif [[ "$output" == *"STATUS:TERMINATED"* ]]; then
        echo "success"
    elif [[ "$output" == *"STATUS:KILLED"* ]]; then
        echo "killed"
    elif [[ "$output" == *"STATUS:FAILED"* ]]; then
        echo "failed"
    elif [[ $exit_code -eq 124 ]]; then
        echo "timeout"
    else
        echo "failed"
    fi
}

_log_kill_status() {
    local node="$1" status="$2" pids="$3" quiet="$4"

    case $status in
        no_processes)
            [[ "$quiet" == false ]] && log_info "[Node: $node] 未找到匹配的进程"
            return 0
            ;;
        success)
            [[ "$quiet" == false ]] && log_info "[Node: $node] 进程已正常终止 (PIDs: $pids)"
            return 0
            ;;
        killed)
            log_warn "[Node: $node] 进程已强制终止 (PIDs: $pids)"
            return 0
            ;;
        timeout)
            log_err "[Node: $node] SSH 连接超时 (${SSH_TIMEOUT}s)"
            return 124
            ;;
        failed)
            log_err "[Node: $node] 无法终止所有进程 (PIDs: $pids)"
            return 1
            ;;
    esac
}

_parse_and_log_kill_result() {
    local node="$1" output="$2" exit_code="$3" quiet="$4"
    local status pids=""

    status=$(_parse_kill_status "$output" "$exit_code")

    if [[ "$output" =~ PIDS:([^[:space:]]+) ]]; then
        pids="${BASH_REMATCH[1]}"
    fi

    _log_kill_status "$node" "$status" "$pids" "$quiet"
}

kill_processes_on_node() {
    local node="$1" dry_run="${2:-false}" quiet="${3:-false}"

    [[ "$quiet" == false ]] && log_info "[Node: $node] 开始检查进程..."

    local pattern
    pattern=$(_build_kill_pattern)

    local remote_cmd
    remote_cmd=$(_gen_kill_remote_script "$pattern" "$KILL_TIMEOUT" "$dry_run")

    local output exit_code=0
    output=$(ssh_run_with_timeout "$node" "$remote_cmd" 2>&1) || exit_code=$?

    _parse_and_log_kill_result "$node" "$output" "$exit_code" "$quiet"
}

# ------------------------------------------
# 用户确认
# ------------------------------------------
confirm_operation() {
    local skip="$1" dry="$2"
    if $skip || $dry; then
        $dry || log_info "跳过确认步骤 (-y 模式)"
        return 0
    fi

    echo "================================================================"
    echo "警告: 此脚本将终止以下节点上的指定进程"
    echo "   目标关键词: ${KEYWORDS[*]}"
    echo "   此操作不可恢复，可能会中断正在运行的任务"
    echo "----------------------------------------------------------------"
    echo "待处理节点:"
    for node in "${NODES[@]}"; do
        echo "  - $node"
    done
    echo "----------------------------------------------------------------"
    read -r -p "输入 'yes' 继续，或其他内容取消: " user_confirm

    if [[ "$user_confirm" != "yes" ]]; then
        log_info "已取消操作，未做任何更改"
        exit 0
    fi
    echo "================================================================"
    log_info "确认继续，开始清理..."
}

# ------------------------------------------
# 结果汇总
# ------------------------------------------
print_summary() {
    local success="$1" failed="$2" timeout="$3"

    echo "================================================================"
    log_info "所有节点处理完成"
    echo "----------------------------------------------------------------"
    echo "  成功:     $success"
    echo "  失败:     $failed"
    echo "  超时:     $timeout"
    echo "----------------------------------------------------------------"

    if [[ ${#FAILED_NODES[@]} -gt 0 ]]; then
        echo "失败的节点:"
        printf '  - %s\n' "${FAILED_NODES[@]}"
    fi

    if [[ ${#TIMEOUT_NODES[@]} -gt 0 ]]; then
        echo "超时的节点:"
        printf '  - %s\n' "${TIMEOUT_NODES[@]}"
    fi
}

# ------------------------------------------
# 主逻辑
# ------------------------------------------
parse_args "$@"

# 检查节点列表文件
if [[ ! -f "$NODE_LIST_FILE" ]]; then
    log_err "节点列表文件未找到: $NODE_LIST_FILE"
    exit 1
fi

# 读取节点列表
NODES=()
while IFS= read -r line; do
    NODES+=("$line")
done < <(read_nodes "$NODE_LIST_FILE")

if [[ ${#NODES[@]} -eq 0 ]]; then
    log_err "节点列表为空: $NODE_LIST_FILE"
    exit 1
fi

# 输出配置信息
log_info "开始多节点进程清理..."
$DRY_RUN && log_info "[DRY RUN 模式] 不会实际终止进程"
log_info "目标关键词: ${KEYWORDS[*]}"
log_info "节点列表文件: $NODE_LIST_FILE"
log_info "节点数量: ${#NODES[@]}"
log_info "最大并发数: $MAX_JOBS"
log_info "终止超时: ${KILL_TIMEOUT}s"
log_info "SSH 超时: ${SSH_TIMEOUT}s"

# 用户确认
confirm_operation "$SKIP_CONFIRM" "$DRY_RUN"

# 并发处理所有节点
declare -i SUCCESS_COUNT=0
 declare -i FAILED_COUNT=0
 declare -i TIMEOUT_COUNT=0

TMP_LOG_DIR=$(mktemp -d "${TMPDIR:-/tmp}/kill_nodes_$$.XXXXXX")
trap 'rm -rf "$TMP_LOG_DIR"' EXIT

for node in "${NODES[@]}"; do
    [[ -z "$node" ]] && continue
    limit_jobs "$MAX_JOBS"

    local_log="${TMP_LOG_DIR}/${node}.log"
    (
        set +e
        kill_processes_on_node "$node" "$DRY_RUN" "$QUIET"
        exit_code=$?
        if [[ $exit_code -eq 0 ]]; then
            echo "RESULT:$node:success" > "$local_log"
        elif [[ $exit_code -eq 124 ]]; then
            echo "RESULT:$node:timeout" > "$local_log"
        else
            echo "RESULT:$node:failed" > "$local_log"
        fi
    ) &
done

set +e
wait
set -e

# 统计结果
if [[ -d "$TMP_LOG_DIR" ]]; then
    for log_file in "$TMP_LOG_DIR"/*.log; do
        [[ -f "$log_file" ]] || continue
        while IFS=: read -r _ node status; do
            case $status in
                success) SUCCESS_COUNT=$((SUCCESS_COUNT + 1)) ;;
                timeout) TIMEOUT_COUNT=$((TIMEOUT_COUNT + 1)); TIMEOUT_NODES+=("$node") ;;
                failed) FAILED_COUNT=$((FAILED_COUNT + 1)); FAILED_NODES+=("$node") ;;
            esac
        done < "$log_file"
    done
fi

# 输出汇总
print_summary "$SUCCESS_COUNT" "$FAILED_COUNT" "$TIMEOUT_COUNT"

# 根据结果返回退出码
if [[ $FAILED_COUNT -eq 0 && $TIMEOUT_COUNT -eq 0 ]]; then
    exit 0
else
    exit 1
fi
