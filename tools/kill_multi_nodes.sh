#!/usr/bin/env bash
# ==============================================================================
# 多节点进程清理脚本 (kill_multi_nodes.sh)
#
# 该脚本通过 SSH 并发连接到多个节点，根据关键字终止指定的进程。
# 脚本会首先尝试温和地终止进程（SIGTERM），超时后若进程仍存活，则强制终止（SIGKILL）。
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/set_env.sh"

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
NODE_LIST_FILE="${NODES_FILE:-${SCRIPT_DIR}/node_list.txt}"
MAX_JOBS="${MAX_JOBS:-16}"
SSH_TIMEOUT="${SSH_TIMEOUT:-10}"
KILL_TIMEOUT="${KILL_TIMEOUT:-3}"

# 定义要 kill 的关键词（支持正则，可通过环境变量扩展）
# 包含 Ray 各组件：raylet, plasma_store, gcs_server, ray::IDLE, dashboard_agent, runtime_env_agent
# 注意：移除了 "python" 因为它太宽泛，容易误杀系统进程
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
declare -A NODE_STATUS  # 节点处理状态: pending|success|failed|timeout
declare -a FAILED_NODES=()
declare -a TIMEOUT_NODES=()

# ------------------------------------------
# 日志函数
# ------------------------------------------
log_info() { echo "[INFO] $(date '+%Y-%m-%d %H:%M:%S') - $*"; }
log_err()  { echo "[ERROR] $(date '+%Y-%m-%d %H:%M:%S') - $*" >&2; }
log_warn() { echo "[WARN] $(date '+%Y-%m-%d %H:%M:%S') - $*"; }

# ------------------------------------------
# 帮助信息
# ------------------------------------------
usage() {
    cat <<EOF
Usage: $0 [OPTIONS] [node_list_file]

Description:
  通过 SSH 并发连接到多个节点，根据关键字终止指定的进程。
  首先尝试 SIGTERM 温和终止，超时后若进程仍存活则使用 SIGKILL 强制终止。

Arguments:
  node_list_file    节点列表文件路径 (默认: $NODE_LIST_FILE)
                    环境变量 NODES_FILE 也可指定此路径

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
  NODES_FILE             节点列表文件路径
  SSH_OPTS               SSH 连接选项
  SSH_USER_HOST_PREFIX   SSH 目标前缀 (如: user@)
  EXTRA_KEYWORDS         额外的关键词，逗号分隔（追加到默认列表）
  MAX_JOBS               最大并发任务数
  KILL_TIMEOUT           终止进程超时时间
  SSH_TIMEOUT            SSH 连接超时时间

Examples:
  $0                                    # 使用默认配置
  $0 /path/to/nodes.txt                 # 指定节点列表文件
  $0 -y                                 # 跳过确认
  $0 -n                                 # 干运行模式，查看会杀哪些进程
  $0 -k "myapp,worker"                  # 自定义关键词（替换默认）
  $0 -y -k "ray" -t 5                   # 强制模式，只杀 ray 进程，超时 5 秒

EOF
}

# ------------------------------------------
# 信号处理：清理后台作业
# ------------------------------------------
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
# 辅助函数
# ------------------------------------------

# 拼接 SSH 目标地址
ssh_target() {
    local node="$1"
    printf "%s%s" "$SSH_USER_HOST_PREFIX" "$node"
}

# 执行带超时的 SSH 命令
ssh_run_with_timeout() {
    local node="$1"
    shift
    local exit_code=0
    
    # 优先使用 timeout 命令，回退到 Perl 实现
    if command -v timeout >/dev/null 2>&1; then
        timeout "$SSH_TIMEOUT" ssh ${SSH_OPTS} "$(ssh_target "$node")" "$@" || exit_code=$?
    else
        perl -e '
            use strict;
            use warnings;
            my $timeout = shift @ARGV;
            my @cmd = @ARGV;
            eval {
                local $SIG{ALRM} = sub { die "TIMEOUT\n" };
                alarm $timeout;
                system(@cmd);
                alarm 0;
            };
            if ($@ eq "TIMEOUT\n") {
                print STDERR "[ERROR] Command timed out after ${timeout}s\n";
                exit 124;
            }
            exit $? >> 8;
        ' "$SSH_TIMEOUT" ssh ${SSH_OPTS} "$(ssh_target "$node")" "$@" || exit_code=$?
    fi
    
    return $exit_code
}

# 转义正则表达式特殊字符
escape_regex() {
    # 转义 . * + ? ^ $ ( ) [ ] { } | \
    sed 's/[.*+?^${}()|[\]/\\&/g' <<< "$1"
}

# 并发数控制
limit_jobs() {
    local max="$1"
    while [[ "$(jobs -rp 2>/dev/null | wc -l)" -ge "$max" ]]; do
        # 使用 wait -n 等待任意一个作业完成，忽略中断信号
        wait -n 2>/dev/null || true
        sleep 0.1
    done
}

# ------------------------------------------
# 核心函数：终止指定节点上的进程
# ------------------------------------------
kill_processes_on_node() {
    local node=$1
    local dry_run=${2:-false}
    local quiet=${3:-false}
    
    [[ "$quiet" == false ]] && log_info "🔎 [Node: $node] 开始检查进程..."

    # 构建转义后的正则表达式
    local escaped_keywords=()
    local kw
    for kw in "${KEYWORDS[@]}"; do
        escaped_keywords+=("$(escape_regex "$kw")")
    done
    local pattern
    pattern=$(IFS='|'; echo "${escaped_keywords[*]}")

    # 构建远程执行命令
    local remote_cmd
    read -r -d '' remote_cmd << 'REMOTE_SCRIPT'
        set -euo pipefail
        
        PATTERN="__PATTERN__"
        KILL_TIMEOUT="__KILL_TIMEOUT__"
        DRY_RUN="__DRY_RUN__"
        
        # 获取匹配关键词的进程 PID（排除 grep 和常见编辑器/IDE 进程）
        get_matching_pids() {
            ps aux | grep -E "$PATTERN" | \
                grep -v grep | \
                grep -v -E '(vscode-server|code-server|sshd:|/bin/sh -c|extension|/agent/|ssh.*:)' | \
                awk '{print $2}' | sort -u | tr '\n' ' ' || true
        }
        
        # 获取进程详细信息
        get_process_info() {
            local pids="$1"
            for pid in $pids; do
                ps -p "$pid" -o pid,ppid,user,%cpu,%mem,etime,args 2>/dev/null || true
            done
        }
        
        # 主逻辑
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
        
        # 温和终止 (SIGTERM)
        echo "ACTION:SIGTERM"
        kill -15 $all_pids 2>/dev/null || true
        
        # 等待进程退出
        sleep "$KILL_TIMEOUT"
        
        # 检查剩余进程
        remaining=""
        for pid in $all_pids; do
            if kill -0 "$pid" 2>/dev/null; then
                remaining="$remaining $pid"
            fi
        done
        remaining="${remaining# }"
        
        if [ -n "$remaining" ]; then
            echo "ACTION:SIGKILL:$remaining"
            kill -9 $remaining 2>/dev/null || true
            sleep 1
            
            # 最终检查
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
            else
                echo "STATUS:KILLED"
            fi
        else
            echo "STATUS:TERMINATED"
        fi
REMOTE_SCRIPT

    # 替换变量
    remote_cmd="${remote_cmd//__PATTERN__/$pattern}"
    remote_cmd="${remote_cmd//__KILL_TIMEOUT__/$KILL_TIMEOUT}"
    remote_cmd="${remote_cmd//__DRY_RUN__/$dry_run}"

    # 执行远程命令并捕获输出
    local output exit_code=0
    output=$(ssh_run_with_timeout "$node" "$remote_cmd" 2>&1) || exit_code=$?

    # 解析输出
    local status="failed"
    if [[ "$output" == *"STATUS:NO_PROCESSES"* ]]; then
        status="no_processes"
    elif [[ "$output" == *"STATUS:TERMINATED"* ]]; then
        status="success"
    elif [[ "$output" == *"STATUS:KILLED"* ]]; then
        status="killed"
    elif [[ "$output" == *"STATUS:FAILED"* ]]; then
        status="failed"
    elif [[ $exit_code -eq 124 ]]; then
        status="timeout"
    fi

    # 提取进程信息用于显示
    local pids=""
    if [[ "$output" =~ PIDS:([^[:space:]]+) ]]; then
        pids="${BASH_REMATCH[1]}"
    fi

    # 输出结果
    case $status in
        no_processes)
            [[ "$quiet" == false ]] && log_info "ℹ️  [Node: $node] 未找到匹配的进程"
            ;;
        success)
            [[ "$quiet" == false ]] && log_info "✅ [Node: $node] 进程已正常终止 (PIDs: $pids)"
            ;;
        killed)
            log_warn "⚠️  [Node: $node] 进程已强制终止 (PIDs: $pids)"
            ;;
        failed)
            log_err "❌ [Node: $node] 无法终止所有进程 (PIDs: $pids)"
            ;;
        timeout)
            log_err "❌ [Node: $node] SSH 连接超时 (${SSH_TIMEOUT}s)"
            ;;
        *)
            log_err "❌ [Node: $node] 处理失败 (退出码: $exit_code)"
            if [[ -n "$output" ]]; then
                echo "$output" | while read -r line; do
                    log_err "    $line"
                done
            fi
            ;;
    esac

    # 返回状态码
    case $status in
        no_processes|success) return 0 ;;
        killed) return 0 ;;  # 虽然用了 SIGKILL，但目标达成了
        timeout) return 124 ;;
        *) return 1 ;;
    esac
}

# ------------------------------------------
# 主逻辑
# ------------------------------------------

# 参数解析
SKIP_CONFIRM=false
DRY_RUN=false
QUIET=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -y|--yes)
            SKIP_CONFIRM=true
            shift
            ;;
        -n|--dry-run)
            DRY_RUN=true
            shift
            ;;
        -q|--quiet)
            QUIET=true
            shift
            ;;
        -k|--keywords)
            if [[ -z "${2:-}" ]] || [[ "$2" == -* ]]; then
                log_err "选项 $1 需要一个参数"
                exit 1
            fi
            IFS=',' read -ra KEYWORDS <<< "$2"
            shift 2
            ;;
        -t|--timeout)
            if [[ -z "${2:-}" ]] || [[ "$2" == -* ]]; then
                log_err "选项 $1 需要一个参数"
                exit 1
            fi
            if ! [[ "$2" =~ ^[0-9]+$ ]]; then
                log_err "超时时间必须是正整数"
                exit 1
            fi
            KILL_TIMEOUT="$2"
            shift 2
            ;;
        -j|--jobs)
            if [[ -z "${2:-}" ]] || [[ "$2" == -* ]]; then
                log_err "选项 $1 需要一个参数"
                exit 1
            fi
            if ! [[ "$2" =~ ^[0-9]+$ ]] || [[ "$2" -lt 1 ]]; then
                log_err "并发数必须是正整数"
                exit 1
            fi
            MAX_JOBS="$2"
            shift 2
            ;;
        --ssh-timeout)
            if [[ -z "${2:-}" ]] || [[ "$2" == -* ]]; then
                log_err "选项 $1 需要一个参数"
                exit 1
            fi
            if ! [[ "$2" =~ ^[0-9]+$ ]]; then
                log_err "SSH 超时时间必须是正整数"
                exit 1
            fi
            SSH_TIMEOUT="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        -*)
            log_err "未知选项: $1"
            usage >&2
            exit 1
            ;;
        *)
            # 非选项参数，视为节点列表文件
            NODE_LIST_FILE="$1"
            shift
            ;;
    esac
done

# 检查节点列表文件
if [[ ! -f "$NODE_LIST_FILE" ]]; then
    log_err "节点列表文件未找到: $NODE_LIST_FILE"
    exit 1
fi

# 读取节点列表
mapfile -t NODES < <(awk 'NF && !/^#/ {print $1}' "$NODE_LIST_FILE")

if [[ ${#NODES[@]} -eq 0 ]]; then
    log_err "节点列表为空: $NODE_LIST_FILE"
    exit 1
fi

# 输出配置信息
log_info "🚀 开始多节点进程清理..."
$DRY_RUN && log_info "📝 [DRY RUN 模式] 不会实际终止进程"
log_info "目标关键词: ${KEYWORDS[*]}"
log_info "节点列表文件: $NODE_LIST_FILE"
log_info "节点数量: ${#NODES[@]}"
log_info "最大并发数: $MAX_JOBS"
log_info "终止超时: ${KILL_TIMEOUT}s"
log_info "SSH 超时: ${SSH_TIMEOUT}s"

# 用户确认（除非使用 -y 跳过或 dry-run 模式）
if ! $SKIP_CONFIRM && ! $DRY_RUN; then
    echo "================================================================"
    echo "⚠️  警告: 此脚本将终止以下节点上的指定进程"
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
else
    $DRY_RUN || log_info "跳过确认步骤 (-y 模式)"
fi

# 并发处理所有节点
declare -i SUCCESS_COUNT=0
declare -i FAILED_COUNT=0
declare -i TIMEOUT_COUNT=0

# 创建临时目录和文件
TMP_LOG_DIR=$(mktemp -d "${TMPDIR:-/tmp}/kill_nodes_$$.XXXXXX")
export TMP_LOG_DIR

# 为每个节点创建独立的日志文件以避免竞争
for node in "${NODES[@]}"; do
    [[ -z "$node" ]] && continue
    limit_jobs "$MAX_JOBS"
    
    # 使用独立的日志文件避免竞争
    local_log="${TMP_LOG_DIR}/${node}.log"
    
    # 使用子 shell 捕获每个节点的结果
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

# 等待所有后台任务完成（禁用 ERR 陷阱避免被信号中断）
set +e
wait
set -e

# 统计结果
if [[ -d "$TMP_LOG_DIR" ]]; then
    for log_file in "$TMP_LOG_DIR"/*.log; do
        [[ -f "$log_file" ]] || continue
        while IFS=: read -r _ node status; do
            case $status in
                success) ((SUCCESS_COUNT++)) ;;
                timeout) ((TIMEOUT_COUNT++)); TIMEOUT_NODES+=("$node") ;;
                failed) ((FAILED_COUNT++)); FAILED_NODES+=("$node") ;;
            esac
        done < "$log_file"
    done
    rm -rf "$TMP_LOG_DIR"
fi

# 输出汇总
echo "================================================================"
log_info "🎉 所有节点处理完成"
echo "----------------------------------------------------------------"
echo "  成功:     $SUCCESS_COUNT"
echo "  失败:     $FAILED_COUNT"
echo "  超时:     $TIMEOUT_COUNT"
echo "----------------------------------------------------------------"

if [[ ${#FAILED_NODES[@]} -gt 0 ]]; then
    echo "失败的节点:"
    printf '  - %s\n' "${FAILED_NODES[@]}"
fi

if [[ ${#TIMEOUT_NODES[@]} -gt 0 ]]; then
    echo "超时的节点:"
    printf '  - %s\n' "${TIMEOUT_NODES[@]}"
fi

# 根据结果返回退出码
if [[ $FAILED_COUNT -eq 0 && $TIMEOUT_COUNT -eq 0 ]]; then
    exit 0
else
    exit 1
fi
