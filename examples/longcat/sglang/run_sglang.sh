#!/bin/bash
#=============================================================================
# LongCat-Flash-Chat SGLang 集群部署脚本
#=============================================================================
# 假设 Docker 容器已在各节点创建好 (--network host)，本脚本通过 SSH 进入
# 各节点容器执行 SGLang 启动命令。
#
# 容器已挂载 /home/jianzhnie/llmtuner/llm/EasyInfer，容器内路径与 host 一致，
# 无需 scp 分发文件。
#
# 用法:
#   bash run_sglang.sh                    # 部署并启动 SGLang 服务
#   bash run_sglang.sh --stop             # 停止所有节点的 SGLang 服务
#   bash run_sglang.sh --status           # 查看各节点服务状态
#   bash run_sglang.sh --logs [N]         # 查看主节点日志 (N=行数, 默认 50)
#   MODEL_PATH=/path/to/model bash run_sglang.sh
#   NODES_FILE=/path/to/nodes.txt TP_SIZE=32 bash run_sglang.sh
#   SGLANG_EXTRA_ARGS="--log-level debug" bash run_sglang.sh
#
# 环境变量 (均可外部覆盖):
#   NODES_FILE          节点列表文件路径
#   MODEL_PATH          模型路径
#   CONTAINER_NAME      Docker 容器名称 (默认: sglang-ascend-env)
#   TP_SIZE             张量并行大小 (默认: 64)
#   SERVER_HOST         服务监听地址 (默认: 0.0.0.0)
#   SERVER_PORT         服务端口 (默认: 6677)
#   MASTER_PORT         通信端口 (默认: 5000)
#   SERVED_MODEL_NAME   服务模型名 (默认: longcat-flash)
#   MEM_FRACTION        显存占用比例 (默认: 0.65)
#   MAX_RUNNING         最大并发请求 (默认: 16)
#   CONTEXT_LENGTH      上下文长度 (默认: 8192)
#   CHUNKED_PREFILL     Chunked Prefill 大小 (默认: 8192)
#   WATCHDOG_TIMEOUT    Watchdog 超时秒数 (默认: 9000)
#   SGLANG_PYTHONPATH   SGLang Python 源码路径
#   SGLANG_EXTRA_ARGS   SGLang 额外命令行参数 (空格分隔)
#   HCCL_SOCKET_IFNAME  HCCL 网络接口名 (默认: enp66s0f0)
#   GLOO_SOCKET_IFNAME  GLOO 网络接口名 (默认: enp66s0f0)
#=============================================================================
set -euo pipefail

# ─── 路径与依赖 ──────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EASYINFER_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
# shellcheck source=../../../scripts/common.sh
source "${EASYINFER_ROOT}/scripts/common.sh"

# ════════════════════════════════════════════════════════════════════════════
# 配置区
# ════════════════════════════════════════════════════════════════════════════

# ─── 节点与模型 ────────────────────────────────────────────────────────────
NODES_FILE="${NODES_FILE:-${EASYINFER_ROOT}/node_list1.txt}"
MODEL_PATH="${MODEL_PATH:-/home/jianzhnie/llmtuner/hfhub/models/meituan-longcat/LongCat-Flash-Chat}"

# ─── Docker ────────────────────────────────────────────────────────────────
CONTAINER_NAME="${CONTAINER_NAME:-sglang-ascend-env}"

# ─── 并行与网络 ────────────────────────────────────────────────────────────
TP_SIZE="${TP_SIZE:-64}"
SERVER_HOST="${SERVER_HOST:-0.0.0.0}"
SERVER_PORT="${SERVER_PORT:-6677}"
MASTER_PORT="${MASTER_PORT:-5000}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-longcat-flash}"

# ─── 显存与调度 ────────────────────────────────────────────────────────────
MEM_FRACTION="${MEM_FRACTION:-0.65}"
MAX_RUNNING="${MAX_RUNNING:-16}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-8192}"
CHUNKED_PREFILL="${CHUNKED_PREFILL:-8192}"
WATCHDOG_TIMEOUT="${WATCHDOG_TIMEOUT:-9000}"

# ─── SGLang 环境 ───────────────────────────────────────────────────────────
SGLANG_PYTHONPATH="${SGLANG_PYTHONPATH:-/home/jianzhnie/llmtuner/llm/sglang/python}"
SGLANG_EXTRA_ARGS="${SGLANG_EXTRA_ARGS:-}"

# ─── 日志 ──────────────────────────────────────────────────────────────────
# 统一日志路径: 容器内与 host 一致，使用 /tmp 避免权限问题
SGLANG_LOG_DIR="${SGLANG_LOG_DIR:-EASYINFER_ROOT}"
SGLANG_LOG_FILE="${SGLANG_LOG_DIR}/sglang_$(basename "${SERVED_MODEL_NAME}").log"

# ─── 网络接口 (export 给 HCCL/GLOO) ────────────────────────────────────────
export HCCL_SOCKET_IFNAME="${HCCL_SOCKET_IFNAME:-enp66s0f0}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-enp66s0f0}"

# ─── SSH 选项 ──────────────────────────────────────────────────────────────
# SC2086: 必须不分词以传递多个 SSH 选项
: "${SSH_OPTS:=-o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10}"

# ════════════════════════════════════════════════════════════════════════════
# 帮助信息
# ════════════════════════════════════════════════════════════════════════════
usage() {
    cat <<'USAGE'
Usage:
  bash run_sglang.sh [OPTIONS]

Options:
  -h, --help                显示帮助信息
  -f, --file <FILE>         节点列表文件路径
  -m, --model-path <PATH>   模型路径
  --stop                    停止所有节点的 SGLang 服务
  --status                  查看各节点服务运行状态
  --logs [N]                查看主节点日志 (默认最后 50 行)
  --restart                 先停止再启动服务

环境变量:
  NODES_FILE, MODEL_PATH, CONTAINER_NAME
  TP_SIZE, SERVER_HOST, SERVER_PORT, MASTER_PORT, SERVED_MODEL_NAME
  MEM_FRACTION, MAX_RUNNING, CONTEXT_LENGTH
  CHUNKED_PREFILL, WATCHDOG_TIMEOUT
  SGLANG_PYTHONPATH, SGLANG_EXTRA_ARGS
  HCCL_SOCKET_IFNAME, GLOO_SOCKET_IFNAME
USAGE
}

# ════════════════════════════════════════════════════════════════════════════
# 参数解析
# ════════════════════════════════════════════════════════════════════════════
parse_args() {
    ACTION="start"
    LOG_LINES=50

    while [[ $# -gt 0 ]]; do
        case "$1" in
            -h|--help) usage; exit 0 ;;
            -f|--file)
                NODES_FILE="${2:?错误: $1 需要一个参数}"
                shift 2 ;;
            -m|--model-path)
                MODEL_PATH="${2:?错误: $1 需要一个参数}"
                shift 2 ;;
            --stop) ACTION="stop"; shift ;;
            --status) ACTION="status"; shift ;;
            --logs)
                if [[ -n "${2:-}" && "$2" != -* ]]; then
                    LOG_LINES="$2"
                    shift 2
                else
                    LOG_LINES=50; shift
                fi ;;
            --restart) ACTION="restart"; shift ;;
            *) log_err "未知参数: $1"; usage; exit 2 ;;
        esac
    done
}

# ════════════════════════════════════════════════════════════════════════════
# 前置校验
# ════════════════════════════════════════════════════════════════════════════
validate_config() {
    [[ -f "$NODES_FILE" ]] || log_fatal "节点列表文件未找到: $NODES_FILE"
    [[ -d "$MODEL_PATH" ]]   || log_fatal "模型路径不存在: $MODEL_PATH"

    NODES=()
    while IFS= read -r line; do
        [[ -n "$line" ]] && NODES+=("$line")
    done < <(read_nodes "$NODES_FILE")

    NNODES=${#NODES[@]}
    [[ "$NNODES" -gt 0 ]] || log_fatal "节点列表为空: $NODES_FILE"

    MASTER_NODE="${NODES[0]}"

    # TP_SIZE 与节点数的合理性检查
    local expected_tp=$(( NNODES * 8 ))
    if [[ "$TP_SIZE" != "$expected_tp" ]]; then
        log_warn "TP_SIZE (${TP_SIZE}) 与节点数 (${NNODES} 节点 × 8 卡 = ${expected_tp}) 不匹配"
        log_warn "请确认这是预期的配置，或设置 TP_SIZE=${expected_tp}"
    fi

    # SGLANG_EXTRA_ARGS 安全检查: 拒绝 shell 元字符，防止命令注入
    if [[ -n "$SGLANG_EXTRA_ARGS" ]] && [[ "$SGLANG_EXTRA_ARGS" =~ [][\;\&\|\$\(\)\{\}\<\>\n\r] ]]; then
        log_fatal "SGLANG_EXTRA_ARGS 包含不安全的 shell 元字符，仅允许字母数字、空格和 - _ . / : = ,"
    fi
}

# ════════════════════════════════════════════════════════════════════════════
# 远程执行
# ════════════════════════════════════════════════════════════════════════════

# 通过 SSH 在容器内后台执行完整脚本 (base64 编码，避免引号嵌套)
remote_exec_script() {
    local node="$1" script="$2" b64
    b64=$(printf '%s' "$script" | base64 | tr -d '\n')

    # shellcheck disable=SC2086,SC2029
    ssh ${SSH_OPTS} "$node" \
        "docker exec -d -i '${CONTAINER_NAME}' bash -c \"echo '${b64}' | base64 -d | bash\""
}

# 通过 SSH 在容器内执行简单命令 (返回输出)
remote_exec_cmd() {
    local node="$1" cmd="$2"

    # shellcheck disable=SC2086,SC2029
    ssh ${SSH_OPTS} "$node" \
        "docker exec '${CONTAINER_NAME}' bash -c \"$cmd\""
}

# ════════════════════════════════════════════════════════════════════════════
# 容器内脚本片段
#
# 每个函数输出一段 bash 脚本，由 build_launch_cmd() 组装后 base64 编码传入容器。
#
# Heredoc 约定:
#   <<'DELIM'  — 引用型，内容原样输出（纯容器代码，无管理节点变量）
#   <<DELIM    — 非引用型，管理节点 ${VAR} 在此展开，容器内变量需 \$ 转义
# ════════════════════════════════════════════════════════════════════════════

# ─── 脚本头 + 日志函数 ─────────────────────────────────────────────────────
fragment_preamble() {
    cat <<'FRAG_PREAMBLE'
set -euo pipefail

# ─── 日志函数 ────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
_log() {
    local c="$1" l="$2"; shift 2
    printf "${c}[%-5s]${NC} %s - %s\n" "$l" "$(date '+%F %T')" "$*"
}
log_info()  { _log "$GREEN"  "INFO"  "$@"; }
log_warn()  { _log "$YELLOW" "WARN"  "$@" >&2; }
log_fatal() { _log "$RED"    "FATAL" "$@" >&2; exit 1; }
FRAG_PREAMBLE
}

# ─── 系统优化 ──────────────────────────────────────────────────────────────
fragment_sys_tuning() {
    cat <<'FRAG_SYS'
# ─── 系统优化 ────────────────────────────────────────────────────────────────
log_info "Applying system optimization..."
echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor 2>/dev/null || true
sysctl -w vm.swappiness=0           2>/dev/null || true
sysctl -w kernel.numa_balancing=0   2>/dev/null || true
sysctl -w kernel.sched_migration_cost_ns=50000 2>/dev/null || true
FRAG_SYS
}

# ─── NPU 环境变量 (管理节点变量在此展开) ──────────────────────────────────────
fragment_npu_env() {
    cat <<FRAG_NPU_ENV
# ─── NPU 环境变量 ────────────────────────────────────────────────────────────
export SGLANG_SET_CPU_AFFINITY=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export STREAMS_PER_DEVICE=32
export SGLANG_DEEPEP_BF16_DISPATCH=1
export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_BUFFSIZE=2048
export MOE_ENABLE_TOPK_NEG_ONE=1
export TRANSFORMERS_VERBOSITY=error
export HCCL_SOCKET_IFNAME=${HCCL_SOCKET_IFNAME}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME}
export PYTHONPATH=${SGLANG_PYTHONPATH}:\${PYTHONPATH:-}
export SERVER_HOST=${SERVER_HOST}
FRAG_NPU_ENV
}

# ─── CANN 环境 ─────────────────────────────────────────────────────────────
fragment_cann_setup() {
    cat <<'FRAG_CANN'
# ─── CANN 环境 (set +u 避免 CANN 脚本中未设置变量导致退出) ────────────────────
set +u
if [[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]]; then
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi
set -u
FRAG_CANN
}

# ─── SGLang 启动 (管理节点变量在此展开) ─────────────────────────────────────
fragment_sglang_launch() {
    local node_rank="$1"
    local master_addr="$2"

    cat <<FRAG_LAUNCH
# ─── SGLang 启动 ────────────────────────────────────────────────────────────
log_info "============================================"
log_info " LongCat-Flash-Chat SGLang Worker"
log_info " 模型路径:       ${MODEL_PATH}"
log_info " 节点数:         ${NNODES}"
log_info " TP 大小:        ${TP_SIZE}"
log_info " 主节点地址:     ${master_addr}"
log_info " 服务地址:       ${SERVER_HOST}:${SERVER_PORT}"
log_info " 当前节点 Rank:  ${node_rank}/$((NNODES - 1))"
log_info "============================================"

log_info "Starting SGLang server..."
nohup python -m sglang.launch_server \
    --trust-remote-code \
    --model-path          "${MODEL_PATH}" \
    --served-model-name   "${SERVED_MODEL_NAME}" \
    --host                "${SERVER_HOST}" \
    --port                "${SERVER_PORT}" \
    --nnodes              "${NNODES}" \
    --node-rank           "${node_rank}" \
    --dist-init-addr      "${master_addr}" \
    --tp-size             "${TP_SIZE}" \
    --mem-fraction-static "${MEM_FRACTION}" \
    --attention-backend   ascend \
    --device              npu \
    --max-running-requests "${MAX_RUNNING}" \
    --context-length       "${CONTEXT_LENGTH}" \
    --disable-radix-cache \
    --chunked-prefill-size "${CHUNKED_PREFILL}" \
    --watchdog-timeout     "${WATCHDOG_TIMEOUT}" \
    --prefill-round-robin-balance \
    --moe-a2a-backend     deepep \
    --deepep-mode         auto \
    ${SGLANG_EXTRA_ARGS:-} \
    > "${SGLANG_LOG_FILE}" 2>&1 &

log_info "SGLang server started in background (PID: \$!)"
log_info "模型加载通常需要 10-20 分钟，查看进度: tail -f ${SGLANG_LOG_FILE}"
FRAG_LAUNCH
}

# ════════════════════════════════════════════════════════════════════════════
# 组装完整容器脚本
# ════════════════════════════════════════════════════════════════════════════
build_launch_cmd() {
    local node_rank="$1"
    local master_addr="${NODES[0]}:${MASTER_PORT}"

    fragment_preamble
    echo ""
    fragment_sys_tuning
    echo ""
    fragment_npu_env
    echo ""
    fragment_cann_setup
    echo ""
    fragment_sglang_launch "$node_rank" "$master_addr"
}

# ════════════════════════════════════════════════════════════════════════════
# 操作: 停止服务
# ════════════════════════════════════════════════════════════════════════════
stop_service() {
    log_info "============================================"
    log_info " 停止 SGLang 服务"
    log_info "============================================"

    local node
    for node in "${NODES[@]}"; do
        log_info "停止 ${node} ..."
        # 使用双引号包裹命令，避免单引号嵌套问题
        remote_exec_cmd "$node" \
            "pkill -f sglang.launch_server 2>/dev/null; sleep 2; pkill -9 -f sglang.launch_server 2>/dev/null || true" \
            2>/dev/null || true
    done

    log_info "所有节点的 SGLang 服务已停止"
}

# ════════════════════════════════════════════════════════════════════════════
# 操作: 查看状态
# ════════════════════════════════════════════════════════════════════════════
show_status() {
    log_info "============================================"
    log_info " SGLang 服务状态"
    log_info "============================================"

    local node
    for node in "${NODES[@]}"; do
        local pid_count
        pid_count=$(remote_exec_cmd "$node" \
            "ps aux | grep sglang.launch_server | grep -v grep | wc -l" 2>/dev/null || echo "0")
        if [[ "$pid_count" -gt 0 ]]; then
            log_info "  ${node}: 运行中 (${pid_count} 个进程)"
        else
            log_warn "  ${node}: 未运行"
        fi
    done
}

# ════════════════════════════════════════════════════════════════════════════
# 操作: 查看日志
# ════════════════════════════════════════════════════════════════════════════
show_logs() {
    local lines="${1:-50}"
    log_info "============================================"
    log_info " 主节点日志 (最后 ${lines} 行)"
    log_info " 文件: ${SGLANG_LOG_FILE}"
    log_info "============================================"

    remote_exec_cmd "$MASTER_NODE" \
        "tail -n ${lines} '${SGLANG_LOG_FILE}' 2>/dev/null || echo '日志文件不存在'"
}

# ════════════════════════════════════════════════════════════════════════════
# 操作: 部署服务
# ════════════════════════════════════════════════════════════════════════════
deploy_service() {
    log_info "============================================"
    log_info " LongCat-Flash-Chat SGLang Deployment"
    log_info "============================================"
    log_info " 节点列表:       ${NODES_FILE}"
    log_info " 节点数:         ${NNODES}"
    log_info " 主节点:         ${MASTER_NODE}"
    log_info " 模型路径:       ${MODEL_PATH}"
    log_info " TP 大小:        ${TP_SIZE}"
    log_info " 服务端口:       ${SERVER_PORT}"
    log_info " 容器名称:       ${CONTAINER_NAME}"
    log_info " 日志文件:       ${SGLANG_LOG_FILE}"
    log_info "============================================"

    # 逐节点启动 (rank 0 必须先就绪, torch.distributed 要求)
    local node launch_cmd
    for i in "${!NODES[@]}"; do
        node="${NODES[$i]}"
        log_info "在 ${node} 上启动 SGLang (rank ${i}/$((NNODES - 1)))..."

        launch_cmd=$(build_launch_cmd "$i")
        if ! remote_exec_script "$node" "$launch_cmd"; then
            log_err "在 ${node} 上启动 SGLang 失败，请检查容器日志"
        fi
    done

    log_info "所有节点的 SGLang 启动命令已发送"
    log_info "等待服务就绪..."

    # 等待主节点端口就绪
    wait_for_port "$MASTER_NODE" "$SERVER_PORT" 300 5 || {
        log_warn "服务可能尚未完全就绪，请稍后手动检查"
    }

    log_info "============================================"
    log_info " 部署完成!"
    log_info " 主节点:   ${MASTER_NODE}:${SERVER_PORT}"
    log_info " 健康检查: http://${MASTER_NODE}:${SERVER_PORT}/health"
    log_info " 模型列表: http://${MASTER_NODE}:${SERVER_PORT}/v1/models"
    log_info " 查看日志: bash run_sglang.sh --logs"
    log_info "============================================"
}

# ════════════════════════════════════════════════════════════════════════════
# 操作: 重启服务
# ════════════════════════════════════════════════════════════════════════════
restart_service() {
    stop_service
    sleep 3
    deploy_service
}

# ════════════════════════════════════════════════════════════════════════════
# 主入口
# ════════════════════════════════════════════════════════════════════════════
main() {
    parse_args "$@"
    validate_config

    case "$ACTION" in
        start)   deploy_service ;;
        stop)    stop_service   ;;
        status)  show_status    ;;
        logs)    show_logs "$LOG_LINES" ;;
        restart) restart_service ;;
    esac
}

main "$@"
