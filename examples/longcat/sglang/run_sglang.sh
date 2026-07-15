#!/bin/bash
#=============================================================================
# LongCat-Flash-Chat SGLang 集群部署脚本
#=============================================================================
# 假设 Docker 容器已在各节点创建好，本脚本通过 SSH 进入各节点容器执行 SGLang 启动。
#
# 容器已挂载 /home/jianzhnie/llmtuner/llm/EasyInfer，容器内路径与 host 一致，
# 无需 scp 分发文件。
#
# 用法:
#   bash run_sglang.sh                    # 部署并启动 SGLang 服务
#   bash run_sglang.sh --stop             # 停止所有节点的 SGLang 服务
#   MODEL_PATH=/path/to/model bash run_sglang.sh
#   NODES_FILE=/path/to/nodes.txt TP_SIZE=32 bash run_sglang.sh
#
# 环境变量 (均可外部覆盖):
#   NODES_FILE      节点列表文件路径
#   MODEL_PATH      模型路径
#   TP_SIZE         张量并行大小 (默认: 64)
#   SERVER_PORT     服务端口 (默认: 6677)
#   MASTER_PORT     通信端口 (默认: 5000)
#   SERVED_MODEL_NAME  服务模型名 (默认: longcat-flash)
#   MEM_FRACTION    显存占用比例 (默认: 0.65)
#   MAX_RUNNING     最大并发请求 (默认: 16)
#   CONTEXT_LENGTH  上下文长度 (默认: 8192)
#=============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EASYINFER_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

# --- 加载共享工具函数 ---
source "${EASYINFER_ROOT}/scripts/common.sh"

#=============================================================================
# 配置参数
#=============================================================================
NODES_FILE="${NODES_FILE:-${EASYINFER_ROOT}/node_list1.txt}"
MODEL_PATH="${MODEL_PATH:-/home/jianzhnie/llmtuner/hfhub/models/meituan-longcat/LongCat-Flash-Chat}"

# --- Docker 配置 ---
CONTAINER_NAME="${CONTAINER_NAME:-sglang-ascend-env}"

# --- SGLang 配置 ---
TP_SIZE="${TP_SIZE:-64}"
SERVER_HOST="${SERVER_HOST:-0.0.0.0}"
SERVER_PORT="${SERVER_PORT:-6677}"
MASTER_PORT="${MASTER_PORT:-5000}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-longcat-flash}"
MEM_FRACTION="${MEM_FRACTION:-0.65}"
MAX_RUNNING="${MAX_RUNNING:-16}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-8192}"
CHUNKED_PREFILL="${CHUNKED_PREFILL:-8192}"
WATCHDOG_TIMEOUT="${WATCHDOG_TIMEOUT:-9000}"

# --- 网络接口配置 ---
export HCCL_SOCKET_IFNAME="${HCCL_SOCKET_IFNAME:-enp66s0f0}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-enp66s0f0}"

#=============================================================================
# 帮助信息
#=============================================================================
usage() {
    cat <<'USAGE'
Usage:
  bash run_sglang.sh [OPTIONS]

Options:
  -h, --help                显示帮助信息
  -f, --file <FILE>         节点列表文件路径
  -m, --model-path <PATH>   模型路径
  --stop                    停止所有节点的 SGLang 服务

环境变量:
  NODES_FILE, MODEL_PATH, CONTAINER_NAME
  TP_SIZE, SERVER_PORT, MASTER_PORT, SERVED_MODEL_NAME
USAGE
}

#=============================================================================
# 参数解析
#=============================================================================
ACTION="start"

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
        *) log_err "未知参数: $1"; usage; exit 2 ;;
    esac
done

#=============================================================================
# 读取节点列表
#=============================================================================
[[ -f "$NODES_FILE" ]] || log_fatal "节点列表文件未找到: $NODES_FILE"
[[ -d "$MODEL_PATH" ]] || log_fatal "模型路径不存在: $MODEL_PATH"

NODES=()
while IFS= read -r line; do
    [[ -n "$line" ]] && NODES+=("$line")
done < <(read_nodes "$NODES_FILE")

NNODES=${#NODES[@]}
[[ "$NNODES" -gt 0 ]] || log_fatal "节点列表为空: $NODES_FILE"

MASTER_NODE="${NODES[0]}"

# --- 验证 TP_SIZE 与节点数匹配 ---
EXPECTED_TP=$(( NNODES * 8 ))
if [[ "$TP_SIZE" != "$EXPECTED_TP" ]]; then
    log_warn "TP_SIZE (${TP_SIZE}) 与节点数 (${NNODES} 节点 × 8 卡 = ${EXPECTED_TP}) 不匹配"
    log_warn "请确认这是预期的配置，或设置 TP_SIZE=${EXPECTED_TP}"
fi

#=============================================================================
# 停止服务
#=============================================================================
if [[ "$ACTION" == "stop" ]]; then
    log_info "============================================"
    log_info " 停止 SGLang 服务"
    log_info "============================================"

    for node in "${NODES[@]}"; do
        log_info "停止 ${node} ..."
        ssh ${SSH_OPTS:--o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10} \
            "$node" "docker exec '${CONTAINER_NAME}' pkill -f 'sglang.launch_server' 2>/dev/null; docker exec '${CONTAINER_NAME}' pkill -9 -f 'sglang.launch_server' 2>/dev/null" 2>/dev/null || true
    done

    log_info "所有节点的 SGLang 服务已停止"
    exit 0
fi

#=============================================================================
# 启动服务
#=============================================================================
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
log_info "============================================"

# 构建节点 IP 字符串
NODES_IPS_STR="${NODES[*]}"

for node in "${NODES[@]}"; do
    log_info "在 ${node} 上启动 SGLang..."

    ssh ${SSH_OPTS:--o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10} \
        "$node" "docker exec -d '${CONTAINER_NAME}' bash -c '
            set -euo pipefail

            # --- 日志函数 ---
            RED=\"\033[0;31m\" GREEN=\"\033[0;32m\" YELLOW=\"\033[1;33m\" NC=\"\033[0m\"
            _log() { local c=\"\$1\" l=\"\$2\"; shift 2; printf \"\${c}[%-5s]\${NC} %s - %s\n\" \"\$l\" \"\$(date +%Y-%m-%d %H:%M:%S)\" \"\$*\"; }
            log_info()  { _log \"\$GREEN\"  \"INFO\"  \"\$@\"; }
            log_warn()  { _log \"\$YELLOW\" \"WARN\"  \"\$@\" >&2; }
            log_fatal() { _log \"\$RED\"    \"FATAL\" \"\$@\" >&2; exit 1; }

            # --- 系统优化 ---
            log_info \"应用系统优化...\"
            echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor 2>/dev/null || true
            sysctl -w vm.swappiness=0 2>/dev/null || true
            sysctl -w kernel.numa_balancing=0 2>/dev/null || true
            sysctl -w kernel.sched_migration_cost_ns=50000 2>/dev/null || true

            # --- 环境变量 ---
            export SGLANG_SET_CPU_AFFINITY=1
            export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
            export STREAMS_PER_DEVICE=32
            export SGLANG_DEEPEP_BF16_DISPATCH=1
            export HCCL_OP_EXPANSION_MODE=\"AIV\"
            export HCCL_BUFFSIZE=2048
            export MOE_ENABLE_TOPK_NEG_ONE=1
            export TRANSFORMERS_VERBOSITY=error
            export PYTHONPATH=/home/jianzhnie/llmtuner/llm/sglang/python:\${PYTHONPATH:-}
            export HCCL_SOCKET_IFNAME=${HCCL_SOCKET_IFNAME}
            export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME}

            # --- CANN 环境 ---
            if [[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]]; then
                source /usr/local/Ascend/ascend-toolkit/set_env.sh
            fi

            # --- 读取节点列表 ---
            NODES=()
            read -ra NODES <<< \"${NODES_IPS_STR}\"
            NNODES=\${#NODES[@]}
            [[ \"\$NNODES\" -gt 0 ]] || log_fatal \"节点列表为空\"

            P_IP=(\"\${NODES[@]}\")
            P_MASTER=\"\${P_IP[0]}:${MASTER_PORT}\"

            # --- 确定当前节点 rank ---
            LOCAL_IPS=\"\$(hostname -I 2>/dev/null || true)\"
            NODE_RANK=\"\"
            for i in \"\${!P_IP[@]}\"; do
                if [[ \" \${LOCAL_IPS} \" == *\" \${P_IP[\$i]} \"* ]]; then
                    NODE_RANK=\"\${i}\"
                    break
                fi
            done
            [[ -n \"\${NODE_RANK}\" ]] || log_fatal \"本地 IP [\${LOCAL_IPS}] 未在节点列表中找到\"

            log_info \"============================================\"
            log_info \" LongCat-Flash-Chat SGLang Worker\"
            log_info \" 模型路径:       ${MODEL_PATH}\"
            log_info \" 节点数:         \${NNODES}\"
            log_info \" TP 大小:        ${TP_SIZE}\"
            log_info \" 主节点:         \${P_MASTER}\"
            log_info \" 服务地址:       ${SERVER_HOST}:${SERVER_PORT}\"
            log_info \" 当前节点 Rank:  \${NODE_RANK}\"
            log_info \"============================================\"

            log_info \"启动 SGLang 服务...\"
            exec python -m sglang.launch_server \
                --trust-remote-code \
                --model-path \"${MODEL_PATH}\" \
                --served-model-name \"${SERVED_MODEL_NAME}\" \
                --host \"${SERVER_HOST}\" \
                --port \"${SERVER_PORT}\" \
                --nnodes \"\${NNODES}\" \
                --node-rank \"\${NODE_RANK}\" \
                --dist-init-addr \"\${P_MASTER}\" \
                --tp-size \"${TP_SIZE}\" \
                --mem-fraction-static \"${MEM_FRACTION}\" \
                --attention-backend ascend \
                --device npu \
                --max-running-requests \"${MAX_RUNNING}\" \
                --context-length \"${CONTEXT_LENGTH}\" \
                --disable-radix-cache \
                --chunked-prefill-size \"${CHUNKED_PREFILL}\" \
                --watchdog-timeout \"${WATCHDOG_TIMEOUT}\" \
                --prefill-round-robin-balance \
                --moe-a2a-backend deepep \
                --deepep-mode auto
        '" 2>/dev/null || {
            log_err "在 ${node} 上启动 SGLang 失败"
        }
done

log_info "SGLang 服务启动命令已发送"
log_info "等待服务就绪..."

# 等待主节点服务就绪
if command -v nc >/dev/null 2>&1; then
    wait_for_port "$MASTER_NODE" "$SERVER_PORT" 300 5 || {
        log_warn "服务可能尚未完全就绪，请稍后手动检查"
    }
fi

log_info "============================================"
log_info " 部署完成!"
log_info " 主节点: ${MASTER_NODE}:${SERVER_PORT}"
log_info " 健康检查: http://${MASTER_NODE}:${SERVER_PORT}/health"
log_info " 模型列表: http://${MASTER_NODE}:${SERVER_PORT}/v1/models"
log_info "============================================"
log_info "部署流程结束"
