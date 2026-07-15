#!/bin/bash
#=============================================================================
# LongCat-Flash-Chat SGLang 多节点部署脚本 (容器内独立运行版本)
#=============================================================================
# 用法:
#   bash run_sglang.sh                    # 使用默认配置
#   MODEL_PATH=/path/to/model bash run_sglang.sh
#   NODES_IPS="10.0.0.1 10.0.0.2" TP_SIZE=32 bash run_sglang.sh
#
# 环境变量 (均可外部覆盖):
#   NODES_FILE      节点列表文件路径
#   NODES_IPS       节点 IP 列表 (空格分隔，优先级高于 NODES_FILE)
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

# --- 内嵌日志函数 (不依赖外部 common.sh) ---
RED='\033[0;31m' GREEN='\033[0;32m' YELLOW='\033[1;33m' CYAN='\033[0;36m' NC='\033[0m'
_log() { local c="$1" l="$2"; shift 2; printf "${c}[%-5s]${NC} %s - %s\n" "$l" "$(date '+%Y-%m-%d %H:%M:%S')" "$*"; }
log_info()  { _log "$GREEN"  "INFO"  "$@"; }
log_warn()  { _log "$YELLOW" "WARN"  "$@" >&2; }
log_err()   { _log "$RED"    "ERROR" "$@" >&2; }
log_fatal() { _log "$RED"    "FATAL" "$@" >&2; exit 1; }

# --- 读取节点列表 (兼容文件和环境变量) ---
read_nodes() {
    local file="$1"
    if [[ -f "$file" ]]; then
        awk 'NF && !/^#/ {print $1}' "$file"
    fi
}

#=============================================================================
# 配置参数
#=============================================================================
NODES_FILE="${NODES_FILE:-}"
MODEL_PATH="${MODEL_PATH:-/home/jianzhnie/llmtuner/hfhub/models/meituan-longcat/LongCat-Flash-Chat}"
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
# 前置检查
#=============================================================================
[[ -d "$MODEL_PATH" ]] || log_fatal "模型路径不存在: $MODEL_PATH"

# --- 读取节点列表 ---
NODES=()
if [[ -n "${NODES_IPS:-}" ]]; then
    read -ra NODES <<< "$NODES_IPS"
elif [[ -n "$NODES_FILE" && -f "$NODES_FILE" ]]; then
    while IFS= read -r line; do
        [[ -n "$line" ]] && NODES+=("$line")
    done < <(read_nodes "$NODES_FILE")
else
    log_fatal "未提供节点列表。请设置 NODES_IPS 环境变量或提供有效的 NODES_FILE"
fi

NNODES=${#NODES[@]}
[[ "$NNODES" -gt 0 ]] || log_fatal "节点列表为空"

# --- 验证 TP_SIZE 与节点数匹配 ---
EXPECTED_TP=$(( NNODES * 8 ))
if [[ "$TP_SIZE" != "$EXPECTED_TP" ]]; then
    log_warn "TP_SIZE (${TP_SIZE}) 与节点数 (${NNODES} 节点 × 8 卡 = ${EXPECTED_TP}) 不匹配"
    log_warn "请确认这是预期的配置，或设置 TP_SIZE=${EXPECTED_TP}"
fi

#=============================================================================
# 构建节点 IP 数组
#=============================================================================
P_IP=("${NODES[@]}")
P_MASTER="${P_IP[0]}:${MASTER_PORT}"

#=============================================================================
# 系统优化
#=============================================================================
log_info "应用系统优化..."
echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor 2>/dev/null || true
sysctl -w vm.swappiness=0 2>/dev/null || true
sysctl -w kernel.numa_balancing=0 2>/dev/null || true
sysctl -w kernel.sched_migration_cost_ns=50000 2>/dev/null || true

#=============================================================================
# 环境变量设置
#=============================================================================
export SGLANG_SET_CPU_AFFINITY=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export STREAMS_PER_DEVICE=32
export SGLANG_DEEPEP_BF16_DISPATCH=1
export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_BUFFSIZE=2048
export MOE_ENABLE_TOPK_NEG_ONE=1
export TRANSFORMERS_VERBOSITY=error

# --- Python Path ---
export PYTHONPATH=/home/jianzhnie/llmtuner/llm/sglang/python:${PYTHONPATH:-}

# --- CANN 环境 ---
if [[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]]; then
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi

#=============================================================================
# 确定当前节点 rank
#=============================================================================
LOCAL_IPS="$(hostname -I 2>/dev/null || true)"
NODE_RANK=""

for i in "${!P_IP[@]}"; do
    if [[ " ${LOCAL_IPS} " == *" ${P_IP[$i]} "* ]]; then
        NODE_RANK="${i}"
        break
    fi
done

if [[ -z "${NODE_RANK}" ]]; then
    log_fatal "本地 IP [${LOCAL_IPS}] 未在节点列表中找到: ${P_IP[*]}"
fi

#=============================================================================
# 打印部署信息
#=============================================================================
log_info "============================================"
log_info " LongCat-Flash-Chat SGLang Deployment"
log_info "============================================"
log_info " 模型路径:       ${MODEL_PATH}"
log_info " 节点数:         ${NNODES}"
log_info " TP 大小:        ${TP_SIZE}"
log_info " 主节点:         ${P_MASTER}"
log_info " 服务地址:       ${SERVER_HOST}:${SERVER_PORT}"
log_info " 模型名称:       ${SERVED_MODEL_NAME}"
log_info " 显存占用:       ${MEM_FRACTION}"
log_info " 最大并发:       ${MAX_RUNNING}"
log_info " 上下文长度:     ${CONTEXT_LENGTH}"
log_info " 本地 IP:        ${LOCAL_IPS}"
log_info " 当前节点 Rank:  ${NODE_RANK}"
log_info "============================================"

#=============================================================================
# 启动 SGLang 服务
#=============================================================================
log_info "启动 SGLang 服务..."

exec python -m sglang.launch_server \
    --trust-remote-code \
    --model-path "${MODEL_PATH}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --host "${SERVER_HOST}" \
    --port "${SERVER_PORT}" \
    --nnodes "${NNODES}" \
    --node-rank "${NODE_RANK}" \
    --dist-init-addr "${P_MASTER}" \
    --tp-size "${TP_SIZE}" \
    --mem-fraction-static "${MEM_FRACTION}" \
    --attention-backend ascend \
    --device npu \
    --max-running-requests "${MAX_RUNNING}" \
    --context-length "${CONTEXT_LENGTH}" \
    --disable-radix-cache \
    --chunked-prefill-size "${CHUNKED_PREFILL}" \
    --watchdog-timeout "${WATCHDOG_TIMEOUT}" \
    --prefill-round-robin-balance \
    --moe-a2a-backend deepep \
    --deepep-mode auto
