#!/usr/bin/env bash
# ==============================================================================
# DeepSeek-V3.2 Multi-Node Deployment Script for Ascend NPU
# ==============================================================================
# 基于 https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/DeepSeek-V3.2.html
#
# 功能:
#   1. 自动读取 node_list.txt 获取多节点列表
#   2. 支持 A2 系列 Ascend NPU 的标准多节点部署
#   3. 自动配置网络环境变量 (HCCL/GLOO/TP 网卡绑定)
#   4. 参考 vllm_model_server.sh 的命令构建方式, 清晰模块化
#   5. 通过 SSH 在各节点并行启动 vllm-ascend 服务
#
# 配置方式 (全部通过环境变量, 无命令行参数):
#   export NIC_NAME=enp66s0f0        # 业务网卡名称 (default: enp66s0f0)
#   export MODEL_PATH=/path/to/model # 模型权重路径
#   export VLLM_PORT=8077            # vLLM 服务端口
#   export DP_RPC_PORT=12890         # Data Parallel RPC 端口
#   export DRY_RUN=1                 # 只打印命令, 不实际执行
#   export SKIP_ENV_CHECK=1          # 跳过 SSH 连通性检查
#
# 使用方法:
#   ./deploy_vllm_multinode.sh
#
# 前置条件:
#   - 各节点之间 SSH 免密登录已配置
#   - 模型权重已下载到共享目录或各节点本地相同路径
#   - vllm-ascend 环境已准备 (Docker 或源码安装)
# ==============================================================================

set -euo pipefail

# ------------------------------------------------------------------------------
# 1. 默认值与常量
# ------------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NODE_LIST_FILE="${SCRIPT_DIR}/node_list.txt"
SSH_OPTS="-o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10"
AUTO_DETECT_FLAGS="${AUTO_DETECT_FLAGS:-1}"

# 部署配置 (可通过环境变量覆盖)
NIC_NAME="${NIC_NAME:-enp66s0f0}"
DRY_RUN="${DRY_RUN:-false}"
SKIP_ENV_CHECK="${SKIP_ENV_CHECK:-false}"

# vLLM 模型与推理配置 (可通过环境变量覆盖)
export MODEL_PATH="${MODEL_PATH:-/llm_workspace_1P/robin/hfhub/models/deepseek-ai/DeepSeek-V3.1}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-deepseek_v3_1}"
export VLLM_PORT="${VLLM_PORT:-8077}"
export DP_RPC_PORT="${DP_RPC_PORT:-12890}"

# 分布式并行配置 (参考 vllm_model_server.sh)
export TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-8}"
export PIPELINE_PARALLEL_SIZE="${PIPELINE_PARALLEL_SIZE:-8}"
export DISTRIBUTED_EXECUTOR_BACKEND="${DISTRIBUTED_EXECUTOR_BACKEND:-ray}"
export ENABLE_EXPERT_PARALLEL="${ENABLE_EXPERT_PARALLEL:-1}"
export EXPERT_PARALLEL_SIZE="${EXPERT_PARALLEL_SIZE:-$((TENSOR_PARALLEL_SIZE * PIPELINE_PARALLEL_SIZE))}"

# A2 每节点 8 卡
NPUS_PER_NODE=8
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.92}"
export PREFIX_CACHING="${PREFIX_CACHING:-0}"
export ENABLE_CHUNKED_PREFILL="${ENABLE_CHUNKED_PREFILL:-0}"

# 颜色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'

log_info()  { echo -e "${GREEN}[INFO]${NC}  $(date '+%Y-%m-%d %H:%M:%S') - $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $(date '+%Y-%m-%d %H:%M:%S') - $*" >&2; }
log_error() { echo -e "${RED}[ERROR]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $*" >&2; }
log_fatal() { echo -e "${RED}[FATAL]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $*" >&2; exit 1; }

# ------------------------------------------------------------------------------
# 2. 读取节点列表
# ------------------------------------------------------------------------------
if [[ ! -f "${NODE_LIST_FILE}" ]]; then
    log_fatal "Node list file not found: ${NODE_LIST_FILE}"
fi

ALL_NODES=()
while IFS= read -r line || [[ -n "${line}" ]]; do
    line="$(echo "${line}" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
    [[ -n "${line}" && ! "${line}" =~ ^# ]] && ALL_NODES+=("${line}")
done < "${NODE_LIST_FILE}"
TOTAL_NODES=${#ALL_NODES[@]}

if [[ ${TOTAL_NODES} -lt 2 ]]; then
    log_fatal "Need at least 2 nodes for multi-node deployment, got ${TOTAL_NODES}"
fi

NODE0="${ALL_NODES[0]}"
log_info "Loaded ${TOTAL_NODES} nodes from ${NODE_LIST_FILE}"
log_info "Master node: ${NODE0}"

# ------------------------------------------------------------------------------
# 3. 配置合法性检查
# ------------------------------------------------------------------------------
TOTAL_CARDS=$((TOTAL_NODES * NPUS_PER_NODE))
CARDS_PER_INSTANCE=$((TENSOR_PARALLEL_SIZE * PIPELINE_PARALLEL_SIZE))

if [[ ${CARDS_PER_INSTANCE} -eq 0 ]]; then
    log_fatal "Invalid config: TENSOR_PARALLEL_SIZE * PIPELINE_PARALLEL_SIZE = 0"
fi

if [[ $((TOTAL_CARDS % CARDS_PER_INSTANCE)) -ne 0 ]]; then
    log_fatal "Card mismatch: TOTAL_CARDS (${TOTAL_CARDS}) is not divisible by CARDS_PER_INSTANCE (${CARDS_PER_INSTANCE}). Please adjust TP/PP."
fi

DP_SIZE=$((TOTAL_CARDS / CARDS_PER_INSTANCE))

if [[ ${DP_SIZE} -lt 1 ]]; then
    log_fatal "Invalid config: DP_SIZE (${DP_SIZE}) must be >= 1. Please reduce TP or PP."
fi

if [[ ${CARDS_PER_INSTANCE} -le ${NPUS_PER_NODE} ]]; then
    DP_SIZE_LOCAL=$((NPUS_PER_NODE / CARDS_PER_INSTANCE))
else
    DP_SIZE_LOCAL=1
fi

log_info "Config check passed: TOTAL_CARDS=${TOTAL_CARDS}, TP=${TENSOR_PARALLEL_SIZE}, PP=${PIPELINE_PARALLEL_SIZE}, DP=${DP_SIZE}, DP_LOCAL=${DP_SIZE_LOCAL}"

# ------------------------------------------------------------------------------
# 4. 获取节点 IP
# ------------------------------------------------------------------------------
get_node_ip() {
    local node=$1
    local nic=$2
    local cmd=""
    if command -v ip >/dev/null 2>&1; then
        cmd="ip -4 addr show ${nic} 2>/dev/null | awk '/inet / {print \$2}' | cut -d/ -f1 | head -n 1"
    elif command -v ifconfig >/dev/null 2>&1; then
        cmd="ifconfig ${nic} 2>/dev/null | awk '/inet / {print \$2}' | head -n 1"
    else
        echo ""
        return
    fi

    local result=""
    if [[ "${node}" == "$(hostname -s)" ]] || [[ "${node}" == "$(hostname)" ]]; then
        result=$(eval "${cmd}")
    else
        result=$(ssh ${SSH_OPTS} "${node}" "${cmd}" 2>/dev/null)
    fi
    if [[ -z "${result}" && ( "${DRY_RUN}" == "true" || "${DRY_RUN}" == "1" ) ]]; then
        echo "192.168.1.$((RANDOM % 254 + 1))"
    else
        echo "${result}"
    fi
}

NODE0_IP=$(get_node_ip "${NODE0}" "${NIC_NAME}")
if [[ -z "${NODE0_IP}" ]]; then
    log_fatal "Failed to get IP address for node ${NODE0} on interface ${NIC_NAME}"
fi
log_info "Node0 IP (DP master): ${NODE0_IP}"

# ------------------------------------------------------------------------------
# 5. SSH 连通性检查
# ------------------------------------------------------------------------------
if [[ "${SKIP_ENV_CHECK}" != "true" && "${DRY_RUN}" != "true" && "${DRY_RUN}" != "1" ]]; then
    log_info "Checking SSH connectivity..."
    failed=0
    for node in "${ALL_NODES[@]}"; do
        if ! ssh ${SSH_OPTS} -o ConnectTimeout=5 "${node}" "echo OK" >/dev/null 2>&1; then
            log_error "SSH failed: ${node}"
            failed=1
        fi
    done
    [[ ${failed} -eq 0 ]] || log_fatal "SSH connectivity check failed"
    log_info "All nodes are reachable via SSH"
fi

# ------------------------------------------------------------------------------
# 6. vLLM 参数探测函数 (参考 vllm_model_server.sh)
# ------------------------------------------------------------------------------
vllm_help() {
    vllm serve --help 2>/dev/null || true
}

choose_flag() {
    local help_text="$1"
    local preferred="$2"
    local fallback="$3"
    if [[ -n "$preferred" && "$help_text" == *"$preferred"* ]]; then
        echo "$preferred"
        return 0
    fi
    if [[ -n "$fallback" && "$help_text" == *"$fallback"* ]]; then
        echo "$fallback"
        return 0
    fi
    echo "$preferred"
}

has_flag() {
    local help_text="$1"
    local flag="$2"
    [[ "$help_text" == *"$flag"* ]]
}

# ------------------------------------------------------------------------------
# 7. 环境变量导出字符串构建
# ------------------------------------------------------------------------------
build_env_exports() {
    local local_ip=$1
    echo "export HCCL_OP_EXPANSION_MODE=AIV"
    echo "export HCCL_IF_IP=${local_ip}"
    echo "export GLOO_SOCKET_IFNAME=${NIC_NAME}"
    echo "export TP_SOCKET_IFNAME=${NIC_NAME}"
    echo "export HCCL_SOCKET_IFNAME=${NIC_NAME}"
    echo "export OMP_PROC_BIND=false"
    echo "export VLLM_USE_V1=1"
    echo "export HCCL_BUFFSIZE=200"
    echo "export VLLM_ASCEND_ENABLE_MLAPO=1"
    echo "export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True"
    echo "export VLLM_ASCEND_ENABLE_FLASHCOMM1=1"
    echo "export OMP_NUM_THREADS=100"
    echo "export HCCL_CONNECT_TIMEOUT=120"
    echo "export HCCL_INTRA_PCIE_ENABLE=1"
    echo "export HCCL_INTRA_ROCE_ENABLE=0"
    # vLLM V1 握手与 RPC 超时放大（大模型加载可能超过 5 分钟默认值）
    echo "export VLLM_V1_FRONTEND_ENGINE_CORE_TIMEOUT=1200"
    echo "export VLLM_RPC_TIMEOUT=600"
}

# ------------------------------------------------------------------------------
# 8. 构建 vLLM 启动参数 (参考 vllm_model_server.sh 的分块构建方式)
# ------------------------------------------------------------------------------
build_vllm_args_declare() {
    local is_headless="$1"
    local node_idx="$2"
    local dp_start_rank="$3"
    local node0_ip="$4"

    local tp_size="${TENSOR_PARALLEL_SIZE}"
    local pp_size="${PIPELINE_PARALLEL_SIZE}"
    local ep_size="${EXPERT_PARALLEL_SIZE}"

    local -a args=()
    args+=(serve "${MODEL_PATH}")
    args+=(--host 0.0.0.0)
    args+=(--port "${VLLM_PORT}")
    args+=(--trust-remote-code)
    args+=(--served-model-name "${SERVED_MODEL_NAME}")
    args+=(--seed 1024)
    args+=(--tensor-parallel-size "${tp_size}")
    args+=(--pipeline-parallel-size "${pp_size}")
    args+=(--data-parallel-size "${DP_SIZE}")
    args+=(--data-parallel-size-local "${DP_SIZE_LOCAL}")
    args+=(--data-parallel-address "${node0_ip}")
    args+=(--data-parallel-rpc-port "${DP_RPC_PORT}")
    args+=(--max-num-seqs "${MAX_NUM_SEQS}")
    args+=(--max-model-len "${MAX_MODEL_LEN}")
    args+=(--max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}")
    args+=(--gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}")
    args+=(--no-enable-prefix-caching)

    if [[ "${is_headless}" == "true" ]]; then
        args+=(--headless)
        args+=(--data-parallel-start-rank "${dp_start_rank}")
    fi

    # A2  compilation-config
    args+=(--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes":[8, 16, 24, 32, 40, 48]}')
    args+=(--additional-config '{"layer_sharding": ["q_b_proj", "o_proj"]}')
    args+=(--speculative-config '{"num_speculative_tokens": 3, "method": "deepseek_mtp"}')

    # 动态探测 Help 信息
    local help_text=""
    if [[ "${AUTO_DETECT_FLAGS}" == "1" ]]; then
        help_text="$(vllm_help)"
    fi

    # Expert Parallel
    if [[ "${ENABLE_EXPERT_PARALLEL}" == "1" ]]; then
        local ep_flag="--enable-expert-parallel"
        if [[ -n "$help_text" ]]; then
            ep_flag="$(choose_flag "$help_text" "--enable-expert-parallel" "--enable_expert_parallel")"
        fi
        args+=("${ep_flag}")
        # 如果 vLLM 版本支持 --expert-parallel-size, 显式设置
        if [[ -n "$help_text" ]]; then
            if has_flag "$help_text" "--expert-parallel-size"; then
                args+=(--expert-parallel-size "${ep_size}")
            fi
        fi
    fi

    # Chunked Prefill
    if [[ "${ENABLE_CHUNKED_PREFILL}" == "1" ]]; then
        if [[ -n "$help_text" ]]; then
            if has_flag "$help_text" "--enable-chunked-prefill"; then
                args+=(--enable-chunked-prefill)
            fi
        else
            args+=(--enable-chunked-prefill)
        fi
    fi

    # Prefix Caching (命令行已经默认加了 --no-enable-prefix-caching, 这里处理启用情况)
    if [[ "${PREFIX_CACHING}" == "1" ]]; then
        # 移除默认的 --no-enable-prefix-caching
        local -a new_args=()
        for a in "${args[@]}"; do
            if [[ "$a" != "--no-enable-prefix-caching" ]]; then
                new_args+=("$a")
            fi
        done
        args=("${new_args[@]}")
        local pc_flag="--enable-prefix-caching"
        if [[ -n "$help_text" ]]; then
            pc_flag="$(choose_flag "$help_text" "--enable-prefix-caching" "--enable_prefix_caching")"
        fi
        args+=("${pc_flag}")
    fi

    # 输出数组定义, 可被远程 bash eval 安全执行
    declare -p args
}

# ------------------------------------------------------------------------------
# 9. 标准多节点部署
# ------------------------------------------------------------------------------
deploy_standard() {
    local tp_size="${TENSOR_PARALLEL_SIZE}"
    local pp_size="${PIPELINE_PARALLEL_SIZE}"
    local ep_size="${EXPERT_PARALLEL_SIZE}"

    log_info "============================================================"
    log_info "Standard Multi-Node Deployment (A2)"
    log_info "Nodes: ${TOTAL_NODES} | DP: ${DP_SIZE} | TP: ${tp_size} | PP: ${pp_size} | EP: ${ep_size}"
    log_info "============================================================"

    local idx=0
    for node in "${ALL_NODES[@]}"; do
        local local_ip
        local_ip=$(get_node_ip "${node}" "${NIC_NAME}")
        [[ -n "${local_ip}" ]] || { log_warn "Skip ${node}: cannot detect IP on ${NIC_NAME}"; continue; }

        local is_headless="false"
        if [[ ${idx} -gt 0 ]]; then
            is_headless="true"
        fi

        local dp_start_rank=$((idx * NPUS_PER_NODE / CARDS_PER_INSTANCE))

        local array_decl
        array_decl=$(build_vllm_args_declare "${is_headless}" "${idx}" "${dp_start_rank}" "${NODE0_IP}")

        local env_exports
        env_exports=$(build_env_exports "${local_ip}")

        # 容器内执行的命令
        local inner_cmd
        inner_cmd="export SCRIPT_DIR='${SCRIPT_DIR}' && cd '${SCRIPT_DIR}' && source set_vlm_env.sh"$'\n'"${env_exports}"$'\n'"${array_decl}"$'\n'"nohup vllm \"\${args[@]}\" > ${SCRIPT_DIR}/vllm_${node}.log 2>&1 &"$'\n'"echo PID:\$!"

        # 远端宿主机命令：进入目录、source 环境变量、然后通过 docker exec 执行容器内命令
        local ssh_cmd
        ssh_cmd="export SCRIPT_DIR='${SCRIPT_DIR}' && cd '${SCRIPT_DIR}' && source set_vlm_env.sh && docker exec -i \"\${CONTAINER_NAME:-vllm-ascend-env-a3}\" bash -s"

        log_info "Launching on ${node} (IP: ${local_ip})..."
        if [[ "${DRY_RUN}" == "true" || "${DRY_RUN}" == "1" ]]; then
            echo "---------- Node: ${node} (host command) ----------"
            echo "${ssh_cmd}"
            echo "---------- Node: ${node} (container inner command) ----------"
            echo "${inner_cmd}"
            echo "-----------------------------------"
        else
            local pid
            pid=$(echo "${inner_cmd}" | ssh ${SSH_OPTS} "${node}" "${ssh_cmd}")
            log_info "Started vLLM on ${node}, PID=${pid}, log=${SCRIPT_DIR}/vllm_${node}.log"
        fi
        idx=$((idx + 1))
    done
}

# ------------------------------------------------------------------------------
# 10. 主流程
# ------------------------------------------------------------------------------
deploy_standard

if [[ "${DRY_RUN}" != "true" && "${DRY_RUN}" != "1" ]]; then
    log_info "============================================================"
    log_info "All vLLM processes launched in background."
    log_info "Check logs: ${SCRIPT_DIR}/vllm_*.log"
    log_info "============================================================"
fi
