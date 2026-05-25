#!/bin/bash
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
NODE_LIST_FILE="${SCRIPT_DIR}/../../node_list.txt"
SSH_OPTS="${SSH_OPTS:--o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10}"
AUTO_DETECT_FLAGS="${AUTO_DETECT_FLAGS:-1}"

# 加载共享工具函数
source "${SCRIPT_DIR}/../../common.sh"
source "${SCRIPT_DIR}/_common.sh"

# 使用带 DRY_RUN 回退的 IP 获取函数
get_node_ip() { _get_node_ip_with_fallback "$@"; }

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

# ------------------------------------------------------------------------------
# 2. 读取节点列表
# ------------------------------------------------------------------------------
load_and_validate_nodes "${NODE_LIST_FILE}" 2

# ------------------------------------------------------------------------------
# 3. 配置合法性检查
# ------------------------------------------------------------------------------
validate_parallelism_config "${TOTAL_NODES}" "${NPUS_PER_NODE}"

# ------------------------------------------------------------------------------
# 4. 获取节点 IP
# ------------------------------------------------------------------------------
resolve_node0_ip "${NODE0}" "${NIC_NAME}"
log_info "Node0 IP (DP master): ${NODE0_IP}"

# ------------------------------------------------------------------------------
# 5. SSH 连通性检查
# ------------------------------------------------------------------------------
if [[ "${SKIP_ENV_CHECK}" != "true" && "${DRY_RUN}" != "true" && "${DRY_RUN}" != "1" ]]; then
    check_ssh_connectivity
fi

# ------------------------------------------------------------------------------
# 6. 构建 vLLM 启动参数
# ------------------------------------------------------------------------------
# 辅助函数 _add_ep_args / _add_chunked_prefill_args / _add_prefix_caching_args
# 定义在 _common.sh 中（被 source 的共享库）

# 构建完整的 vLLM 参数数组并输出 declare -p
build_vllm_args_declare() {
    local is_headless="$1"
    # shellcheck disable=SC2034
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

    # A2 编译配置
    args+=(--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture sizes":[8, 16, 24, 32, 40, 48]}')
    args+=(--additional-config '{"layer_sharding": ["q_b_proj", "o_proj"]}')
    args+=(--speculative-config '{"num_speculative_tokens": 3, "method": "deepseek_mtp"}')

    local help_text=""
    [[ "${AUTO_DETECT_FLAGS}" == "1" ]] && help_text="$(vllm_help)"

    _add_ep_args args "$help_text" "$ep_size"
    _add_chunked_prefill_args args "$help_text"
    _add_prefix_caching_args args "$help_text"

    declare -p args
}

# ------------------------------------------------------------------------------
# 7. 在远程节点启动 vLLM 的辅助函数
# ------------------------------------------------------------------------------
launch_on_node() {
    local node="$1" local_ip="$2" is_headless="$3" idx="$4"

    local dp_start_rank=$((idx * NPUS_PER_NODE / CARDS_PER_INSTANCE))
    local array_decl env_exports
    array_decl=$(build_vllm_args_declare "${is_headless}" "${idx}" "${dp_start_rank}" "${NODE0_IP}")
    env_exports=$(build_env_exports "${local_ip}")

    local inner_cmd ssh_cmd
    inner_cmd="export SCRIPT_DIR='${SCRIPT_DIR}' && cd '${SCRIPT_DIR}' && source ../set_env.sh"$'\n'"${env_exports}"$'\n'"${array_decl}"$'\n'"nohup vllm \"\${args[@]}\" > ${SCRIPT_DIR}/vllm_${node}.log 2>&1 &"$'\n'"echo PID:\$!"
    ssh_cmd="export SCRIPT_DIR='${SCRIPT_DIR}' && cd '${SCRIPT_DIR}' && source ../set_env.sh && docker exec -i \"\${CONTAINER_NAME:-vllm-ascend-env-a3}\" bash -s"

    log_info "Launching on ${node} (IP: ${local_ip})..."
    if [[ "${DRY_RUN}" == "true" || "${DRY_RUN}" == "1" ]]; then
        echo "---------- Node: ${node} (host command) ----------"
        echo "${ssh_cmd}"
        echo "---------- Node: ${node} (container inner command) ----------"
        echo "${inner_cmd}"
        echo "-----------------------------------"
        return
    fi

    local pid
    # shellcheck disable=SC2086,SC2029
    pid=$(echo "${inner_cmd}" | ssh ${SSH_OPTS} "${node}" "${ssh_cmd}")
    log_info "Started vLLM on ${node}, PID=${pid}, log=${SCRIPT_DIR}/vllm_${node}.log"
}

# ------------------------------------------------------------------------------
# 8. 标准多节点部署
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
    local node local_ip is_headless
    for node in "${ALL_NODES[@]}"; do
        local_ip=$(get_node_ip "${node}" "${NIC_NAME}")
        [[ -n "${local_ip}" ]] || { log_warn "Skip ${node}: cannot detect IP on ${NIC_NAME}"; continue; }

        is_headless="false"
        [[ ${idx} -gt 0 ]] && is_headless="true"

        launch_on_node "${node}" "${local_ip}" "${is_headless}" "${idx}"
        idx=$((idx + 1))
    done
}

# ------------------------------------------------------------------------------
# 8. 主流程
# ------------------------------------------------------------------------------
deploy_standard

if [[ "${DRY_RUN}" != "true" && "${DRY_RUN}" != "1" ]]; then
    log_info "============================================================"
    log_info "All vLLM processes launched in background."
    log_info "Check logs: ${SCRIPT_DIR}/vllm_*.log"
    log_info "============================================================"
fi
