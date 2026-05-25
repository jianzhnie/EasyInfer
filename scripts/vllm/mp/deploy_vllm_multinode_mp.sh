#!/bin/bash
# ==============================================================================
# DeepSeek-V3.2 Multi-Node Deployment Script for Ascend NPU
# ==============================================================================
# 基于 https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/DeepSeek-V3.2.html
# 以及 https://docs.vllm.ai/en/stable/serving/parallelism_scaling/#running-vllm-with-multiprocessing
#
# 功能:
#   1. 自动读取 node_list.txt 获取多节点列表
#   2. 支持 A2 系列 Ascend NPU 的标准多节点部署 (Multiprocessing 后端)
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
SSH_OPTS="${SSH_OPTS:--o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10}"
AUTO_DETECT_FLAGS="${AUTO_DETECT_FLAGS:-1}"
NODE_LIST_FILE=$(parse_nodes_file_arg "$@")

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
export MODEL_PATH="${MODEL_PATH:-/llm_workspace_1P/robin/hfhub/models/moonshotai/Kimi-K2-Base}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-kimi_k2_base}"
export VLLM_PORT="${VLLM_PORT:-8077}"
export DP_RPC_PORT="${DP_RPC_PORT:-12890}"

# 分布式并行配置 (参考 vllm_model_server.sh)
export TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-8}"
export PIPELINE_PARALLEL_SIZE="${PIPELINE_PARALLEL_SIZE:-8}"
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
load_and_validate_nodes "${NODE_LIST_FILE}" 1

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
# 6. 构建 vLLM 启动参数 (参考 vllm_model_server.sh 的分块构建方式)
# ------------------------------------------------------------------------------

_build_base_args() {
    local array_name="$1"
    eval "${array_name}+=(serve \"${MODEL_PATH}\")"
    eval "${array_name}+=(--host 0.0.0.0)"
    eval "${array_name}+=(--port \"${VLLM_PORT}\")"
    eval "${array_name}+=(--trust-remote-code)"
    eval "${array_name}+=(--served-model-name \"${SERVED_MODEL_NAME}\")"
    eval "${array_name}+=(--seed 1024)"
    eval "${array_name}+=(--tensor-parallel-size \"${TENSOR_PARALLEL_SIZE}\")"
    eval "${array_name}+=(--pipeline-parallel-size \"${PIPELINE_PARALLEL_SIZE}\")"
    eval "${array_name}+=(--max-num-seqs \"${MAX_NUM_SEQS}\")"
    eval "${array_name}+=(--max-model-len \"${MAX_MODEL_LEN}\")"
    eval "${array_name}+=(--max-num-batched-tokens \"${MAX_NUM_BATCHED_TOKENS}\")"
    eval "${array_name}+=(--gpu-memory-utilization \"${GPU_MEMORY_UTILIZATION}\")"
    eval "${array_name}+=(--no-enable-prefix-caching)"
}

_build_mp_args() {
    local array_name="$1"
    local node_rank="$2" master_addr="$3" nnodes="$4"
    if [[ "${nnodes}" -gt 1 ]]; then
        eval "${array_name}+=(--distributed-executor-backend mp)"
        eval "${array_name}+=(--nnodes \"${nnodes}\")"
        eval "${array_name}+=(--node-rank \"${node_rank}\")"
        eval "${array_name}+=(--master-addr \"${master_addr}\")"
    fi
}

_build_dp_args() {
    local array_name="$1"
    local is_headless="$2" dp_size_local="$3" dp_start_rank="$4"
    if [[ "${DP_SIZE}" -gt 1 ]]; then
        eval "${array_name}+=(--data-parallel-size \"${DP_SIZE}\")"
        eval "${array_name}+=(--data-parallel-size-local \"${dp_size_local}\")"
        eval "${array_name}+=(--data-parallel-address \"${NODE0_IP}\")"
        eval "${array_name}+=(--data-parallel-rpc-port \"${DP_RPC_PORT}\")"
        if [[ "${is_headless}" == "true" ]]; then
            eval "${array_name}+=(--headless)"
            eval "${array_name}+=(--data-parallel-start-rank \"${dp_start_rank}\")"
        fi
    else
        if [[ "${is_headless}" == "true" ]]; then
            eval "${array_name}+=(--headless)"
        fi
    fi
}

_build_a2_compile_args() {
    local array_name="$1"
    eval "${array_name}+=(--compilation-config '{\"cudagraph_mode\": \"FULL_DECODE_ONLY\", \"cudagraph_capture sizes\":[8, 16, 24, 32, 40, 48]}')"
    eval "${array_name}+=(--additional-config '{\"layer_sharding\": [\"q_b_proj\", \"o_proj\"]}')"
    eval "${array_name}+=(--speculative-config '{\"num_speculative_tokens\": 3, \"method\": \"deepseek_mtp\"}')"
}

build_vllm_args_declare() {
    local is_headless="$1"
    local node_rank="$2"
    local dp_start_rank="$3"
    local dp_size_local="$4"
    local master_addr="$5"
    local nnodes="$6"
    local vllm_port="$7"
    local use_internal_dp="$8"

    local tp_size="${TENSOR_PARALLEL_SIZE}"
    local pp_size="${PIPELINE_PARALLEL_SIZE}"
    local ep_size="${EXPERT_PARALLEL_SIZE}"

    local -a args=()
    _build_base_args args
    _build_mp_args args "${node_rank}" "${master_addr}" "${nnodes}"

    if [[ "${use_internal_dp}" == "true" ]]; then
        _build_dp_args args "${is_headless}" "${dp_size_local}" "${dp_start_rank}"
    else
        if [[ "${is_headless}" == "true" ]]; then
            args+=(--headless)
        fi
    fi

    _build_a2_compile_args args

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
    local node="$1" local_ip="$2" is_headless="$3" node_rank="$4"
    local dp_start_rank="$5" dp_size_local="$6" master_addr="$7" nnodes="$8"
    local vllm_port="$9" use_internal_dp="${10}"
    local env_exports prefix
    env_exports=$(build_env_exports "${local_ip}")
    prefix="export SCRIPT_DIR='${SCRIPT_DIR}' && cd '${SCRIPT_DIR}' && source ../set_env.sh"$'\n'"${env_exports}"
    _launch_vllm_on_node "$node" "$local_ip" "$prefix" \
        "build_vllm_args_declare '${is_headless}' '${node_rank}' '${dp_start_rank}' '${dp_size_local}' '${master_addr}' '${nnodes}' '${vllm_port}' '${use_internal_dp}'" \
        "_${vllm_port}"
}

# ------------------------------------------------------------------------------
# 8. 标准多节点部署
# ------------------------------------------------------------------------------

# 场景 A: 单个模型实例跨多个节点
_deploy_multinode_instance() {
    log_info "Mode: multi-node per instance (${NODES_PER_INSTANCE} nodes per instance, independent instances)"

    if [[ ${DP_SIZE} -gt 1 ]]; then
        log_warn "Internal DP is disabled because each instance spans multiple nodes."
        log_warn "Please configure an external load balancer for ports ${VLLM_PORT}-$((VLLM_PORT + DP_SIZE - 1)) on master nodes."
    fi

    local dp_idx instance_start_node instance_master_idx instance_master_node instance_master_ip instance_port
    local offset node_idx node local_ip is_headless
    for ((dp_idx = 0; dp_idx < DP_SIZE; dp_idx++)); do
        instance_start_node=$((dp_idx * NODES_PER_INSTANCE))
        instance_master_idx=${instance_start_node}
        instance_master_node="${ALL_NODES[$instance_master_idx]}"
        instance_master_ip=$(get_node_ip "${instance_master_node}" "${NIC_NAME}")
        instance_port=$((VLLM_PORT + dp_idx))

        log_info "Deploying instance ${dp_idx}/${DP_SIZE} on nodes ${instance_start_node}..$((instance_start_node + NODES_PER_INSTANCE - 1)), master=${instance_master_node}:${instance_port}"

        for ((offset = 0; offset < NODES_PER_INSTANCE; offset++)); do
            node_idx=$((instance_start_node + offset))
            node="${ALL_NODES[$node_idx]}"
            local_ip=$(get_node_ip "${node}" "${NIC_NAME}")
            [[ -n "${local_ip}" ]] || { log_warn "Skip ${node}: cannot detect IP on ${NIC_NAME}"; continue; }

            is_headless="false"
            [[ ${offset} -gt 0 ]] && is_headless="true"

            launch_on_node "${node}" "${local_ip}" "${is_headless}" "${offset}" "0" "1" "${instance_master_ip}" "${NODES_PER_INSTANCE}" "${instance_port}" "false"
        done
    done
}

# 场景 B: 单个模型实例可放在一个节点内 (单节点 TP/PP，多节点 DP)
_deploy_singlenode_instance() {
    log_info "Mode: single-node per instance, ${DP_SIZE_LOCAL} instances per node"

    local node_idx node local_ip local_dp dp_rank port is_headless
    for ((node_idx = 0; node_idx < TOTAL_NODES; node_idx++)); do
        node="${ALL_NODES[$node_idx]}"
        local_ip=$(get_node_ip "${node}" "${NIC_NAME}")
        [[ -n "${local_ip}" ]] || { log_warn "Skip ${node}: cannot detect IP on ${NIC_NAME}"; continue; }

        for ((local_dp = 0; local_dp < DP_SIZE_LOCAL; local_dp++)); do
            dp_rank=$((node_idx * DP_SIZE_LOCAL + local_dp))
            port=$((VLLM_PORT + local_dp))
            is_headless="false"

            # 只有全局第一个实例 (Node0, local_dp=0) 启动 API server
            if [[ ${node_idx} -gt 0 || ${local_dp} -gt 0 ]]; then
                is_headless="true"
            fi

            launch_on_node "${node}" "${local_ip}" "${is_headless}" "0" "${dp_rank}" "${DP_SIZE_LOCAL}" "${local_ip}" "1" "${port}" "true"
        done
    done
}

deploy_standard() {
    local tp_size="${TENSOR_PARALLEL_SIZE}"
    local pp_size="${PIPELINE_PARALLEL_SIZE}"
    local ep_size="${EXPERT_PARALLEL_SIZE}"

    log_info "============================================================"
    log_info "Standard Multi-Node Deployment (A2) via Multiprocessing"
    log_info "Nodes: ${TOTAL_NODES} | DP: ${DP_SIZE} | TP: ${tp_size} | PP: ${pp_size} | EP: ${ep_size}"
    log_info "============================================================"

    if [[ ${CARDS_PER_INSTANCE} -gt ${NPUS_PER_NODE} ]]; then
        _deploy_multinode_instance
    else
        _deploy_singlenode_instance
    fi
}

# ------------------------------------------------------------------------------
# 9. 主流程
# ------------------------------------------------------------------------------
deploy_standard

if [[ "${DRY_RUN}" != "true" && "${DRY_RUN}" != "1" ]]; then
    log_info "============================================================"
    log_info "All vLLM processes launched in background."
    log_info "Check logs: ${SCRIPT_DIR}/vllm_*.log"
    log_info "============================================================"
fi
