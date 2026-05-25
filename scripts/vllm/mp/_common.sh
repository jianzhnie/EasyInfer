#!/bin/bash
#
# 共享函数 — vLLM 多节点部署脚本 (deploy_vllm_multinode.sh / deploy_vllm_multinode_mp.sh)
#
# 注意: 本文件被 source 而非直接执行，不设 set -euo pipefail。
# 依赖: 调用方需先 source common.sh，设置 SSH_OPTS, DRY_RUN, NIC_NAME 等全局变量。

# ------------------------------------------------------------------------------
# 获取节点 IP（支持 DRY_RUN 随机 IP 回退）
# 调用方已 source common.sh，get_node_ip 已在环境中；本函数仅添加 DRY_RUN 回退。
# ------------------------------------------------------------------------------
_get_node_ip_with_fallback() {
    local node="$1" nic="$2"
    local result
    result=$(get_node_ip "$node" "$nic")

    # 如果无法获取且处于 DRY_RUN 模式，返回随机测试 IP
    if [[ -z "${result}" ]] && [[ "${DRY_RUN:-false}" == "true" || "${DRY_RUN:-false}" == "1" ]]; then
        printf '192.168.1.%s\n' "$((RANDOM % 254 + 1))"
        return
    fi

    printf '%s\n' "${result}"
}

# ------------------------------------------------------------------------------
# 环境变量导出字符串（HCCL/网络/NPU 配置）
# ------------------------------------------------------------------------------
build_env_exports() {
    local local_ip="$1"
    printf 'export HCCL_OP_EXPANSION_MODE=AIV\n'
    printf 'export HCCL_IF_IP=%s\n' "$local_ip"
    printf 'export GLOO_SOCKET_IFNAME=%s\n' "${NIC_NAME}"
    printf 'export TP_SOCKET_IFNAME=%s\n' "${NIC_NAME}"
    printf 'export HCCL_SOCKET_IFNAME=%s\n' "${NIC_NAME}"
    printf 'export OMP_PROC_BIND=false\n'
    printf 'export VLLM_USE_V1=1\n'
    printf 'export HCCL_BUFFSIZE=200\n'
    printf 'export VLLM_ASCEND_ENABLE_MLAPO=1\n'
    printf 'export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True\n'
    printf 'export VLLM_ASCEND_ENABLE_FLASHCOMM1=1\n'
    printf 'export OMP_NUM_THREADS=100\n'
    printf 'export HCCL_CONNECT_TIMEOUT=120\n'
    printf 'export HCCL_INTRA_PCIE_ENABLE=1\n'
    printf 'export HCCL_INTRA_ROCE_ENABLE=0\n'
    printf 'export VLLM_V1_FRONTEND_ENGINE_CORE_TIMEOUT=1200\n'
    printf 'export VLLM_RPC_TIMEOUT=600\n'
}

# ------------------------------------------------------------------------------
# vLLM 参数构建辅助函数（供 deploy_vllm_multinode.sh / deploy_vllm_multinode_mp.sh 使用）
# ------------------------------------------------------------------------------

# 添加 Expert Parallel 参数（通过数组变量名操作，兼容 bash 4.2）
_add_ep_args() {
    local array_name="$1"
    local help_text="$2" ep_size="$3"
    [[ "${ENABLE_EXPERT_PARALLEL}" != "1" ]] && return
    local ep_flag="--enable-expert-parallel"
    [[ -n "$help_text" ]] && ep_flag="$(choose_flag "$help_text" "--enable-expert-parallel" "--enable_expert_parallel")"
    eval "${array_name}+=(\"${ep_flag}\")"
    if [[ -n "$help_text" ]] && has_flag "$help_text" "--expert-parallel-size"; then
        eval "${array_name}+=(--expert-parallel-size \"${ep_size}\")"
    fi
}

# 添加 Chunked Prefill 参数
_add_chunked_prefill_args() {
    local array_name="$1"
    local help_text="$2"
    [[ "${ENABLE_CHUNKED_PREFILL}" != "1" ]] && return
    if [[ -n "$help_text" ]]; then
        has_flag "$help_text" "--enable-chunked-prefill" && eval "${array_name}+=(--enable-chunked-prefill)"
    else
        eval "${array_name}+=(--enable-chunked-prefill)"
    fi
}

# 处理 Prefix Caching 参数：从数组中移除 --no-enable-prefix-caching，按需添加 --enable-prefix-caching
# shellcheck disable=SC2034
_add_prefix_caching_args() {
    local array_name="$1"
    local help_text="$2"
    [[ "${PREFIX_CACHING}" != "1" ]] && return
    local -a filtered=()
    local item
    eval 'for item in "${'"${array_name}"'[@]}"; do [[ "$item" != "--no-enable-prefix-caching" ]] && filtered+=("$item"); done'
    eval "${array_name}=(\"\${filtered[@]}\")"
    local pc_flag="--enable-prefix-caching"
    [[ -n "$help_text" ]] && pc_flag="$(choose_flag "$help_text" "--enable-prefix-caching" "--enable_prefix_caching")"
    eval "${array_name}+=(\"${pc_flag}\")"
}

# ------------------------------------------------------------------------------
# 共享多节点部署逻辑
# ------------------------------------------------------------------------------

# 读取并验证节点列表
load_and_validate_nodes() {
    local nodes_file="$1"
    local min_nodes="${2:-2}"

    if [[ ! -f "${nodes_file}" ]]; then
        log_fatal "Node list file not found: ${nodes_file}"
    fi

    ALL_NODES=()
    while IFS= read -r line; do
        ALL_NODES+=("$line")
    done < <(read_nodes "${nodes_file}")
    TOTAL_NODES=${#ALL_NODES[@]}

    if [[ ${TOTAL_NODES} -lt ${min_nodes} ]]; then
        log_fatal "Need at least ${min_nodes} nodes, got ${TOTAL_NODES}"
    fi

    NODE0="${ALL_NODES[0]}"
    log_info "Loaded ${TOTAL_NODES} nodes from ${nodes_file}"
    log_info "Master node: ${NODE0}"
}

# 验证并行配置合法性
validate_parallelism_config() {
    local total_nodes="$1"
    local npus_per_node="$2"

    TOTAL_CARDS=$((total_nodes * npus_per_node))
    CARDS_PER_INSTANCE=$((TENSOR_PARALLEL_SIZE * PIPELINE_PARALLEL_SIZE))

    if [[ ${CARDS_PER_INSTANCE} -eq 0 ]]; then
        log_fatal "Invalid config: TENSOR_PARALLEL_SIZE * PIPELINE_PARALLEL_SIZE = 0"
    fi

    if [[ $((TOTAL_CARDS % CARDS_PER_INSTANCE)) -ne 0 ]]; then
        log_fatal "Card mismatch: TOTAL_CARDS (${TOTAL_CARDS}) is not divisible by CARDS_PER_INSTANCE (${CARDS_PER_INSTANCE})"
    fi

    DP_SIZE=$((TOTAL_CARDS / CARDS_PER_INSTANCE))
    if [[ ${DP_SIZE} -lt 1 ]]; then
        log_fatal "Invalid config: DP_SIZE (${DP_SIZE}) must be >= 1"
    fi

    if [[ ${CARDS_PER_INSTANCE} -le ${npus_per_node} ]]; then
        DP_SIZE_LOCAL=$((npus_per_node / CARDS_PER_INSTANCE))
    else
        DP_SIZE_LOCAL=1
    fi

    if [[ ${CARDS_PER_INSTANCE} -gt ${npus_per_node} ]]; then
        NODES_PER_INSTANCE=$((CARDS_PER_INSTANCE / npus_per_node))
        if [[ $((total_nodes % NODES_PER_INSTANCE)) -ne 0 ]]; then
            log_fatal "Node mismatch: each instance needs ${NODES_PER_INSTANCE} nodes"
        fi
        if [[ $((DP_SIZE * NODES_PER_INSTANCE)) -ne ${total_nodes} ]]; then
            log_fatal "Config mismatch: DP_SIZE * NODES_PER_INSTANCE != TOTAL_NODES"
        fi
    else
        NODES_PER_INSTANCE=1
    fi

    log_info "Config: TOTAL_CARDS=${TOTAL_CARDS}, TP=${TENSOR_PARALLEL_SIZE}, PP=${PIPELINE_PARALLEL_SIZE}, DP=${DP_SIZE}, DP_LOCAL=${DP_SIZE_LOCAL}, NODES_PER_INSTANCE=${NODES_PER_INSTANCE}"
}

# 获取 node0 IP
resolve_node0_ip() {
    local node0="$1"
    local nic_name="$2"
    NODE0_IP=$(get_node_ip "${node0}" "${nic_name}")
    if [[ -z "${NODE0_IP}" ]]; then
        log_fatal "Failed to get IP for node ${node0} on interface ${nic_name}"
    fi
    log_info "Node0 IP: ${NODE0_IP}"
}

# ------------------------------------------------------------------------------
# SSH 连通性检查（检测全部节点）
# 依赖全局数组: ALL_NODES, SSH_OPTS
# ------------------------------------------------------------------------------
check_ssh_connectivity() {
    log_info "Checking SSH connectivity..."
    local failed=0
    for node in "${ALL_NODES[@]}"; do
        # shellcheck disable=SC2086
        if ! ssh ${SSH_OPTS:-} -o ConnectTimeout=5 "$(ssh_target "$node")" "echo OK" >/dev/null 2>&1; then
            log_err "SSH failed: ${node}"
            failed=1
        fi
    done
    [[ ${failed} -eq 0 ]] || log_fatal "SSH connectivity check failed"
    log_info "All nodes are reachable via SSH"
}

# ------------------------------------------------------------------------------
# 共享的远程节点 vLLM 启动逻辑
# 用法: _launch_vllm_on_node <node> <local_ip> <inner_cmd_prefix> <array_decl_cmd> <log_suffix>
#   inner_cmd_prefix: source env + env_exports 的前置命令
#   array_decl_cmd:   生成 declare -p args 的命令字符串
#   log_suffix:       日志文件标识 (如 "_node1" 或 "_node1_8077")
# ------------------------------------------------------------------------------
_launch_vllm_on_node() {
    local node="$1" local_ip="$2" inner_cmd_prefix="$3" array_decl_cmd="$4" log_suffix="${5:-}"

    local array_decl
    array_decl=$(eval "$array_decl_cmd")

    local inner_cmd ssh_cmd
    inner_cmd="${inner_cmd_prefix}"$'\n'"${array_decl}"$'\n'"nohup vllm \"\${args[@]}\" > ${SCRIPT_DIR}/vllm_${node}${log_suffix}.log 2>&1 &"$'\n'"echo PID:\$!"

    ssh_cmd="export SCRIPT_DIR='${SCRIPT_DIR}' && cd '${SCRIPT_DIR}' && source ../set_env.sh && docker exec -i \"\${CONTAINER_NAME:-vllm-ascend-env-a3}\" bash -s"

    log_info "Launching on ${node} (IP: ${local_ip})..."
    if [[ "${DRY_RUN}" == "true" || "${DRY_RUN}" == "1" ]]; then
        echo "---------- Node: ${node} (host command) ----------"
        echo "${ssh_cmd}"
        echo "---------- Node: ${node} (container inner command) ----------"
        echo "${inner_cmd}"
        echo "-----------------------------------"
    else
        local pid
        # shellcheck disable=SC2086,SC2029
        pid=$(echo "${inner_cmd}" | ssh ${SSH_OPTS} "${node}" "${ssh_cmd}")
        log_info "Started vLLM on ${node}, PID=${pid}, log=${SCRIPT_DIR}/vllm_${node}${log_suffix}.log"
    fi
}
