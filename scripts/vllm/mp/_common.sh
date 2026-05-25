#!/bin/bash
#
# 共享函数 — vLLM 多节点部署脚本 (deploy_vllm_multinode.sh / deploy_vllm_multinode_mp.sh)
#
# 注意: 本文件被 source 而非直接执行，不设 set -euo pipefail。
# 依赖: 调用方需先 source common.sh，设置 SSH_OPTS, DRY_RUN, NIC_NAME 等全局变量。

# ------------------------------------------------------------------------------
# 获取节点 IP
# ------------------------------------------------------------------------------
get_node_ip() {
    local node="$1" nic="$2" cmd=""
    if command -v ip >/dev/null 2>&1; then
        cmd="ip -4 addr show ${nic} 2>/dev/null | awk '/inet / {print \$2}' | cut -d/ -f1 | head -n 1"
    elif command -v ifconfig >/dev/null 2>&1; then
        cmd="ifconfig ${nic} 2>/dev/null | awk '/inet / {print \$2}' | head -n 1"
    else
        printf '\n'
        return
    fi

    local result=""
    if [[ "${node}" == "$(hostname -s)" || "${node}" == "$(hostname)" ]]; then
        result=$(eval "${cmd}")
    else
        # shellcheck disable=SC2086,SC2029
        result=$(ssh ${SSH_OPTS} "${node}" "${cmd}" 2>/dev/null)
    fi
    if [[ -z "${result}" && ( "${DRY_RUN}" == "true" || "${DRY_RUN}" == "1" ) ]]; then
        printf '192.168.1.%s\n' "$((RANDOM % 254 + 1))"
    else
        printf '%s\n' "${result}"
    fi
}

# ------------------------------------------------------------------------------
# vLLM 帮助信息
# ------------------------------------------------------------------------------
vllm_help() {
    vllm serve --help 2>/dev/null || true
}

# ------------------------------------------------------------------------------
# 参数探测：从 vLLM help 中选择支持的 flag
# ------------------------------------------------------------------------------
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

has_flag() {
    local help_text="$1" flag="$2"
    [[ "$help_text" == *"$flag"* ]]
}

# ------------------------------------------------------------------------------
# 环境变量导出字符串（HCCL/网络/NPU 配置）
# ------------------------------------------------------------------------------
build_env_exports() {
    local local_ip="$1"
    printf 'export HCCL_OP_EXPANSION_MODE=AIV\n'
    printf 'export HCCL_IF_IP=%s\n' "$local_ip"
    printf 'export GLOO_SOCKET_IFNAME=%s\n' "$NIC_NAME"
    printf 'export TP_SOCKET_IFNAME=%s\n' "$NIC_NAME"
    printf 'export HCCL_SOCKET_IFNAME=%s\n' "$NIC_NAME"
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
# SSH 连通性检查（检测全部节点）
# 依赖全局数组: ALL_NODES, SSH_OPTS
# ------------------------------------------------------------------------------
check_ssh_connectivity() {
    log_info "Checking SSH connectivity..."
    local failed=0
    for node in "${ALL_NODES[@]}"; do
        # shellcheck disable=SC2086
        if ! ssh ${SSH_OPTS} -o ConnectTimeout=5 "${node}" "echo OK" >/dev/null 2>&1; then
            log_err "SSH failed: ${node}"
            failed=1
        fi
    done
    [[ ${failed} -eq 0 ]] || log_fatal "SSH connectivity check failed"
    log_info "All nodes are reachable via SSH"
}