#!/bin/bash
# =============================================================================
# LongCat-Flash-Chat — Multi-Node Multiprocessing Deployment (no Ray)
# =============================================================================
# Deploys LongCat with TP=64 across 8 nodes using vLLM multiprocessing backend.
# Each node runs one vLLM process with --node-rank, coordinating via torch distributed.
#
# Usage:
#   bash run_vllm_mp.sh --file /path/to/node_list_8.txt
#   NIC_NAME=enp66s0f1 bash run_vllm_mp.sh --file node_list.txt
#   DRY_RUN=1 bash run_vllm_mp.sh --file node_list.txt
#
# Requirements:
#   - 8 nodes with vllm-ascend-env container running
#   - MC2 patch applied on all nodes
#   - SSH passwordless access to all nodes
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EASYINFER_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

# Load common functions
# shellcheck disable=SC1091
source "${EASYINFER_ROOT}/scripts/common.sh"

# ------------------------------------------------------------------------------
# Configuration (all overridable via env vars)
# ------------------------------------------------------------------------------
readonly MODEL_PATH="${MODEL_PATH:-/home/jianzhnie/llmtuner/hfhub/models/meituan-longcat/expand/LongCat-Flash-Chat-1024E-512Zero-E-Topk24-v2}"
readonly SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-longcat-flash}"
readonly HOST="${HOST:-0.0.0.0}"
readonly PORT="${PORT:-8010}"
readonly TP="${TP:-64}"
readonly PP="${PP:-1}"
readonly MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
readonly MAX_NUM_SEQS="${MAX_NUM_SEQS:-128}"
readonly GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
readonly MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"

# Network interface for HCCL communication
readonly NIC_NAME="${NIC_NAME:-enp66s0f1}"

# Container name
readonly CONTAINER_NAME="${CONTAINER_NAME:-vllm-ascend-env}"

# Dry run mode
DRY_RUN="${DRY_RUN:-false}"

# SSH options
SSH_OPTS="${SSH_OPTS:--o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10}"

# ------------------------------------------------------------------------------
# Parse node list
# ------------------------------------------------------------------------------
NODE_LIST_FILE=$(parse_nodes_file_arg "$@")

ALL_NODES=()
while IFS= read -r line; do
    ALL_NODES+=("$line")
done < <(read_nodes "${NODE_LIST_FILE}")

TOTAL_NODES=${#ALL_NODES[@]}
NNODES="${TOTAL_NODES}"

if [[ ${TOTAL_NODES} -lt 1 ]]; then
    log_fatal "Need at least 1 node, got ${TOTAL_NODES}"
fi

MASTER_ADDR="${ALL_NODES[0]}"
log_info "Loaded ${TOTAL_NODES} nodes, master: ${MASTER_ADDR}"

# Validate: TP * PP should equal TOTAL_NODES * 8
TOTAL_NPUS=$((TOTAL_NODES * 8))
REQUIRED_NPUS=$((TP * PP))
if [[ ${REQUIRED_NPUS} -ne ${TOTAL_NPUS} ]]; then
    log_warn "TP*PP=${REQUIRED_NPUS} != total NPUs=${TOTAL_NPUS}. Ensure config is correct."
fi

# ------------------------------------------------------------------------------
# Build vLLM command for a given node
# ------------------------------------------------------------------------------
build_vllm_cmd() {
    local node_rank="$1"

    cat <<EOF
vllm serve ${MODEL_PATH} \\
    --host ${HOST} \\
    --port ${PORT} \\
    --served-model-name ${SERVED_MODEL_NAME} \\
    --trust-remote-code \\
    --dtype bfloat16 \\
    --tensor-parallel-size ${TP} \\
    --pipeline-parallel-size ${PP} \\
    --distributed-executor-backend mp \\
    --nnodes ${NNODES} \\
    --node-rank ${node_rank} \\
    --master-addr ${MASTER_ADDR} \\
    --gpu-memory-utilization ${GPU_MEM_UTIL} \\
    --max-model-len ${MAX_MODEL_LEN} \\
    --max-num-seqs ${MAX_NUM_SEQS} \\
    --max-num-batched-tokens ${MAX_NUM_BATCHED_TOKENS} \\
    --enable-chunked-prefill \\
    --no-enable-prefix-caching \\
    --enforce-eager \\
    --seed 1024
EOF
}

# Build environment exports
build_env_block() {
    local node_ip="$1"
    cat <<EOF
export HCCL_OP_EXPANSION_MODE=AIV
export HCCL_IF_IP=${node_ip}
export HCCL_SOCKET_IFNAME=${NIC_NAME}
export GLOO_SOCKET_IFNAME=${NIC_NAME}
export TP_SOCKET_IFNAME=${NIC_NAME}
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export HCCL_BUFFSIZE=800
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_CONNECT_TIMEOUT=3600
export HCCL_EXEC_TIMEOUT=3600
export VLLM_USE_MODELSCOPE=False
export VLLM_ASCEND_BALANCE_SCHEDULING=1
export VLLM_USE_V1=1
EOF
}

# ------------------------------------------------------------------------------
# Launch on each node
# ------------------------------------------------------------------------------
LOG_DIR="${SCRIPT_DIR}"

echo "============================================"
log_info "LongCat-Flash-Chat — MP Multi-Node Deployment"
log_info "Model: ${MODEL_PATH}"
log_info "TP=${TP} PP=${PP} | Nodes=${NNODES} | Master=${MASTER_ADDR}"
log_info "Host: ${HOST}:${PORT}"
log_info "NIC: ${NIC_NAME}"
echo "============================================"

for ((idx = 0; idx < TOTAL_NODES; idx++)); do
    node="${ALL_NODES[$idx]}"
    node_ip="${node}"

    vllm_cmd=$(build_vllm_cmd "$idx")
    env_block=$(build_env_block "$node_ip")
    log_file="${LOG_DIR}/vllm_${node}_mp.log"

    inner_cmd="${env_block}
${vllm_cmd} > ${log_file} 2>&1 &
echo PID:\$!"

    if [[ "${DRY_RUN}" == "true" || "${DRY_RUN}" == "1" ]]; then
        echo "---------- Node ${idx}: ${node} ----------"
        echo "${inner_cmd}"
        echo "-------------------------------------------"
    else
        log_info "Launching node-rank ${idx} on ${node}..."
        # shellcheck disable=SC2086,SC2029
        result=$(ssh ${SSH_OPTS} "${node}" "docker exec -i ${CONTAINER_NAME} bash -l" <<< "${inner_cmd}" 2>&1)
        log_info "  ${node}: ${result}"
    fi
done

if [[ "${DRY_RUN}" != "true" && "${DRY_RUN}" != "1" ]]; then
    echo ""
    log_info "============================================"
    log_info "All ${TOTAL_NODES} vLLM processes launched."
    log_info "Logs: ${LOG_DIR}/vllm_*_mp.log"
    log_info "Service endpoint: http://${MASTER_ADDR}:${PORT}/v1"
    log_info "============================================"
    echo ""
    log_info "Monitor with:"
    log_info "  ssh ${MASTER_ADDR} \"docker exec ${CONTAINER_NAME} tail -f ${log_file}\""
fi
