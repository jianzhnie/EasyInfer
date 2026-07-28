#!/bin/bash
# ==============================================================================
# PP 部署脚本 — 启动 Ray 集群 + 单实例 vLLM (PP=2/4)
# ==============================================================================
# 用法: bash deploy_pp.sh deploy   (默认 PP=2)
#       PP=4 bash deploy_pp.sh deploy
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 第一个参数可以是 config 文件路径
if [[ "${1:-}" == *.conf ]] && [[ -f "$1" ]]; then
    source "$1"
    shift
fi
PP="${PP:-2}"

GREEN='\033[0;32m'; NC='\033[0m'
info() { echo -e "${GREEN}[INFO]${NC} $(date +%H:%M:%S) $*"; }

_ssh() { ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 "root@$1" "$@"; }

# ==============================================================================
# start_ray — 在 PP 组上启动 Ray 集群 (支持 2/4 节点)
# ==============================================================================
start_ray() {
    local head="$1"; shift
    info "Ray: head=$head workers=$*"
    for n in "$head" "$@"; do
        _ssh "$n" "docker exec ${CONTAINER_NAME} bash -c 'ray stop --force 2>/dev/null; true'" 2>/dev/null || true
    done
    sleep 2
    _ssh "$head" "docker exec -d ${CONTAINER_NAME} bash -c 'ray start --head --port=${RAY_PORT}'" 2>/dev/null
    sleep 3
    for worker in "$@"; do
        _ssh "$worker" "docker exec -d ${CONTAINER_NAME} bash -c 'ray start --address=${head}:${RAY_PORT}'" 2>/dev/null
    done
    sleep 3
    info "Ray cluster ready: ${head}:${RAY_PORT}"
}

# ==============================================================================
# deploy_pnode — 在 PP head 上启动 Prefill vLLM
# ==============================================================================
deploy_pnode() {
    local group_idx="$1" head="$2"; shift 2  # remaining args are workers
    local port=$((P_VLLM_START_PORT + group_idx))
    local engine_id="$group_idx"

    info "PNode PP=$PP group=$group_idx: head=$head port=$port"

    _ssh "$head" "docker exec -d ${CONTAINER_NAME} bash -l -c '
export LOCAL_IP='$head' NIC_NAME='$NIC_NAME' MODEL_PATH='$MODEL_PATH' LOG_DIR='$LOG_DIR' ENGINE_ID='$engine_id'
export HCCL_OP_EXPANSION_MODE=AIV HCCL_IF_IP='$head'
export GLOO_SOCKET_IFNAME='$NIC_NAME' TP_SOCKET_IFNAME='$NIC_NAME' HCCL_SOCKET_IFNAME='$NIC_NAME'
export OMP_PROC_BIND=false OMP_NUM_THREADS=1 PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_BUFFSIZE=400 ASCEND_AGGREGATE_ENABLE=1 ACL_OP_INIT_MODE=1
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1 VLLM_ASCEND_ENABLE_FUSED_MC2=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=3600 VLLM_ENGINE_READY_TIMEOUT_S=3600

LOG_FILE=\"glm5_pnode_pp${PP}_g${group_idx}_\$(date +%Y%m%d_%H%M%S).log\"
mkdir -p \"$LOG_DIR\"

nohup vllm serve \"$MODEL_PATH\" \
    --host 0.0.0.0 --port '$port' \
    --tensor-parallel-size '$P_TP_SIZE' \
    --pipeline-parallel-size '$PP' \
    --data-parallel-size '$P_DP_SIZE' \
    --data-parallel-rank '$group_idx' \
    --data-parallel-address '$P_DP_ADDRESS' \
    --data-parallel-rpc-port '$P_DP_RPC_PORT' \
    --distributed-executor-backend ray \
    --enable-expert-parallel --seed 1024 --served-model-name glm-52 \
    --max-model-len 135168 \
    --speculative-config \"'\"'\"'\"'\"'{\"num_speculative_tokens\": 1, \"method\":\"deepseek_mtp\"}'\"'\"'\"'\"'\" \
    --additional-config \"'\"'\"'\"'\"'{\"enable_sparse_c8\":false,\"fuse_muls_add\":true,\"multistream_overlap_shared_expert\":true,\"recompute_scheduler_enable\":true,\"ascend_compilation_config\":{\"enable_npugraph_ex\":true},\"enable_dsa_cp\":true}'\"'\"'\"'\"'\" \
    --max-num-batched-tokens 4096 --trust-remote-code --max-num-seqs 64 \
    --async-scheduling --quantization ascend --gpu-memory-utilization 0.90 \
    --enforce-eager --enable-chunked-prefill --enable-prefix-caching \
    --enable-auto-tool-choice --tool-call-parser glm47 --reasoning-parser glm45 \
    --kv-transfer-config \"'\"'\"'\"'\"'{\"kv_connector\":\"MooncakeConnector\",\"kv_role\":\"kv_producer\",\"kv_port\":\"$((30000 + group_idx))\",\"engine_id\":\"'$engine_id'\",\"kv_connector_module_path\":\"vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector\",\"kv_connector_extra_config\":{\"use_ascend_direct\":true,\"prefill\":{\"dp_size\":'$P_DP_SIZE',\"tp_size\":'$P_TP_SIZE'},\"decode\":{\"dp_size\":'$D_DP_SIZE',\"tp_size\":'$D_TP_SIZE'}}}'\"'\"'\"'\"'\" \
    < /dev/null > \"$LOG_DIR/\$LOG_FILE\" 2>&1 &
'" 2>/dev/null
    info "PNode group=$group_idx launched"
}

deploy_dnode() {
    local group_idx="$1" head="$2"; shift 2
    local port=$((D_VLLM_START_PORT + group_idx))
    local engine_id=$((4 + group_idx))

    info "DNode PP=$PP group=$group_idx: head=$head port=$port"

    _ssh "$head" "docker exec -d ${CONTAINER_NAME} bash -l -c '
export LOCAL_IP='$head' NIC_NAME='$NIC_NAME' MODEL_PATH='$MODEL_PATH' LOG_DIR='$LOG_DIR' ENGINE_ID='$engine_id'
export HCCL_OP_EXPANSION_MODE=AIV HCCL_IF_IP='$head'
export GLOO_SOCKET_IFNAME='$NIC_NAME' TP_SOCKET_IFNAME='$NIC_NAME' HCCL_SOCKET_IFNAME='$NIC_NAME'
export OMP_PROC_BIND=false OMP_NUM_THREADS=1 PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_BUFFSIZE=500 ASCEND_AGGREGATE_ENABLE=1 ACL_OP_INIT_MODE=1 TASK_QUEUE_ENABLE=1
export DYNAMIC_EPLB=1 VLLM_ASCEND_ENABLE_FUSED_MC2=1 VLLM_ASCEND_ENABLE_MLAPO=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=3600 VLLM_ENGINE_READY_TIMEOUT_S=3600

LOG_FILE=\"glm5_dnode_pp${PP}_g${group_idx}_\$(date +%Y%m%d_%H%M%S).log\"
mkdir -p \"$LOG_DIR\"

nohup vllm serve \"$MODEL_PATH\" \
    --host 0.0.0.0 --port '$port' \
    --tensor-parallel-size '$D_TP_SIZE' \
    --pipeline-parallel-size '$PP' \
    --data-parallel-size '$D_DP_SIZE' \
    --data-parallel-rank '$group_idx' \
    --data-parallel-address '$D_DP_ADDRESS' \
    --data-parallel-rpc-port '$D_DP_RPC_PORT' \
    --distributed-executor-backend ray \
    --enable-expert-parallel --seed 1024 --served-model-name glm-52 \
    --max-model-len 135168 --max-num-batched-tokens 164 \
    --compilation-config \"'\"'\"'\"'\"'{\"cudagraph_mode\":\"FULL_DECODE_ONLY\"}'\"'\"'\"'\"'\" \
    --speculative-config \"'\"'\"'\"'\"'{\"num_speculative_tokens\": 1, \"method\":\"deepseek_mtp\"}'\"'\"'\"'\"'\" \
    --additional-config \"'\"'\"'\"'\"'{\"enable_sparse_c8\":false,\"fuse_muls_add\":true,\"multistream_overlap_shared_expert\":true,\"recompute_scheduler_enable\":true,\"ascend_compilation_config\":{\"enable_npugraph_ex\":true}}'\"'\"'\"'\"'\" \
    --trust-remote-code --max-num-seqs 48 \
    --gpu-memory-utilization 0.90 --async-scheduling --enable-prefix-caching \
    --quantization ascend --enable-auto-tool-choice \
    --tool-call-parser glm47 --reasoning-parser glm45 \
    --kv-transfer-config \"'\"'\"'\"'\"'{\"kv_connector\":\"MooncakeConnector\",\"kv_role\":\"kv_consumer\",\"kv_port\":\"$((30100 + group_idx))\",\"engine_id\":\"'$engine_id'\",\"kv_connector_module_path\":\"vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector\",\"kv_connector_extra_config\":{\"use_ascend_direct\":true,\"prefill\":{\"dp_size\":'$P_DP_SIZE',\"tp_size\":'$P_TP_SIZE'},\"decode\":{\"dp_size\":'$D_DP_SIZE',\"tp_size\":'$D_TP_SIZE'}}}'\"'\"'\"'\"'\" \
    < /dev/null > \"$LOG_DIR/\$LOG_FILE\" 2>&1 &
'" 2>/dev/null
    info "DNode group=$group_idx launched"
}

# ==============================================================================
cmd_deploy() {
    echo "============================================"
    echo " GLM-5.2 PD+PP=$PP 部署"
    echo " P: ${#P_PP_GROUPS[@]} 组 PP=$PP, TP=$P_TP_SIZE"
    echo " D: ${#D_PP_GROUPS[@]} 组 PP=$PP, TP=$D_TP_SIZE"
    echo "============================================"

    # 部署 PNode PP 组
    for i in "${!P_PP_GROUPS[@]}"; do
        read -ra nodes <<< "${P_PP_GROUPS[$i]}"
        start_ray "${nodes[@]}"
        deploy_pnode "$i" "${nodes[@]}"
    done

    # 部署 DNode PP 组
    for i in "${!D_PP_GROUPS[@]}"; do
        read -ra nodes <<< "${D_PP_GROUPS[$i]}"
        start_ray "${nodes[@]}"
        deploy_dnode "$i" "${nodes[@]}"
    done

    echo "部署完成。检查:"
    echo "  PNode: http://10.42.11.194:9081/v1/models"
    echo "  DNode: http://10.42.11.198:9900/v1/models"
}

cmd_stop() {
    for g in "${P_PP_GROUPS[@]}" "${D_PP_GROUPS[@]}"; do
        read -ra nodes <<< "$g"
        for n in "${nodes[@]}"; do
            _ssh "$n" "docker exec ${CONTAINER_NAME} bash -c 'pkill -f vllm 2>/dev/null; ray stop --force 2>/dev/null; true'" 2>/dev/null || true
        done
    done
    echo "stopped"
}

CMD="${1:-deploy}"
case "$CMD" in
    deploy|stop) "cmd_${CMD}" ;;
    *) echo "Usage: $0 {deploy|stop}"; exit 1 ;;
esac
