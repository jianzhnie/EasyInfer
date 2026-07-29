#!/bin/bash
# =============================================================================
# GLM-5.2 W8A8 — 1M 上下文 Ray 单实例部署
# =============================================================================
# 架构: Ray 集群 + vllm serve 单实例（--distributed-executor-backend ray）
#   本脚本只负责在已有 Ray 集群上启动 vLLM；Ray 集群的启停由专用脚本管理:
#     bash scripts/ray_cluster/manage_ray_cluster.sh start  -f nodes/node_list3.txt
#     bash scripts/ray_cluster/manage_ray_cluster.sh status -f nodes/node_list3.txt
#
# 默认拓扑: TP=16 PP=4 DP=1（8 节点 × 8 卡 = 64 NPU），MTP 关
#   - PP>1 + MTP 在 v0.23.0rc1 共部署不支持（#11076 仅 main）→ ENABLE_MTP=0
#   - Ray 仅适用于 DP=1 且 TP/PP 跨节点；DP>1 请用 dp1m 套件（原生多节点 DP）
#   - ⚠️ TP=16 PP=4 未实测，建议先 TP=16 PP=2（4 节点）起步验证
#
# 已知约束（A2 + W8A8 实测）：FUSED_MC2 必须 0（跨节点 EP 崩溃），本脚本强制覆盖
#
# Usage:
#   bash run_1m.sh                  # 部署（默认 TP=16 PP=4 DP=1）
#   TP=16 PP=2 bash run_1m.sh       # 4 节点起步
#   bash run_1m.sh status           # vLLM / API 状态
#   bash run_1m.sh stop             # 停止 vLLM（不动 Ray 集群）
# =============================================================================
set -euo pipefail

REPO=/home/jianzhnie/llmtuner/llm/EasyInfer
CONTAINER=vllm-ascend-env
NODES_FILE="${NODES_FILE:-$REPO/nodes/node_list3.txt}"
NIC_NAME="${NIC_NAME:-enp66s0f0}"
RAY_PORT=6379

TP="${TP:-16}"
PP="${PP:-4}"
DP="${DP:-1}"
ENABLE_MTP="${ENABLE_MTP:-0}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-32768}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"

mapfile -t NODES < <(grep -v '^\s*\(#\|$\)' "$NODES_FILE")
HEAD="${NODES[0]}"

# ---- 参数校验 ----------------------------------------------------------------
need=$((TP * PP * DP)); have=$(( ${#NODES[@]} * 8 ))
(( need <= have )) || { echo "[ERROR] TP×PP×DP=$need 超过总卡数 $have（${#NODES[@]} 节点）" >&2; exit 1; }
(( PP > 1 && ENABLE_MTP == 1 )) && { echo "[ERROR] PP>1 时 MTP 不可用（#11076），ENABLE_MTP=0" >&2; exit 1; }

in_head() { ssh -o BatchMode=yes -o ConnectTimeout=10 "$HEAD" "docker exec $CONTAINER bash -c '$1'"; }

# ---- Ray 集群前置检查（不启停，只校验） ---------------------------------------
check_ray() {
    local active
    active=$(in_head "ray status 2>/dev/null | grep -cE '^\s*1 node_'" || echo 0)
    if [[ "$active" -lt "${#NODES[@]}" ]]; then
        echo "[ERROR] Ray 集群未就绪（活跃节点 $active/${#NODES[@]}）。请先启动：" >&2
        echo "  bash $REPO/scripts/ray_cluster/manage_ray_cluster.sh start -f $NODES_FILE" >&2
        exit 1
    fi
    echo "[OK] Ray 集群就绪（$active/${#NODES[@]} 节点，head: $HEAD:$RAY_PORT）"
}

cmd_deploy() {
    echo "============================================"
    echo " GLM-5.2 W8A8 1M — Ray 单实例"
    echo " TP=$TP PP=$PP DP=$DP MTP=$ENABLE_MTP head=$HEAD"
    echo "============================================"
    check_ray
    ssh -o BatchMode=yes "$HEAD" "docker exec -d $CONTAINER bash -c '
        cd $REPO/examples/glm-5.2_w8a8/vllm
        RAY_ADDRESS=$HEAD:$RAY_PORT NIC_NAME=$NIC_NAME HCCL_IF_IP=$HEAD \
        VLLM_ASCEND_ENABLE_FUSED_MC2=0 \
        TP=$TP PP=$PP DP=$DP ENABLE_MTP=$ENABLE_MTP \
        MAX_NUM_SEQS=$MAX_NUM_SEQS MAX_NUM_BATCHED_TOKENS=$MAX_NUM_BATCHED_TOKENS \
        GPU_MEM_UTIL=$GPU_MEM_UTIL \
        bash run_vllm_1m.sh > $LOG_FILE 2>&1'"
    echo "[OK] vLLM 已在 $HEAD 容器内后台启动"
    echo "     日志: $LOG_FILE    跟踪: tail -f $LOG_FILE"
    echo "     就绪后入口: http://$HEAD:8007 (model: glm-5.2)"
}

cmd_status() {
    echo "vLLM 进程: $(in_head "pgrep -fc 'vllm serve' || echo 0")"
    echo "API ($HEAD:8007): $(in_head "curl -s -o /dev/null -w '%{http_code}' --max-time 5 http://localhost:8007/v1/models" || echo DOWN)"
}

cmd_stop() {
    in_head "pkill -f 'vllm serve' 2>/dev/null || true"
    echo "[OK] vLLM 已停止（Ray 集群未动；崩溃后的显存残留需 ssh $HEAD docker restart $CONTAINER）"
}

LOG_FILE="$REPO/logs/glm5_ray1m_$(date +%Y%m%d_%H%M%S).log"
case "${1:-deploy}" in
    deploy) cmd_deploy ;;
    status) cmd_status ;;
    stop)   cmd_stop ;;
    *) echo "Usage: $0 {deploy|status|stop}"; exit 1 ;;
esac
