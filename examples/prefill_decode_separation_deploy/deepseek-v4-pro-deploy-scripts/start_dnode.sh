#!/bin/bash
# ==============================================================================
# start_dnode.sh — 启动指定编号的 Decode 节点
# ==============================================================================
# 用法:
#   ./start_dnode.sh <node_index>
#
# 示例（在 10.18.1.14 上执行）:
#   ./start_dnode.sh 0
#
# 本脚本读取 deploy.conf，自动确定:
#   - local_ip       = DNODE_IPS[node_index]
#   - dp_rank_start  = DNODE_RANK_STARTS[node_index]  (0/2/4/6，非节点序号)
#   - dp_address     = D_DP_ADDRESS (rank-0 节点的 IP)
# 然后调用 launch_online_dp.py 启动 vLLM。
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- 加载配置 ---------------------------------------------------------------
source "${SCRIPT_DIR}/deploy.conf"

# ---- 参数检查 ---------------------------------------------------------------
if [ $# -ne 1 ]; then
    echo "用法: $0 <node_index>"
    echo "  node_index: DNode 节点编号 (0-$(( ${#DNODE_IPS[@]} - 1 )))"
    echo "  对应关系:"
    for i in "${!DNODE_IPS[@]}"; do
        echo "    $i -> ${DNODE_IPS[$i]}  (dp-rank-start ${DNODE_RANK_STARTS[$i]})"
    done
    exit 1
fi

node_index="$1"

if ! [[ "$node_index" =~ ^[0-9]+$ ]] || \
   [ "$node_index" -ge "${#DNODE_IPS[@]}" ]; then
    echo "错误: node_index 必须是 0 到 $(( ${#DNODE_IPS[@]} - 1 )) 之间的整数"
    exit 1
fi

# ---- 确定本节点参数 ---------------------------------------------------------
local_ip="${DNODE_IPS[$node_index]}"
dp_rank_start="${DNODE_RANK_STARTS[$node_index]}"   # DNode: 0/2/4/6

# ---- 注入环境变量（供 dnode.sh 读取）----------------------------------------
export LOCAL_IP="$local_ip"
export NIC_NAME="$NIC_NAME"
export MODEL_PATH="$MODEL_PATH"
export LOG_DIR="$LOG_DIR"

echo "============================================================"
echo "  启动 DNode $node_index"
echo "  Local IP        : $local_ip"
echo "  DP Size         : $D_DP_SIZE"
echo "  TP Size         : $D_TP_SIZE"
echo "  DP Size Local   : $D_DP_SIZE_LOCAL"
echo "  DP Rank Start   : $dp_rank_start"
echo "  DP Address      : $D_DP_ADDRESS"
echo "  DP RPC Port     : $D_DP_RPC_PORT"
echo "  vLLM Start Port : $D_VLLM_START_PORT"
echo "  Model Path      : $MODEL_PATH"
echo "============================================================"

# ---- 清理旧日志 -------------------------------------------------------------
	rm -f "$LOG_DIR"/dnode_*.log

# ---- 调用通用启动器 ---------------------------------------------------------
cd "$SCRIPT_DIR"
python launch_online_dp.py \
    --script ./dnode.sh \
    --dp-size "$D_DP_SIZE" \
    --tp-size "$D_TP_SIZE" \
    --dp-size-local "$D_DP_SIZE_LOCAL" \
    --dp-rank-start "$dp_rank_start" \
    --dp-address "$D_DP_ADDRESS" \
    --dp-rpc-port "$D_DP_RPC_PORT" \
    --vllm-start-port "$D_VLLM_START_PORT"
