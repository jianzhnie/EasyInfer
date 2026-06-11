#!/bin/bash
# DeepSeek-V4-Pro W4A8 — 4-Node Data-Parallel Deployment for A2 (64G×8)
# Reference: https://docs.vllm.ai/projects/ascend/zh-cn/releases-v0.20.2rc/tutorials/models/DeepSeek-V4-Pro.html
# Each node runs its own Ray + vllm serve with DP coordination.
#
# Usage (run on EACH node):
#   MASTER_IP=10.16.201.40 DP_RANK=0 bash run_vllm_dp.sh   # Node 0 (master)
#   MASTER_IP=10.16.201.40 DP_RANK=1 bash run_vllm_dp.sh   # Node 1
#   MASTER_IP=10.16.201.40 DP_RANK=2 bash run_vllm_dp.sh   # Node 2
#   MASTER_IP=10.16.201.40 DP_RANK=3 bash run_vllm_dp.sh   # Node 3
set -eo pipefail

set +u
if [[ -f "/usr/local/Ascend/cann/set_env.sh" ]]; then
    source /usr/local/Ascend/cann/set_env.sh
fi
if [[ -f "/usr/local/Ascend/nnal/atb/set_env.sh" ]]; then
    source /usr/local/Ascend/nnal/atb/set_env.sh
fi
set -u

# --- Node configuration ---
MASTER_IP="${MASTER_IP:-10.16.201.40}"
DP_RANK="${DP_RANK:-0}"
DP_SIZE="${DP_SIZE:-4}"
DP_RPC_PORT="${DP_RPC_PORT:-13399}"
LOCAL_IP="${LOCAL_IP:-$(hostname -I | awk '{print $1}')}"
NIC="${NIC:-enp66s0f5}"

# --- Model configuration ---
BASE_MODEL_PATH="/home/jianzhnie/llmtuner/hfhub/models/Eco-Tech"
MODEL_PATH="${MODEL_PATH:-$BASE_MODEL_PATH/DeepSeek-V4-Pro-w4a8-mtp}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
TP="${TP:-8}"
PP="${PP:-1}"

# --- Official env vars for V4-Pro A2 multi-node ---
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:${LD_PRELOAD:-}
export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_BUFFSIZE="${HCCL_BUFFSIZE:-1024}"
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export TASK_QUEUE_ENABLE=1
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
export VLLM_ASCEND_APPLY_DSV4_PATCH=1
export VLLM_RPC_TIMEOUT=3600000
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
export HCCL_EXEC_TIMEOUT="${HCCL_EXEC_TIMEOUT:-204}"
export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-1200}"
export VLLM_HOST_IP="$LOCAL_IP"
export HCCL_IF_IP="$LOCAL_IP"
export GLOO_SOCKET_IFNAME="$NIC"
export TP_SOCKET_IFNAME="$NIC"
export HCCL_SOCKET_IFNAME="$NIC"

# Workers use headless mode; master serves API
HEADLESS_FLAG=""
if [[ "$DP_RANK" != "0" ]]; then
    HEADLESS_FLAG="--headless"
fi

echo "============================================"
echo "[INFO] DeepSeek-V4-Pro W4A8 — DP Deployment"
echo "[INFO] DP Rank: $DP_RANK/$DP_SIZE | Master: $MASTER_IP:$DP_RPC_PORT"
echo "[INFO] TP=$TP PP=$PP PORT=$PORT Headless: ${HEADLESS_FLAG:-no}"
echo "[INFO] Local IP: $LOCAL_IP NIC: $NIC"
echo "============================================"

vllm serve "$MODEL_PATH" \
    $HEADLESS_FLAG \
    --host "$HOST" \
    --port "$PORT" \
    --served-model-name deepseek-v4-pro \
    --trust-remote-code \
    --dtype bfloat16 \
    --tensor-parallel-size "$TP" \
    --pipeline-parallel-size "$PP" \
    --data-parallel-size "$DP_SIZE" \
    --data-parallel-size-local 1 \
    --data-parallel-start-rank "$DP_RANK" \
    --data-parallel-address "$MASTER_IP" \
    --data-parallel-rpc-port "$DP_RPC_PORT" \
    --distributed-executor-backend ray \
    --enable-expert-parallel \
    --quantization ascend \
    --gpu-memory-utilization 0.90 \
    --max-model-len 131072 \
    --max-num-seqs 8 \
    --max-num-batched-tokens 8192 \
    --enable-chunked-prefill \
    --enable-prefix-caching \
    --enforce-eager \
    --speculative-config '{"num_speculative_tokens": 1, "method": "mtp", "enforce_eager": true}' \
    --enable-auto-tool-choice \
    --tool-call-parser deepseek_v4 \
    --seed 1024 \
    "$@"
