#!/bin/bash
# =============================================================================
# DeepSeek-V4-Flash W8A8 MTP — SGLang launch_server 部署
# Architecture: DeepseekV4ForCausalLM | 256 Experts | MLA | MTP=1
# Max Position: 1048576 | Deploy: 64K context (override with CONTEXT_LEN)
#
# 4 节点部署: TP=32 PP=1 EP=32 (32 NPU total = 4 nodes × 8 NPU)
# 单节点部署: TP=8 PP=1 EP=8
#
# SGLang 特性:
#   - RadixAttention 自动前缀缓存（无需额外配置）
#   - torch.distributed 原生多节点（无需 Ray）
#   - EAGLE 推测解码（MTP 模型）
# =============================================================================
set -eo pipefail

# Load Ascend CANN environment (required for libascend_hal.so)
# CANN scripts reference unset vars; disable nounset during source
set +u
if [[ -f "/usr/local/Ascend/cann/set_env.sh" ]]; then
    source /usr/local/Ascend/cann/set_env.sh
fi
if [[ -f "/usr/local/Ascend/nnal/atb/set_env.sh" ]]; then
    source /usr/local/Ascend/nnal/atb/set_env.sh
fi
set -u

MODEL_PATH="${MODEL_PATH:-/home/jianzhnie/llmtuner/hfhub/models/Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
TP="${TP:-32}"
PP="${PP:-1}"
EP="${EP:-32}"
CONTEXT_LEN="${CONTEXT_LEN:-65536}"
MAX_RUNNING_REQS="${MAX_RUNNING_REQS:-16}"
MEM_FRACTION="${MEM_FRACTION:-0.90}"

# Multi-node settings (torch.distributed)
NNODES="${NNODES:-4}"
NODE_RANK="${NODE_RANK:-0}"
DIST_INIT_ADDR="${DIST_INIT_ADDR:-10.16.201.193}"
DIST_INIT_PORT="${DIST_INIT_PORT:-5000}"

# HCCL/NPU performance optimizations
export HCCL_OP_EXPANSION_MODE="${HCCL_OP_EXPANSION_MODE:-AIV}"
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export HCCL_BUFFSIZE=200
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export SGLANG_ASCEND_BALANCE_SCHEDULING=1

echo "============================================"
echo "[INFO] DeepSeek-V4-Flash W8A8 MTP — SGLang Deployment"
echo "[INFO] TP=$TP PP=$PP EP=$EP PORT=$PORT"
echo "[INFO] CONTEXT_LEN=$CONTEXT_LEN MAX_RUNNING_REQS=$MAX_RUNNING_REQS"
echo "[INFO] MEM_FRACTION=$MEM_FRACTION"
echo "[INFO] NNODES=$NNODES NODE_RANK=$NODE_RANK DIST_INIT_ADDR=$DIST_INIT_ADDR"
echo "[INFO] Prefix Caching: RadixAttention (automatic)"
echo "[INFO] Speculative Decoding: EAGLE (MTP=1)"
echo "============================================"

python -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --host "$HOST" \
    --port "$PORT" \
    --served-model-name deepseek-v4-flash \
    --trust-remote-code \
    --dtype bfloat16 \
    --tp "$TP" \
    --pp "$PP" \
    --ep "$EP" \
    --device npu \
    --quantization modelopt_fp8 \
    --mem-fraction-static "$MEM_FRACTION" \
    --context-length "$CONTEXT_LEN" \
    --max-running-requests "$MAX_RUNNING_REQS" \
    --max-total-tokens 8192 \
    --chunked-prefill-size 8192 \
    --enable-torch-compile \
    --speculative-algorithm EAGLE \
    --speculative-num-draft-tokens 3 \
    --enable-tool-call \
    --tool-call-parser deepseek_v3 \
    --nnodes "$NNODES" \
    --node-rank "$NODE_RANK" \
    --dist-init-addr "$DIST_INIT_ADDR" \
    --dist-init-port "$DIST_INIT_PORT" \
    --log-level info \
    "$@"
