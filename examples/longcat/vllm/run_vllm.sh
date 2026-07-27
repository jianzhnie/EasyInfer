#!/bin/bash
# =============================================================================
# LongCat-Flash-Chat — Direct vllm serve deployment
# =============================================================================
# Architecture: LongcatFlashForCausalLM | 512 Routed Experts + 256 Zero | MLA
# Default: TP=64 EP=64 PP=1 (multi-node via Ray, 8 nodes × 8 NPU)
# Note: Massive MoE model (~560B params, 512 experts, topk=12), requires 64
#       NPUs minimum. Uses --trust-remote-code for custom modeling code.
#       No quantization (bfloat16 native weights).
#
# Usage:
#   bash run_vllm.sh                          # EP mode (default, EP=1)
#   EP=0 bash run_vllm.sh                     # pure TP mode
#   TP=64 MAX_MODEL_LEN=8192 bash run_vllm.sh
#
# Notes:
#   - MLA attention kernel only supports block size 128; baked in
#     (override with BLOCK_SIZE=<n>).
#   - Chunked prefill is disabled by default (CHUNKED_PREFILL=1 to enable).
#   - MC2 MoE comm is incompatible with zero-expert weight zeroing
#     (MoeDistributeCombineV2 shape check fails -> collective hang).
#     The EasyInfer plugin overrides the comm method to ALLGATHER via
#     EASYINFER_MOE_COMM=allgather (set automatically when EP=1).
#
# Reference:
#   https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/index.html
# =============================================================================
set -euo pipefail

# Load Ascend CANN environment
set +u
if [[ -f "/usr/local/Ascend/cann/set_env.sh" ]]; then
    source /usr/local/Ascend/cann/set_env.sh
fi
if [[ -f "/usr/local/Ascend/nnal/atb/set_env.sh" ]]; then
    source /usr/local/Ascend/nnal/atb/set_env.sh
fi
set -u

# Base configuration
readonly BASE_MODEL_PATH="/home/jianzhnie/llmtuner/hfhub/models/meituan-longcat"
readonly MODEL_PATH="${MODEL_PATH:-$BASE_MODEL_PATH/LongCat-Flash-Chat}"
readonly HOST="${HOST:-0.0.0.0}"
readonly PORT="${PORT:-8010}"
readonly TP="${TP:-64}"
readonly PP="${PP:-1}"
readonly DP="${DP:-1}"
readonly ENABLE_EP="${EP:-1}"
readonly EXECUTOR="${EXECUTOR:-ray}"
readonly MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
readonly MAX_NUM_SEQS="${MAX_NUM_SEQS:-128}"
readonly GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
readonly MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
readonly SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-longcat-flash}"
readonly DTYPE="${DTYPE:-bfloat16}"
# MLA attention kernel only supports block size 128 on this image.
readonly BLOCK_SIZE="${BLOCK_SIZE:-128}"
# Chunked prefill conflicts with EP token dispatch; disable by default.
readonly CHUNKED_PREFILL="${CHUNKED_PREFILL:-0}"
# enforce-eager 会禁用 cudagraph (FULL_DECODE_ONLY 随之失效)。默认开启
# (历史稳定路径); 设 ENFORCE_EAGER=0 启用 decode graph 以提升吞吐, 待验证。
readonly ENFORCE_EAGER="${ENFORCE_EAGER:-1}"

# ------------------------------------------------------------------------------
# Ensure EasyInfer plugins are registered (required for the EP fixes)
# ------------------------------------------------------------------------------
pip install --no-build-isolation --no-deps -e /home/jianzhnie/llmtuner/llm/EasyInfer --quiet 2>/dev/null || true

# ------------------------------------------------------------------------------
# Log file (default: <repo>/logs/vllm_longcat_<timestamp>.log, override with LOG_FILE)
# ------------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs}"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/vllm_longcat_$(date +%Y%m%d_%H%M%S).log}"
readonly LOG_FILE
echo "[INFO] Log file: $LOG_FILE"
exec > >(tee -a "$LOG_FILE") 2>&1

# ------------------------------------------------------------------------------
# NPU environment variables
# ------------------------------------------------------------------------------
# Auto-detect network interface
if [[ -z "${HCCL_SOCKET_IFNAME:-}" ]]; then
    HCCL_SOCKET_IFNAME="$(ip -o -4 route show default | awk '{print $5}' | head -1)"
    HCCL_SOCKET_IFNAME="${HCCL_SOCKET_IFNAME:-enp66s0f5}"
fi
if [[ -z "${GLOO_SOCKET_IFNAME:-}" ]]; then
    GLOO_SOCKET_IFNAME="$HCCL_SOCKET_IFNAME"
fi

export HCCL_OP_EXPANSION_MODE=AIV
export HCCL_SOCKET_IFNAME
export GLOO_SOCKET_IFNAME
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export HCCL_BUFFSIZE="${HCCL_BUFFSIZE:-800}"
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_USE_MODELSCOPE=False

# HCCL multi-node communication
export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-1800}"
export HCCL_EXEC_TIMEOUT="${HCCL_EXEC_TIMEOUT:-1800}"

# Scheduling
export VLLM_ASCEND_BALANCE_SCHEDULING=1
export VLLM_ASCEND_ENABLE_FLASHCOMM1="${VLLM_ASCEND_ENABLE_FLASHCOMM1:-1}"
export VLLM_ASCEND_ENABLE_MLAPO="${VLLM_ASCEND_ENABLE_MLAPO:-1}"
# 主机侧任务队列, 大 batch / 长 prefill 下降低下发开销 (GLM-5.2 教程同款)
export TASK_QUEUE_ENABLE="${TASK_QUEUE_ENABLE:-1}"
if [[ "$PP" -gt 1 || "$TP" -gt 8 ]]; then
    export VLLM_ASCEND_ENABLE_FUSED_MC2=1
else
    export VLLM_ASCEND_ENABLE_FUSED_MC2=0
fi

# ------------------------------------------------------------------------------
# Expert Parallel (optional)
# ------------------------------------------------------------------------------
if [[ "$ENABLE_EP" == "1" ]]; then
    export ENABLE_EXPERT_PARALLEL=1
    # MC2 MoE comm breaks with zero-expert weight zeroing (MoeDistributeCombineV2
    # shape check fails -> collective hang).  The EasyInfer plugin overrides the
    # comm method to ALLGATHER (see fix_ep_zero_expert.py); default to allgather
    # here since MC2 is a known hang for this model.  Set EASYINFER_MOE_COMM=""
    # explicitly only if you want to experiment with the native MC2 path.
    export EASYINFER_MOE_COMM="${EASYINFER_MOE_COMM:-allgather}"
fi

# 前置检查
command -v vllm >/dev/null 2>&1 || { echo "[ERROR] vllm not found" >&2; exit 127; }
[[ -d "$MODEL_PATH" ]] || { echo "[ERROR] MODEL_PATH not found: $MODEL_PATH" >&2; exit 2; }

# 层数必须能被 PP 均分 (LongCat-Flash 共 28 层)
if (( 28 % PP != 0 )); then
    echo "[ERROR] PP=$PP 无法整除模型 28 层, 可选 PP=1/2/4/7/14/28" >&2
    exit 2
fi

# Ray 集群 NPU 数必须 >= TP×PP×DP; 集群退化时 vllm 会无限期等 placement
# group (无报错挂起), 提前退出更省时间
REQUIRED_NPUS=$((TP * PP * DP))
if [[ "$EXECUTOR" == "ray" ]] && command -v ray >/dev/null 2>&1; then
    AVAIL_NPUS=$(ray status 2>/dev/null | grep -oE '[0-9]+\.[0-9]+/[0-9]+\.[0-9]+ NPU' \
        | head -1 | cut -d/ -f2 | cut -d. -f1)
    if [[ -n "$AVAIL_NPUS" ]]; then
        if (( AVAIL_NPUS < REQUIRED_NPUS )); then
            echo "[ERROR] Ray 集群只有 ${AVAIL_NPUS} NPU, 当前配置需要 ${REQUIRED_NPUS}" \
                "(TP=$TP × PP=$PP × DP=$DP)。请检查节点掉线或调小并行度。" >&2
            exit 3
        fi
        echo "[INFO] Ray 集群 NPU: ${AVAIL_NPUS} 可用 / 需要 ${REQUIRED_NPUS}"
    else
        echo "[WARN] 无法从 ray status 解析 NPU 数, 跳过集群容量检查"
    fi
fi

# EP + 空 EASYINFER_MOE_COMM = 走 MC2, 对本模型是已知挂起点
if [[ "$ENABLE_EP" == "1" && -z "${EASYINFER_MOE_COMM}" ]]; then
    echo "[WARN] EP=1 且 EASYINFER_MOE_COMM 为空: MC2 与零号专家不兼容," \
        "大概率 collective hang (建议 allgather)"
fi

#=============================================================================
# 启动命令
#=============================================================================

echo "============================================"
echo "[INFO] LongCat-Flash-Chat — Deployment"
echo "[INFO] Model: $MODEL_PATH"
echo "[INFO] TP=$TP PP=$PP DP=$DP EP=$ENABLE_EP Backend=$EXECUTOR"
echo "[INFO] Host: ${HOST}:${PORT}"
echo "[INFO] MAX_MODEL_LEN=$MAX_MODEL_LEN MAX_NUM_SEQS=$MAX_NUM_SEQS"
echo "[INFO] MAX_NUM_BATCHED_TOKENS=$MAX_NUM_BATCHED_TOKENS CHUNKED_PREFILL=$CHUNKED_PREFILL BLOCK_SIZE=$BLOCK_SIZE"
echo "[INFO] GPU_MEM_UTIL=$GPU_MEM_UTIL EASYINFER_MOE_COMM=${EASYINFER_MOE_COMM:-<unset>}"
echo "============================================"

EP_FLAGS=()
if [[ "$ENABLE_EP" == "1" ]]; then
    EP_FLAGS=(--enable-expert-parallel)
fi

PREFILL_FLAGS=(--enable-chunked-prefill)
if [[ "$CHUNKED_PREFILL" == "0" ]]; then
    PREFILL_FLAGS=(--no-enable-chunked-prefill)
fi

EAGER_FLAGS=(--enforce-eager)
if [[ "$ENFORCE_EAGER" == "0" ]]; then
    EAGER_FLAGS=()
fi

vllm serve "$MODEL_PATH" \
    --host "$HOST" \
    --port "$PORT" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --trust-remote-code \
    --dtype "$DTYPE" \
    --tensor-parallel-size "$TP" \
    --pipeline-parallel-size "$PP" \
    --data-parallel-size "$DP" \
    "${EP_FLAGS[@]}" \
    "${PREFILL_FLAGS[@]}" \
    --block-size "$BLOCK_SIZE" \
    --distributed-executor-backend "$EXECUTOR" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
    --no-enable-prefix-caching \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
    "${EAGER_FLAGS[@]}" \
    --seed 1024 \
    "$@"
