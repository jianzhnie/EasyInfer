#!/bin/bash
# =============================================================================
# Qwen3.8-27B-W8A8 - Direct vllm serve deployment
# =============================================================================
# Architecture: Qwen3_5ForConditionalGeneration | Dense hybrid attention
# 64 layers (48 linear attention + 16 full attention) | Vision + MTP=1
# Default: TP=2 PP=1 DP=1 (single Atlas 800 A2/A3 node)
#
# Usage:
#   bash run_vllm.sh
#   ENABLE_MTP=0 MAX_MODEL_LEN=32768 bash run_vllm.sh
#   RAY_ADDRESS=<head>:6379 TP=2 PP=2 bash run_vllm.sh
#
# Reference:
#   https://docs.vllm.ai/projects/ascend/zh-cn/latest/tutorials/models/Qwen3.8-27B.html
# =============================================================================
set -euo pipefail

# Load Ascend CANN environment when running inside the vLLM-Ascend container.
set +u
if [[ -f "/usr/local/Ascend/cann/set_env.sh" ]]; then
    source "/usr/local/Ascend/cann/set_env.sh"
fi
if [[ -f "/usr/local/Ascend/nnal/atb/set_env.sh" ]]; then
    source "/usr/local/Ascend/nnal/atb/set_env.sh"
fi
set -u

# ---- Model and server -------------------------------------------------------
readonly BASE_MODEL_PATH="/home/jianzhnie/llmtuner/hfhub/models/Eco-Tech"
readonly MODEL_PATH="${MODEL_PATH:-$BASE_MODEL_PATH/Qwen3.8-27B-w8a8}"
readonly HOST="${HOST:-0.0.0.0}"
readonly PORT="${PORT:-8022}"
readonly SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3.8}"

# ---- Parallelism ------------------------------------------------------------
readonly TP="${TP:-2}"
readonly PP="${PP:-1}"
readonly DP="${DP:-1}"
readonly DISTRIBUTED_EXECUTOR_BACKEND="${DISTRIBUTED_EXECUTOR_BACKEND:-}"
readonly RAY_ADDRESS="${RAY_ADDRESS:-}"

# ---- Memory and scheduling --------------------------------------------------
readonly DTYPE="${DTYPE:-bfloat16}"
readonly QUANTIZATION="${QUANTIZATION:-ascend}"
readonly MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
readonly MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
readonly MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-16384}"
readonly GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
readonly ENABLE_MTP="${ENABLE_MTP:-1}"
readonly ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-1}"
readonly ENABLE_CHUNKED_PREFILL="${ENABLE_CHUNKED_PREFILL:-1}"
readonly ENABLE_CPU_BINDING="${ENABLE_CPU_BINDING:-1}"
readonly LANGUAGE_MODEL_ONLY="${LANGUAGE_MODEL_ONLY:-0}"

# ---- NPU environment --------------------------------------------------------
readonly FLASHCOMM1="${FLASHCOMM1:-0}"
readonly MLAPO="${MLAPO:-0}"
readonly HCCL_BUFFSIZE="${HCCL_BUFFSIZE:-512}"
readonly NIC_NAME="${NIC_NAME:-}"
readonly HCCL_IF_IP="${HCCL_IF_IP:-}"

export VLLM_USE_MODELSCOPE=False
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_OP_EXPANSION_MODE=AIV
export HCCL_BUFFSIZE
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export TASK_QUEUE_ENABLE=1
export VLLM_ASCEND_BALANCE_SCHEDULING=1
export VLLM_ASCEND_ENABLE_FLASHCOMM1="$FLASHCOMM1"
export VLLM_ASCEND_ENABLE_MLAPO="$MLAPO"
if [[ -n "$RAY_ADDRESS" ]]; then
    export RAY_ADDRESS
fi

if [[ -n "$HCCL_IF_IP" ]]; then
    export HCCL_IF_IP
fi
if [[ -n "$NIC_NAME" ]]; then
    export GLOO_SOCKET_IFNAME="$NIC_NAME"
    export TP_SOCKET_IFNAME="$NIC_NAME"
    export HCCL_SOCKET_IFNAME="$NIC_NAME"
fi

# A PP deployment spans nodes in the documented cluster setup. Ray must be
# explicit in that case so the Engine Core process joins the existing cluster.
if [[ ( "$PP" -gt 1 || "$TP" -gt 8 ) && -z "$RAY_ADDRESS" ]]; then
    echo "ERROR: PP>1 or TP>8 requires RAY_ADDRESS=<ray-head>:6379 for multi-node deployment" >&2
    exit 1
fi

if [[ "$MODEL_PATH" == /* && ! -d "$MODEL_PATH" ]]; then
    echo "ERROR: MODEL_PATH does not exist: $MODEL_PATH" >&2
    exit 1
fi

# vLLM defaults to multiprocessing for a local multi-NPU process. Select Ray
# automatically when the caller provides a Ray GCS address.
EXECUTOR_BACKEND="$DISTRIBUTED_EXECUTOR_BACKEND"
if [[ -z "$EXECUTOR_BACKEND" && -n "$RAY_ADDRESS" ]]; then
    EXECUTOR_BACKEND="ray"
fi
if [[ -z "$EXECUTOR_BACKEND" && ( "$TP" -gt 1 || "$PP" -gt 1 || "$DP" -gt 1 ) ]]; then
    EXECUTOR_BACKEND="mp"
fi

VLLM_ARGS=(
    --host "$HOST"
    --port "$PORT"
    --served-model-name "$SERVED_MODEL_NAME"
    --trust-remote-code
    --dtype "$DTYPE"
    --tensor-parallel-size "$TP"
    --pipeline-parallel-size "$PP"
    --data-parallel-size "$DP"
    --gpu-memory-utilization "$GPU_MEM_UTIL"
    --max-model-len "$MAX_MODEL_LEN"
    --max-num-seqs "$MAX_NUM_SEQS"
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
    --seed 1024
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'
)

if [[ "$QUANTIZATION" != "none" ]]; then
    VLLM_ARGS+=(--quantization "$QUANTIZATION")
fi
if [[ "$LANGUAGE_MODEL_ONLY" == "1" ]]; then
    VLLM_ARGS+=(--language-model-only)
else
    VLLM_ARGS+=(
        --mm-encoder-tp-mode data
        --allowed-local-media-path /home/jianzhnie/llmtuner/
    )
fi

if [[ "$ENABLE_PREFIX_CACHING" == "1" ]]; then
    VLLM_ARGS+=(--enable-prefix-caching)
else
    VLLM_ARGS+=(--no-enable-prefix-caching)
fi
if [[ "$ENABLE_CHUNKED_PREFILL" == "1" ]]; then
    VLLM_ARGS+=(--enable-chunked-prefill)
fi
if [[ "$ENABLE_MTP" == "1" ]]; then
    VLLM_ARGS+=(--speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":3,"enforce_eager":true}')
fi
if [[ -n "$EXECUTOR_BACKEND" ]]; then
    VLLM_ARGS+=(--distributed-executor-backend "$EXECUTOR_BACKEND")
fi

if [[ "$ENABLE_CPU_BINDING" == "1" ]]; then
    readonly ADDITIONAL_CONFIG='{"enable_cpu_binding":true}'
else
    readonly ADDITIONAL_CONFIG='{}'
fi
VLLM_ARGS+=(--additional-config "$ADDITIONAL_CONFIG")

echo "============================================"
echo "[INFO] Qwen3.8-27B-W8A8 - vLLM-Ascend Deployment"
echo "[INFO] Model:       $MODEL_PATH"
echo "[INFO] Served name: $SERVED_MODEL_NAME"
echo "[INFO] TP=$TP PP=$PP DP=$DP PORT=$PORT"
echo "[INFO] DTYPE=$DTYPE QUANTIZATION=$QUANTIZATION"
echo "[INFO] MAX_MODEL_LEN=$MAX_MODEL_LEN MAX_NUM_SEQS=$MAX_NUM_SEQS"
echo "[INFO] MAX_NUM_BATCHED_TOKENS=$MAX_NUM_BATCHED_TOKENS GPU_MEM_UTIL=$GPU_MEM_UTIL"
echo "[INFO] MTP=$ENABLE_MTP PrefixCaching=$ENABLE_PREFIX_CACHING ChunkedPrefill=$ENABLE_CHUNKED_PREFILL"
echo "[INFO] LANGUAGE_MODEL_ONLY=$LANGUAGE_MODEL_ONLY"
echo "[INFO] FLASHCOMM1=$FLASHCOMM1 MLAPO=$MLAPO HCCL_BUFFSIZE=$HCCL_BUFFSIZE"
echo "[INFO] Executor backend=${EXECUTOR_BACKEND:-default}"
echo "============================================"

exec vllm serve "$MODEL_PATH" "${VLLM_ARGS[@]}" "$@"
