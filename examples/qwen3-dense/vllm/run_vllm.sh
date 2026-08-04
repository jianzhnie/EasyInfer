#!/bin/bash
# =============================================================================
# Qwen3-Dense (Qwen3-8B) — vllm serve deployment (BF16 / optional YaRN)
# =============================================================================
# Architecture: Qwen3 dense | 8.2B total / 6.95B non-embedding | GQA (32H/8KVH)
# First supported: v0.8.4rc2 | Reasoning parser: qwen3 | Native ctx: 32768
#
# Hardware:
#   - BF16: 1x Atlas 800 A3 or A2 node (64G)
#
# Usage:
#   bash run_vllm.sh
#   ENABLE_YARN=1 MAX_MODEL_LEN=131072 bash run_vllm.sh
#   TP=2 DISTRIBUTED_EXECUTOR_BACKEND=mp bash run_vllm.sh
#   DISTRIBUTED_EXECUTOR_BACKEND=ray RAY_ADDRESS=<head>:6379 TP=2 bash run_vllm.sh
#   QUANTIZATION=ascend DTYPE=float16 MODEL_PATH=<quantized-model> bash run_vllm.sh
#
# Reference:
#   https://docs.vllm.ai/projects/ascend/zh-cn/latest/tutorials/models/Qwen3-Dense.html
#   https://qwen.readthedocs.io/en/latest/deployment/vllm.html
# =============================================================================
set -euo pipefail

# Load Ascend CANN environment
set +u
if [[ -f "/usr/local/Ascend/cann/set_env.sh" ]]; then
    source "/usr/local/Ascend/cann/set_env.sh"
fi
if [[ -f "/usr/local/Ascend/nnal/atb/set_env.sh" ]]; then
    source "/usr/local/Ascend/nnal/atb/set_env.sh"
fi
set -u

# ---- Model & Server ----
readonly BASE_MODEL_PATH="/home/jianzhnie/llmtuner/hfhub/models/Qwen"
readonly MODEL_PATH="${MODEL_PATH:-$BASE_MODEL_PATH/Qwen3-8B}"
readonly HOST="${HOST:-0.0.0.0}"
readonly PORT="${PORT:-8021}"
readonly SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3}"

# ---- Parallelism ----
readonly TP="${TP:-1}"
readonly PP="${PP:-1}"
readonly DP="${DP:-1}"
readonly DISTRIBUTED_EXECUTOR_BACKEND="${DISTRIBUTED_EXECUTOR_BACKEND:-}"

# ---- Memory & Context ----
readonly DTYPE="${DTYPE:-bfloat16}"
readonly QUANTIZATION="${QUANTIZATION:-none}"
readonly MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
readonly MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
readonly MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
readonly GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
readonly ENABLE_YARN="${ENABLE_YARN:-0}"
readonly ROPE_SCALING="${ROPE_SCALING:-{\"rope_type\":\"yarn\",\"factor\":4.0,\"original_max_position_embeddings\":32768}}"

# ---- Qwen3 parsing ----
readonly ENABLE_REASONING="${ENABLE_REASONING:-1}"
readonly ENABLE_TOOL_CALLING="${ENABLE_TOOL_CALLING:-1}"

# ---- NPU environment ----
readonly FLASHCOMM1="${FLASHCOMM1:-0}"
readonly MLAPO="${MLAPO:-0}"
readonly HCCL_BUFFSIZE="${HCCL_BUFFSIZE:-512}"

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

if [[ "$ENABLE_YARN" == "1" && "$MAX_MODEL_LEN" -lt 131072 ]]; then
    echo "[WARN] ENABLE_YARN=1 is usually paired with MAX_MODEL_LEN=131072"
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
    --chat-template-content-format string
    --enable-prefix-caching
    --enable-chunked-prefill
    --seed 1024
)

if [[ "$ENABLE_REASONING" == "1" ]]; then
    VLLM_ARGS+=(--reasoning-parser qwen3)
fi

if [[ "$ENABLE_TOOL_CALLING" == "1" ]]; then
    VLLM_ARGS+=(--enable-auto-tool-choice --tool-call-parser hermes)
fi

if [[ "$ENABLE_YARN" == "1" ]]; then
    VLLM_ARGS+=(--rope-scaling "$ROPE_SCALING")
fi

EXECUTOR_BACKEND="$DISTRIBUTED_EXECUTOR_BACKEND"
if [[ -z "$EXECUTOR_BACKEND" ]]; then
    if [[ -n "${RAY_ADDRESS:-}" ]]; then
        EXECUTOR_BACKEND="ray"
    elif [[ "$TP" -gt 1 || "$PP" -gt 1 || "$DP" -gt 1 ]]; then
        EXECUTOR_BACKEND="mp"
    fi
fi
if [[ -n "$EXECUTOR_BACKEND" ]]; then
    VLLM_ARGS+=(--distributed-executor-backend "$EXECUTOR_BACKEND")
fi

echo "============================================"
echo "[INFO] Qwen3-Dense (Qwen3-8B) — vLLM-Ascend Deployment"
echo "[INFO] Model:       $MODEL_PATH"
echo "[INFO] Served name:  $SERVED_MODEL_NAME"
echo "[INFO] TP=$TP  PP=$PP  DP=$DP  PORT=$PORT"
echo "[INFO] DTYPE=$DTYPE  QUANTIZATION=$QUANTIZATION"
echo "[INFO] MAX_MODEL_LEN=$MAX_MODEL_LEN  MAX_NUM_SEQS=$MAX_NUM_SEQS"
echo "[INFO] MAX_NUM_BATCHED_TOKENS=$MAX_NUM_BATCHED_TOKENS"
echo "[INFO] GPU_MEM_UTIL=$GPU_MEM_UTIL"
echo "[INFO] Reasoning=qwen3  ToolCalling=hermes  YaRN=$ENABLE_YARN"
echo "[INFO] FLASHCOMM1=$FLASHCOMM1  MLAPO=$MLAPO  HCCL_BUFFSIZE=$HCCL_BUFFSIZE"
echo "[INFO] Executor backend=${EXECUTOR_BACKEND:-default}"
echo "============================================"

if [[ "$QUANTIZATION" != "none" ]]; then
    VLLM_ARGS+=(--quantization "$QUANTIZATION")
fi

exec vllm serve "$MODEL_PATH" "${VLLM_ARGS[@]}" "$@"
