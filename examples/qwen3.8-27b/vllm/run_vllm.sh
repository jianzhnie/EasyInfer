#!/bin/bash
# =============================================================================
# Qwen3.8-27B-W8A8 - Direct vllm serve deployment
# =============================================================================
# Architecture: Qwen3_5ForConditionalGeneration | Dense hybrid attention
# 64 layers (48 linear attention + 16 full attention) | Vision + MTP=1
# Default profile: throughput (TP=2, DP=4, all 8 NPUs)
# Optional profile: long-context-1m (TP=4, DP=2, static YaRN 4x)
#
# Usage:
#   bash run_vllm.sh
#   DEFAULT_REASONING_EFFORT=medium bash run_vllm.sh
#   MODEL_PATH=Eco-Tech/Qwen3.8-27B-w8a8 VLLM_USE_MODELSCOPE=True bash run_vllm.sh
#   DEPLOY_PROFILE=long-context-1m bash run_vllm.sh
#   DEPLOY_PROFILE=long-context-1m DRY_RUN=1 bash run_vllm.sh
#   CUDAGRAPH_MODE=PIECEWISE DRY_RUN=1 bash run_vllm.sh  # troubleshooting only
#   DP=1 DP_LOCAL=1 MAX_NUM_SEQS=32 \
#     MAX_NUM_BATCHED_TOKENS=16384 bash run_vllm.sh  # fallback
#   ENABLE_MTP=0 MAX_MODEL_LEN=32768 bash run_vllm.sh
#   RAY_ADDRESS=<head>:6379 DISTRIBUTED_EXECUTOR_BACKEND=ray \
#     TP=2 PP=2 DP=1 DP_LOCAL=1 bash run_vllm.sh
#
# Reference:
#   https://docs.vllm.ai/projects/ascend/zh-cn/main/tutorials/models/Qwen3.8-27B.html#9
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

# ---- Deployment profile ----------------------------------------------------
readonly DEPLOY_PROFILE="${DEPLOY_PROFILE:-throughput}"
case "$DEPLOY_PROFILE" in
    throughput)
        readonly DEFAULT_TP=2
        readonly DEFAULT_PP=1
        readonly DEFAULT_DP=4
        readonly DEFAULT_MAX_MODEL_LEN=131072
        readonly DEFAULT_MAX_NUM_SEQS=64
        readonly DEFAULT_MAX_NUM_BATCHED_TOKENS=32768
        readonly DEFAULT_ENABLE_MTP=1
        readonly DEFAULT_LANGUAGE_MODEL_ONLY=0
        readonly DEFAULT_ENABLE_YARN=0
        ;;
    long-context-1m)
        readonly DEFAULT_TP=8
        readonly DEFAULT_PP=1
        readonly DEFAULT_DP=1
        readonly DEFAULT_MAX_MODEL_LEN=1000000
        readonly DEFAULT_MAX_NUM_SEQS=1
        readonly DEFAULT_MAX_NUM_BATCHED_TOKENS=32768
        readonly DEFAULT_ENABLE_MTP=0
        readonly DEFAULT_LANGUAGE_MODEL_ONLY=1
        readonly DEFAULT_ENABLE_YARN=1
        ;;
    *)
        echo "ERROR: DEPLOY_PROFILE must be throughput or long-context-1m" >&2
        exit 1
        ;;
esac

# ---- Parallelism ------------------------------------------------------------
readonly TP="${TP:-$DEFAULT_TP}"
readonly PP="${PP:-$DEFAULT_PP}"
readonly DP="${DP:-$DEFAULT_DP}"
readonly DP_LOCAL="${DP_LOCAL:-$DP}"
readonly DP_RANK_START="${DP_RANK_START:-0}"
readonly DP_RPC_PORT="${DP_RPC_PORT:-13390}"
readonly DP_BACKEND="${DP_BACKEND:-mp}"
readonly DISTRIBUTED_EXECUTOR_BACKEND="${DISTRIBUTED_EXECUTOR_BACKEND:-}"
readonly RAY_ADDRESS="${RAY_ADDRESS:-}"
readonly LOCAL_NPU_COUNT="${LOCAL_NPU_COUNT:-8}"

# ---- Memory and scheduling --------------------------------------------------
readonly DTYPE="${DTYPE:-bfloat16}"
readonly QUANTIZATION="${QUANTIZATION:-ascend}"
readonly MAX_MODEL_LEN="${MAX_MODEL_LEN:-$DEFAULT_MAX_MODEL_LEN}"
readonly MAX_NUM_SEQS="${MAX_NUM_SEQS:-$DEFAULT_MAX_NUM_SEQS}"
readonly MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-$DEFAULT_MAX_NUM_BATCHED_TOKENS}"
readonly GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
readonly ENABLE_MTP="${ENABLE_MTP:-$DEFAULT_ENABLE_MTP}"
readonly ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-1}"
readonly ENABLE_CHUNKED_PREFILL="${ENABLE_CHUNKED_PREFILL:-1}"
readonly ENABLE_CPU_BINDING="${ENABLE_CPU_BINDING:-1}"
readonly LANGUAGE_MODEL_ONLY="${LANGUAGE_MODEL_ONLY:-$DEFAULT_LANGUAGE_MODEL_ONLY}"
readonly ENABLE_BALANCE_SCHEDULING="${ENABLE_BALANCE_SCHEDULING:-0}"
readonly ENABLE_YARN="${ENABLE_YARN:-$DEFAULT_ENABLE_YARN}"
readonly YARN_FACTOR="${YARN_FACTOR:-4.0}"
readonly YARN_ORIGINAL_MAX_MODEL_LEN="${YARN_ORIGINAL_MAX_MODEL_LEN:-262144}"
readonly DEFAULT_REASONING_EFFORT="${DEFAULT_REASONING_EFFORT:-xhigh}"
readonly ENABLE_AUTO_TOOL_CHOICE="${ENABLE_AUTO_TOOL_CHOICE:-1}"
readonly TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-qwen3xml}"
readonly CUDAGRAPH_MODE="${CUDAGRAPH_MODE:-FULL_DECODE_ONLY}"
readonly DRY_RUN="${DRY_RUN:-0}"
readonly VLLM_USE_V1="${VLLM_USE_V1:-1}"
readonly VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-1800}"
readonly VLLM_USE_MODELSCOPE="${VLLM_USE_MODELSCOPE:-False}"

readonly YARN_HF_OVERRIDES="{\"text_config\":{\"rope_parameters\":{\"mrope_interleaved\":true,\"mrope_section\":[11,11,10],\"rope_type\":\"yarn\",\"rope_theta\":10000000,\"partial_rotary_factor\":0.25,\"factor\":$YARN_FACTOR,\"original_max_position_embeddings\":$YARN_ORIGINAL_MAX_MODEL_LEN}}}"
readonly HF_OVERRIDES="${HF_OVERRIDES:-$YARN_HF_OVERRIDES}"

# ---- NPU environment --------------------------------------------------------
readonly FLASHCOMM1="${FLASHCOMM1:-0}"
readonly MLAPO="${MLAPO:-0}"
readonly HCCL_BUFFSIZE="${HCCL_BUFFSIZE:-512}"
readonly NIC_NAME="${NIC_NAME:-}"
readonly HCCL_IF_IP="${HCCL_IF_IP:-10.16.201.229}"
readonly DP_ADDRESS="${DP_ADDRESS:-$HCCL_IF_IP}"

if [[ "$VLLM_USE_MODELSCOPE" != "True" && "$VLLM_USE_MODELSCOPE" != "False" ]]; then
    echo "ERROR: VLLM_USE_MODELSCOPE must be True or False" >&2
    exit 1
fi
export VLLM_USE_MODELSCOPE
export VLLM_USE_V1
export VLLM_ENGINE_READY_TIMEOUT_S
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_OP_EXPANSION_MODE=AIV
export HCCL_BUFFSIZE
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export TASK_QUEUE_ENABLE=1
export VLLM_ASCEND_BALANCE_SCHEDULING="$ENABLE_BALANCE_SCHEDULING"
export VLLM_ASCEND_ENABLE_FLASHCOMM1="$FLASHCOMM1"
export VLLM_ASCEND_ENABLE_MLAPO="$MLAPO"
if [[ "$ENABLE_YARN" == "1" ]]; then
    export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
fi
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

if [[ "$MODEL_PATH" == /* && ! -d "$MODEL_PATH" ]]; then
    echo "ERROR: MODEL_PATH does not exist: $MODEL_PATH" >&2
    exit 1
fi

# vLLM's dtype is the compute/storage dtype for non-quantized parameters and
# intermediate tensors. INT8 is selected by the Ascend quantization method,
# not by --dtype. Keep the accepted spellings explicit so a typo cannot make
# the engine fall back to an unintended dtype.
case "$DTYPE" in
    auto|bfloat16|float16|half|float32|float) ;;
    *)
        echo "ERROR: DTYPE must be auto, bfloat16, float16, half, float32, or float (got $DTYPE)" >&2
        exit 1
        ;;
esac

case "$QUANTIZATION" in
    ascend|none) ;;
    *)
        echo "ERROR: QUANTIZATION must be ascend or none for this deployment (got $QUANTIZATION)" >&2
        exit 1
        ;;
esac

# This directory is a ModelSlim W8A8_DYNAMIC checkpoint. Fail early when its
# quantization metadata is missing or when the caller attempts to load it
# through the ordinary (unquantized) path. Model IDs are resolved by
# ModelScope/Hugging Face and are intentionally not inspected locally.
if [[ "$MODEL_PATH" == /* && -d "$MODEL_PATH" ]]; then
    readonly QUANT_DESCRIPTION="$MODEL_PATH/quant_model_description.json"
    readonly QUANT_INDEX="$MODEL_PATH/quant_model_weights.safetensors.index.json"
    if [[ "$QUANTIZATION" == "ascend" ]]; then
        if [[ ! -f "$QUANT_DESCRIPTION" || ! -f "$QUANT_INDEX" ]]; then
            echo "ERROR: QUANTIZATION=ascend requires quant_model_description.json and quant_model_weights.safetensors.index.json in MODEL_PATH" >&2
            exit 1
        fi
    elif [[ "$QUANTIZATION" == "none" && ( -f "$QUANT_DESCRIPTION" || -f "$QUANT_INDEX" ) ]]; then
        echo "ERROR: MODEL_PATH contains ModelSlim W8A8 metadata; keep QUANTIZATION=ascend. Use an unquantized BF16 checkpoint for QUANTIZATION=none." >&2
        exit 1
    fi
fi

# Also protect the model-specific remote ID form (for example
# Eco-Tech/Qwen3.8-27B-w8a8), where the files are not available for a local
# preflight check yet.
if [[ "$QUANTIZATION" == "none" && "${MODEL_PATH,,}" == *w8a8* ]]; then
    echo "ERROR: MODEL_PATH looks like a W8A8 checkpoint; keep QUANTIZATION=ascend. Use an unquantized BF16 checkpoint for QUANTIZATION=none." >&2
    exit 1
fi

if [[ "$QUANTIZATION" == "ascend" && "$DTYPE" != "bfloat16" && "$DTYPE" != "auto" ]]; then
    echo "WARNING: W8A8_DYNAMIC is recommended with DTYPE=bfloat16 (or auto, which follows the model's BF16 config); got DTYPE=$DTYPE" >&2
fi

for integer_var in PORT TP PP DP DP_LOCAL DP_RPC_PORT MAX_MODEL_LEN \
    MAX_NUM_SEQS MAX_NUM_BATCHED_TOKENS LOCAL_NPU_COUNT \
    YARN_ORIGINAL_MAX_MODEL_LEN HCCL_BUFFSIZE VLLM_ENGINE_READY_TIMEOUT_S; do
    if [[ ! "${!integer_var}" =~ ^[1-9][0-9]*$ ]]; then
        echo "ERROR: $integer_var must be a positive integer (got ${!integer_var})" >&2
        exit 1
    fi
done
if [[ ! "$DP_RANK_START" =~ ^(0|[1-9][0-9]*)$ ]]; then
    echo "ERROR: DP_RANK_START must be a non-negative integer" >&2
    exit 1
fi
if (( PORT > 65535 || DP_RPC_PORT > 65535 )); then
    echo "ERROR: PORT and DP_RPC_PORT must be <= 65535" >&2
    exit 1
fi
if [[ "$DP" -gt 1 && "$PORT" == "$DP_RPC_PORT" ]]; then
    echo "ERROR: PORT and DP_RPC_PORT must be different when DP>1" >&2
    exit 1
fi
for boolean_var in ENABLE_MTP ENABLE_PREFIX_CACHING ENABLE_CHUNKED_PREFILL \
    ENABLE_CPU_BINDING LANGUAGE_MODEL_ONLY ENABLE_BALANCE_SCHEDULING \
    ENABLE_YARN ENABLE_AUTO_TOOL_CHOICE DRY_RUN VLLM_USE_V1; do
    if [[ "${!boolean_var}" != "0" && "${!boolean_var}" != "1" ]]; then
        echo "ERROR: $boolean_var must be 0 or 1 (got ${!boolean_var})" >&2
        exit 1
    fi
done
if [[ ! "$YARN_FACTOR" =~ ^(0\.[0-9]*[1-9][0-9]*|[1-9][0-9]*([.][0-9]+)?)$ ]]; then
    echo "ERROR: YARN_FACTOR must be a positive number" >&2
    exit 1
fi
if [[ ! "$GPU_MEM_UTIL" =~ ^(0\.[0-9]*[1-9][0-9]*|1([.]0+)?)$ ]]; then
    echo "ERROR: GPU_MEM_UTIL must be in (0, 1] (got $GPU_MEM_UTIL)" >&2
    exit 1
fi
if [[ "$CUDAGRAPH_MODE" != "FULL_DECODE_ONLY" && \
      "$CUDAGRAPH_MODE" != "PIECEWISE" ]]; then
    echo "ERROR: CUDAGRAPH_MODE must be FULL_DECODE_ONLY or PIECEWISE" >&2
    exit 1
fi
if [[ "$DEFAULT_REASONING_EFFORT" != "xhigh" && \
      "$DEFAULT_REASONING_EFFORT" != "medium" && \
      "$DEFAULT_REASONING_EFFORT" != "low" ]]; then
    echo "ERROR: DEFAULT_REASONING_EFFORT must be xhigh, medium, or low" >&2
    exit 1
fi
if [[ -z "$TOOL_CALL_PARSER" || "$TOOL_CALL_PARSER" =~ [[:space:]] ]]; then
    echo "ERROR: TOOL_CALL_PARSER must be a non-empty parser name without whitespace" >&2
    exit 1
fi
if [[ "$VLLM_USE_V1" == "0" && "$ENABLE_CHUNKED_PREFILL" == "1" ]]; then
    echo "ERROR: ENABLE_CHUNKED_PREFILL=1 requires VLLM_USE_V1=1" >&2
    exit 1
fi
if [[ "$DEPLOY_PROFILE" == "long-context-1m" && "$ENABLE_CHUNKED_PREFILL" != "1" ]]; then
    echo "ERROR: long-context-1m requires ENABLE_CHUNKED_PREFILL=1" >&2
    exit 1
fi

# A PP deployment spans nodes in the documented cluster setup. Ray must be
# explicit in that case so the Engine Core process joins the existing cluster.
if (( PP > 1 || TP > 8 )) && [[ -z "$RAY_ADDRESS" ]]; then
    echo "ERROR: PP>1 or TP>8 requires RAY_ADDRESS=<ray-head>:6379 for multi-node deployment" >&2
    exit 1
fi
if (( DP_LOCAL > DP || DP_RANK_START + DP_LOCAL > DP )); then
    echo "ERROR: DP_RANK_START + DP_LOCAL must be <= DP" >&2
    exit 1
fi
if (( DP > 1 )) && [[ -z "$DP_ADDRESS" ]]; then
    echo "ERROR: DP>1 requires DP_ADDRESS or HCCL_IF_IP" >&2
    exit 1
fi
if [[ "$DP_BACKEND" != "mp" && "$DP_BACKEND" != "ray" ]]; then
    echo "ERROR: DP_BACKEND must be mp or ray (got $DP_BACKEND)" >&2
    exit 1
fi
if [[ "$ENABLE_BALANCE_SCHEDULING" == "1" ]]; then
    echo "ERROR: Qwen3.8 is Dense; ENABLE_BALANCE_SCHEDULING=1 selects a MoE-only DP engine in this image" >&2
    exit 1
fi
if (( MAX_MODEL_LEN > YARN_ORIGINAL_MAX_MODEL_LEN )) && [[ "$ENABLE_YARN" != "1" ]]; then
    echo "ERROR: MAX_MODEL_LEN>$YARN_ORIGINAL_MAX_MODEL_LEN requires ENABLE_YARN=1" >&2
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
case "$EXECUTOR_BACKEND" in
    ""|mp|ray|uni|external_launcher) ;;
    *)
        echo "ERROR: DISTRIBUTED_EXECUTOR_BACKEND must be mp, ray, uni, or external_launcher" >&2
        exit 1
        ;;
esac
if [[ "$EXECUTOR_BACKEND" == "ray" && -z "$RAY_ADDRESS" ]]; then
    echo "ERROR: DISTRIBUTED_EXECUTOR_BACKEND=ray requires RAY_ADDRESS=<ray-head>:6379" >&2
    exit 1
fi
if (( PP > 1 || TP > 8 )) && [[ "$EXECUTOR_BACKEND" != "ray" ]]; then
    echo "ERROR: PP>1 or TP>8 requires DISTRIBUTED_EXECUTOR_BACKEND=ray" >&2
    exit 1
fi
if [[ "$EXECUTOR_BACKEND" == "mp" ]] && (( TP * PP * DP_LOCAL > LOCAL_NPU_COUNT )); then
    echo "ERROR: local profile needs $((TP * PP * DP_LOCAL)) NPUs, but LOCAL_NPU_COUNT=$LOCAL_NPU_COUNT" >&2
    exit 1
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
    --default-chat-template-kwargs "{\"reasoning_effort\":\"$DEFAULT_REASONING_EFFORT\"}"
    --seed 1024
    --compilation-config "{\"cudagraph_mode\":\"$CUDAGRAPH_MODE\"}"
)

if (( DP > 1 )); then
    # vLLM's built-in DP coordinator needs an explicit local replica count
    # when multiple replicas share one node. Dense DP with TP=2/DP=4 is
    # verified on v0.23.0rc1-a3, provided Balance Scheduling remains disabled.
    VLLM_ARGS+=(
        --data-parallel-size-local "$DP_LOCAL"
        --data-parallel-start-rank "$DP_RANK_START"
        --data-parallel-address "$DP_ADDRESS"
        --data-parallel-rpc-port "$DP_RPC_PORT"
        --data-parallel-backend "$DP_BACKEND"
    )
fi

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
if [[ "$ENABLE_AUTO_TOOL_CHOICE" == "1" ]]; then
    VLLM_ARGS+=(--enable-auto-tool-choice --tool-call-parser "$TOOL_CALL_PARSER")
fi
if [[ "$ENABLE_YARN" == "1" ]]; then
    VLLM_ARGS+=(--hf-overrides "$HF_OVERRIDES")
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
echo "[INFO] Profile:     $DEPLOY_PROFILE"
echo "[INFO] Model:       $MODEL_PATH"
echo "[INFO] Served name: $SERVED_MODEL_NAME"
echo "[INFO] TP=$TP PP=$PP DP=$DP PORT=$PORT"
echo "[INFO] DP_LOCAL=$DP_LOCAL DP_ADDRESS=${DP_ADDRESS:-unset} DP_RPC_PORT=$DP_RPC_PORT DP_BACKEND=$DP_BACKEND"
echo "[INFO] DTYPE=$DTYPE QUANTIZATION=$QUANTIZATION"
echo "[INFO] MAX_MODEL_LEN=$MAX_MODEL_LEN MAX_NUM_SEQS=$MAX_NUM_SEQS"
echo "[INFO] MAX_NUM_BATCHED_TOKENS=$MAX_NUM_BATCHED_TOKENS GPU_MEM_UTIL=$GPU_MEM_UTIL"
echo "[INFO] MTP=$ENABLE_MTP PrefixCaching=$ENABLE_PREFIX_CACHING ChunkedPrefill=$ENABLE_CHUNKED_PREFILL"
echo "[INFO] LANGUAGE_MODEL_ONLY=$LANGUAGE_MODEL_ONLY"
echo "[INFO] CUDAGRAPH_MODE=$CUDAGRAPH_MODE VLLM_USE_V1=$VLLM_USE_V1"
echo "[INFO] VLLM_ENGINE_READY_TIMEOUT_S=$VLLM_ENGINE_READY_TIMEOUT_S"
echo "[INFO] YARN=$ENABLE_YARN YARN_FACTOR=$YARN_FACTOR"
if [[ "$ENABLE_YARN" == "1" ]]; then
    echo "[INFO] HF_OVERRIDES=$HF_OVERRIDES"
fi
echo "[INFO] DEFAULT_REASONING_EFFORT=$DEFAULT_REASONING_EFFORT"
echo "[INFO] AUTO_TOOL_CHOICE=$ENABLE_AUTO_TOOL_CHOICE TOOL_CALL_PARSER=$TOOL_CALL_PARSER"
echo "[INFO] BALANCE_SCHEDULING=$ENABLE_BALANCE_SCHEDULING"
echo "[INFO] FLASHCOMM1=$FLASHCOMM1 MLAPO=$MLAPO HCCL_BUFFSIZE=$HCCL_BUFFSIZE"
echo "[INFO] Executor backend=${EXECUTOR_BACKEND:-default}"
echo "============================================"

if [[ "$DRY_RUN" == "1" ]]; then
    printf '[DRY RUN]'
    printf ' %q' vllm serve "$MODEL_PATH" "${VLLM_ARGS[@]}" "$@"
    printf '\n'
    exit 0
fi

exec vllm serve "$MODEL_PATH" "${VLLM_ARGS[@]}" "$@"
