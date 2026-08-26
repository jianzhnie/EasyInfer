#!/usr/bin/env bash
# Qwen3.8-27B-W8A8 deployment for vLLM-Ascend v0.23.0rc1-a3.
#
# Common usage:
#   bash run_vllm.sh
#   DEPLOY_PROFILE=long-context-1m bash run_vllm.sh
#   DEPLOY_PROFILE=long-context-1m TP=4 DP=2 DP_LOCAL=2 bash run_vllm.sh
#   DEFAULT_REASONING_EFFORT=low bash run_vllm.sh
#   DRY_RUN=1 bash run_vllm.sh
set -euo pipefail

# vLLM is installed in the deployment container. Load CANN when the image has
# not already done so in its entrypoint.
set +u
[[ -f /usr/local/Ascend/cann/set_env.sh ]] && source /usr/local/Ascend/cann/set_env.sh
[[ -f /usr/local/Ascend/nnal/atb/set_env.sh ]] && source /usr/local/Ascend/nnal/atb/set_env.sh
set -u

# ---- User-facing settings --------------------------------------------------
MODEL_PATH="${MODEL_PATH:-/home/jianzhnie/llmtuner/hfhub/models/Eco-Tech/Qwen3.8-27B-w8a8}"
PORT="${PORT:-8022}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3.8}"
DEPLOY_PROFILE="${DEPLOY_PROFILE:-throughput}"
DEFAULT_REASONING_EFFORT="${DEFAULT_REASONING_EFFORT:-xhigh}"
DRY_RUN="${DRY_RUN:-0}"

# Optional topology overrides. Profile defaults use all 8 local NPUs.
RAY_ADDRESS="${RAY_ADDRESS:-}"
DP_ADDRESS="${DP_ADDRESS:-10.16.201.229}"

case "$DEPLOY_PROFILE" in
    throughput)
        TP="${TP:-2}"
        DP="${DP:-4}"
        DP_LOCAL="${DP_LOCAL:-$DP}"
        MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
        MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
        MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-32768}"
        ;;
    long-context-1m)
        TP="${TP:-4}"
        DP="${DP:-2}"
        DP_LOCAL="${DP_LOCAL:-$DP}"
        MAX_MODEL_LEN="${MAX_MODEL_LEN:-1000000}"
        MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
        MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-32768}"
        ;;
    *)
        echo "ERROR: DEPLOY_PROFILE must be throughput or long-context-1m" >&2
        exit 1
        ;;
esac

PP="${PP:-1}"

# ---- Small preflight -------------------------------------------------------
[[ -d "$MODEL_PATH" ]] || {
    echo "ERROR: MODEL_PATH does not exist: $MODEL_PATH" >&2
    exit 1
}
for file in quant_model_description.json quant_model_weights.safetensors.index.json; do
    [[ -f "$MODEL_PATH/$file" ]] || {
        echo "ERROR: W8A8 metadata is missing: $MODEL_PATH/$file" >&2
        exit 1
    }
done
for name in PORT TP PP DP DP_LOCAL MAX_MODEL_LEN MAX_NUM_SEQS MAX_NUM_BATCHED_TOKENS; do
    [[ "${!name}" =~ ^[1-9][0-9]*$ ]] || {
        echo "ERROR: $name must be a positive integer (got ${!name})" >&2
        exit 1
    }
done
(( PORT <= 65535 )) || {
    echo "ERROR: PORT must be <= 65535" >&2
    exit 1
}
[[ "$DEFAULT_REASONING_EFFORT" =~ ^(xhigh|medium|low)$ ]] || {
    echo "ERROR: DEFAULT_REASONING_EFFORT must be xhigh, medium, or low" >&2
    exit 1
}
[[ "$DRY_RUN" == 0 || "$DRY_RUN" == 1 ]] || {
    echo "ERROR: DRY_RUN must be 0 or 1" >&2
    exit 1
}
(( DP_LOCAL <= DP )) || {
    echo "ERROR: DP_LOCAL cannot exceed DP" >&2
    exit 1
}
if (( MAX_MODEL_LEN > 262144 )) && [[ "$DEPLOY_PROFILE" != long-context-1m ]]; then
    echo "ERROR: contexts above 262144 require DEPLOY_PROFILE=long-context-1m (static YaRN 4x)" >&2
    exit 1
fi

if [[ -n "$RAY_ADDRESS" ]]; then
    EXECUTOR_BACKEND=ray
    export RAY_ADDRESS
else
    EXECUTOR_BACKEND=mp
    (( TP * PP * DP_LOCAL <= 8 )) || {
        echo "ERROR: local topology needs $((TP * PP * DP_LOCAL)) NPUs; only 8 are available" >&2
        exit 1
    }
    (( PP == 1 )) || {
        echo "ERROR: PP>1 requires RAY_ADDRESS=<ray-head>:6379" >&2
        exit 1
    }
fi

# Fixed values below are model/image-specific best practices, not general
# tuning knobs. W8A8 is selected by --quantization ascend; bfloat16 is the
# compute dtype for unquantized parameters and intermediate tensors.
export VLLM_USE_V1=1
export VLLM_USE_MODELSCOPE=False
export VLLM_ENGINE_READY_TIMEOUT_S=1800
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_OP_EXPANSION_MODE=AIV
export HCCL_BUFFSIZE=512
export HCCL_IF_IP="$DP_ADDRESS"
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export TASK_QUEUE_ENABLE=1
export VLLM_ASCEND_BALANCE_SCHEDULING=0
export VLLM_ASCEND_ENABLE_FLASHCOMM1=0
export VLLM_ASCEND_ENABLE_MLAPO=0

VLLM_ARGS=(
    --host 0.0.0.0
    --port "$PORT"
    --served-model-name "$SERVED_MODEL_NAME"
    --trust-remote-code
    --dtype bfloat16
    --quantization ascend
    --tensor-parallel-size "$TP"
    --pipeline-parallel-size "$PP"
    --data-parallel-size "$DP"
    --gpu-memory-utilization 0.85
    --max-model-len "$MAX_MODEL_LEN"
    --max-num-seqs "$MAX_NUM_SEQS"
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
    --enable-prefix-caching
    --enable-chunked-prefill
    --enable-auto-tool-choice
    --tool-call-parser qwen3_xml
    --default-chat-template-kwargs "{\"reasoning_effort\":\"$DEFAULT_REASONING_EFFORT\"}"
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'
    --additional-config '{"enable_cpu_binding":true}'
    --distributed-executor-backend "$EXECUTOR_BACKEND"
    --seed 1024
)

if (( DP > 1 )); then
    VLLM_ARGS+=(
        --data-parallel-size-local "$DP_LOCAL"
        --data-parallel-start-rank 0
        --data-parallel-address "$DP_ADDRESS"
        --data-parallel-rpc-port 13390
        --data-parallel-backend mp
    )
fi

if [[ "$DEPLOY_PROFILE" == throughput ]]; then
    VLLM_ARGS+=(
        --mm-encoder-tp-mode data
        --allowed-local-media-path /home/jianzhnie/llmtuner/
        --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":3,"enforce_eager":true}'
    )
else
    VLLM_ARGS+=(
        --language-model-only
        --hf-overrides '{"text_config":{"rope_parameters":{"mrope_interleaved":true,"mrope_section":[11,11,10],"rope_type":"yarn","rope_theta":10000000,"partial_rotary_factor":0.25,"factor":4.0,"original_max_position_embeddings":262144}}}'
    )
fi

echo "============================================"
echo "Qwen3.8-27B-W8A8 / $DEPLOY_PROFILE"
echo "model=$MODEL_PATH name=$SERVED_MODEL_NAME port=$PORT"
echo "TP=$TP PP=$PP DP=$DP DP_LOCAL=$DP_LOCAL backend=$EXECUTOR_BACKEND"
echo "max_len=$MAX_MODEL_LEN seqs=$MAX_NUM_SEQS batched_tokens=$MAX_NUM_BATCHED_TOKENS"
echo "reasoning=$DEFAULT_REASONING_EFFORT tools=qwen3_xml"
echo "============================================"

if [[ "$DRY_RUN" == 1 ]]; then
    printf '[DRY RUN]'
    printf ' %q' vllm serve "$MODEL_PATH" "${VLLM_ARGS[@]}" "$@"
    printf '\n'
    exit 0
fi

exec vllm serve "$MODEL_PATH" "${VLLM_ARGS[@]}" "$@"
