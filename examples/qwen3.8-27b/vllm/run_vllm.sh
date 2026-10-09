#!/usr/bin/env bash
# Qwen3.8-27B-W8A8: single-node 8-NPU, 1M-context deployment.
#
# Usage:
#   bash run_vllm.sh
#   DEFAULT_REASONING_EFFORT=medium bash run_vllm.sh
#   MODEL_PATH=/path/to/model PORT=8022 SERVED_MODEL_NAME=qwen3.8 bash run_vllm.sh
#   DRY_RUN=1 bash run_vllm.sh
set -euo pipefail

# Load CANN/ATB when the container entrypoint has not already done so.
set +u
[[ -f /usr/local/Ascend/cann/set_env.sh ]] && source /usr/local/Ascend/cann/set_env.sh
[[ -f /usr/local/Ascend/nnal/atb/set_env.sh ]] && source /usr/local/Ascend/nnal/atb/set_env.sh
set -u

# User-facing settings. The NPU topology and performance parameters below are
# fixed as one deployment shape so accidental overrides cannot desync it.
MODEL_PATH="${MODEL_PATH:-/home/jianzhnie/llmtuner/hfhub/models/Eco-Tech/Qwen3.8-27B-w8a8}"
PORT="${PORT:-8022}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3.8}"
DEFAULT_REASONING_EFFORT="${DEFAULT_REASONING_EFFORT:-medium}"
DRY_RUN="${DRY_RUN:-0}"
TP_SIZE="${TP_SIZE:-2}"
DP_SIZE="${DP_SIZE:-1}"
PP_SIZE="${PP_SIZE:-1}"

# ---- Preflight -------------------------------------------------------------
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
[[ "$PORT" =~ ^[1-9][0-9]*$ ]] && (( PORT <= 65535 )) || {
    echo "ERROR: PORT must be an integer between 1 and 65535 (got $PORT)" >&2
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

port_is_listening() {
    command -v ss >/dev/null 2>&1 &&
        ss -H -ltn "sport = :$1" 2>/dev/null | grep -q .
}

if [[ "$DRY_RUN" == 0 ]] && port_is_listening "$PORT"; then
    echo "ERROR: port $PORT is already listening" >&2
    exit 1
fi

# Stable vLLM-Ascend v0.23.0rc1-a3 settings for this dense W8A8 model.
export VLLM_USE_V1=1
export VLLM_USE_MODELSCOPE=False
export VLLM_ENGINE_READY_TIMEOUT_S=1800
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_OP_EXPANSION_MODE=AIV
export HCCL_BUFFSIZE=512
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export TASK_QUEUE_ENABLE=1
export VLLM_ASCEND_BALANCE_SCHEDULING=0
export VLLM_ASCEND_ENABLE_MLAPO=0

# TP4 x DP1 uses one four-NPU replica. Static YaRN 4x extends the native 262K
# context to 1M; the remaining four local NPUs are intentionally left idle.
VLLM_ARGS=(
    --host 0.0.0.0
    --port "$PORT"
    --served-model-name "$SERVED_MODEL_NAME"
    --trust-remote-code
    --dtype bfloat16
    --quantization ascend
    --tensor-parallel-size "$TP_SIZE"
    --pipeline-parallel-size "$PP_SIZE"
    --data-parallel-size "$DP_SIZE"
    --gpu-memory-utilization 0.92
    --max-model-len 1000000
    --max-num-seqs 16
    --max-num-batched-tokens 16384
    --enable-prefix-caching
    --enable-chunked-prefill
    --language-model-only
    --enable-auto-tool-choice
    --tool-call-parser qwen3_coder
    --reasoning-parser qwen3
    --default-chat-template-kwargs "{\"reasoning_effort\":\"$DEFAULT_REASONING_EFFORT\"}"
    --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":3,"enforce_eager":true}'
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'
    --additional-config '{"enable_cpu_binding":true}'
    --hf-overrides '{"text_config":{"rope_parameters":{"mrope_interleaved":true,"mrope_section":[11,11,10],"rope_type":"yarn","rope_theta":10000000,"partial_rotary_factor":0.25,"factor":4.0,"original_max_position_embeddings":262144}}}'
    --seed 1024
)

echo "Qwen3.8-27B-W8A8: TP=$TP_SIZE PP=$PP_SIZE DP=$DP_SIZE, 1M context, MTP=3"
echo "model=$MODEL_PATH name=$SERVED_MODEL_NAME port=$PORT reasoning=$DEFAULT_REASONING_EFFORT"

if [[ "$DRY_RUN" == 1 ]]; then
    printf '[DRY RUN]'
    printf ' %q' vllm serve "$MODEL_PATH" "${VLLM_ARGS[@]}"
    printf '\n'
    exit 0
fi

exec vllm serve "$MODEL_PATH" "${VLLM_ARGS[@]}"
