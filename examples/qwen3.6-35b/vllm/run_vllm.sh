#!/bin/bash
# =============================================================================
# Qwen3.6-35B-A3B — vllm serve deployment (W8A8 / BF16)
# =============================================================================
# Architecture: Qwen3.6 MoE (sparse) | 35B total / 3B activated | Hybrid Attn
# First supported: v0.18.0rc1 | Max Position: 262144
#
# Compact MoE model — single node sufficient for both BF16 and W8A8.
# Uses Qwen3.5-style hybrid attention, prefix caching ON by default.
#
# Hardware (official):
#   - W8A8/BF16: 1× Atlas 800I A3 (64G×16) or 1× Atlas 800I A2 (64G×8)
#
# Usage:
#   bash run_vllm.sh                                    # default: TP=2 W8A8
#   TP=2 DP=4 bash run_vllm.sh                          # high-throughput
#   MAX_MODEL_LEN=262144 bash run_vllm.sh               # long context (262K)
#   ENABLE_MTP=1 bash run_vllm.sh                       # MTP speculative decode
#   QUANTIZATION=none bash run_vllm.sh                  # BF16 full precision
#
# Reference:
#   https://docs.vllm.ai/projects/ascend/zh-cn/latest/tutorials/models/Qwen3.6-35B-A3B.html
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

# ---- Model & Server ----
readonly BASE_MODEL_PATH="/home/jianzhnie/llmtuner/hfhub/models/Eco-Tech"
readonly MODEL_PATH="${MODEL_PATH:-$BASE_MODEL_PATH/Qwen3.6-35B-A3B-w8a8}"
readonly HOST="${HOST:-0.0.0.0}"
readonly PORT="${PORT:-8020}"
readonly SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3.6}"

# ---- Parallelism ----
readonly TP="${TP:-2}"
readonly DP="${DP:-1}"
readonly DP_LOCAL="${DP_LOCAL:-$DP}"
readonly DP_RANK_START="${DP_RANK_START:-0}"

# ---- Memory & Context ----
readonly MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
readonly MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
readonly MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-16384}"
readonly GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"

# ---- Quantization ----
readonly QUANTIZATION="${QUANTIZATION:-ascend}"

# ---- MTP speculative decoding ----
readonly ENABLE_MTP="${ENABLE_MTP:-0}"

# ---- NPU environment (official recommendations) ----
export VLLM_USE_MODELSCOPE=True
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_OP_EXPANSION_MODE=AIV
export HCCL_BUFFSIZE="${HCCL_BUFFSIZE:-1024}"
export OMP_NUM_THREADS=1
export TASK_QUEUE_ENABLE=1

# CPU / memory tuning (official)
echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor 2>/dev/null || true
sysctl -w vm.swappiness=0 2>/dev/null || true
sysctl -w kernel.numa_balancing=0 2>/dev/null || true
sysctl -w kernel.sched_migration_cost_ns=50000 2>/dev/null || true
if [[ -f "/usr/lib/aarch64-linux-gnu/libjemalloc.so.2" ]]; then
    export LD_PRELOAD="/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD"
fi

# ---- Multi-node network ----
readonly NIC_NAME="${NIC_NAME:-}"
readonly HCCL_IF_IP="${HCCL_IF_IP:-}"
if [[ -n "$HCCL_IF_IP" ]]; then
    export HCCL_IF_IP
fi
if [[ -n "$NIC_NAME" ]]; then
    export GLOO_SOCKET_IFNAME="$NIC_NAME"
    export TP_SOCKET_IFNAME="$NIC_NAME"
    export HCCL_SOCKET_IFNAME="$NIC_NAME"
fi

# ---- Base vllm args ----
VLLM_ARGS=(
    --host "$HOST"
    --port "$PORT"
    --served-model-name "$SERVED_MODEL_NAME"
    --trust-remote-code
    --tensor-parallel-size "$TP"
    --data-parallel-size "$DP"
    --enable-expert-parallel
    --max-num-seqs "$MAX_NUM_SEQS"
    --max-model-len "$MAX_MODEL_LEN"
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
    --gpu-memory-utilization "$GPU_MEM_UTIL"
    --enable-prefix-caching
    --async-scheduling
    --seed 1024
)

# ---- Quantization ----
if [[ "$QUANTIZATION" != "none" ]]; then
    VLLM_ARGS+=(--quantization "$QUANTIZATION")
fi

# ---- DP multi-process (high-throughput TP=2 DP=4) ----
if [[ "$DP" -gt 1 ]]; then
    readonly DP_ADDRESS="${DP_ADDRESS:-$HCCL_IF_IP}"
    readonly DP_RPC_PORT="${DP_RPC_PORT:-12345}"
    if [[ -z "$DP_ADDRESS" ]]; then
        echo "ERROR: DP>1 requires DP_ADDRESS or HCCL_IF_IP"
        exit 1
    fi
    VLLM_ARGS+=(
        --data-parallel-size-local "$DP_LOCAL"
        --data-parallel-start-rank "$DP_RANK_START"
        --data-parallel-address "$DP_ADDRESS"
        --data-parallel-rpc-port "$DP_RPC_PORT"
    )
fi

# ---- MTP speculative decoding ----
if [[ "$ENABLE_MTP" == "1" ]]; then
    VLLM_ARGS+=(
        --speculative-config '{"method": "qwen3_5_mtp", "num_speculative_tokens": 3, "enforce_eager": true}'
    )
fi

# ---- Compilation & additional config ----
readonly COMPILATION_CONFIG='{"cudagraph_mode": "FULL_DECODE_ONLY"}'

ADDITIONAL_JSON="\"enable_flashcomm1\": true"
if [[ "${ENABLE_CPU_BINDING:-1}" == "1" ]]; then
    ADDITIONAL_JSON="$ADDITIONAL_JSON, \"enable_cpu_binding\": true"
fi
if [[ "${ENABLE_SHARED_EXPERT_OVERLAP:-1}" == "1" ]]; then
    ADDITIONAL_JSON="$ADDITIONAL_JSON, \"multistream_overlap_shared_expert\": true"
fi
readonly ADDITIONAL_CONFIG="{$ADDITIONAL_JSON}"

echo "============================================"
echo "[INFO] Qwen3.6-35B-A3B — vLLM-Ascend Deployment"
echo "[INFO] Model:       $MODEL_PATH"
echo "[INFO] TP=$TP  DP=$DP  PORT=$PORT"
echo "[INFO] Quantization: $QUANTIZATION"
echo "[INFO] MAX_MODEL_LEN=$MAX_MODEL_LEN  MAX_NUM_SEQS=$MAX_NUM_SEQS"
echo "[INFO] MAX_NUM_BATCHED_TOKENS=$MAX_NUM_BATCHED_TOKENS"
echo "[INFO] GPU_MEM_UTIL=$GPU_MEM_UTIL"
echo "[INFO] MTP=$ENABLE_MTP  (qwen3_5_mtp, 3 tokens)"
echo "[INFO] Prefix caching: ENABLED (official recommendation)"
echo "[INFO] 35B MoE (3B activated), hybrid attention, EP enabled"
echo "[INFO] official: https://docs.vllm.ai/projects/ascend/zh-cn/latest/tutorials/models/Qwen3.6-35B-A3B.html"
echo "============================================"

vllm serve "$MODEL_PATH" \
    "${VLLM_ARGS[@]}" \
    --compilation-config "$COMPILATION_CONFIG" \
    --additional-config "$ADDITIONAL_CONFIG" \
    "$@"
