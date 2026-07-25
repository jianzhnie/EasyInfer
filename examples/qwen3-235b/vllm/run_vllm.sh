#!/bin/bash
# =============================================================================
# Qwen3-235B-A22B — vllm serve deployment (W8A8 / BF16)
# =============================================================================
# Architecture: Qwen3MoeForCausalLM | 128 Experts | GQA (64H/4KVH) | MoE
# First supported: v0.8.4rc2 | Validated: v0.21.0+ | Max Position: 262144
#
# Hardware (official):
#   - BF16/W8A8: 1× Atlas 800I A3 (64G×16) or 1× Atlas 800I A2 (64G×8)
#
# Usage:
#   bash run_vllm.sh                                    # default: TP=8 W8A8
#   TP=4 DP=4 HCCL_IF_IP=<ip> DP_ADDRESS=<ip> bash ...  # high-throughput A3
#   TP=16 bash run_vllm.sh                              # low-latency A3
#   QUANTIZATION=none bash run_vllm.sh                  # BF16 full precision
#   MAX_MODEL_LEN=135000 bash run_vllm.sh               # long context
#
# Reference:
#   https://docs.vllm.ai/projects/ascend/zh-cn/latest/tutorials/models/Qwen3-235B-A22B.html
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
readonly BASE_MODEL_PATH="/home/jianzhnie/llmtuner/hfhub/models/Qwen"
readonly MODEL_PATH="${MODEL_PATH:-$BASE_MODEL_PATH/Qwen3-235B-A22B-Instruct-2507}"
readonly HOST="${HOST:-0.0.0.0}"
readonly PORT="${PORT:-8018}"
readonly SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3}"

# ---- Parallelism ----
readonly TP="${TP:-8}"
readonly DP="${DP:-1}"
readonly DP_LOCAL="${DP_LOCAL:-$DP}"
readonly DP_RANK_START="${DP_RANK_START:-0}"

# ---- Memory & Context ----
readonly MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
readonly MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
readonly MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8096}"
readonly GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.95}"

# ---- Quantization (official: W8A8 using ascend backend) ----
readonly QUANTIZATION="${QUANTIZATION:-ascend}"

# ---- NPU environment (official recommendations) ----
export HCCL_OP_EXPANSION_MODE=AIV
export HCCL_BUFFSIZE="${HCCL_BUFFSIZE:-512}"
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export OMP_PROC_BIND=false
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

# ---- Fused MC2 (official: on for high-throughput TP4 DP4, off for TP16 low-latency) ----
if [[ "$DP" -gt 1 || "$TP" -gt 8 ]]; then
    FUSED_MC2="${FUSED_MC2:-1}"
else
    FUSED_MC2="${FUSED_MC2:-0}"
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
    --no-enable-prefix-caching
    --async-scheduling
    --seed 1024
)

# ---- Quantization ----
if [[ "$QUANTIZATION" != "none" ]]; then
    VLLM_ARGS+=(--quantization "$QUANTIZATION")
fi

# ---- DP multi-process ----
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

# ---- Compilation & additional config ----
readonly COMPILATION_CONFIG='{"cudagraph_mode": "FULL_DECODE_ONLY"}'

ADDITIONAL_JSON="\"enable_flashcomm1\": true"
if [[ "$FUSED_MC2" == "1" ]]; then
    ADDITIONAL_JSON="$ADDITIONAL_JSON, \"enable_fused_mc2\": 1"
fi
if [[ "${ENABLE_CPU_BINDING:-1}" == "1" ]]; then
    ADDITIONAL_JSON="$ADDITIONAL_JSON, \"enable_cpu_binding\": true"
fi
readonly ADDITIONAL_CONFIG="{$ADDITIONAL_JSON}"

echo "============================================"
echo "[INFO] Qwen3-235B-A22B — vLLM-Ascend Deployment"
echo "[INFO] Model:       $MODEL_PATH"
echo "[INFO] TP=$TP  DP=$DP  PORT=$PORT"
echo "[INFO] Quantization: $QUANTIZATION"
echo "[INFO] MAX_MODEL_LEN=$MAX_MODEL_LEN  MAX_NUM_SEQS=$MAX_NUM_SEQS"
echo "[INFO] MAX_NUM_BATCHED_TOKENS=$MAX_NUM_BATCHED_TOKENS"
echo "[INFO] GPU_MEM_UTIL=$GPU_MEM_UTIL  FUSED_MC2=$FUSED_MC2"
echo "[INFO] Attention: GQA (64H/4KVH) — MLA disabled"
echo "[INFO] 128 experts, EP enabled, prefix caching OFF"
echo "============================================"

vllm serve "$MODEL_PATH" \
    "${VLLM_ARGS[@]}" \
    --compilation-config "$COMPILATION_CONFIG" \
    --additional-config "$ADDITIONAL_CONFIG" \
    "$@"
