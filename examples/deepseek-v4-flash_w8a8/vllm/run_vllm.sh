#!/bin/bash
# =============================================================================
# DeepSeek-V4-Flash W8A8 MTP — vllm serve deployment
# =============================================================================
# Architecture: DeepseekV4ForCausalLM | 256 Experts (+1 shared) | MLA | MTP=1
# Native support since v0.22.1rc1 | Max Position: 1,048,576 (1M)
#
# Hardware (official):
#   - W8A8: 1× Atlas 800I A3 (128G×8, TP=4 DP=4) or A2 (64G×8, TP=8 DP=1)
#
# Usage:
#   bash run_vllm.sh                                    # default: TP=4 DP=4 (1M ctx)
#   TP=8 DP=1 bash run_vllm.sh                          # A2 single-node
#   ENABLE_MTP=0 bash run_vllm.sh                       # disable MTP
#
# Reference:
#   https://docs.vllm.ai/projects/ascend/zh-cn/latest/tutorials/models/DeepSeek-V4-Flash.html
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
readonly MODEL_PATH="${MODEL_PATH:-$BASE_MODEL_PATH/DeepSeek-V4-Flash-w8a8-mtp}"
readonly HOST="${HOST:-0.0.0.0}"
readonly PORT="${PORT:-8000}"
readonly SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-dsv4}"

# ---- Parallelism (official: TP=4 DP=4 high-throughput) ----
readonly TP="${TP:-4}"
readonly DP="${DP:-4}"
readonly PP="${PP:-1}"
readonly DP_LOCAL="${DP_LOCAL:-$DP}"
readonly DP_RANK_START="${DP_RANK_START:-0}"

# ---- Memory & Context ----
readonly MAX_MODEL_LEN="${MAX_MODEL_LEN:-1048576}"
readonly MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
readonly MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-10240}"
readonly GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"

# ---- MTP speculative decoding ----
readonly ENABLE_MTP="${ENABLE_MTP:-1}"

# ---- NPU environment (official recommendations) ----
export OMP_PROC_BIND=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-10}"
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_OP_EXPANSION_MODE=AIV
export HCCL_BUFFSIZE="${HCCL_BUFFSIZE:-1024}"
export TASK_QUEUE_ENABLE=1

# ---- NPU optimization env vars (retained from original) ----
export VLLM_ASCEND_ENABLE_FLASHCOMM1="${VLLM_ASCEND_ENABLE_FLASHCOMM1:-1}"
export VLLM_ASCEND_ENABLE_MLAPO="${VLLM_ASCEND_ENABLE_MLAPO:-1}"
export VLLM_ASCEND_BALANCE_SCHEDULING="${VLLM_ASCEND_BALANCE_SCHEDULING:-1}"
export USE_MULTI_GROUPS_KV_CACHE="${USE_MULTI_GROUPS_KV_CACHE:-1}"
export USE_MULTI_BLOCK_POOL="${USE_MULTI_BLOCK_POOL:-1}"
export ACL_OP_INIT_MODE="${ACL_OP_INIT_MODE:-1}"

# Timeout settings for large model loading
export VLLM_RPC_TIMEOUT="${VLLM_RPC_TIMEOUT:-3600000}"
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-30000}"
export HCCL_EXEC_TIMEOUT="${HCCL_EXEC_TIMEOUT:-204}"
export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-1200}"

# Fused MC2 (retained from original)
if [[ "$PP" -gt 1 || "$TP" -gt 8 ]]; then
    export VLLM_ASCEND_ENABLE_FUSED_MC2="${VLLM_ASCEND_ENABLE_FUSED_MC2:-1}"
else
    export VLLM_ASCEND_ENABLE_FUSED_MC2="${VLLM_ASCEND_ENABLE_FUSED_MC2:-0}"
fi

# CPU / memory tuning
echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor 2>/dev/null || true
sysctl -w vm.swappiness=0 2>/dev/null || true
sysctl -w kernel.numa_balancing=0 2>/dev/null || true
sysctl -w kernel.sched_migration_cost_ns=50000 2>/dev/null || true
export LD_PRELOAD="${LD_PRELOAD:-/usr/lib/aarch64-linux-gnu/libjemalloc.so.2}"

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
    --pipeline-parallel-size "$PP"
    --enable-expert-parallel
    --distributed-executor-backend ray
    --max-num-seqs "$MAX_NUM_SEQS"
    --max-model-len "$MAX_MODEL_LEN"
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
    --gpu-memory-utilization "$GPU_MEM_UTIL"
    --block-size 128
    --quantization ascend
    --safetensors-load-strategy prefetch
    --model-loader-extra-config '{"enable_multithread_load": "true", "num_threads": 128}'
    --tokenizer-mode deepseek_v4
    --tool-call-parser deepseek_v4
    --enable-auto-tool-choice
    --reasoning-parser deepseek_v4
    --no-disable-hybrid-kv-cache-manager
    --enable-chunked-prefill
    --enable-prefix-caching
    --async-scheduling
    --seed 1024
)

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

# ---- MTP speculative decoding ----
if [[ "$ENABLE_MTP" == "1" ]]; then
    VLLM_ARGS+=(
        --speculative-config '{"num_speculative_tokens": 1, "method": "mtp", "enforce_eager": true}'
    )
fi

# ---- Compilation & additional config ----
readonly COMPILATION_CONFIG='{"cudagraph_mode": "FULL_DECODE_ONLY"}'

ADDITIONAL_JSON="\"enable_cpu_binding\": true"
ADDITIONAL_JSON="$ADDITIONAL_JSON, \"multistream_overlap_shared_expert\": true"
if [[ "${ENABLE_SHARED_EXPERT_DP:-1}" == "1" ]]; then
    ADDITIONAL_JSON="$ADDITIONAL_JSON, \"enable_shared_expert_dp\": true"
fi
readonly ADDITIONAL_CONFIG="{$ADDITIONAL_JSON}"

echo "============================================"
echo "[INFO] DeepSeek-V4-Flash W8A8 MTP — vLLM-Ascend Deployment"
echo "[INFO] Model:       $MODEL_PATH"
echo "[INFO] TP=$TP  DP=$DP  PP=$PP  PORT=$PORT"
echo "[INFO] MAX_MODEL_LEN=$MAX_MODEL_LEN  MAX_NUM_SEQS=$MAX_NUM_SEQS"
echo "[INFO] MAX_NUM_BATCHED_TOKENS=$MAX_NUM_BATCHED_TOKENS"
echo "[INFO] GPU_MEM_UTIL=$GPU_MEM_UTIL  MTP=$ENABLE_MTP"
echo "[INFO] FUSED_MC2=$VLLM_ASCEND_ENABLE_FUSED_MC2  FLASHCOMM1=$VLLM_ASCEND_ENABLE_FLASHCOMM1"
echo "[INFO] Hybrid KV Cache: enabled | Prefix Caching: enabled | Chunked Prefill: enabled"
echo "[INFO] 256 experts (+1 shared), MLA, block_size=128, deepseek_v4 tokenizer"
echo "[INFO] official: https://docs.vllm.ai/projects/ascend/zh-cn/latest/tutorials/models/DeepSeek-V4-Flash.html"
echo "============================================"

vllm serve "$MODEL_PATH" \
    "${VLLM_ARGS[@]}" \
    --compilation-config "$COMPILATION_CONFIG" \
    --additional-config "$ADDITIONAL_CONFIG" \
    "$@"
