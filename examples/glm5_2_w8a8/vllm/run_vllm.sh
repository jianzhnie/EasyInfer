#!/bin/bash
# =============================================================================
# GLM-5.2 W8A8 — Direct vllm serve deployment (official config)
# =============================================================================
# Architecture: GlmMoeDsaForCausalLM | 256 Experts | MLA | MTP=1
# Max Position: 1048576 | Deploy: 32K context (override with MAX_MODEL_LEN)
# Note: GLM-5.2 does not support Pipeline Parallelism; use large TP across nodes.
#
# Hardware:
#   - Atlas 800 A3 (128G x 16): TP=8 DP=2 (single node)
#   - Atlas 800 A2 (64G x 8):   TP=8 DP=1 (single node, low context)
#                                TP=16 (2 nodes, 32K+ context)
#
# Usage:
#   bash run_vllm.sh                      # TP=8 DP=1 single node (A2)
#   DP=2 bash run_vllm.sh                 # TP=8 DP=2 single node (A3, 16 NPUs)
#   TP=16 MAX_MODEL_LEN=131072 bash run_vllm.sh  # 2 nodes (A2, 32K+ context)
#   TP=32 MAX_MODEL_LEN=202752 bash run_vllm.sh  # 4 nodes
#
# Reference:
#   https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/GLM5.2.html
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
readonly BASE_MODEL_PATH="/home/jianzhnie/llmtuner/hfhub/models/Eco-Tech"
readonly MODEL_PATH="${MODEL_PATH:-$BASE_MODEL_PATH/GLM-5.2-w8a8}"
readonly HOST="${HOST:-0.0.0.0}"
readonly PORT="${PORT:-8007}"
readonly TP="${TP:-8}"
readonly PP="${PP:-1}"
readonly DP="${DP:-1}"
readonly MAX_MODEL_LEN="${MAX_MODEL_LEN:-31744}"
readonly MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
readonly MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"
readonly GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.95}"

# NPU environment variables (official docs)
export HCCL_OP_EXPANSION_MODE=AIV
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export HCCL_BUFFSIZE=200
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_USE_MODELSCOPE=False
export VLLM_ASCEND_BALANCE_SCHEDULING=1
export VLLM_ASCEND_ENABLE_FLASHCOMM1=0
export VLLM_ASCEND_ENABLE_MLAPO=1
export VLLM_USE_V1=1

# Runtime / debug
export ASCEND_LAUNCH_BLOCKING=0
export VLLM_ENGINE_READY_TIMEOUT_S=1800

# =============================================================================
# Multi-node network interface binding (optional)
# Required for multi-node TP>8 deployments. Set NIC_NAME to your high-speed
# interface (e.g., enp66s0f1). Leave empty for single-node auto-detect.
#
# RAY_ADDRESS: For multi-node (TP>8), set this to the Ray head node address
#   (e.g., RAY_ADDRESS=10.42.11.130:6379) to allow Engine Core subprocess
#   to connect to the existing Ray cluster instead of starting a local one.
#   Auto-detect: ray.init(address='auto') → get_runtime_context().gcs_address
#
# Note: GLM-5.2 does NOT support TP=16 due to MLA dimension incompatibility
#   (head_dim=192 × num_kv_heads=3 = 576, not divisible by 16).
#   For A2 (64GB) hardware, TP=8 OOMs (~60.4 GiB weights per NPU).
#   Use A3 (128GB) hardware with TP=8 for production deployment.
# =============================================================================
readonly RAY_ADDRESS="${RAY_ADDRESS:-}"
readonly NIC_NAME="${NIC_NAME:-}"
readonly HCCL_IF_IP="${HCCL_IF_IP:-}"
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

# Compilation config (official docs)
readonly COMPILATION_CONFIG='{"cudagraph_mode": "FULL_DECODE_ONLY"}'
readonly ADDITIONAL_CONFIG='{"multistream_overlap_shared_expert": true}'
# MTP speculative decoding. NOTE: PP>1 + MTP is rejected by vLLM 0.23.0
# ("PP+MTP is only supported on PD-disaggregated P nodes"), so MTP defaults
# to off; enable only with PP=1.
readonly ENABLE_MTP="${ENABLE_MTP:-0}"
SPEC_ARGS=()
if [[ "$ENABLE_MTP" == "1" ]]; then
    SPEC_ARGS+=(--speculative-config '{"num_speculative_tokens": 3, "method": "deepseek_mtp", "enforce_eager": true}')
fi

echo "============================================"
echo "[INFO] GLM-5.2 W8A8 — vLLM-Ascend Deployment (Official Config)"
echo "[INFO] Model:    $MODEL_PATH"
echo "[INFO] TP=$TP  PP=$PP  DP=$DP  PORT=$PORT"
echo "[INFO] MAX_MODEL_LEN=$MAX_MODEL_LEN  MAX_NUM_SEQS=$MAX_NUM_SEQS"
echo "[INFO] MAX_NUM_BATCHED_TOKENS=$MAX_NUM_BATCHED_TOKENS"
echo "[INFO] GPU_MEM_UTIL=$GPU_MEM_UTIL"
echo "[INFO] RAY_ADDRESS=${RAY_ADDRESS:-auto-detect}"
echo "[INFO] MTP: 3 tokens, method=deepseek_mtp"
echo "[INFO] Tool Calling: glm47 parser + glm45 reasoning"
echo "[INFO] Features: chunked-prefill, prefix-caching, eager MTP"
echo "[INFO] Hardware: A3 (128G) recommended. A2 (64G) TP=8 OOMs."
echo "============================================"

vllm serve "$MODEL_PATH" \
    --host "$HOST" \
    --port "$PORT" \
    --served-model-name "glm-5.2" \
    --trust-remote-code \
    --dtype bfloat16 \
    --tensor-parallel-size "$TP" \
    --pipeline-parallel-size "$PP" \
    --data-parallel-size "$DP" \
    --distributed-executor-backend ray \
    --quantization ascend \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
    --chat-template-content-format string \
    --enable-chunked-prefill \
    --enable-prefix-caching \
    --enable-expert-parallel \
    --enable-auto-tool-choice \
    --tool-call-parser glm47 \
    --reasoning-parser glm45 \
    --async-scheduling \
    ${SPEC_ARGS[@]+"${SPEC_ARGS[@]}"} \
    --additional-config "$ADDITIONAL_CONFIG" \
    --compilation-config "$COMPILATION_CONFIG" \
    --seed 1024 \
    "$@"
