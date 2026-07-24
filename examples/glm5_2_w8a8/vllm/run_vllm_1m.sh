#!/bin/bash
# =============================================================================
# GLM-5.2 W8A8 — 1M context vllm serve deployment
# =============================================================================
# Architecture: GlmMoeDsaForCausalLM | 256 Experts | MLA | MTP=1
# Max Position: 1048576 | Deploy: 1M context with DSA CP
# Reference: https://docs.vllm.ai/projects/ascend/zh-cn/latest/tutorials/models/GLM5.2.html#1m
#
# Hardware:
#   - Atlas 800 A3 (128G x 16): TP=16 DP=1 (single node, 1M context)
#   - Atlas 800 A3 (128G x 8):  TP=8 PP=2 (2 nodes, 1M context co-located)
#
# Note: TP=16 may fail on some configurations due to MLA dimension
#   incompatibility (head_dim=192 × num_kv_heads=3 = 576 not divisible by 16).
#   If TP=16 fails, use TP=8 PP=2 multi-node instead. PP>1 blocks MTP.
#
# Usage:
#   bash run_vllm_1m.sh                           # TP=16 DP=1 single node (A3)
#   TP=8 PP=2 DP=2 bash run_vllm_1m.sh            # 2-node A3
#   ENABLE_MTP=1 bash run_vllm_1m.sh              # MTP (PP=1 only)
# =============================================================================
set -euo pipefail

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
readonly TP="${TP:-16}"
readonly PP="${PP:-1}"
readonly DP="${DP:-1}"
readonly ENABLE_MTP="${ENABLE_MTP:-1}"
readonly MAX_MODEL_LEN="${MAX_MODEL_LEN:-1024000}"
readonly MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
readonly MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-16384}"
readonly GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.8}"

# 1M context NPU environment (official docs)
export VLLM_ASCEND_ENABLE_NZ=1
export HCCL_OP_EXPANSION_MODE="AIV"
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=20
export HCCL_BUFFSIZE=768
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_USE_MODELSCOPE=False
export VLLM_ASCEND_BALANCE_SCHEDULING=1
export VLLM_ASCEND_ENABLE_FLASHCOMM1=0
# Fused MC2: enable for multi-node, disable for single-node
if [[ "$PP" -gt 1 || "$TP" -gt 8 ]]; then
    export VLLM_ASCEND_ENABLE_FUSED_MC2=1
else
    export VLLM_ASCEND_ENABLE_FUSED_MC2=0
fi
export VLLM_ASCEND_ENABLE_MLAPO=1
export VLLM_USE_V1=1
export ASCEND_LAUNCH_BLOCKING=0
export VLLM_ENGINE_READY_TIMEOUT_S=1800
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TASK_QUEUE_ENABLE=1

# =============================================================================
# Multi-node network interface binding (optional)
# Set NIC_NAME to your high-speed interface (e.g., enp66s0f1).
# Leave empty for single-node auto-detect.
#
# RAY_ADDRESS: For multi-node, set this to the Ray head node address
#   (e.g., RAY_ADDRESS=10.42.11.130:6379) so Engine Core subprocesses
#   connect to the existing Ray cluster instead of starting a local one.
#
# Note: TP=16 may fail due to MLA dimension incompatibility
#   (head_dim=192 × num_kv_heads=3 = 576, not divisible by 16).
#   If affected, use TP=8 PP=2 multi-node instead.
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

# Compilation config (official 1M context docs)
readonly COMPILATION_CONFIG='{"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [4, 16, 128]}'
readonly ADDITIONAL_CONFIG='{"enable_flashcomm1": false, "enable_dsa_cp": true, \
    "ascend_compilation_config": {"enable_npugraph_ex": true, "enable_static_kernel": false}, \
    "fuse_muls_add": true, "multistream_overlap_shared_expert": true, "enable_mc2_hierarchy_comm": false, \
    "enable_sparse_sfa_c8": true, "enable_sparse_li_c8": true, "enable_cpu_binding": true, \
    "recompute_scheduler_enable": false}'

# MTP speculative decoding. NOTE: PP>1 + MTP is rejected by vLLM 0.23.0
# ("PP+MTP is only supported on PD-disaggregated P nodes"), so MTP defaults
# to on for single-node; set ENABLE_MTP=0 with PP>1.
SPEC_ARGS=()
if [[ "$ENABLE_MTP" == "1" ]]; then
    SPEC_ARGS+=(--speculative-config '{"num_speculative_tokens": 3, "method": "deepseek_mtp", "enforce_eager": true}')
fi

echo "============================================"
echo "[INFO] GLM-5.2 W8A8 — 1M Context vLLM-Ascend Deployment"
echo "[INFO] Model:    $MODEL_PATH"
echo "[INFO] TP=$TP  PP=$PP  DP=$DP  PORT=$PORT"
echo "[INFO] MAX_MODEL_LEN=$MAX_MODEL_LEN  MAX_NUM_SEQS=$MAX_NUM_SEQS"
echo "[INFO] MAX_NUM_BATCHED_TOKENS=$MAX_NUM_BATCHED_TOKENS"
echo "[INFO] GPU_MEM_UTIL=$GPU_MEM_UTIL"
echo "[INFO] RAY_ADDRESS=${RAY_ADDRESS:-auto-detect}"
echo "[INFO] MTP: $([[ "$ENABLE_MTP" == "1" ]] && echo 'ON (3 tokens, deepseek_mtp)' || echo 'OFF')"
echo "[INFO] DSA CP: prefill_cp=1 decode_cp=16 interleave=128"
echo "[INFO] FLASHCOMM1=0 (DSA CP incompatible)"
echo "[INFO] FUSED_MC2=$VLLM_ASCEND_ENABLE_FUSED_MC2"
echo "[INFO] Tool Calling: glm47 parser + glm45 reasoning"
echo "[INFO] Features: chunked-prefill, prefix-caching, async-scheduling"
echo "[INFO] Hardware: A3 (128G) TP=16 single-node or TP=8 PP=2 two-node"
echo "============================================"

vllm serve "$MODEL_PATH" \
    --host "$HOST" \
    --port "$PORT" \
    --api-server-count 1 \
    --served-model-name "glm-5.2" \
    --trust-remote-code \
    --tensor-parallel-size "$TP" \
    --pipeline-parallel-size "$PP" \
    --data-parallel-size "$DP" \
    --prefill-context-parallel-size 1 \
    --decode-context-parallel-size 16 \
    --cp-kv-cache-interleave-size 128 \
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
    "${SPEC_ARGS[@]}" \
    --additional-config "$ADDITIONAL_CONFIG" \
    --compilation-config "$COMPILATION_CONFIG" \
    --safetensors-load-strategy prefetch \
    --seed 1024 \
    "$@"
