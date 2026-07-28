#!/bin/bash
# =============================================================================
# GLM-5.2 W8A8 — Direct vllm serve deployment (32K context)
# =============================================================================
# Architecture: GlmMoeDsaForCausalLM | 256 Experts | MLA | MTP=1
# Max Position: 1048576 | Deploy: 32K context (override with MAX_MODEL_LEN)
#
# Hardware:
#   - Atlas 800 A3 (128G x 8):  TP=8 DP=1 PP=1 (single node)
#   - Atlas 800 A3 (64G x 16):  TP=8 DP=2 PP=1 (single node, official)
#   - Atlas 800 A2 (64G x 8):   TP=8 PP=2 (2 nodes, verified)
#
# Note: TP=16 is feasible (attention heads 64/16=4, indexer heads 32/16=2;
#   official GLM-5.2 1M single-node config uses TP=16). PP>1 + MTP is not
#   supported on v0.23.0rc1 mixed deployments (fixed on main, #11076).
#
# Usage:
#   bash run_vllm.sh                           # TP=8 DP=1 PP=1 single node
#   DP=2 bash run_vllm.sh                      # TP=8 DP=2 single node (A3, 16 NPU)
#   PP=2 RAY_ADDRESS=<head>:6379 bash run_vllm.sh  # 2-node A2 (verified)
#   ENABLE_MTP=1 bash run_vllm.sh              # MTP speculative decode (PP=1 only)
#
# Reference:
#   https://docs.vllm.ai/projects/ascend/zh-cn/latest/tutorials/models/GLM5.2.html
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
readonly MAX_NUM_SEQS="${MAX_NUM_SEQS:-128}"
readonly MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-32768}"
readonly GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.95}"

# NPU environment variables (official docs + W8A8 specifics)
export HCCL_OP_EXPANSION_MODE=AIV
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_USE_MODELSCOPE=False
export VLLM_ASCEND_BALANCE_SCHEDULING=1
# DSA path is incompatible with FLASHCOMM1 (GLM-5.2 index_topk:2048 triggers DSA CP)
export VLLM_ASCEND_ENABLE_FLASHCOMM1=0
# Fused MC2 + HCCL_BUFFSIZE: multi-node (PP>1, TP>8, or DP spanning nodes)
# vs single-node (official docs). Override FUSED_MC2 explicitly via env if
# needed (e.g. single-node DP=2 on a 16-NPU A3).
if [[ "$PP" -gt 1 || "$TP" -gt 8 || "$DP" -gt 1 ]]; then
    export VLLM_ASCEND_ENABLE_FUSED_MC2="${VLLM_ASCEND_ENABLE_FUSED_MC2:-1}"
    export HCCL_BUFFSIZE=400
else
    export VLLM_ASCEND_ENABLE_FUSED_MC2="${VLLM_ASCEND_ENABLE_FUSED_MC2:-0}"
    export HCCL_BUFFSIZE=200
fi
export VLLM_ASCEND_ENABLE_MLAPO=1
export VLLM_USE_V1=1

# Runtime / debug
export ASCEND_LAUNCH_BLOCKING=0
export VLLM_ENGINE_READY_TIMEOUT_S=1800

# =============================================================================
# Multi-node network interface binding (optional)
# Set NIC_NAME to your high-speed interface (e.g., enp66s0f1). Leave empty for
# single-node auto-detect.
#
# RAY_ADDRESS: For multi-node (TP>8 or PP>1), set this to the Ray head node
#   address (e.g., RAY_ADDRESS=10.42.11.130:6379) so Engine Core subprocesses
#   connect to the existing Ray cluster instead of starting a local one.
#   Auto-detect: ray.init(address='auto') → get_runtime_context().gcs_address
#
# Note: TP=16 is feasible (attention heads 64/16=4, indexer heads 32/16=2;
#   official GLM-5.2 1M config uses TP=16). For A2 (64GB), TP=8 alone OOMs
#   (~60.4 GiB/NPU); use PP=2 or TP=16 (2 nodes). A3 (128GB) handles TP=8
#   single-node.
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
readonly ADDITIONAL_CONFIG='{"enable_dsa_cp": true,"enable_sparse_sfa_c8": false, "enable_sparse_li_c8": true,"enable_balance_scheduling": true,"multistream_overlap_shared_expert":true}'

# MTP speculative decoding. NOTE: on vllm-ascend v0.23.0rc1, PP>1 + MTP is
# rejected in co-located (mixed) deployment — mixed PP+MTP support (#11076)
# is only on main, not in any release tag; PD-disaggregated P nodes support
# PP+MTP since v0.22.1rc1 (#10199). MTP defaults to off; enable only with PP=1.
ENABLE_MTP="${ENABLE_MTP:-0}"
if [[ "$ENABLE_MTP" == "1" && "$PP" -gt 1 ]]; then
    echo "[WARN] v0.23.0rc1 共部署不支持 PP>1+MTP（#11076 仅在 main），已自动禁用 MTP"
    ENABLE_MTP=0
fi
readonly ENABLE_MTP
SPEC_ARGS=()
if [[ "$ENABLE_MTP" == "1" ]]; then
    SPEC_ARGS+=(--speculative-config '{"num_speculative_tokens": 3, "method": "deepseek_mtp", "enforce_eager": true}')
fi

# Expert parallel. Official low-latency single-node config (dp1tp16) disables
# it; default on for throughput-oriented DP deployments.
ENABLE_EP="${ENABLE_EP:-1}"
readonly ENABLE_EP
EP_ARGS=()
if [[ "$ENABLE_EP" == "1" ]]; then
    EP_ARGS+=(--enable-expert-parallel)
fi

echo "============================================"
echo "[INFO] GLM-5.2 W8A8 — vLLM-Ascend Deployment"
echo "[INFO] Model:    $MODEL_PATH"
echo "[INFO] TP=$TP  PP=$PP  DP=$DP  PORT=$PORT"
echo "[INFO] MAX_MODEL_LEN=$MAX_MODEL_LEN  MAX_NUM_SEQS=$MAX_NUM_SEQS"
echo "[INFO] MAX_NUM_BATCHED_TOKENS=$MAX_NUM_BATCHED_TOKENS"
echo "[INFO] GPU_MEM_UTIL=$GPU_MEM_UTIL"
echo "[INFO] RAY_ADDRESS=${RAY_ADDRESS:-auto-detect}"
echo "[INFO] MTP: $([[ "$ENABLE_MTP" == "1" ]] && echo 'ON (3 tokens, deepseek_mtp)' || echo 'OFF')"
echo "[INFO] FLASHCOMM1=0 (DSA CP incompatible)"
echo "[INFO] FUSED_MC2=$VLLM_ASCEND_ENABLE_FUSED_MC2  EP=$ENABLE_EP"
echo "[INFO] Tool Calling: glm47 parser + glm45 reasoning"
echo "[INFO] Features: chunked-prefill, prefix-caching, async-scheduling"
echo "[INFO] Hardware: A3 (128G) TP=8 single-node; A2 (64G) TP=8 PP=2 two-node"
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
    --distributed-executor-backend ray \
    --quantization ascend \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
    --chat-template-content-format string \
    --enable-chunked-prefill \
    --enable-prefix-caching \
    "${EP_ARGS[@]}" \
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
