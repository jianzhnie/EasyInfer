#!/bin/bash
# =============================================================================
# Deploy LongCat-Flash-Chat-2layer inside Docker container (EP mode)
# =============================================================================
set -euo pipefail

# Ensure EasyInfer plugins are installed (entry_points for vllm discovery)
pip install -e /home/jianzhnie/llmtuner/llm/EasyInfer --quiet 2>&1 | tail -1

export PYTHONPATH=/home/jianzhnie/llmtuner/llm/EasyInfer:${PYTHONPATH}
export HCCL_OP_EXPANSION_MODE=AIV
export HCCL_SOCKET_IFNAME="enp66s0f5"
export GLOO_SOCKET_IFNAME="enp66s0f5"
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export HCCL_BUFFSIZE=4096
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_USE_MODELSCOPE=False
export HCCL_CONNECT_TIMEOUT=1800
export HCCL_EXEC_TIMEOUT=1800
export ENABLE_EXPERT_PARALLEL=1

MODEL_PATH="/home/jianzhnie/llmtuner/hfhub/models/meituan-longcat/LongCat-Flash-Chat/expand/LongCat-Flash-Chat-2layer"

echo "============================================"
echo "[INFO] LongCat-Flash-Chat-2layer Deployment"
echo "[INFO] Model:  $MODEL_PATH"
echo "[INFO] TP=2, EP=enabled, mp backend"
echo "[INFO] HCCL_BUFFSIZE=4096"
echo "[INFO] Port: 8010"
echo "============================================"

vllm serve "$MODEL_PATH" \
    --host 0.0.0.0 \
    --port 8010 \
    --served-model-name longcat-flash-2layer \
    --trust-remote-code \
    --dtype bfloat16 \
    --tensor-parallel-size 2 \
    --pipeline-parallel-size 1 \
    --enable-expert-parallel \
    --distributed-executor-backend mp \
    --gpu-memory-utilization 0.85 \
    --max-model-len 2048 \
    --max-num-seqs 64 \
    --max-num-batched-tokens 2048 \
    --no-enable-prefix-caching \
    --enforce-eager \
    --seed 1024
