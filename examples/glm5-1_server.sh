#!/bin/bash

# 加载共享环境配置（VLLM_HOST_IP, NPU 网络接口等）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SET_ENV_FILE="${SCRIPT_DIR}/../scripts/vllm/set_env.sh"
if [[ -f "$SET_ENV_FILE" ]]; then
    set +u
    source "$SET_ENV_FILE" 2>/dev/null || true
    set -u
fi

export HCCL_OP_EXPANSION_MODE="AIV"
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export HCCL_BUFFSIZE=200
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_ASCEND_BALANCE_SCHEDULING=1

export MODEL_PATH="${MODEL_PATH:-/home/jianzhnie/llmtuner/hfhub/models/Eco-Tech/GLM-5.1-w8a8}"
vllm serve \
     $MODEL_PATH \
     --host 0.0.0.0 \
     --port 8077 \
     --seed 1024 \
     --data-parallel-size 1 \
     --tensor-parallel-size 32 \
     --enable-expert-parallel \
     --served-model-name glm-5.1 \
     --max-num-seqs 2 \
     --max-model-len 131072 \
     --max-num-batched-tokens 4096 \
     --trust-remote-code \
     --gpu-memory-utilization 0.95 \
     --quantization ascend \
     --distributed-executor-backend ray \
     --tool-call-parser glm47 \
     --reasoning-parser glm45 \
     --enable-chunked-prefill \
     --enable-prefix-caching \
     --enable-auto-tool-choice \
     --chat-template-content-format=string \
     --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
     --additional-config '{"fuse_muls_add": true, "multistream_overlap_shared_expert": true, "ascend_compilation_config": {"enable_npugraph_ex": true}}' \
     --speculative-config '{"num_speculative_tokens": 3, "method": "deepseek_mtp"}'