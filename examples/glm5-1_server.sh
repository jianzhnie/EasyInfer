#!/bin/bash

export HCCL_OP_EXPANSION_MODE="AIV"
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export HCCL_BUFFSIZE=200
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_ASCEND_BALANCE_SCHEDULING=1

model_path=models/vllm-ascend/GLM-5-w4a8 
vllm serve \
     $model_path \
     --host 0.0.0.0 \
     --port 8077 \
     --data-parallel-size 1 \
     --tensor-parallel-size 8 \
     --enable-expert-parallel \
     --seed 1024 \
     --served-model-name glm-5.1 \
     --max-num-seqs 2 \
     --max-model-len 32768 \
     --max-num-batched-tokens 4096 \
     --trust-remote-code \
     --gpu-memory-utilization 0.95 \
     --quantization ascend \
     --tool-call-parser glm47 \
     --reasoning-parser glm45 \
     --enable-chunked-prefill \
     --enable-prefix-caching \
     --enable-auto-tool-choice \
     --chat-template-content-format=string \
     --async-scheduling \
     --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
     --additional-config '{"fuse_muls_add": true, "multistream_overlap_shared_expert": true, "ascend_compilation_config": {"enable_npugraph_ex": true}}' \
     --speculative-config '{"num_speculative_tokens": 3, "method": "deepseek_mtp"}'