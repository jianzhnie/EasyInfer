#!/bin/bash/
source set_env.sh

hfhub="/llm_workspace_1P/robin/hfhub/models"
model_path="${hfhub}/Qwen/Qwen3-32B"
model_name="Qwen/Qwen3-32B"

num_gpus=8
max_model_len=32768  # ✅ 支持 32k 上下文
gpu_memory_utilization=0.9  # ✅ 提高内存利用率

vllm serve \
  --model $model_path \
  --trust-remote-code \
  --served-model-name $model_name \
  --tensor-parallel-size $num_gpus \
  --gpu-memory-utilization $gpu_memory_utilization \
  --max-model-len $max_model_len  \
  --max-num-seqs 256 \
  --enable-prefix-caching \
  --enforce-eager \
  --port 8000
