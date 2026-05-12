#!/bin/bash
## 进入容器
# docker exec -it vllm-ascend-env-a3 /bin/bash

# Qwen3-32B 模型配置
# ------------------------------------------------------------------------------
export MODEL_PATH="${MODEL_PATH:-/home/jianzhnie/llmtuner/hfhub/models/Qwen/Qwen3-32B}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-32b}"
export TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-8}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"

bash  /home/jianzhnie/llmtuner/llm/EasyInfer/examples/qwen3_server.sh