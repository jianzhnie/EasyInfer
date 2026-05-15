#!/bin/bash

# 设置国内镜像
export HF_ENDPOINT="https://hf-mirror.com"
export HF_HUB_ENABLE_HF_TRANSFER=0

# 模型下载
## HuggingFace 模型下载
hf download Qwen/Qwen3-0.6B --local-dir ~/hfhub/models/Qwen/Qwen3-0.6B
# hf download Qwen/Qwen2.5-0.5B --local-dir ~/hfhub/models/Qwen/Qwen2.5-0.5B
# hf download facebook/opt-125m  --local-dir ~/hfhub/models/facebook/opt-125m
# hf download deepseek-ai/DeepSeek-V3-Base --local-dir ~/hfhub/models/deepseek-ai/DeepSeek-V3-Base --exclude "*.safetensors"
# hf download moonshotai/Kimi-K2-Base --local-dir ~/hfhub/models/moonshotai/Kimi-K2-Base --exclude "*.safetensors"
# hf download Qwen/Qwen3-32B --local-dir ~/hfhub/models/Qwen/Qwen3-32B --exclude "*.safetensors"

## ModelScope 模型下载
# modelscope download --model 'LLM-Research/Meta-Llama-3.1-405B' --include '*.json' --local_dir ~/hfhub/models/LLM-Research/Meta-Llama-3.1-405B

# 数据集下载
# hf download --repo-type dataset openai/gsm8k --local-dir ~/hfhub/datasets/openai/gsm8k
# hf download --repo-type dataset tatsu-lab/alpaca --local-dir ~/hfhub/datasets/tatsu-lab/alpaca
hf download --repo-type dataset openai/gsm8k --local-dir ~/hfhub/datasets/openai/gsm8k
hf download --repo-type dataset cais/mmlu --local-dir ~/hfhub/datasets/cais/mmlu