#!/usr/bin/env bash
#
# HuggingFace 模型与数据集下载脚本
# 默认使用国内镜像 hf-mirror.com
#
# 用法:
#   bash hf_downlaod.sh                          # 使用默认配置下载
#   HF_HUB_DIR=/data/hfhub bash hf_downlaod.sh   # 自定义缓存根目录
#   HF_ENDPOINT=https://hf.co bash hf_downlaod.sh # 自定义镜像源
#

set -euo pipefail

# 设置国内镜像
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"

# 前置检查
if ! command -v hf >/dev/null 2>&1; then
    echo "[ERROR] 'hf' command not found. Install: pip install huggingface_hub" >&2
    exit 127
fi

# 目录配置
HF_HUB_DIR="${HF_HUB_DIR:-$HOME/hfhub}"
HF_MODEL_DIR="${HF_MODEL_DIR:-${HF_HUB_DIR}/models}"
HF_DATASETS_DIR="${HF_DATASETS_DIR:-${HF_HUB_DIR}/datasets}"

# ---- 模型下载 ----

hf download Qwen/Qwen3-0.6B --local-dir "${HF_MODEL_DIR}/Qwen/Qwen3-0.6B"
# hf download Qwen/Qwen2.5-0.5B --local-dir "${HF_MODEL_DIR}/Qwen/Qwen2.5-0.5B"
# hf download facebook/opt-125m --local-dir "${HF_MODEL_DIR}/facebook/opt-125m"
# hf download deepseek-ai/DeepSeek-V3-Base --local-dir "${HF_MODEL_DIR}/deepseek-ai/DeepSeek-V3-Base" --exclude "*.safetensors"
# hf download moonshotai/Kimi-K2-Base --local-dir "${HF_MODEL_DIR}/moonshotai/Kimi-K2-Base" --exclude "*.safetensors"
# hf download Qwen/Qwen3-32B --local-dir "${HF_MODEL_DIR}/Qwen/Qwen3-32B" --exclude "*.safetensors"

# ---- 数据集下载 ----

hf download --repo-type dataset openai/gsm8k --local-dir "${HF_DATASETS_DIR}/openai/gsm8k"
hf download --repo-type dataset cais/mmlu --local-dir "${HF_DATASETS_DIR}/cais/mmlu"
# hf download --repo-type dataset tatsu-lab/alpaca --local-dir "${HF_DATASETS_DIR}/tatsu-lab/alpaca"
