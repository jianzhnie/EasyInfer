#!/bin/bash
#
# ModelScope 模型与数据集下载脚本
#
# 用法:
#   bash ms_download.sh                          # 使用默认配置下载
#   MS_CACHE_DIR=/data/ms_cache bash ms_download.sh  # 自定义缓存目录
#

set -euo pipefail

# 前置检查
if ! command -v modelscope >/dev/null 2>&1; then
    echo "[ERROR] 'modelscope' command not found. Install: pip install modelscope" >&2
    exit 127
fi

# 目录配置
MS_CACHE_DIR="${MS_CACHE_DIR:-$HOME/modelscope}"
MS_MODEL_DIR="${MS_MODEL_DIR:-${MS_CACHE_DIR}/models}"
MS_DATASETS_DIR="${MS_DATASETS_DIR:-${MS_CACHE_DIR}/datasets}"

# ---- 模型下载 ----

# modelscope download --model 'LLM-Research/Meta-Llama-3.1-405B' --include '*.json' --local_dir "${MS_MODEL_DIR}/LLM-Research/Meta-Llama-3.1-405B"
# modelscope download --model 'Qwen/Qwen3-32B' --local_dir "${MS_MODEL_DIR}/Qwen/Qwen3-32B"

# ---- 数据集下载 ----

# modelscope download --dataset 'openai/gsm8k' --local_dir "${MS_DATASETS_DIR}/openai/gsm8k"
