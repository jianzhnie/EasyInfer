#!/usr/bin/env bash
#
# lm-evaluation-harness 运行脚本
# 用法: 环境变量覆盖: MODEL_PATH=/path/to/model bash lm_eval.sh
#

set -euo pipefail

# 设置缓存目录
export HF_HOME="${HF_HOME:-/llm_workspace_1P/robin/hfhub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/llm_workspace_1P/robin/hfhub/datasets}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-0}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"

MODEL_PATH="${MODEL_PATH:-/llm_workspace_1P/robin/hfhub/pcl-kimi2-stage2/kimi2-mcore2hf_step450}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/mcore2hf_step450}"
URL="${URL:-0.0.0.0}"
PORT="${PORT:-8080}"
TASKS="${TASKS:-mmlu}"
FEWSHOT="${FEWSHOT:-5}"

bash tools/eval/run_lmeval.sh \
    --model-path "$MODEL_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --backend api \
    --url "$URL" \
    --port "$PORT" \
    --tasks "$TASKS" \
    --fewshot "$FEWSHOT"
