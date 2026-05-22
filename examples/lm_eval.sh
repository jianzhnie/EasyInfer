#!/bin/bash
#
# lm-evaluation-harness 评测示例脚本
# 支持使用本地 HuggingFace 数据集进行评测
#
# 用法:
#   # 基本用法（使用本地 HF 缓存）
#   bash examples/lm_eval.sh
#
#   # 指定模型和任务
#   MODEL_PATH=/data/model TASKS=mmlu,gsm8k bash examples/lm_eval.sh
#
#   # 使用自定义 YAML 任务配置（本地 JSON/CSV 数据集）
#   TASK_DIR=/data/custom_tasks TASKS=my_custom_task bash examples/lm_eval.sh
#
#   # 离线模式（要求所有数据集已在本地缓存）
#   HF_DATASETS_OFFLINE=1 bash examples/lm_eval.sh
#

set -euo pipefail

# ---------------------------------------------------------------------------
# HF 缓存配置 — 指向本地数据目录，避免重复下载
# ---------------------------------------------------------------------------
export HF_HOME="${HF_HOME:-/llm_workspace_1P/robin/hfhub/cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

# ---------------------------------------------------------------------------
# 评测参数
# ---------------------------------------------------------------------------
MODEL_PATH="${MODEL_PATH:-/llm_workspace_1P/robin/hfhub/models/meituan-longcat/expand/LongCat-Flash-Chat-1024E-512Zero-E-Topk24-v2}"
OUTPUT_DIR="${OUTPUT_DIR:-/llm_workspace_1P/robin/EasyInfer/output/LongCat-Flash-Chat-1024E-512Zero-E-Topk24-v2}"
MODEL_NAME="${MODEL_NAME:-longcat-flash}"
PORT="${PORT:-8080}"
TASKS="${TASKS:-mmlu}"
FEWSHOT="${FEWSHOT:-5}"
BACKEND="${BACKEND:-api}"

# math 500 
# hendrycks_math500,minerva_math500   
# ceval
# ceval-valid   
# ---------------------------------------------------------------------------
# 执行评测
# ---------------------------------------------------------------------------
bash tools/eval/run_lmeval.sh \
    --model-path "$MODEL_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --model-name "${MODEL_NAME}" \
    --backend "$BACKEND" \
    --port "$PORT" \
    --tasks "$TASKS" \
    --fewshot "$FEWSHOT"