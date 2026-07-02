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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Source project common library
# shellcheck source=../scripts/common.sh
source "${SCRIPT_DIR}/../scripts/common.sh"

# ---------------------------------------------------------------------------
# HF 缓存配置 — 指向本地数据目录，避免重复下载
# ---------------------------------------------------------------------------
export HF_HOME="${HF_HOME:-/home/jianzhnie/llmtuner/hfhub/cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
# export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
# export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
# export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

# ---------------------------------------------------------------------------
# 评测参数
# ---------------------------------------------------------------------------
MODEL_PATH="${MODEL_PATH:-/home/jianzhnie/llmtuner/hfhub/models/meituan-longcat/expand/LongCat-Flash-Chat-1024E-512Zero-E-Topk24}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/jianzhnie/llmtuner/llm/EasyInfer/output/LongCat-Flash-Chat-1024E-512Zero-E-Topk24}"
MODEL_NAME="${MODEL_NAME:-longcat-flash}"
PORT="${PORT:-8000}"
TASKS="${TASKS:-mmlu}"
FEWSHOT="${FEWSHOT:-5}"
BACKEND="${BACKEND:-api}"
MAX_GEN_TOKS="${MAX_GEN_TOKS:-256}"
# 额外生成参数，通过 --gen-kwargs 透传给 lm_eval（与 --max-gen-toks 合并）
# 注意：不要在此设置 max_tokens，会覆盖上面的 MAX_GEN_TOKS
GEN_KWARGS_EXTRA="${GEN_KWARGS_EXTRA:-}"

# math 500
# hendrycks_math500,minerva_math500
# ceval
# ceval-valid
# ---------------------------------------------------------------------------
# 执行评测
# ---------------------------------------------------------------------------
log_info "Starting evaluation: model=$MODEL_NAME, tasks=$TASKS, backend=$BACKEND"

bash "${PROJECT_ROOT}/tools/eval/run_lmeval.sh" \
    --model-path "$MODEL_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --model-name "${MODEL_NAME}" \
    --backend "$BACKEND" \
    --port "$PORT" \
    --tasks "$TASKS" \
    --fewshot "$FEWSHOT" \
    --max-gen-toks "$MAX_GEN_TOKS" \
    ${GEN_KWARGS_EXTRA:+--gen-kwargs "$GEN_KWARGS_EXTRA"}
