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
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Source project common library
# shellcheck source=../../scripts/common.sh
source "${SCRIPT_DIR}/../../scripts/common.sh"

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
MODEL_PATH="${MODEL_PATH:-/home/jianzhnie/llmtuner/hfhub/models/meituan-longcat/expand/LongCat-Flash-Chat-combined}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/jianzhnie/llmtuner/llm/EasyInfer/output/LongCat-Flash-Chat}"
MODEL_NAME="${MODEL_NAME:-longcat-flash}"
PORT="${PORT:-8000}"
TASKS="${TASKS:-mmlu}"
FEWSHOT="${FEWSHOT:-5}"
BACKEND="${BACKEND:-api}"
# max_model_len → 在 API 模式下映射为 max_length（上下文总长度，含 prompt + 生成）
# max_gen_toks 控制生成长度上限，未设置时由各后端决定（默认 256）
# 注意: 必须 ≤ 模型部署时的 MAX_MODEL_LEN，否则请求会被拒绝
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"

# 可选任务:
#   mmlu, gsm8k, ceval-valid, hendrycks_math500, minerva_math500
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
    --max-model-len "$MAX_MODEL_LEN" \
    --num-concurrent 4
