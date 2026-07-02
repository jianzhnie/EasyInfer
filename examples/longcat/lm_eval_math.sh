#!/bin/bash
#
# Math Benchmark Script (via lm-evaluation-harness, API backend)
# =============================================================================
# Usage:
#   bash examples/longcat/lm_eval_math.sh
#
#   # 环境变量覆盖
#   TASKS=gsm8k MAX_GEN_TOKS=1024 bash examples/longcat/lm_eval_math.sh
#
#   # 快速验证
#   LIMIT=10 bash examples/longcat/lm_eval_math.sh
#
#   # 全量（不限样本数）
#   LIMIT=none bash examples/longcat/lm_eval_math.sh
# =============================================================================
# Math Tasks (generative, compatible with --chat):
#   gsm8k                      Grade school math (5-shot, CoT)
#   math500 / minerva_math500  MATH benchmark 500-sample subset (4-shot)
#   hendrycks_math             Full MATH (12K problems, slow)
#
# Math Tasks (loglikelihood, do NOT use with --chat):
#   gpqa_main                  Graduate-level physics QA
#   mmlu_college_mathematics   MMLU college math subset
#   mmlu_high_school_mathematics
#   cmmlu_math / ceval-math    Chinese math subsets
#
# To mix generative + loglikelihood tasks, remove --chat and run separately.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# shellcheck source=../../scripts/common.sh
source "${SCRIPT_DIR}/../../scripts/common.sh"

# ---------------------------------------------------------------------------
# HF 缓存
# ---------------------------------------------------------------------------
export HF_HOME="${HF_HOME:-/home/jianzhnie/llmtuner/hfhub/cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"

# ---------------------------------------------------------------------------
# 评测参数
# ---------------------------------------------------------------------------
MODEL_PATH="${MODEL_PATH:-/home/jianzhnie/llmtuner/hfhub/models/meituan-longcat/expand/LongCat-Flash-Chat-combined}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/jianzhnie/llmtuner/llm/EasyInfer/output/LongCat-Flash-Chat}"
MODEL_NAME="${MODEL_NAME:-longcat-flash}"
PORT="${PORT:-8000}"
BACKEND="${BACKEND:-api}"

# 默认：生成类数学任务（适用 --chat --apply-chat-template）
TASKS="${TASKS:-gsm8k,math500}"
FEWSHOT="${FEWSHOT:-5}"
MAX_GEN_TOKS="${MAX_GEN_TOKS:-512}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
LIMIT="${LIMIT:-none}"

# 可选 math 任务组合:
#   快速: gsm8k,math500
#   完整: gsm8k,hendrycks_math
#   中文: ceval-valid (loglikelihood, 去掉 --chat --apply-chat-template)
#   混合: 先用 --chat 跑生成类, 再单独跑 loglikelihood 类

# ---------------------------------------------------------------------------
# 执行评测
# ---------------------------------------------------------------------------
log_info "Math Benchmarks: model=$MODEL_NAME, tasks=$TASKS"
log_info "fewshot=$FEWSHOT, max_gen_toks=$MAX_GEN_TOKS, max_model_len=$MAX_MODEL_LEN"

ARGS=(
    --model-path "$MODEL_PATH"
    --model-name "$MODEL_NAME"
    --backend "$BACKEND"
    --port "$PORT"
    --tasks "$TASKS"
    --fewshot "$FEWSHOT"
    --max-model-len "$MAX_MODEL_LEN"
    --max-gen-toks "$MAX_GEN_TOKS"
    --output-dir "${OUTPUT_DIR}/math"
    --num-concurrent 4
    --chat
    --apply-chat-template
)

[[ "$LIMIT" != "none" ]] && ARGS+=(--limit "$LIMIT")

bash "${PROJECT_ROOT}/tools/eval/run_lmeval.sh" "${ARGS[@]}"
