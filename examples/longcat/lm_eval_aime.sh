#!/bin/bash
#
# AIME Benchmark Script (via lm-evaluation-harness, API backend)
# =============================================================================
# AIME 竞赛级数学推理评测，默认使用采样 + 投票 (pass@64)。
#
# Usage:
#   bash examples/longcat/lm_eval_aime.sh
#
#   # 环境变量覆盖（注意：do_sample=True 必须显式设置以覆盖 YAML 默认值）
#   GEN_KWARGS='do_sample=True,temperature=0.8,top_p=0.9,n=32' bash examples/longcat/lm_eval_aime.sh
#
#   # 快速验证（每题只采样 4 次，限 5 题）
#   GEN_KWARGS='do_sample=True,temperature=0.6,top_p=0.95,top_k=40,n=4' LIMIT=5 bash examples/longcat/lm_eval_aime.sh
#
#   # greedy 单次生成
#   GEN_KWARGS='do_sample=False,n=1' bash examples/longcat/lm_eval_aime.sh
# =============================================================================
# AIME Tasks (generative, requires --chat):
#   aime24                     AIME 2024 (30 problems, 0-shot CoT)
#   aime25                     AIME 2025 (30 problems, 0-shot CoT)
#
# Important: aime24/aime25 任务 YAML 内置了 do_sample=False 和 max_gen_toks=32768，
# 必须通过 GEN_KWARGS 显式覆盖这两个值，否则采样不会生效且生成长度过大。
#
# Sampling strategy:
#   do_sample=True, n=64, temperature=0.6, top_p=0.95, top_k=40  → pass@64
#   do_sample=False, n=1                                          → pass@1 (greedy)
#
# 注意: n=64 时每题 64 次生成，60 题共计 3840 次 API 请求，耗时较长。
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
# export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
# export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
# export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

# ---------------------------------------------------------------------------
# 评测参数
# ---------------------------------------------------------------------------
# 模型路径 (本地 tokenizer 路径，用于 lm-eval 做 tokenization)
MODEL_PATH="${MODEL_PATH:-/home/jianzhnie/llmtuner/hfhub/models/meituan-longcat/LongCat-Flash-Chat}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/jianzhnie/llmtuner/llm/EasyInfer/output/LongCat-Flash}"
# API 中注册的模型名 (served-model-name)
MODEL_NAME="${MODEL_NAME:-longcat-flash}"
# 服务地址
API_HOST="${API_HOST:-localhost}"
PORT="${PORT:-8200}"
BACKEND="${BACKEND:-api}"

# AIME 默认：aime24 + aime25，采样 64 条做 pass@64
TASKS="${TASKS:-aime24,aime25}"
FEWSHOT="${FEWSHOT:-0}"
# max_model_len → API 模式下映射为 max_length（上下文总长度，含 prompt + 生成）
# max_gen_toks 控制生成长度上限，AIME 推理链较长需要较大值
# 注意: max_model_len 必须 ≤ 模型部署时的 MAX_MODEL_LEN，否则请求会被拒绝
MAX_GEN_TOKS="${MAX_GEN_TOKS:-32768}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-34816}"
LIMIT="${LIMIT:-none}"
# AIME 默认生成参数：采样模式，64 条候选做 majority voting
# do_sample=True 和 max_gen_toks 必须显式设置，覆盖任务 YAML 中的默认值
#   - aime24/aime25 YAML 内置 do_sample=False（greedy），需要 CLI 覆盖
#   - aime24/aime25 YAML 内置 max_gen_toks=32768，需要 CLI 覆盖为合理值
GEN_KWARGS="${GEN_KWARGS:-do_sample=True,temperature=0.6,top_p=0.95,top_k=40,n=32,max_gen_toks=32768}"
# API 请求超时（秒），n=64 时单题耗时较长，建议 ≥600
TIMEOUT="${TIMEOUT:-360000}"
# 并发数，n=64 时适当降低避免服务端排队过深
NUM_CONCURRENT="${NUM_CONCURRENT:-8}"

# ---------------------------------------------------------------------------
# 执行评测
# ---------------------------------------------------------------------------
log_info "AIME Benchmarks: model=$MODEL_NAME, tasks=$TASKS"
log_info "fewshot=$FEWSHOT, max_gen_toks=$MAX_GEN_TOKS, max_model_len=$MAX_MODEL_LEN"
log_info "num_concurrent=$NUM_CONCURRENT, timeout=$TIMEOUT"
log_info "gen_kwargs=$GEN_KWARGS"

ARGS=(
    --model-path "$MODEL_PATH"
    --model-name "$MODEL_NAME"
    --backend "$BACKEND"
    --port "$PORT"
    --tasks "$TASKS"
    --fewshot "$FEWSHOT"
    --max-model-len "$MAX_MODEL_LEN"
    --max-gen-toks "$MAX_GEN_TOKS"
    --output-dir "${OUTPUT_DIR}/aime"
    --num-concurrent "$NUM_CONCURRENT"
    --chat
    --apply-chat-template
    --gen-kwargs "$GEN_KWARGS"
    --timeout "$TIMEOUT"
)

# API 模式下显式指定远程 URL（支持非本地服务）
if [[ "$BACKEND" == "api" ]]; then
    ARGS+=(--url "http://${API_HOST}:${PORT}/v1/chat/completions")
fi

[[ "$LIMIT" != "none" ]] && ARGS+=(--limit "$LIMIT")

bash "${PROJECT_ROOT}/tools/eval/run_lmeval.sh" "${ARGS[@]}"
