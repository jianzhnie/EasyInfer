#!/bin/bash
# =============================================================================
# MATH Benchmark Script (via lm-evaluation-harness, API backend)
# Usage: MODEL_NAME=glm-5 PORT=8001 bash bench_math.sh
#
# MATH task settings (from lm_eval):
#   - Format: "Problem: {{problem}}\nAnswer:" (completion, NOT chat)
#   - fewshot: 4 | until: ["Problem:"]
#   - max_gen_toks: 1024 (complex math proofs need space)
#   - 7 subtasks: algebra, counting_and_prob, geometry, intermediate_algebra,
#                  num_theory, prealgebra, precalc
# =============================================================================
set -eo pipefail

MODEL_PATH="${MODEL_PATH:-/home/jianzhnie/llmtuner/hfhub/models/Eco-Tech/GLM-5-w4a8}"
MODEL_NAME="${MODEL_NAME:-glm-5}"
PORT="${PORT:-8001}"
LIMIT="${LIMIT:-100}"
FEWSHOT="${FEWSHOT:-4}"
MAX_GEN_TOKS="${MAX_GEN_TOKS:-1024}"

export HF_HOME="${HF_HOME:-/home/jianzhnie/llmtuner/hfhub/cache}"

echo "[INFO] Benchmark: ${MODEL_NAME} hendrycks_math (port=${PORT}, fewshot=${FEWSHOT}, max_tokens=${MAX_GEN_TOKS}, limit=${LIMIT})"

bash /home/jianzhnie/llmtuner/llm/npuslim/tools/eval/run_lmeval.sh \
    "$MODEL_PATH" \
    --backend api \
    --port "$PORT" \
    --url "http://127.0.0.1:${PORT}/v1/completions" \
    --model-name "$MODEL_NAME" \
    --tasks hendrycks_math \
    --fewshot "$FEWSHOT" \
    --max-gen-toks "$MAX_GEN_TOKS" \
    --output-dir "${OUTPUT_DIR:-/tmp/benchmark_${MODEL_NAME}_math}" \
    --limit "$LIMIT"
