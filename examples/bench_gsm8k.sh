#!/bin/bash
# =============================================================================
# GSM8K Benchmark Script (via lm-evaluation-harness, API backend)
# Usage: MODEL_NAME=glm-5 PORT=8001 bash bench_gsm8k.sh
#
# GSM8K task settings (from lm_eval):
#   - Format: "Question: {{question}}\nAnswer:" (completion, NOT chat)
#   - fewshot: 5 | until: ["Question:", "</s>", "<|im_end|>"]
#   - max_gen_toks: 512 (chain-of-thought needs space)
# =============================================================================
set -eo pipefail

MODEL_PATH="${MODEL_PATH:-/home/jianzhnie/llmtuner/hfhub/models/Eco-Tech/GLM-5-w4a8}"
MODEL_NAME="${MODEL_NAME:-glm-5}"
PORT="${PORT:-8001}"
LIMIT="${LIMIT:-100}"
FEWSHOT="${FEWSHOT:-5}"
MAX_GEN_TOKS="${MAX_GEN_TOKS:-512}"

export HF_HOME="${HF_HOME:-/home/jianzhnie/llmtuner/hfhub/cache}"

echo "[INFO] Benchmark: ${MODEL_NAME} gsm8k (port=${PORT}, fewshot=${FEWSHOT}, max_tokens=${MAX_GEN_TOKS}, limit=${LIMIT})"

bash /home/jianzhnie/llmtuner/llm/npuslim/tools/eval/run_lmeval.sh \
    "$MODEL_PATH" \
    --backend api \
    --port "$PORT" \
    --url "http://127.0.0.1:${PORT}/v1/completions" \
    --model-name "$MODEL_NAME" \
    --tasks gsm8k \
    --fewshot "$FEWSHOT" \
    --max-gen-toks "$MAX_GEN_TOKS" \
    --output-dir "${OUTPUT_DIR:-/tmp/benchmark_${MODEL_NAME}_gsm8k}" \
    --limit "$LIMIT"
