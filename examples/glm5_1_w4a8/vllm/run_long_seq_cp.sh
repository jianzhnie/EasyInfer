#!/bin/bash
# GLM-5.1 W4A8 — 长序列上下文并行 (Context Parallelism)
# 注意: GLM-5.1 W4A8 的 DSA CP 路径不兼容，需要 A3 设备
# 参考: https://docs.vllm.ai/projects/ascend/zh-cn/releases-v0.20.2rc/tutorials/features/long_sequence_context_parallel_single_node.html
#
# 用法:
#   # A3 单节点: TP=16 DCP=2 MAX_MODEL_LEN=131072 bash run_long_seq_cp.sh
set -eo pipefail

set +u
if [[ -f "/usr/local/Ascend/cann/set_env.sh" ]]; then
    source /usr/local/Ascend/cann/set_env.sh
fi
if [[ -f "/usr/local/Ascend/nnal/atb/set_env.sh" ]]; then
    source /usr/local/Ascend/nnal/atb/set_env.sh
fi
set -u

BASE_MODEL_PATH="/home/jianzhnie/llmtuner/hfhub/models/Eco-Tech"
MODEL_PATH="${MODEL_PATH:-$BASE_MODEL_PATH/GLM-5.1-w4a8}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8002}"
TP="${TP:-16}"
PP="${PP:-1}"
PCP_SIZE="${PCP_SIZE:-2}"
DCP_SIZE="${DCP_SIZE:-2}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-131072}"

export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export HCCL_BUFFSIZE=512
export VLLM_ASCEND_BALANCE_SCHEDULING=0
export HCCL_OP_EXPANSION_MODE=AIV
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_ASCEND_ENABLE_FLASHCOMM1=0
export VLLM_ASCEND_ENABLE_MLAPO=1
export TASK_QUEUE_ENABLE=1
export VLLM_USE_MODELSCOPE=False

echo "============================================"
echo "[INFO] GLM-5.1 W4A8 — Long Sequence Context Parallel"
echo "[INFO] TP=$TP PP=$PP PCP=$PCP_SIZE DCP=$DCP_SIZE"
echo "[INFO] MAX_MODEL_LEN=$MAX_MODEL_LEN"
echo "[WARN] 需要 Atlas A3 设备"
echo "[WARN] FLASHCOMM1=0 (DSA CP 兼容性)"
echo "============================================"

vllm serve "$MODEL_PATH" \
    --host "$HOST" \
    --port "$PORT" \
    --served-model-name "glm-5.1" \
    --trust-remote-code \
    --dtype bfloat16 \
    --tensor-parallel-size "$TP" \
    --pipeline-parallel-size "$PP" \
    --prefill-context-parallel-size "$PCP_SIZE" \
    --decode-context-parallel-size "$DCP_SIZE" \
    --distributed-executor-backend ray \
    --quantization ascend \
    --gpu-memory-utilization 0.95 \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
    --enable-chunked-prefill \
    --enable-expert-parallel \
    --enable-auto-tool-choice \
    --tool-call-parser glm47 \
    --reasoning-parser glm45 \
    --speculative-config '{"num_speculative_tokens": 3, "method": "mtp"}' \
    --no-enable-prefix-caching \
    --seed 1024 \
    "$@"
