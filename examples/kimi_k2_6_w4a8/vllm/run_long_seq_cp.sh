#!/bin/bash
# Kimi-K2.6 W4A8 — 长序列上下文并行 (Context Parallelism)
# 功能: 通过 Context Parallelism (CP) 突破单卡序列长度限制
# 架构: KimiK25ForConditionalGeneration | MLA | 支持 CP on A3
# 参考: https://docs.vllm.ai/projects/ascend/zh-cn/releases-v0.20.2rc/tutorials/features/long_sequence_context_parallel_single_node.html
#
# 约束:
#   - tp_size 必须能被 dcp_size 整除
#   - dcp_size ≤ max_dcp_size = tp_size // num_kv_heads
#   - 当前仅支持 Atlas A3 设备
#   - Kimi-K2.6: MLA kv_lora_rank=512, q_lora_rank=1536, head_dim=128
#
# 用法:
#   # A3 单节点 16 卡: TP=16 DCP=2
#   TP=16 DCP=2 MAX_MODEL_LEN=131072 bash run_long_seq_cp.sh
#
#   # A3 双节点 PP: TP=8 PP=2 DCP=2
#   TP=8 PP=2 DCP=2 MAX_MODEL_LEN=131072 bash run_long_seq_cp.sh
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
MODEL_PATH="${MODEL_PATH:-$BASE_MODEL_PATH/Kimi-K2.6-w4a8}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8003}"
TP="${TP:-16}"
PP="${PP:-1}"
DP="${DP:-1}"
PCP_SIZE="${PCP_SIZE:-2}"
DCP_SIZE="${DCP_SIZE:-2}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-131072}"

export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export HCCL_BUFFSIZE=800
export VLLM_ASCEND_BALANCE_SCHEDULING=0
export HCCL_OP_EXPANSION_MODE=AIV
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_ASCEND_ENABLE_MLAPO=1
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
export TASK_QUEUE_ENABLE=1
export VLLM_USE_MODELSCOPE=False

echo "============================================"
echo "[INFO] Kimi-K2.6 W4A8 — Long Sequence Context Parallel"
echo "[INFO] Model: $MODEL_PATH"
echo "[INFO] TP=$TP PP=$PP DP=$DP PCP=$PCP_SIZE DCP=$DCP_SIZE"
echo "[INFO] MAX_MODEL_LEN=$MAX_MODEL_LEN MAX_NUM_SEQS=$MAX_NUM_SEQS"
echo "[WARN] 需要 Atlas A3 设备 (A2 不支持 CP)"
echo "============================================"

vllm serve "$MODEL_PATH" \
    --host "$HOST" \
    --port "$PORT" \
    --served-model-name "kimi-k2.6" \
    --trust-remote-code \
    --dtype bfloat16 \
    --tensor-parallel-size "$TP" \
    --pipeline-parallel-size "$PP" \
    --data-parallel-size "$DP" \
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
    --tool-call-parser kimi_k2 \
    --language-model-only \
    --mm-encoder-tp-mode data \
    --allowed-local-media-path /home/jianzhnie/llmtuner/ \
    --no-enable-prefix-caching \
    --seed 1024 \
    "$@"
