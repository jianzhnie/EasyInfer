#!/bin/bash
# =============================================================================
# Kimi-K2.6 W4A8 部署示例 (华为 NPU 环境)
# =============================================================================
# 调用 vllm_model_server.sh 部署 Kimi-K2.6 W4A8 量化模型
# Kimi-K2.6 基于 DeepSeek V3 架构，MoE (384 专家)，支持多模态 (Vision)
#
# 硬件要求:
#   - Atlas 800 A2 (64G × 8):   单节点 W4A8 部署
#   - Atlas 800 A3 (64G × 16):  单节点 W4A8 部署
#   - 多节点:                   8×8 卡 PP/DP 扩展
#
# 关键特性:
#   - 384 路由专家 + 1 共享专家 (DeepSeek MoE)
#   - W4A8 Ascend 量化
#   - 262K 原生上下文窗口
#   - 多模态支持 (Vision Transformer)
#   - DeepSeek V3 架构 (MLA + MoE)
#
# 用法:
#   # 默认 W4A8 单节点
#   ./vllm_server.sh
#
#   # 16 卡部署
#   TENSOR_PARALLEL_SIZE=16 MAX_MODEL_LEN=131072 ./vllm_server.sh
#
#   # 8 节点多节点部署
#   PIPELINE_PARALLEL_SIZE=8 DATA_PARALLEL_SIZE=8 ./vllm_server.sh
#
# 参考文档:
#   https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/index.html
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLLM_SCRIPT="${SCRIPT_DIR}/../../scripts/vllm/vllm_model_server.sh"

if [[ ! -f "$VLLM_SCRIPT" ]]; then
    echo "[ERROR] vLLM startup script not found: $VLLM_SCRIPT" >&2
    exit 1
fi

# ------------------------------------------------------------------------------
# 模型路径与基础配置
# ------------------------------------------------------------------------------
export MODEL_PATH="${MODEL_PATH:-/home/jianzhnie/llmtuner/hfhub/models/Eco-Tech/Kimi-K2.6-w4a8}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-kimi-k2.6}"
export HOST="${HOST:-0.0.0.0}"
export PORT="${PORT:-8003}"

# ------------------------------------------------------------------------------
# 华为 NPU 环境变量
# ------------------------------------------------------------------------------
export HCCL_OP_EXPANSION_MODE="${HCCL_OP_EXPANSION_MODE:-AIV}"
export OMP_PROC_BIND="${OMP_PROC_BIND:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export HCCL_BUFFSIZE="${HCCL_BUFFSIZE:-800}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"
export VLLM_ASCEND_BALANCE_SCHEDULING="${VLLM_ASCEND_BALANCE_SCHEDULING:-1}"
export TASK_QUEUE_ENABLE="${TASK_QUEUE_ENABLE:-1}"
export VLLM_ASCEND_ENABLE_FLASHCOMM1="${VLLM_ASCEND_ENABLE_FLASHCOMM1:-1}"
export VLLM_ASCEND_ENABLE_MLAPO="${VLLM_ASCEND_ENABLE_MLAPO:-1}"

# ------------------------------------------------------------------------------
# 并行配置 (Kimi-K2.6 MoE, 384 专家)
# ------------------------------------------------------------------------------
# Kimi-K2.6 基于 DeepSeek V3 架构，384 专家
# 推荐: TP=8 (填满单节点), PP=1 (单节点), EP 由 EP_SIZE 自动计算
# EP 需整除 384，单节点 EP=8 即每个 expert group 处理 48 专家
export TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-8}"
export PIPELINE_PARALLEL_SIZE="${PIPELINE_PARALLEL_SIZE:-1}"
# MoE 384 专家 → 必须启用专家并行
export ENABLE_EXPERT_PARALLEL="${ENABLE_EXPERT_PARALLEL:-1}"
export DATA_PARALLEL_SIZE="${DATA_PARALLEL_SIZE:-1}"

# ------------------------------------------------------------------------------
# 量化与内存配置 (W4A8)
# ------------------------------------------------------------------------------
export DTYPE="${DTYPE:-bfloat16}"
export QUANTIZATION="${QUANTIZATION:-ascend}"
export LOAD_FORMAT="${LOAD_FORMAT:-auto}"
# W4A8 量化降低显存，可适当提高利用率
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.92}"
export SWAP_SPACE="${SWAP_SPACE:-32}"

# ------------------------------------------------------------------------------
# 序列调度 (W4A8, 262K 原生上下文)
# ------------------------------------------------------------------------------
# A2 (8卡): 32k 安全值; A3 (16卡): 131k; 多节点: 更大
if [[ -z "${MAX_MODEL_LEN:-}" ]]; then
    if [[ "${TENSOR_PARALLEL_SIZE:-8}" -ge 16 ]]; then
        export MAX_MODEL_LEN=131072
    else
        export MAX_MODEL_LEN=32768
    fi
fi
if [[ -z "${MAX_NUM_SEQS:-}" ]]; then
    export MAX_NUM_SEQS=64
fi
export ENABLE_CHUNKED_PREFILL="${ENABLE_CHUNKED_PREFILL:-1}"
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-16384}"
export MAX_TOKENS_PER_SEQUENCE="${MAX_TOKENS_PER_SEQUENCE:-40000}"
export CHAT_TEMPLATE_CONTENT_FORMAT="${CHAT_TEMPLATE_CONTENT_FORMAT:-string}"

# ------------------------------------------------------------------------------
# 加速特性
# ------------------------------------------------------------------------------
# Prefix caching 对 Agent 场景效果显著 (Claude Code 系统提示缓存复用)
export PREFIX_CACHING="${PREFIX_CACHING:-1}"
export ENFORCE_EAGER="${ENFORCE_EAGER:-1}"
# 注意: --num-scheduler-steps 在 vLLM-Ascend 0.18.0rc1 不支持
# export NUM_SCHEDULER_STEPS="${NUM_SCHEDULER_STEPS:-8}"

# Kimi-K2.6 无 MTP (num_nextn_predict_layers=0)，不启用投机解码
# export SPECULATIVE_METHOD="deepseek_mtp"  # 不适用

# ------------------------------------------------------------------------------
# NPU 编译优化
# ------------------------------------------------------------------------------
export CUDAGRAPH_MODE="${CUDAGRAPH_MODE:-FULL_DECODE_ONLY}"
export ENABLE_NPUGRAPH_EX="${ENABLE_NPUGRAPH_EX:-true}"
export FUSE_MULS_ADD="${FUSE_MULS_ADD:-true}"
export MULTISTREAM_OVERLAP_SHARED_EXPERT="${MULTISTREAM_OVERLAP_SHARED_EXPERT:-true}"

# ------------------------------------------------------------------------------
# 异步调度 (W4A8 量化模型推荐)
# ------------------------------------------------------------------------------
# 注意: --async-scheduling 不支持 Ray backend (仅 mp/external_launcher)
# 注意: --num-scheduler-steps 在 vLLM-Ascend 0.18.0rc1 不支持
# 异步调度已禁用 (Ray 兼容性)
# export NUM_SCHEDULER_STEPS="${NUM_SCHEDULER_STEPS:-8}"
# export ENABLE_ASYNC_SCHEDULING="${ENABLE_ASYNC_SCHEDULING:-1}"  # Ray 不支持

# ------------------------------------------------------------------------------
# 工具调用 (Claude Code 集成)
# ------------------------------------------------------------------------------
export ENABLE_TOOL_CALLING="${ENABLE_TOOL_CALLING:-1}"
# Kimi-K2.6 基于 DeepSeek V3 架构，使用 kimi_k2 tool parser (适配 Kimi tokenizer)
export TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-kimi_k2}"

# ------------------------------------------------------------------------------
# 监控与日志
# ------------------------------------------------------------------------------
export ENABLE_METRICS="${ENABLE_METRICS:-1}"
export LOG_LEVEL="${LOG_LEVEL:-info}"
export MAX_RETRIES="${MAX_RETRIES:-3}"
export RETRY_DELAY="${RETRY_DELAY:-10}"

# ------------------------------------------------------------------------------
# 启动参数
# ------------------------------------------------------------------------------
EXTRA_ARGS=(
    --seed 1024
    --trust-remote-code
    --enable-prefix-caching
    --allowed-local-media-path /
    --mm-encoder-tp-mode data
)

# 编译配置 (NPU 专用)
if [[ "$QUANTIZATION" == "ascend" ]]; then
    EXTRA_ARGS+=(
        --additional-config "{\"fuse_muls_add\": $FUSE_MULS_ADD, \"multistream_overlap_shared_expert\": $MULTISTREAM_OVERLAP_SHARED_EXPERT, \"ascend_compilation_config\": {\"enable_npugraph_ex\": $ENABLE_NPUGRAPH_EX}}"
        --compilation-config "{\"cudagraph_mode\": \"$CUDAGRAPH_MODE\"}"
    )
fi

# ------------------------------------------------------------------------------
# 启动信息
# ------------------------------------------------------------------------------
echo "[INFO] Starting Kimi-K2.6 W4A8 server"
echo "[INFO] Model:     ${MODEL_PATH}"
echo "[INFO] Hardware:  TP=$TENSOR_PARALLEL_SIZE, PP=$PIPELINE_PARALLEL_SIZE, DP=$DATA_PARALLEL_SIZE"
echo "[INFO] Quant:     W4A8 (ascend), dtype=$DTYPE"
echo "[INFO] Memory:    max_len=$MAX_MODEL_LEN, max_seqs=$MAX_NUM_SEQS, gpu_util=$GPU_MEMORY_UTILIZATION"
echo "[INFO] Features:  MoE (384 experts), Vision (multimodal), Async Scheduling"
echo "[INFO] HCCL:      OP_EXPANSION_MODE=$HCCL_OP_EXPANSION_MODE, BUFFSIZE=${HCCL_BUFFSIZE}MB, TASK_QUEUE=$TASK_QUEUE_ENABLE"

exec bash "$VLLM_SCRIPT" "${EXTRA_ARGS[@]}" "$@"
