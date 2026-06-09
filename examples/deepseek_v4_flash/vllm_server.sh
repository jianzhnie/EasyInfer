#!/bin/bash
# =============================================================================
# DeepSeek-V4-Flash W8A8 MTP 部署示例 (华为 NPU 环境)
# =============================================================================
# ⚠️ 兼容性警告: vLLM-Ascend 0.18.0rc1 不支持 DeepseekV4ForCausalLM 架构。
# 需要升级到支持 DeepSeek V4 的版本。参见 run_vllm.sh 了解直接部署方式。
# 调用 vllm_model_server.sh 部署 DeepSeek-V4-Flash 模型
# DeepSeek V4 采用 MoE 架构 + Multi-Token Prediction (MTP)，支持超长上下文
#
# 硬件要求:
#   - 单节点: Atlas 800 A3 (64G × 16) 推荐 (W8A8 量化)
#   - 多节点: 8 节点 × 8 NPU (64 卡) 支持 PP/EP 扩展
#
# 关键特性:
#   - 256 路由专家 + 1 共享专家 (MoE)
#   - Multi-Token Prediction (MTP, 投机解码)
#   - W8A8 Ascend 量化
#   - 1M 原生上下文窗口
#   - DeepSeek V4 Flash 架构 (MLA + DeepSeek MoE)
#
# 用法:
#   # 单节点默认 W8A8
#   ./vllm_server.sh
#
#   # 多节点 8×8 卡
#   PIPELINE_PARALLEL_SIZE=8 ./vllm_server.sh
#
#   # 自定义上下文长度
#   MAX_MODEL_LEN=131072 ./vllm_server.sh
#
# 参考文档:
#   https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/index.html
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLLM_SCRIPT="${SCRIPT_DIR}/../../scripts/vllm/vllm_model_server.sh"

# 检查启动脚本是否存在
if [[ ! -f "$VLLM_SCRIPT" ]]; then
    echo "[ERROR] vLLM startup script not found: $VLLM_SCRIPT" >&2
    exit 1
fi

# ------------------------------------------------------------------------------
# 模型路径与基础配置
# ------------------------------------------------------------------------------
export MODEL_PATH="${MODEL_PATH:-/home/jianzhnie/llmtuner/hfhub/models/Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-deepseek-v4-flash}"
export HOST="${HOST:-0.0.0.0}"
export PORT="${PORT:-8000}"

# ------------------------------------------------------------------------------
# 华为 NPU 环境变量
# ------------------------------------------------------------------------------
export HCCL_OP_EXPANSION_MODE="${HCCL_OP_EXPANSION_MODE:-AIV}"
export OMP_PROC_BIND="${OMP_PROC_BIND:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export HCCL_BUFFSIZE="${HCCL_BUFFSIZE:-200}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"
export VLLM_ASCEND_BALANCE_SCHEDULING="${VLLM_ASCEND_BALANCE_SCHEDULING:-1}"

# ------------------------------------------------------------------------------
# 并行配置 (DeepSeek V4 Flash MoE + MTP)
# ------------------------------------------------------------------------------
# 默认单节点 TP=8 (8 卡填满)
# 多节点: PIPELINE_PARALLEL_SIZE=N (N 为节点数)
export TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-8}"
export PIPELINE_PARALLEL_SIZE="${PIPELINE_PARALLEL_SIZE:-1}"
# 专家并行 (MoE 模型必需，256 专家)
export ENABLE_EXPERT_PARALLEL="${ENABLE_EXPERT_PARALLEL:-1}"
# 数据并行
export DATA_PARALLEL_SIZE="${DATA_PARALLEL_SIZE:-1}"

# ------------------------------------------------------------------------------
# 量化与内存配置 (W8A8)
# ------------------------------------------------------------------------------
export DTYPE="${DTYPE:-bfloat16}"
# W8A8 Ascend 量化
export QUANTIZATION="${QUANTIZATION:-ascend}"
export LOAD_FORMAT="${LOAD_FORMAT:-auto}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
export SWAP_SPACE="${SWAP_SPACE:-32}"

# ------------------------------------------------------------------------------
# 序列调度 (W8A8 量化，兼顾内存与上下文)
# ------------------------------------------------------------------------------
# DeepSeek V4 Flash 原生支持 1M 上下文；W8A8 量化下可根据显存调整
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
export ENABLE_CHUNKED_PREFILL="${ENABLE_CHUNKED_PREFILL:-1}"
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
export MAX_TOKENS_PER_SEQUENCE="${MAX_TOKENS_PER_SEQUENCE:-65536}"
export CHAT_TEMPLATE_CONTENT_FORMAT="${CHAT_TEMPLATE_CONTENT_FORMAT:-string}"

# ------------------------------------------------------------------------------
# 加速特性
# ------------------------------------------------------------------------------
export PREFIX_CACHING="${PREFIX_CACHING:-1}"
export ENFORCE_EAGER="${ENFORCE_EAGER:-1}"
export NUM_SCHEDULER_STEPS="${NUM_SCHEDULER_STEPS:-8}"

# ------------------------------------------------------------------------------
# 投机解码 (MTP - Multi-Token Prediction)
# ------------------------------------------------------------------------------
# DeepSeek V4 Flash 支持 MTP (num_nextn_predict_layers=1)
export SPECULATIVE_METHOD="${SPECULATIVE_METHOD:-deepseek_mtp}"
export SPECULATIVE_NUM_TOKENS="${SPECULATIVE_NUM_TOKENS:-3}"

# ------------------------------------------------------------------------------
# NPU 编译优化 (W8A8 量化)
# ------------------------------------------------------------------------------
export CUDAGRAPH_MODE="${CUDAGRAPH_MODE:-FULL_DECODE_ONLY}"
export ENABLE_NPUGRAPH_EX="${ENABLE_NPUGRAPH_EX:-true}"
export FUSE_MULS_ADD="${FUSE_MULS_ADD:-true}"
export MULTISTREAM_OVERLAP_SHARED_EXPERT="${MULTISTREAM_OVERLAP_SHARED_EXPERT:-true}"

# ------------------------------------------------------------------------------
# 异步调度 (量化模型推荐)
# ------------------------------------------------------------------------------
export ENABLE_ASYNC_SCHEDULING="${ENABLE_ASYNC_SCHEDULING:-1}"

# ------------------------------------------------------------------------------
# 工具调用 (Claude Code 集成)
# ------------------------------------------------------------------------------
export ENABLE_TOOL_CALLING="${ENABLE_TOOL_CALLING:-1}"
export TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-deepseek_v3}"

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
)

# 投机解码配置 (MTP)
if [[ "$SPECULATIVE_METHOD" == "deepseek_mtp" ]]; then
    EXTRA_ARGS+=(
        --speculative-config "{\"num_speculative_tokens\": $SPECULATIVE_NUM_TOKENS, \"method\": \"$SPECULATIVE_METHOD\"}"
    )
fi

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
echo "[INFO] Starting DeepSeek-V4-Flash W8A8 MTP server"
echo "[INFO] Model:     ${MODEL_PATH}"
echo "[INFO] Hardware:  TP=$TENSOR_PARALLEL_SIZE, PP=$PIPELINE_PARALLEL_SIZE, DP=$DATA_PARALLEL_SIZE"
echo "[INFO] Quant:     W8A8 (ascend), dtype=$DTYPE"
echo "[INFO] Memory:    max_len=$MAX_MODEL_LEN, max_seqs=$MAX_NUM_SEQS, gpu_util=$GPU_MEMORY_UTILIZATION"
echo "[INFO] Features:  MoE (256 experts), MTP (spec=$SPECULATIVE_METHOD, tokens=$SPECULATIVE_NUM_TOKENS)"
echo "[INFO] HCCL:      OP_EXPANSION_MODE=$HCCL_OP_EXPANSION_MODE, BUFFSIZE=${HCCL_BUFFSIZE}MB"

# ------------------------------------------------------------------------------
# 启动
# ------------------------------------------------------------------------------
exec bash "$VLLM_SCRIPT" "${EXTRA_ARGS[@]}" "$@"
