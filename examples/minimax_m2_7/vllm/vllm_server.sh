#!/bin/bash
# =============================================================================
# MiniMax-M2.7 W8A8 QuaRot MTP 部署示例 (华为 NPU 环境)
# =============================================================================
# 调用 vllm_model_server.sh 部署 MiniMax-M2.7 W8A8 QuaRot 量化模型
# MiniMax-M2.7 基于 MiniMaxM2ForCausalLM 架构，256 专家 MoE，支持 MTP
#
# 硬件要求:
#   - Atlas 800 A2 (64G × 8):   单节点 W8A8 部署 (TP=4 官方推荐)
#   - Atlas 800 A3 (64G × 16):  单节点 W8A8 部署
#
# 关键特性:
#   - 256 路由专家 (MoE)
#   - W8A8 QuaRot Ascend 量化
#   - 204K 原生上下文窗口
#   - 官方推荐 TP=4 (A2 环境)
#
# 用法:
#   # 默认 W8A8 单节点 (A2, TP=4)
#   ./vllm_server.sh
#
#   # A3 16 卡部署
#   TENSOR_PARALLEL_SIZE=8 MAX_MODEL_LEN=65536 ./vllm_server.sh
#
#   # 多节点部署
#   TENSOR_PARALLEL_SIZE=8 PIPELINE_PARALLEL_SIZE=2 ./vllm_server.sh
#
# 参考文档:
#   https://docs.vllm.ai/projects/ascend/zh-cn/releases-v0.20.2rc/tutorials/models/MiniMax-M2.5.html
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
export MODEL_PATH="${MODEL_PATH:-/home/jianzhnie/llmtuner/hfhub/models/Eco-Tech/MiniMax-M2.7-w8a8-QuaRot}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-minimax-m2.7}"
export HOST="${HOST:-0.0.0.0}"
export PORT="${PORT:-8004}"

# ------------------------------------------------------------------------------
# 华为 NPU 环境变量
# ------------------------------------------------------------------------------
export HCCL_OP_EXPANSION_MODE="${HCCL_OP_EXPANSION_MODE:-AIV}"
export OMP_PROC_BIND="${OMP_PROC_BIND:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export HCCL_BUFFSIZE="${HCCL_BUFFSIZE:-1024}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"
export TASK_QUEUE_ENABLE="${TASK_QUEUE_ENABLE:-1}"
export VLLM_ASCEND_ENABLE_FUSED_MC2="${VLLM_ASCEND_ENABLE_FUSED_MC2:-1}"
export VLLM_ASCEND_ENABLE_FLASHCOMM1="${VLLM_ASCEND_ENABLE_FLASHCOMM1:-1}"
export VLLM_ASCEND_BALANCE_SCHEDULING="${VLLM_ASCEND_BALANCE_SCHEDULING:-1}"

# ------------------------------------------------------------------------------
# 并行配置 (MiniMax-M2.7 MoE, 256 专家)
# ------------------------------------------------------------------------------
# 官方推荐 A2 环境 TP=4 (W8A8 量化)
export TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-4}"
export PIPELINE_PARALLEL_SIZE="${PIPELINE_PARALLEL_SIZE:-1}"
export ENABLE_EXPERT_PARALLEL="${ENABLE_EXPERT_PARALLEL:-1}"
export DATA_PARALLEL_SIZE="${DATA_PARALLEL_SIZE:-1}"

# ------------------------------------------------------------------------------
# 量化与内存配置 (W8A8 QuaRot)
# ------------------------------------------------------------------------------
export DTYPE="${DTYPE:-bfloat16}"
export QUANTIZATION="${QUANTIZATION:-ascend}"
export LOAD_FORMAT="${LOAD_FORMAT:-auto}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
export SWAP_SPACE="${SWAP_SPACE:-32}"

# ------------------------------------------------------------------------------
# 序列调度 (W8A8, 204K 原生上下文)
# ------------------------------------------------------------------------------
if [[ -z "${MAX_MODEL_LEN:-}" ]]; then
    if [[ "${TENSOR_PARALLEL_SIZE:-4}" -ge 8 ]]; then
        export MAX_MODEL_LEN=65536
    else
        export MAX_MODEL_LEN=32768
    fi
fi
if [[ -z "${MAX_NUM_SEQS:-}" ]]; then
    export MAX_NUM_SEQS=16
fi
export ENABLE_CHUNKED_PREFILL="${ENABLE_CHUNKED_PREFILL:-1}"
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
export MAX_TOKENS_PER_SEQUENCE="${MAX_TOKENS_PER_SEQUENCE:-32768}"
export CHAT_TEMPLATE_CONTENT_FORMAT="${CHAT_TEMPLATE_CONTENT_FORMAT:-string}"

# ------------------------------------------------------------------------------
# 加速特性
# ------------------------------------------------------------------------------
export PREFIX_CACHING="${PREFIX_CACHING:-1}"
export ENFORCE_EAGER="${ENFORCE_EAGER:-1}"

# ------------------------------------------------------------------------------
# 投机解码 (MTP) — MiniMax-M2.7 不支持 vLLM-Ascend 0.20.2 的 mtp 方法
# ------------------------------------------------------------------------------
# export SPECULATIVE_METHOD="${SPECULATIVE_METHOD:-mtp}"
# export SPECULATIVE_NUM_TOKENS="${SPECULATIVE_NUM_TOKENS:-3}"

# ------------------------------------------------------------------------------
# NPU 编译优化
# ------------------------------------------------------------------------------
export CUDAGRAPH_MODE="${CUDAGRAPH_MODE:-FULL_DECODE_ONLY}"
export ENABLE_NPUGRAPH_EX="${ENABLE_NPUGRAPH_EX:-true}"
export FUSE_MULS_ADD="${FUSE_MULS_ADD:-true}"
export MULTISTREAM_OVERLAP_SHARED_EXPERT="${MULTISTREAM_OVERLAP_SHARED_EXPERT:-true}"

# ------------------------------------------------------------------------------
# 工具调用
# ------------------------------------------------------------------------------
export ENABLE_TOOL_CALLING="${ENABLE_TOOL_CALLING:-1}"
export TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-minimax_m2}"

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
echo "[INFO] Starting MiniMax-M2.7 W8A8 QuaRot MTP server"
echo "[INFO] Model:     ${MODEL_PATH}"
echo "[INFO] Hardware:  TP=$TENSOR_PARALLEL_SIZE, PP=$PIPELINE_PARALLEL_SIZE, DP=$DATA_PARALLEL_SIZE"
echo "[INFO] Quant:     W8A8 QuaRot (ascend), dtype=$DTYPE"
echo "[INFO] Memory:    max_len=$MAX_MODEL_LEN, max_seqs=$MAX_NUM_SEQS, gpu_util=$GPU_MEMORY_UTILIZATION"
echo "[INFO] Features:  MoE (256 experts), MTP (tokens=$SPECULATIVE_NUM_TOKENS, method=$SPECULATIVE_METHOD)"
echo "[INFO] HCCL:      OP_EXPANSION_MODE=$HCCL_OP_EXPANSION_MODE, BUFFSIZE=${HCCL_BUFFSIZE}MB, TASK_QUEUE=$TASK_QUEUE_ENABLE"
echo "[INFO] Env:       FUSED_MC2=$VLLM_ASCEND_ENABLE_FUSED_MC2, FLASHCOMM1=$VLLM_ASCEND_ENABLE_FLASHCOMM1"

exec bash "$VLLM_SCRIPT" "${EXTRA_ARGS[@]}" "$@"
