#!/bin/bash
# =============================================================================
# GLM-5.1 W4A8 部署示例 (华为 NPU 环境)
# =============================================================================
# 调用 vllm_model_server.sh 部署 GLM-5.1 W4A8 量化模型
# GLM-5.1 是 GLM-5 的升级版，采用相同的 MoE 架构 (256 专家) + MTP
#
# 与 GLM-5 的主要区别:
#   - 改进的训练数据和后训练流程
#   - 更强的推理和 agent 能力
#   - 相同的模型架构和硬件需求
#
# 硬件要求:
#   - Atlas 800 A2 (64G × 8):   单节点 W4A8 部署
#   - Atlas 800 A3 (64G × 16):  单节点 W4A8 部署 (支持更大上下文)
#
# 用法:
#   ./vllm_server.sh
#   TENSOR_PARALLEL_SIZE=16 MAX_MODEL_LEN=200000 ./vllm_server.sh
#   PIPELINE_PARALLEL_SIZE=8 ./vllm_server.sh
#
# 参考文档:
#   https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/GLM5.html
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
export MODEL_PATH="${MODEL_PATH:-/home/jianzhnie/llmtuner/hfhub/models/Eco-Tech/GLM-5.1-w4a8}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-glm-5.1}"
export HOST="${HOST:-0.0.0.0}"
export PORT="${PORT:-8002}"

# ------------------------------------------------------------------------------
# 华为 NPU 环境变量
# ------------------------------------------------------------------------------
export HCCL_OP_EXPANSION_MODE="${HCCL_OP_EXPANSION_MODE:-AIV}"
export OMP_PROC_BIND="${OMP_PROC_BIND:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export HCCL_BUFFSIZE="${HCCL_BUFFSIZE:-200}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"
export VLLM_ASCEND_BALANCE_SCHEDULING="${VLLM_ASCEND_BALANCE_SCHEDULING:-1}"
export VLLM_ASCEND_ENABLE_FLASHCOMM1="${VLLM_ASCEND_ENABLE_FLASHCOMM1:-1}"
export VLLM_ASCEND_ENABLE_MLAPO="${VLLM_ASCEND_ENABLE_MLAPO:-1}"

# ------------------------------------------------------------------------------
# 并行配置 (GLM-5.1 MoE, 256 专家)
# ------------------------------------------------------------------------------
export TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-8}"
export PIPELINE_PARALLEL_SIZE="${PIPELINE_PARALLEL_SIZE:-1}"
export ENABLE_EXPERT_PARALLEL="${ENABLE_EXPERT_PARALLEL:-1}"
export DATA_PARALLEL_SIZE="${DATA_PARALLEL_SIZE:-1}"

# ------------------------------------------------------------------------------
# 量化与内存配置 (W4A8)
# ------------------------------------------------------------------------------
export DTYPE="${DTYPE:-bfloat16}"
export QUANTIZATION="${QUANTIZATION:-ascend}"
export LOAD_FORMAT="${LOAD_FORMAT:-auto}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.95}"
export SWAP_SPACE="${SWAP_SPACE:-16}"

# ------------------------------------------------------------------------------
# 序列调度
# ------------------------------------------------------------------------------
if [[ -z "${MAX_MODEL_LEN:-}" ]]; then
    if [[ "${TENSOR_PARALLEL_SIZE:-8}" -ge 16 ]]; then
        export MAX_MODEL_LEN=200000
    else
        export MAX_MODEL_LEN=32768
    fi
fi
if [[ -z "${MAX_NUM_SEQS:-}" ]]; then
    if [[ "${TENSOR_PARALLEL_SIZE:-8}" -ge 16 ]]; then
        export MAX_NUM_SEQS=8
    else
        export MAX_NUM_SEQS=2
    fi
fi
export ENABLE_CHUNKED_PREFILL="${ENABLE_CHUNKED_PREFILL:-1}"
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"
export MAX_TOKENS_PER_SEQUENCE="${MAX_TOKENS_PER_SEQUENCE:-40000}"
export CHAT_TEMPLATE_CONTENT_FORMAT="${CHAT_TEMPLATE_CONTENT_FORMAT:-string}"

# ------------------------------------------------------------------------------
# 加速特性
# ------------------------------------------------------------------------------
export PREFIX_CACHING="${PREFIX_CACHING:-1}"
export ENFORCE_EAGER="${ENFORCE_EAGER:-1}"
export NUM_SCHEDULER_STEPS="${NUM_SCHEDULER_STEPS:-4}"

# ------------------------------------------------------------------------------
# 投机解码 (MTP)
# ------------------------------------------------------------------------------
export SPECULATIVE_METHOD="${SPECULATIVE_METHOD:-deepseek_mtp}"
export SPECULATIVE_NUM_TOKENS="${SPECULATIVE_NUM_TOKENS:-3}"

# ------------------------------------------------------------------------------
# NPU 编译优化
# ------------------------------------------------------------------------------
export CUDAGRAPH_MODE="${CUDAGRAPH_MODE:-FULL_DECODE_ONLY}"
export ENABLE_NPUGRAPH_EX="${ENABLE_NPUGRAPH_EX:-true}"
export FUSE_MULS_ADD="${FUSE_MULS_ADD:-true}"
export MULTISTREAM_OVERLAP_SHARED_EXPERT="${MULTISTREAM_OVERLAP_SHARED_EXPERT:-true}"

# ------------------------------------------------------------------------------
# 异步调度
# ------------------------------------------------------------------------------
export ENABLE_ASYNC_SCHEDULING="${ENABLE_ASYNC_SCHEDULING:-1}"

# ------------------------------------------------------------------------------
# 工具调用
# ------------------------------------------------------------------------------
export ENABLE_TOOL_CALLING="${ENABLE_TOOL_CALLING:-1}"
export TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-glm47}"

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

if [[ "$SPECULATIVE_METHOD" == "deepseek_mtp" ]]; then
    EXTRA_ARGS+=(
        --speculative-config "{\"num_speculative_tokens\": $SPECULATIVE_NUM_TOKENS, \"method\": \"$SPECULATIVE_METHOD\"}"
    )
fi

if [[ "$QUANTIZATION" == "ascend" ]]; then
    EXTRA_ARGS+=(
        --additional-config "{\"fuse_muls_add\": $FUSE_MULS_ADD, \"multistream_overlap_shared_expert\": $MULTISTREAM_OVERLAP_SHARED_EXPERT, \"ascend_compilation_config\": {\"enable_npugraph_ex\": $ENABLE_NPUGRAPH_EX}}"
        --compilation-config "{\"cudagraph_mode\": \"$CUDAGRAPH_MODE\"}"
    )
fi

# ------------------------------------------------------------------------------
# 启动信息
# ------------------------------------------------------------------------------
echo "[INFO] Starting GLM-5.1 W4A8 server"
echo "[INFO] Model:     ${MODEL_PATH}"
echo "[INFO] Hardware:  TP=$TENSOR_PARALLEL_SIZE, PP=$PIPELINE_PARALLEL_SIZE, DP=$DATA_PARALLEL_SIZE"
echo "[INFO] Quant:     W4A8 (ascend), dtype=$DTYPE"
echo "[INFO] Memory:    max_len=$MAX_MODEL_LEN, max_seqs=$MAX_NUM_SEQS, gpu_util=$GPU_MEMORY_UTILIZATION"
echo "[INFO] Features:  MoE (256 experts), MTP (tokens=$SPECULATIVE_NUM_TOKENS), Async Scheduling"
echo "[INFO] HCCL:      OP_EXPANSION_MODE=$HCCL_OP_EXPANSION_MODE, BUFFSIZE=${HCCL_BUFFSIZE}MB"

exec bash "$VLLM_SCRIPT" "${EXTRA_ARGS[@]}" "$@"
