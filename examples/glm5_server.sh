#!/usr/bin/env bash
# =============================================================================
# GLM-5.1 部署示例 (华为 NPU 环境)
# =============================================================================
# 调用 vllm_model_server.sh 部署 GLM-5.1 模型
# GLM-5.1 采用 MoE 架构，支持华为 NPU Ascend 量化加速
#
# 硬件要求:
#   - 8x NPU (单节点部署)
#   - 16x NPU (2节点部署)
#   - 32x NPU (4节点部署)
#
# 特性:
#   - Ascend W4A8 量化 (4-bit 权重, 8-bit 激活)
#   - 专家并行 (Expert Parallel)
#   - 投机解码 (Speculative Decoding with MTP)
#   - Chunked Prefill
#   - Prefix Caching
#
# 用法:
#   ./glm5_server.sh                                # 默认配置启动
#   PORT=8077 ./glm5_server.sh                      # 覆盖端口
#   MAX_MODEL_LEN=65536 ./glm5_server.sh            # 扩大上下文窗口
#   TENSOR_PARALLEL_SIZE=4 PIPELINE_PARALLEL_SIZE=2 ./glm5_server.sh  # 多节点部署
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLLM_SCRIPT="${SCRIPT_DIR}/../scripts/vllm/vllm_model_server.sh"

# 检查启动脚本是否存在
if [[ ! -f "$VLLM_SCRIPT" ]]; then
    echo "[ERROR] vLLM startup script not found: $VLLM_SCRIPT" >&2
    exit 1
fi

# ------------------------------------------------------------------------------
# GLM-5.1 模型配置
# ------------------------------------------------------------------------------
export MODEL_PATH="${MODEL_PATH:-/root/.cache/modelscope/hub/models/vllm-ascend/GLM-5-w4a8}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-glm-5}"
export HOST="${HOST:-0.0.0.0}"
export PORT="${PORT:-8077}"

# ------------------------------------------------------------------------------
# 华为 NPU 环境变量 (针对 GLM-5.1 优化)
# ------------------------------------------------------------------------------
# HCCL 操作扩展模式 (华为集合通信库优化)
export HCCL_OP_EXPANSION_MODE="${HCCL_OP_EXPANSION_MODE:-AIV}"
# OpenMP 线程绑定 (禁用避免干扰 NPU 调度)
export OMP_PROC_BIND="${OMP_PROC_BIND:-false}"
# OpenMP 线程数 (减少 CPU 线程数，降低调度开销)
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
# HCCL 缓冲区大小 (MB)
export HCCL_BUFFSIZE="${HCCL_BUFFSIZE:-200}"
# PyTorch NPU 内存分配配置
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"
# vLLM Ascend 负载均衡调度
export VLLM_ASCEND_BALANCE_SCHEDULING="${VLLM_ASCEND_BALANCE_SCHEDULING:-1}"

# ------------------------------------------------------------------------------
# 并行配置 (GLM-5.1 MoE 架构)
# ------------------------------------------------------------------------------
# 单节点 8 卡部署 (默认)
export TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-8}"
# 流水线并行 (跨节点部署时使用)
export PIPELINE_PARALLEL_SIZE="${PIPELINE_PARALLEL_SIZE:-1}"
# 专家并行 (MoE 模型必需)
export ENABLE_EXPERT_PARALLEL="${ENABLE_EXPERT_PARALLEL:-1}"

# ------------------------------------------------------------------------------
# 内存与量化配置
# ------------------------------------------------------------------------------
# Ascend W4A8 量化 (4-bit 权重, 8-bit 激活)
export DTYPE="${DTYPE:-bfloat16}"
export QUANTIZATION="${QUANTIZATION:-ascend}"
export LOAD_FORMAT="${LOAD_FORMAT:-auto}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.95}"
# NPU 环境下 swap 需求较小
export SWAP_SPACE="${SWAP_SPACE:-16}"

# ------------------------------------------------------------------------------
# 序列调度
# ------------------------------------------------------------------------------
# GLM-5.1 支持长上下文，根据显存调整
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
# Ascend 量化后吞吐量较高，可适当降低并发
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-2}"
export ENABLE_CHUNKED_PREFILL="${ENABLE_CHUNKED_PREFILL:-1}"
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"

# ------------------------------------------------------------------------------
# 加速特性
# ------------------------------------------------------------------------------
# 前缀缓存 (多轮对话场景推荐启用)
export PREFIX_CACHING="${PREFIX_CACHING:-1}"
# NPU 环境强制 Eager 模式
export ENFORCE_EAGER="${ENFORCE_EAGER:-0}"
export NUM_SCHEDULER_STEPS="${NUM_SCHEDULER_STEPS:-4}"

# ------------------------------------------------------------------------------
# 投机解码 (Speculative Decoding)
# ------------------------------------------------------------------------------
# GLM-5.1 支持 DeepSeek MTP 投机解码
# 推测 token 数量: 建议 3-5，越大加速效果越好但风险越高
export SPECULATIVE_NUM_TOKENS="${SPECULATIVE_NUM_TOKENS:-3}"
export SPECULATIVE_METHOD="${SPECULATIVE_METHOD:-deepseek_mtp}"

# ------------------------------------------------------------------------------
# NPU 编译优化配置
# ------------------------------------------------------------------------------
# CUDA Graph 模式 (NPU 环境推荐 FULL_DECODE_ONLY)
export CUDAGRAPH_MODE="${CUDAGRAPH_MODE:-FULL_DECODE_ONLY}"
# NPU 图编译优化
export ENABLE_NPUGRAPH_EX="${ENABLE_NPUGRAPH_EX:-true}"
# 算子融合优化
export FUSE_MULS_ADD="${FUSE_MULS_ADD:-true}"
# 多流重叠共享专家
export MULTISTREAM_OVERLAP_SHARED_EXPERT="${MULTISTREAM_OVERLAP_SHARED_EXPERT:-true}"

# ------------------------------------------------------------------------------
# 工具调用 (Claude Code 集成)
# ------------------------------------------------------------------------------
export ENABLE_TOOL_CALLING="${ENABLE_TOOL_CALLING:-1}"
export TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-hermes}"

# ------------------------------------------------------------------------------
# 监控与日志
# ------------------------------------------------------------------------------
export ENABLE_METRICS="${ENABLE_METRICS:-1}"
export LOG_LEVEL="${LOG_LEVEL:-info}"
export MAX_RETRIES="${MAX_RETRIES:-3}"
export RETRY_DELAY="${RETRY_DELAY:-10}"

# ------------------------------------------------------------------------------
# 启动参数 (传递给 vLLM)
# ------------------------------------------------------------------------------
# 构建额外参数
EXTRA_ARGS=(
    --seed 1024
    --trust-remote-code
)

# 投机解码配置
if [[ "$SPECULATIVE_METHOD" == "deepseek_mtp" ]]; then
    EXTRA_ARGS+=(
        --speculative-config "{\"num_speculative_tokens\": $SPECULATIVE_NUM_TOKENS, \"method\": \"$SPECULATIVE_METHOD\"}"
    )
fi

# 编译配置 (NPU 专用)
if [[ "$QUANTIZATION" == "ascend" ]]; then
    EXTRA_ARGS+=(
        --compilation-config "{\"cudagraph_mode\": \"$CUDAGRAPH_MODE\"}"
        --additional-config "{\"fuse_muls_add\": $FUSE_MULS_ADD, \"multistream_overlap_shared_expert\": $MULTISTREAM_OVERLAP_SHARED_EXPERT, \"ascend_compilation_config\": {\"enable_npugraph_ex\": $ENABLE_NPUGRAPH_EX}}"
    )
fi

# ------------------------------------------------------------------------------
# 启动
# ------------------------------------------------------------------------------
echo "[INFO] Starting GLM-5.1 server (TP=$TENSOR_PARALLEL_SIZE, PP=$PIPELINE_PARALLEL_SIZE, quant=$QUANTIZATION, max_len=$MAX_MODEL_LEN)"
echo "[INFO] NPU optimizations: Ascend W4A8, Expert Parallel, Speculative Decoding ($SPECULATIVE_METHOD, tokens=$SPECULATIVE_NUM_TOKENS)"
echo "[INFO] HCCL config: OP_EXPANSION_MODE=$HCCL_OP_EXPANSION_MODE, BUFFSIZE=${HCCL_BUFFSIZE}MB"
exec bash "$VLLM_SCRIPT" "${EXTRA_ARGS[@]}" "$@"