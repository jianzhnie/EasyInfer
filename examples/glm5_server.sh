#!/bin/bash
# =============================================================================
# GLM-5/GLM-5.1 部署示例 (华为 NPU 环境)
# =============================================================================
# 调用 vllm_model_server.sh 部署 GLM-5/GLM-5.1 模型
# GLM-5 采用 MoE 架构，支持华为 NPU Ascend 量化加速
#
# 硬件要求:
#   - Atlas 800 A2 (64G × 8):   W4A8 单节点部署
#   - Atlas 800 A3 (64G × 16):  W4A8/W8A8 单节点部署
#   - 多节点:                   BF16/W4A8/W8A8 跨节点部署
#
# 量化选项:
#   - w4a8:  4-bit 权重 + 8-bit 激活 (推荐 A2/A3)
#   - w8a8:  8-bit 权重 + 8-bit 激活 (仅 A3)
#   - bf16:  原生 BF16 (需多节点)
#
# 特性:
#   - Ascend 量化 (W4A8/W8A8)
#   - 专家并行 (Expert Parallel)
#   - 投机解码 (Speculative Decoding with DeepSeek MTP)
#   - Chunked Prefill
#   - Prefix Caching
#   - Async Scheduling
#   - NPU Graph Optimization
#
# 用法:
#   # 默认 W4A8 量化 (A2 8卡)
#   ./glm5_server.sh
#
#   # W4A8 量化 (A3 16卡)
#   QUANT_TYPE=w4a8 TENSOR_PARALLEL_SIZE=16 MAX_MODEL_LEN=200000 MAX_NUM_SEQS=8 ./glm5_server.sh
#
#   # W8A8 量化 (A3 16卡)
#   QUANT_TYPE=w8a8 TENSOR_PARALLEL_SIZE=16 MAX_MODEL_LEN=40960 MAX_NUM_SEQS=8 ./glm5_server.sh
#
#   # BF16 (多节点 2×16卡)
#   QUANT_TYPE=bf16 TENSOR_PARALLEL_SIZE=16 PIPELINE_PARALLEL_SIZE=2 ./glm5_server.sh
#
# 参考文档:
#   https://docs.vllm.ai/projects/ascend/en/v0.18.0/tutorials/models/GLM5.html
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
# 量化类型选择 (w4a8 / w8a8 / bf16)
# ------------------------------------------------------------------------------
QUANT_TYPE="${QUANT_TYPE:-w4a8}"

# ------------------------------------------------------------------------------
# 根据量化类型配置模型路径和参数
# ------------------------------------------------------------------------------
case "$QUANT_TYPE" in
    w4a8)
        # W4A8 量化 (4-bit 权重 + 8-bit 激活)
        # A2: 8卡部署 (GLM-5-w4a8), A3: 16卡部署 (GLM5-w4a8)
        # 模型路径根据硬件自动判断
        if [[ "${TENSOR_PARALLEL_SIZE:-8}" -ge 16 ]]; then
            # A3 (16卡): 使用 GLM5-w4a8 (无连字符)
            export MODEL_PATH="${MODEL_PATH:-/root/.cache/modelscope/hub/models/vllm-ascend/GLM5-w4a8}"
        else
            # A2 (8卡): 使用 GLM-5-w4a8 (有连字符)
            export MODEL_PATH="${MODEL_PATH:-/root/.cache/modelscope/hub/models/vllm-ascend/GLM-5-w4a8}"
        fi
        export QUANTIZATION="ascend"
        # W4A8 不需要 MLAPO
        unset VLLM_ASCEND_ENABLE_MLAPO
        ;;
    w8a8)
        # W8A8 量化 (8-bit 权重 + 8-bit 激活)
        # 仅支持 A3 (16卡)
        export MODEL_PATH="${MODEL_PATH:-/root/.cache/modelscope/hub/models/vllm-ascend/GLM5-w8a8}"
        export QUANTIZATION="ascend"
        # W8A8 需要启用 MLAPO (Model Layer Parallel Optimization)
        export VLLM_ASCEND_ENABLE_MLAPO="${VLLM_ASCEND_ENABLE_MLAPO:-1}"
        ;;
    bf16)
        # BF16 原生精度 (无量化)
        # 需要多节点部署 (至少 2×16卡)
        export MODEL_PATH="${MODEL_PATH:-/root/.cache/modelscope/hub/models/vllm-ascend/GLM5-bf16}"
        export QUANTIZATION="none"
        unset VLLM_ASCEND_ENABLE_MLAPO
        ;;
    *)
        echo "[ERROR] Invalid QUANT_TYPE: $QUANT_TYPE (expected: w4a8, w8a8, bf16)" >&2
        exit 1
        ;;
esac

export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-glm-5}"
export HOST="${HOST:-0.0.0.0}"
export PORT="${PORT:-8077}"

# ------------------------------------------------------------------------------
# 华为 NPU 环境变量 (针对 GLM-5 优化)
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
# 并行配置 (GLM-5 MoE 架构)
# ------------------------------------------------------------------------------
# 默认配置:
#   - A2 (8卡):  TP=8, PP=1
#   - A3 (16卡): TP=16, PP=1
#   - 多节点:    根据节点数调整 PP
export TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-8}"
export PIPELINE_PARALLEL_SIZE="${PIPELINE_PARALLEL_SIZE:-1}"
# 专家并行 (MoE 模型必需)
export ENABLE_EXPERT_PARALLEL="${ENABLE_EXPERT_PARALLEL:-1}"
# 数据并行 (单节点默认为 1)
export DATA_PARALLEL_SIZE="${DATA_PARALLEL_SIZE:-1}"

# ------------------------------------------------------------------------------
# 内存与量化配置
# ------------------------------------------------------------------------------
export DTYPE="${DTYPE:-bfloat16}"
export LOAD_FORMAT="${LOAD_FORMAT:-auto}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.95}"
export SWAP_SPACE="${SWAP_SPACE:-16}"

# ------------------------------------------------------------------------------
# 序列调度 (根据量化类型和硬件调整)
# ------------------------------------------------------------------------------
# 默认值根据量化类型自动设置
if [[ -z "${MAX_MODEL_LEN:-}" ]]; then
    case "$QUANT_TYPE" in
        w4a8)
            # A2: 32k, A3: 200k (根据 TP 自动判断)
            if [[ "$TENSOR_PARALLEL_SIZE" -ge 16 ]]; then
                export MAX_MODEL_LEN=200000
            else
                export MAX_MODEL_LEN=32768
            fi
            ;;
        w8a8)
            # W8A8 仅支持 A3, 推荐 40k
            export MAX_MODEL_LEN=40960
            ;;
        bf16)
            # BF16 多节点, 推荐 8k
            export MAX_MODEL_LEN=8192
            ;;
    esac
fi

# 默认 max_num_seqs 根据量化类型调整
if [[ -z "${MAX_NUM_SEQS:-}" ]]; then
    case "$QUANT_TYPE" in
        w4a8)
            # A2: 2, A3: 8
            if [[ "$TENSOR_PARALLEL_SIZE" -ge 16 ]]; then
                export MAX_NUM_SEQS=8
            else
                export MAX_NUM_SEQS=2
            fi
            ;;
        w8a8)
            export MAX_NUM_SEQS=8
            ;;
        bf16)
            export MAX_NUM_SEQS=16
            ;;
    esac
fi

export ENABLE_CHUNKED_PREFILL="${ENABLE_CHUNKED_PREFILL:-1}"
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"
export MAX_TOKENS_PER_SEQUENCE="${MAX_TOKENS_PER_SEQUENCE:-40000}"
export CHAT_TEMPLATE_CONTENT_FORMAT="${CHAT_TEMPLATE_CONTENT_FORMAT:-string}"

# ------------------------------------------------------------------------------
# 加速特性
# ------------------------------------------------------------------------------
export PREFIX_CACHING="${PREFIX_CACHING:-1}"
# NPU 环境 CUDA Graph 配置 (非 Eager 模式)
export ENFORCE_EAGER="${ENFORCE_EAGER:-0}"
export NUM_SCHEDULER_STEPS="${NUM_SCHEDULER_STEPS:-4}"

# ------------------------------------------------------------------------------
# 投机解码 (Speculative Decoding)
# ------------------------------------------------------------------------------
# GLM-5 支持 DeepSeek MTP 投机解码
export SPECULATIVE_NUM_TOKENS="${SPECULATIVE_NUM_TOKENS:-3}"
export SPECULATIVE_METHOD="${SPECULATIVE_METHOD:-deepseek_mtp}"

# ------------------------------------------------------------------------------
# NPU 编译优化配置
# ------------------------------------------------------------------------------
export CUDAGRAPH_MODE="${CUDAGRAPH_MODE:-FULL_DECODE_ONLY}"
export ENABLE_NPUGRAPH_EX="${ENABLE_NPUGRAPH_EX:-true}"
export FUSE_MULS_ADD="${FUSE_MULS_ADD:-true}"
export MULTISTREAM_OVERLAP_SHARED_EXPERT="${MULTISTREAM_OVERLAP_SHARED_EXPERT:-true}"

# ------------------------------------------------------------------------------
# 异步调度 (Async Scheduling)
# ------------------------------------------------------------------------------
# 优化大规模模型推理效率，提高并发和吞吐
export ENABLE_ASYNC_SCHEDULING="${ENABLE_ASYNC_SCHEDULING:-1}"

# ------------------------------------------------------------------------------
# 工具调用 (Claude Code 集成)
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
# 启动参数 (传递给 vLLM)
# ------------------------------------------------------------------------------
EXTRA_ARGS=(
    --seed 1024
    --trust-remote-code
)

# 数据并行参数 (多节点部署)
if [[ "$DATA_PARALLEL_SIZE" -gt 1 ]]; then
    # 多节点部署需要额外参数 (用户需手动配置)
    echo "[INFO] Multi-node deployment detected (DP=$DATA_PARALLEL_SIZE)"
    echo "[INFO] Please set the following environment variables on each node:"
    echo "  - HCCL_IF_IP, GLOO_SOCKET_IFNAME, TP_SOCKET_IFNAME, HCCL_SOCKET_IFNAME"
    echo "  - DATA_PARALLEL_SIZE_LOCAL, DATA_PARALLEL_ADDRESS, DATA_PARALLEL_RPC_PORT"
    echo "[INFO] Refer to official docs for multi-node configuration details"
fi

# 异步调度 (仅量化模型)
if [[ "$QUANTIZATION" == "ascend" ]] && [[ "$ENABLE_ASYNC_SCHEDULING" == "1" ]]; then
    EXTRA_ARGS+=(--async-scheduling)
fi

# 投机解码配置
if [[ "$SPECULATIVE_METHOD" == "deepseek_mtp" ]]; then
    EXTRA_ARGS+=(
        --speculative-config "{\"num_speculative_tokens\": $SPECULATIVE_NUM_TOKENS, \"method\": \"$SPECULATIVE_METHOD\"}"
    )
fi

# 编译配置 (NPU 专用，严格按照文档顺序: additional-config -> compilation-config)
if [[ "$QUANTIZATION" == "ascend" ]]; then
    # 注意: 文档中 additional-config 在 compilation-config 之前
    EXTRA_ARGS+=(
        --additional-config "{\"fuse_muls_add\": $FUSE_MULS_ADD, \"multistream_overlap_shared_expert\": $MULTISTREAM_OVERLAP_SHARED_EXPERT, \"ascend_compilation_config\": {\"enable_npugraph_ex\": $ENABLE_NPUGRAPH_EX}}"
        --compilation-config "{\"cudagraph_mode\": \"$CUDAGRAPH_MODE\"}"
    )
fi

# ------------------------------------------------------------------------------
# 启动信息
# ------------------------------------------------------------------------------
echo "[INFO] Starting GLM-5 server"
echo "[INFO] Hardware: TP=$TENSOR_PARALLEL_SIZE, PP=$PIPELINE_PARALLEL_SIZE, DP=$DATA_PARALLEL_SIZE"
echo "[INFO] Quantization: $QUANT_TYPE ($QUANTIZATION)"
echo "[INFO] Memory: max_len=$MAX_MODEL_LEN, max_seqs=$MAX_NUM_SEQS, gpu_util=$GPU_MEMORY_UTILIZATION"
echo "[INFO] Acceleration: Expert Parallel, Speculative Decoding ($SPECULATIVE_METHOD, tokens=$SPECULATIVE_NUM_TOKENS)"
if [[ "$QUANTIZATION" == "ascend" ]]; then
    echo "[INFO] NPU Optimizations: Async Scheduling, NPU Graph, Multi-stream Overlap"
fi
echo "[INFO] HCCL Config: OP_EXPANSION_MODE=$HCCL_OP_EXPANSION_MODE, BUFFSIZE=${HCCL_BUFFSIZE}MB"
if [[ -n "${VLLM_ASCEND_ENABLE_MLAPO:-}" ]]; then
    echo "[INFO] MLAPO Enabled: VLLM_ASCEND_ENABLE_MLAPO=$VLLM_ASCEND_ENABLE_MLAPO"
fi

# ------------------------------------------------------------------------------
# 启动
# ------------------------------------------------------------------------------
exec bash "$VLLM_SCRIPT" "${EXTRA_ARGS[@]}" "$@"