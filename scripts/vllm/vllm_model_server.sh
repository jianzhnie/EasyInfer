#!/usr/bin/env bash

# =============================================================================
# vLLM Model Server Startup Script (MoE Models on NPU Cluster)
# =============================================================================
# 所有变量均支持通过环境变量或 set_env.sh 外部覆盖。
# 变量详细说明参见 vllm_server_env_template.sh —— 本脚本只保留操作摘要。
#
# 用法:
#   1. 默认启动: ./vllm_model_server.sh
#   2. 环境变量覆盖: MODEL_PATH=/path/to/model ./vllm_model_server.sh
#   3. 外部配置: SET_ENV_FILE=/path/to/env.sh ./vllm_model_server.sh
#   4. 命令行参数: ./vllm_model_server.sh --port 8080 --tensor-parallel-size 4
# =============================================================================

set -euo pipefail

# -----------------------------------------------------------------------------
# 配置加载
# -----------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 首先加载 set_env.sh
SET_ENV_FILE="${SCRIPT_DIR}/set_env.sh"
if [[ -f "$SET_ENV_FILE" ]]; then
    source "$SET_ENV_FILE" 2>/dev/null || echo "[WARN] Failed to source ${SET_ENV_FILE}, continuing..." >&2
fi

# 然后加载 vllm_server_env.sh（用户自定义覆盖）
VLLM_ENV_FILE="${VLLM_ENV_FILE:-${SCRIPT_DIR}/vllm_server_env.sh}"
if [[ -f "$VLLM_ENV_FILE" ]]; then
    source "$VLLM_ENV_FILE" 2>/dev/null || echo "[WARN] Failed to source ${VLLM_ENV_FILE}, continuing..." >&2
fi

# ------------------------------------------------------------------------------
# 1. 基础环境变量 — 详见 vllm_server_env_template.sh
# ------------------------------------------------------------------------------
# 模型路径: 指向 Hugging Face 模型目录
# 必须包含 config.json, tokenizer 文件和模型权重
MODEL_PATH="${MODEL_PATH:-moonshotai/Kimi-K2-Base}"
# 服务对外暴露的模型名称
# 客户端调用 API 时使用此名称
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-kimi-k2-base}"
# 服务监听地址
# 0.0.0.0 表示监听所有网络接口，127.0.0.1 仅监听本地
HOST="${HOST:-0.0.0.0}"
# 服务监听端口
PORT="${PORT:-8000}"
# 日志级别: debug, info, warning, error
LOG_LEVEL="${LOG_LEVEL:-info}"

# ------------------------------------------------------------------------------
# 命令行参数解析 (支持 --key=value 或 --key value 格式)
# ------------------------------------------------------------------------------
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model-path) MODEL_PATH="$2"; shift 2 ;;
        --served-model-name) SERVED_MODEL_NAME="$2"; shift 2 ;;
        --host) HOST="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        --tensor-parallel-size) TENSOR_PARALLEL_SIZE="$2"; shift 2 ;;
        --pipeline-parallel-size) PIPELINE_PARALLEL_SIZE="$2"; shift 2 ;;
        --dtype) DTYPE="$2"; shift 2 ;;
        --quantization) QUANTIZATION="$2"; shift 2 ;;
        --gpu-memory-utilization) GPU_MEMORY_UTILIZATION="$2"; shift 2 ;;
        --max-model-len) MAX_MODEL_LEN="$2"; shift 2 ;;
        --api-key) API_KEY="$2"; shift 2 ;;
        --*) EXTRA_ARGS+=("$1"); shift ;;
        *) EXTRA_ARGS+=("$1"); shift ;;
    esac
done

# ------------------------------------------------------------------------------
# 2. 分布式并行配置 — 详见 vllm_server_env_template.sh
# ------------------------------------------------------------------------------
# 推荐配置参考 (Kimi-K2 MoE, 384 experts):
#   128 NPU (16*8): TP=8, PP=16, EP=128
#   64 NPU  (8*8):  TP=8, PP=8,  EP=64
#   32 NPU  (4*8):  TP=8, PP=4,  EP=32
#   16 NPU  (2*8):  TP=8, PP=2,  EP=16
#   8 NPU   (1*8):  TP=8, PP=1,  EP=8
# ------------------------------------------------------------------------------
# 张量并行大小 (Tensor Parallel)
# 建议: 节点内 NPU 数量，通常设为 8
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-8}"
# 流水线并行大小 (Pipeline Parallel)
# 建议: 根据节点数设置，跨节点并行
PIPELINE_PARALLEL_SIZE="${PIPELINE_PARALLEL_SIZE:-1}"
# 分布式执行后端: ray, mp; 留空→自动选择
DISTRIBUTED_EXECUTOR_BACKEND="${DISTRIBUTED_EXECUTOR_BACKEND:-}"
# 专家并行开关 (Expert Parallel)
# MoE 模型强烈建议启用，可显著提升性能
ENABLE_EXPERT_PARALLEL="${ENABLE_EXPERT_PARALLEL:-1}"
# EP 自动计算为 TP×PP（Kimi-K2 有 384 experts，须能整除）
: "${EXPERT_PARALLEL_SIZE:=$((TENSOR_PARALLEL_SIZE * PIPELINE_PARALLEL_SIZE))}"

# ------------------------------------------------------------------------------
# 3. 内存与量化 — 详见 vllm_server_env_template.sh
# ------------------------------------------------------------------------------
# 模型数据类型
# 可选: float16, bfloat16, float32
# 即使权重是 FP8 量化，激活仍使用此类型
DTYPE="${DTYPE:-bfloat16}"
# 量化方式
# 可选: fp8, awq, gptq, squeezellm, marlin, 或留空表示无
QUANTIZATION="${QUANTIZATION:-fp8}"
# 模型加载格式
# 可选: safetensors, pt, auto
LOAD_FORMAT="${LOAD_FORMAT:-safetensors}"
# GPU(NPU) 内存利用率 (0.0 - 1.0)
# 较大的值使用更多显存用于 KV Cache，提高吞吐
# 建议范围: 0.88 - 0.95
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
# CPU 交换空间大小 (GiB)
# 用于 KV Cache 驱逐时的缓冲，MoE 模型建议设置较大值
# 注意：当 TP*PP 较大时，总交换空间 = SWAP_SPACE * TP，需要确保不超过系统内存
SWAP_SPACE="${SWAP_SPACE:-128}"

# ------------------------------------------------------------------------------
# 4. 吞吐量与序列调度 — 详见 vllm_server_env_template.sh
# ------------------------------------------------------------------------------
# 最大模型长度 (上下文窗口)
# 模型原生支持 131072，但为内存和吞吐折中，可限制为 32k-64k
# 如需更长序列，请确保有足够的 NPU 内存
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
# 最大并发请求数
# 根据预期负载和硬件能力调整，MoE 模型吞吐量较高
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1024}"
# 分块预填充开关 (Chunked Prefill)
# 强烈建议启用，解耦 Prefill 和 Decode 阶段，提升并发
ENABLE_CHUNKED_PREFILL="${ENABLE_CHUNKED_PREFILL:-1}"
# 每个 step 处理的最大 token 数
# 较大的值提高吞吐量，较小的值降低延迟
# 建议: 4096 - 16384
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
# 每个序列的最大 tokens (prefill + decode)
# 用于限制单个请求的资源占用，防止单个请求占用过多资源
MAX_TOKENS_PER_SEQUENCE="${MAX_TOKENS_PER_SEQUENCE:-32768}"


# ------------------------------------------------------------------------------
# 5. 高级加速特性 — 详见 vllm_server_env_template.sh
# ------------------------------------------------------------------------------
# 前缀缓存开关 (Prefix Caching)
# 对于多轮对话或大量重复 system prompt 极其有效，强烈建议启用
PREFIX_CACHING="${PREFIX_CACHING:-1}"
# 强制 Eager 模式
# 1 = 禁用 CUDA Graph/编译图 (推荐 NPU 环境)
# 0 = 启用 CUDA Graph (如果底层支持)
ENFORCE_EAGER="${ENFORCE_EAGER:-1}"
# CUDA Graph 捕获的最大序列长度
# 仅在 ENFORCE_EAGER=0 时有效，对于 MoE 模型建议保持较小值
MAX_SEQ_LEN_TO_CAPTURE="${MAX_SEQ_LEN_TO_CAPTURE:-8192}"
# 多步调度步数 (Multi-step Scheduling)
# 减少框架在各个 NPU 之间的调度通信开销
# 建议值: 4-8，较大的值提高吞吐但增加延迟
NUM_SCHEDULER_STEPS="${NUM_SCHEDULER_STEPS:-8}"
# 自动检测 vLLM 版本支持的参数
# 1 = 自动检测，0 = 使用预设参数
AUTO_DETECT_FLAGS="${AUTO_DETECT_FLAGS:-1}"


# ------------------------------------------------------------------------------
# 6. API 和监控 — 详见 vllm_server_env_template.sh
#     注意: 模板 ENABLE_METRICS=1（推荐生产开启），本脚本保守默认 0
# ------------------------------------------------------------------------------
# API 密钥 (生产环境强烈建议设置)
# 留空表示不启用认证
API_KEY="${API_KEY:-}"
# 工具调用开关 (Claude Code 集成必需)
# 1 = 启用，0 = 禁用
ENABLE_TOOL_CALLING="${ENABLE_TOOL_CALLING:-1}"
# 工具调用解析器
# 根据模型选择: hermes (Qwen), llama (Llama), mistral, deepseekv3 等
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-hermes}"
# Prometheus 指标导出开关
# 1 = 启用，0 = 禁用
ENABLE_METRICS="${ENABLE_METRICS:-0}"
# Prometheus 指标导出端口
METRICS_PORT="${METRICS_PORT:-8001}"
# 禁用请求日志开关
# 1 = 禁用 (减少日志量)，0 = 启用
DISABLE_LOG_REQUESTS="${DISABLE_LOG_REQUESTS:-0}"
# CORS 允许的源
# * 表示允许所有，或设置特定域名如 "https://example.com"
ALLOWED_ORIGINS="${ALLOWED_ORIGINS:-*}"

# ------------------------------------------------------------------------------
# 7. 启动与重试
# ------------------------------------------------------------------------------
# 最大重试次数
# 服务崩溃后自动重启的次数
MAX_RETRIES="${MAX_RETRIES:-3}"
RETRY_DELAY="${RETRY_DELAY:-10}"

# -----------------------------------------------------------------------------
# 辅助函数
# -----------------------------------------------------------------------------
has_flag() {
    [[ "${HELP_TEXT:-}" == *"$1"* ]]
}

# -----------------------------------------------------------------------------
# 前置检查
# -----------------------------------------------------------------------------
command -v vllm >/dev/null 2>&1 || { echo "[ERROR] vllm not found" >&2; exit 127; }
[[ -e "$MODEL_PATH" ]] || { echo "[ERROR] MODEL_PATH not found: $MODEL_PATH" >&2; exit 2; }
[[ -f "$MODEL_PATH/config.json" ]] || { echo "[ERROR] config.json not found" >&2; exit 2; }

# -----------------------------------------------------------------------------
# 动态检测 vLLM 支持的参数
# -----------------------------------------------------------------------------
HELP_TEXT=""
[[ "$AUTO_DETECT_FLAGS" == "1" ]] && HELP_TEXT="$(vllm serve --help 2>/dev/null || true)"

# -----------------------------------------------------------------------------
# 构建启动参数
# -----------------------------------------------------------------------------
args=(
    serve "$MODEL_PATH"
    --trust-remote-code
    --served-model-name "$SERVED_MODEL_NAME"
    --host "$HOST"
    --port "$PORT"
    --dtype "$DTYPE"
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
    --pipeline-parallel-size "$PIPELINE_PARALLEL_SIZE"
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --max-num-seqs "$MAX_NUM_SEQS"
    --max-model-len "$MAX_MODEL_LEN"
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
)

# 条件参数
[[ -n "$QUANTIZATION" && "$QUANTIZATION" != "none" ]] && args+=(--quantization "$QUANTIZATION")
[[ -n "$LOAD_FORMAT" ]] && args+=(--load-format "$LOAD_FORMAT")
# 分布式执行后端 (留空则由 vLLM 自动选择)
[[ -n "$DISTRIBUTED_EXECUTOR_BACKEND" ]] && args+=(--distributed-executor-backend "$DISTRIBUTED_EXECUTOR_BACKEND")
# Chunked Prefill
[[ "$ENABLE_CHUNKED_PREFILL" == "1" ]] && args+=(--enable-chunked-prefill)
# Swap Space (vLLM v1 已移除此参数)
has_flag "--swap-space" && args+=(--swap-space "$SWAP_SPACE")
# API Key
[[ -n "$API_KEY" ]] && args+=(--api-key "$API_KEY")
# Tool Calling
if [[ "$ENABLE_TOOL_CALLING" == "1" ]]; then
    args+=(--enable-auto-tool-choice)
    [[ -n "$TOOL_CALL_PARSER" ]] && args+=(--tool-call-parser "$TOOL_CALL_PARSER")
fi
# max_tokens_per_sequence
[[ -n "${MAX_TOKENS_PER_SEQUENCE:-}" ]] && has_flag "--max-tokens-per-sequence" && \
    args+=(--max-tokens-per-sequence "$MAX_TOKENS_PER_SEQUENCE")
# num-scheduler-steps
has_flag "--num-scheduler-steps" && args+=(--num-scheduler-steps "$NUM_SCHEDULER_STEPS")

# 动态特性检测
if [[ "$AUTO_DETECT_FLAGS" == "1" ]]; then
    # Expert Parallel
    if [[ "$ENABLE_EXPERT_PARALLEL" == "1" ]] && has_flag "--enable-expert-parallel"; then
        args+=("--enable-expert-parallel")
    fi

    # Prefix Caching — 优先 enable，否则按 disable 处理
    if [[ "$PREFIX_CACHING" == "1" ]] && has_flag "--enable-prefix-caching"; then
        args+=("--enable-prefix-caching")
    elif [[ "$PREFIX_CACHING" == "0" ]] && has_flag "--disable-prefix-caching"; then
        args+=("--disable-prefix-caching")
    fi

    # CUDA Graph (enforce-eager 须检测版本支持)
    if [[ "$ENFORCE_EAGER" == "1" ]]; then
        has_flag "--enforce-eager" && args+=(--enforce-eager)
    elif has_flag "--max-seq-len-to-capture"; then
        args+=(--max-seq-len-to-capture "$MAX_SEQ_LEN_TO_CAPTURE")
    fi

    # 日志级别
    has_flag "--log-level" && args+=(--log-level "$LOG_LEVEL")

    # Metrics
    if [[ "$ENABLE_METRICS" == "1" ]] && has_flag "--enable-metrics"; then
        args+=(--enable-metrics)
        has_flag "--metrics-port" && args+=(--metrics-port "$METRICS_PORT")
    fi

    has_flag "--allowed-origins" && args+=(--allowed-origins "$ALLOWED_ORIGINS")
    [[ "$DISABLE_LOG_REQUESTS" == "1" ]] && has_flag "--disable-log-requests" && args+=(--disable-log-requests)
fi

# 额外参数去重
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
    for extra_arg in "${EXTRA_ARGS[@]}"; do
        skip=false
        flag_name="${extra_arg%%=*}"
        for existing in "${args[@]}"; do
            existing_name="${existing%%=*}"
            if [[ "$existing_name" == "$flag_name" ]]; then
                skip=true
                break
            fi
        done
        [[ "$skip" == false ]] && args+=("$extra_arg")
    done
fi

# -----------------------------------------------------------------------------
# 配置摘要
# -----------------------------------------------------------------------------
cat << EOF
================================================================================
[INFO] vLLM Server Configuration
================================================================================
  Model:        $MODEL_PATH
  Name:         $SERVED_MODEL_NAME
  Listen:       $HOST:$PORT
--------------------------------------------------------------------------------
  Parallel:     TP=$TENSOR_PARALLEL_SIZE, PP=$PIPELINE_PARALLEL_SIZE, EP=$EXPERT_PARALLEL_SIZE
  Backend:      ${DISTRIBUTED_EXECUTOR_BACKEND:-auto}
--------------------------------------------------------------------------------
  Memory:       dtype=$DTYPE, quant=$QUANTIZATION, gpu_util=$GPU_MEMORY_UTILIZATION
--------------------------------------------------------------------------------
  Scheduling:   max_seqs=$MAX_NUM_SEQS, max_len=$MAX_MODEL_LEN, batched=$MAX_NUM_BATCHED_TOKENS, scheduler_steps=$NUM_SCHEDULER_STEPS
  Features:     chunked=$ENABLE_CHUNKED_PREFILL, prefix=$PREFIX_CACHING, tool_call=$ENABLE_TOOL_CALLING
--------------------------------------------------------------------------------
  Metrics:      enabled=$ENABLE_METRICS, port=$METRICS_PORT
================================================================================
[INFO] Command: vllm ${args[*]}
================================================================================
EOF

# -----------------------------------------------------------------------------
# 启动 (带重试)
# -----------------------------------------------------------------------------
RETRY_COUNT=0
while [[ $RETRY_COUNT -lt $MAX_RETRIES ]]; do
    vllm "${args[@]}"
    EXIT_CODE=$?

    if [[ $EXIT_CODE -eq 0 ]]; then
        echo "[INFO] vLLM server exited normally."
        break
    elif [[ $EXIT_CODE -eq 130 || $EXIT_CODE -eq 137 ]]; then
        echo "[INFO] Terminated by signal (exit $EXIT_CODE)."
        exit 0
    else
        RETRY_COUNT=$((RETRY_COUNT + 1))
        if [[ $RETRY_COUNT -lt $MAX_RETRIES ]]; then
            echo "[WARN] Crashed (exit $EXIT_CODE), retrying in ${RETRY_DELAY}s... ($RETRY_COUNT/$MAX_RETRIES)"
            sleep "$RETRY_DELAY"
        else
            echo "[FATAL] Max retries reached."
            exit "$EXIT_CODE"
        fi
    fi
done
