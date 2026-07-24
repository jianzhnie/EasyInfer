#!/bin/bash
# =============================================================================
# DeepSeek-V4-Flash W8A8 MTP — SGLang 包装器部署脚本
# =============================================================================
# 通过环境变量驱动 sglang launch_server，支持 ${VAR:-default} 覆盖模式。
#
# 用法:
#   1. 默认启动: ./sglang_server.sh
#   2. 环境变量覆盖: MODEL_PATH=/path/to/model ./sglang_server.sh
#   3. 命令行参数: ./sglang_server.sh --port 8080 --tp 16
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# -----------------------------------------------------------------------------
# 配置加载
# -----------------------------------------------------------------------------
# Load Ascend CANN environment
set +u
if [[ -f "/usr/local/Ascend/cann/set_env.sh" ]]; then
    source /usr/local/Ascend/cann/set_env.sh
fi
if [[ -f "/usr/local/Ascend/nnal/atb/set_env.sh" ]]; then
    source /usr/local/Ascend/nnal/atb/set_env.sh
fi
set -u

# -----------------------------------------------------------------------------
# 1. 基础配置
# -----------------------------------------------------------------------------
MODEL_PATH="${MODEL_PATH:-/home/jianzhnie/llmtuner/hfhub/models/Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-deepseek-v4-flash}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
LOG_LEVEL="${LOG_LEVEL:-info}"

# -----------------------------------------------------------------------------
# 2. 并行配置
# -----------------------------------------------------------------------------
TP="${TP:-32}"
PP="${PP:-1}"
EP="${EP:-32}"
NNODES="${NNODES:-4}"
NODE_RANK="${NODE_RANK:-0}"
DIST_INIT_ADDR="${DIST_INIT_ADDR:-10.16.201.193}"
DIST_INIT_PORT="${DIST_INIT_PORT:-5000}"

# -----------------------------------------------------------------------------
# 3. 内存与性能
# -----------------------------------------------------------------------------
DTYPE="${DTYPE:-bfloat16}"
QUANTIZATION="${QUANTIZATION:-modelopt_fp8}"
MEM_FRACTION="${MEM_FRACTION:-0.90}"
CONTEXT_LEN="${CONTEXT_LEN:-65536}"
MAX_RUNNING_REQS="${MAX_RUNNING_REQS:-16}"
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-8192}"
CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-8192}"
ENABLE_TORCH_COMPILE="${ENABLE_TORCH_COMPILE:-1}"

# -----------------------------------------------------------------------------
# 4. 特性开关
# -----------------------------------------------------------------------------
ENABLE_TOOL_CALL="${ENABLE_TOOL_CALL:-1}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-deepseek_v3}"
ENABLE_SPECULATIVE="${ENABLE_SPECULATIVE:-1}"
SPECULATIVE_ALGORITHM="${SPECULATIVE_ALGORITHM:-EAGLE}"
SPECULATIVE_NUM_DRAFT_TOKENS="${SPECULATIVE_NUM_DRAFT_TOKENS:-3}"

# -----------------------------------------------------------------------------
# 5. HCCL/NPU 环境变量
# -----------------------------------------------------------------------------
export HCCL_OP_EXPANSION_MODE="${HCCL_OP_EXPANSION_MODE:-AIV}"
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export HCCL_BUFFSIZE=200
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export SGLANG_ASCEND_BALANCE_SCHEDULING=1

# -----------------------------------------------------------------------------
# 命令行参数覆盖
# -----------------------------------------------------------------------------
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model-path) MODEL_PATH="$2"; shift 2 ;;
        --served-model-name) SERVED_MODEL_NAME="$2"; shift 2 ;;
        --host) HOST="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        --tp) TP="$2"; shift 2 ;;
        --pp) PP="$2"; shift 2 ;;
        --ep) EP="$2"; shift 2 ;;
        --nnodes) NNODES="$2"; shift 2 ;;
        --node-rank) NODE_RANK="$2"; shift 2 ;;
        --dist-init-addr) DIST_INIT_ADDR="$2"; shift 2 ;;
        --context-length) CONTEXT_LEN="$2"; shift 2 ;;
        --quantization) QUANTIZATION="$2"; shift 2 ;;
        --*) EXTRA_ARGS+=("$1"); shift ;;
        *) EXTRA_ARGS+=("$1"); shift ;;
    esac
done

# -----------------------------------------------------------------------------
# 前置检查
# -----------------------------------------------------------------------------
command -v python3 >/dev/null 2>&1 || { echo "[ERROR] python3 not found"; exit 127; }
[[ -e "$MODEL_PATH" ]] || { echo "[ERROR] MODEL_PATH not found: $MODEL_PATH"; exit 3; }
[[ -f "$MODEL_PATH/config.json" ]] || { echo "[ERROR] config.json not found in: $MODEL_PATH"; exit 3; }

# -----------------------------------------------------------------------------
# 构建启动参数
# -----------------------------------------------------------------------------
args=(
    --model-path "$MODEL_PATH"
    --host "$HOST"
    --port "$PORT"
    --served-model-name "$SERVED_MODEL_NAME"
    --trust-remote-code
    --dtype "$DTYPE"
    --tp "$TP"
    --pp "$PP"
    --ep "$EP"
    --device npu
    --quantization "$QUANTIZATION"
    --mem-fraction-static "$MEM_FRACTION"
    --context-length "$CONTEXT_LEN"
    --max-running-requests "$MAX_RUNNING_REQS"
    --max-total-tokens "$MAX_TOTAL_TOKENS"
    --chunked-prefill-size "$CHUNKED_PREFILL_SIZE"
    --nnodes "$NNODES"
    --node-rank "$NODE_RANK"
    --dist-init-addr "$DIST_INIT_ADDR"
    --dist-init-port "$DIST_INIT_PORT"
    --log-level "$LOG_LEVEL"
)

# 条件参数
[[ "$ENABLE_TORCH_COMPILE" == "1" ]] && args+=(--enable-torch-compile)

if [[ "$ENABLE_TOOL_CALL" == "1" ]]; then
    args+=(--enable-tool-call)
    [[ -n "$TOOL_CALL_PARSER" ]] && args+=(--tool-call-parser "$TOOL_CALL_PARSER")
fi

if [[ "$ENABLE_SPECULATIVE" == "1" ]]; then
    args+=(--speculative-algorithm "$SPECULATIVE_ALGORITHM")
    args+=(--speculative-num-draft-tokens "$SPECULATIVE_NUM_DRAFT_TOKENS")
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
[INFO] SGLang Server Configuration — DeepSeek-V4-Flash W8A8 MTP
================================================================================
  Model:        $MODEL_PATH
  Name:         $SERVED_MODEL_NAME
  Listen:       $HOST:$PORT
--------------------------------------------------------------------------------
  Parallel:     TP=$TP, PP=$PP, EP=$EP
  Cluster:      NNODES=$NNODES, NODE_RANK=$NODE_RANK
  Init Addr:    $DIST_INIT_ADDR:$DIST_INIT_PORT
--------------------------------------------------------------------------------
  Memory:       dtype=$DTYPE, quant=$QUANTIZATION, mem_frac=$MEM_FRACTION
  Context:      max_len=$CONTEXT_LEN, max_reqs=$MAX_RUNNING_REQS
  Prefill:      chunked=$CHUNKED_PREFILL_SIZE, max_tokens=$MAX_TOTAL_TOKENS
--------------------------------------------------------------------------------
  Features:     tool_call=$ENABLE_TOOL_CALL ($TOOL_CALL_PARSER)
                speculative=$ENABLE_SPECULATIVE ($SPECULATIVE_ALGORITHM)
                torch_compile=$ENABLE_TORCH_COMPILE
                prefix_cache=RadixAttention (automatic)
================================================================================
[INFO] Command: python -m sglang.launch_server ${args[*]}
================================================================================
EOF

# -----------------------------------------------------------------------------
# 启动
# -----------------------------------------------------------------------------
exec python -m sglang.launch_server "${args[@]}"
