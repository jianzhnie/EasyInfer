#!/bin/bash
# =============================================================================
# GLM-5 (Full Parameter) vLLM 推理服务部署脚本
# =============================================================================
# 参考自: examples/glm5-1_server.sh
# 
# 用法:
#   bash examples/glm5_full_server.sh
#
# 环境变量（均可外部覆盖）:
#   MODEL_PATH              - 模型路径 (默认: /llm_workspace_1P/robin/hfhub/models/ZhipuAI/GLM-5)
#   PORT                    - 服务端口 (默认: 8077)
#   MAX_MODEL_LEN           - 最大上下文长度 (默认: 131072)
#   TENSOR_PARALLEL_SIZE    - 张量并行度 (默认: 32, GLM-5 全量模型建议 32 或以上)
#   GPU_MEMORY_UTILIZATION  - 显存利用率 (默认: 0.95)
#   VLLM_HOST_IP            - 节点 IP (默认: 自动检测)
# =============================================================================

set -euo pipefail

# shellcheck disable=SC2034
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./_common.sh
source "${SCRIPT_DIR}/_common.sh"

# -----------------------------------------------------------------------------
# 服务配置（针对 GLM-5 全量参数模型优化）
# -----------------------------------------------------------------------------
MODEL_PATH="${MODEL_PATH:-/llm_workspace_1P/robin/hfhub/models/ZhipuAI/GLM-5}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-glm-5}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8077}"
API_KEY="${API_KEY:-}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-64}"
PIPELINE_PARALLEL_SIZE="${PIPELINE_PARALLEL_SIZE:-1}"
DATA_PARALLEL_SIZE="${DATA_PARALLEL_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.95}"
DTYPE="${DTYPE:-bfloat16}"
# 全量模型不使用量化参数
QUANTIZATION="${QUANTIZATION:-}" 
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
SEED="${SEED:-1024}"

# -----------------------------------------------------------------------------
# NPU / HCCL 环境变量 (针对 MoE 架构优化)
# -----------------------------------------------------------------------------
export HCCL_OP_EXPANSION_MODE="${HCCL_OP_EXPANSION_MODE:-AIV}"
export OMP_PROC_BIND="${OMP_PROC_BIND:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export HCCL_BUFFSIZE="${HCCL_BUFFSIZE:-200}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"
# export VLLM_ASCEND_BALANCE_SCHEDULING="${VLLM_ASCEND_BALANCE_SCHEDULING:-1}"
# export VLLM_ASCEND_ENABLE_FLASHCOMM1=1

# -----------------------------------------------------------------------------
# 前置检查
# -----------------------------------------------------------------------------
check_prereqs "$MODEL_PATH"

# -----------------------------------------------------------------------------
# 打印配置
# -----------------------------------------------------------------------------
cat <<EOF
================================================================================
  GLM-5 (Full Parameter) vLLM Server
================================================================================
  Model:          $MODEL_PATH
  Served name:    $SERVED_MODEL_NAME
  Listen:         $HOST:$PORT
  API Key:        ${API_KEY:+******** (set)}${API_KEY:-(not set)}
--------------------------------------------------------------------------------
  Tensor Parallel:      TP=$TENSOR_PARALLEL_SIZE
  Pipeline Parallel:    PP=$PIPELINE_PARALLEL_SIZE
  Data Parallel:        DP=$DATA_PARALLEL_SIZE
  Distributed Backend:  ray
--------------------------------------------------------------------------------
  dtype:           $DTYPE
  quantization:    ${QUANTIZATION:-None (Full Parameter)}
  GPU memory:      ${GPU_MEMORY_UTILIZATION}
  max_model_len:   $MAX_MODEL_LEN
  max_num_seqs:    $MAX_NUM_SEQS
  max_batched_tok: $MAX_NUM_BATCHED_TOKENS
--------------------------------------------------------------------------------
================================================================================
EOF

# -----------------------------------------------------------------------------
# 构建 vllm serve 参数
# -----------------------------------------------------------------------------
vllm_args=(
    serve "$MODEL_PATH"
    --host "$HOST"
    --port "$PORT"
    --served-model-name "$SERVED_MODEL_NAME"
    --seed "$SEED"
    --dtype "$DTYPE"

    # 并行配置
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
    --pipeline-parallel-size "$PIPELINE_PARALLEL_SIZE"
    --data-parallel-size "$DATA_PARALLEL_SIZE"
    --enable-expert-parallel
    --distributed-executor-backend ray

    # 内存与上下文
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --max-model-len "$MAX_MODEL_LEN"
    --max-num-seqs "$MAX_NUM_SEQS"
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"

    # 全量模型不显式传递 --quantization 如果为空
    ${QUANTIZATION:+--quantization "$QUANTIZATION"}
    --trust-remote-code

    # 加速特性
    --enable-chunked-prefill
    --enable-prefix-caching
    # --async-scheduling # Ray 后端暂不支持异步调度

    # 投机解码 (如果出现权重未初始化错误，建议先禁用)
    # --speculative-config '{"num_speculative_tokens": 3, "method": "mtp"}'

    # 编译优化
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [64]}'
    --additional-config '{"fuse_muls_add": true, "multistream_overlap_shared_expert": true, "ascend_compilation_config": {"enable_npugraph_ex": true}}'

    # 工具调用
    --enable-auto-tool-choice
    --tool-call-parser glm47
    --reasoning-parser glm45
    --chat-template-content-format string
)

# API Key（可选认证）
if [[ -n "$API_KEY" ]]; then
    vllm_args+=(--api-key "$API_KEY")
fi

echo "[INFO] Starting vLLM server (this may take a few minutes)..."
echo "[INFO] Full command: vllm ${vllm_args[*]}"
echo ""

# -----------------------------------------------------------------------------
# 启动 vLLM
# -----------------------------------------------------------------------------
cleanup() {
    if [[ -n "${VLLM_PID:-}" ]] && kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "[INFO] Cleaning up vLLM process (PID: $VLLM_PID)..."
        kill "$VLLM_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

vllm "${vllm_args[@]}" &
VLLM_PID=$!

# -----------------------------------------------------------------------------
# 等待服务就绪
# -----------------------------------------------------------------------------
wait_for_server "$HOST" "$PORT" "$VLLM_PID" 900

# 把控制权交还给 vLLM 进程
wait "$VLLM_PID"
