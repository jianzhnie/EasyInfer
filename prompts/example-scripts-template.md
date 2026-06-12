# EasyInfer 示例脚本模板与规范

本文件定义 `examples/<model>/vllm/` 目录下 Shell 脚本的统一格式。所有新模型示例脚本必须按此模板生成，现有脚本逐步对齐。

## 1. 通用 Shell 规范

### 1.1 文件头

每个直接执行的 `.sh` 脚本必须以统一横幅开头：

```bash
#!/bin/bash
# =============================================================================
# <模型名> <量化> — <一句话用途>
# =============================================================================
# <补充说明：架构、默认配置、关键约束>
#
# Usage:
#   ./<script>.sh
#   VAR=value ./<script>.sh
#
# Reference:
#   <vLLM-Ascend 官方文档链接>
# =============================================================================
```

### 1.2 Shell 选项

- 直接执行的脚本：`set -euo pipefail`
- 被 source 的库文件：不设 shell 选项
- CANN 环境加载前必须用 `set +u` / `set -u` 包裹

### 1.3 变量规范

| 类型 | 命名 | 声明方式 | 示例 |
|------|------|----------|------|
| 环境变量/可覆盖配置 | `UPPER_SNAKE_CASE` | `${VAR:-default}` | `TP="${TP:-8}"` |
| 本地常量 | `UPPER_SNAKE_CASE` | `readonly` | `readonly BASE_MODEL_PATH="..."` |
| 函数局部变量 | `snake_case` | `local` | `local elapsed` |

### 1.4 关键约束

- 所有变量引用必须双引号：`"$VAR"`、`"${VAR}"`
- 条件判断用 `[[ ]]`，命令替换用 `$(command)`
- 函数内变量用 `local`，常量用 `readonly`
- 4 空格缩进，最大行宽 120 字符
- 单脚本不超过 400 行，单函数不超过 50 行
- 禁止 `eval` 执行动态构建的命令
- 必须通过 `bash -n` 语法检查

---

## 2. `run_vllm.sh` 模板

```bash
#!/bin/bash
# =============================================================================
# <Model> <Quant> — Direct vllm serve deployment
# =============================================================================
# Architecture: <Arch> | <Experts> Experts | <MoE/MLA/...>
# Default: TP=<tp> PP=1 (single-node)
# Note: <model-specific notes>
#
# Usage:
#   bash run_vllm.sh
#   TP=<tp> MAX_MODEL_LEN=<len> bash run_vllm.sh
#
# Reference:
#   <url>
# =============================================================================
set -euo pipefail

# Load Ascend CANN environment
set +u
if [[ -f "/usr/local/Ascend/cann/set_env.sh" ]]; then
    source /usr/local/Ascend/cann/set_env.sh
fi
if [[ -f "/usr/local/Ascend/nnal/atb/set_env.sh" ]]; then
    source /usr/local/Ascend/nnal/atb/set_env.sh
fi
set -u

# Base configuration
readonly BASE_MODEL_PATH="/home/jianzhnie/llmtuner/hfhub/models/Eco-Tech"
readonly MODEL_PATH="${MODEL_PATH:-$BASE_MODEL_PATH/<MODEL_REL_PATH>}"
readonly HOST="${HOST:-0.0.0.0}"
readonly PORT="${PORT:-<PORT>}"
readonly TP="${TP:-<TP>}"
readonly PP="${PP:-1}"
readonly DP="${DP:-1}"          # 仅当模型支持 DP 时使用
readonly MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
readonly MAX_NUM_SEQS="${MAX_NUM_SEQS:-<N>}"
readonly GPU_MEM_UTIL="${GPU_MEM_UTIL:-<0.XX>}"

# NPU environment variables
export HCCL_OP_EXPANSION_MODE=AIV
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export HCCL_BUFFSIZE=<BUFFSIZE>
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_USE_MODELSCOPE=False

# Fallback variables for older versions
export VLLM_ASCEND_BALANCE_SCHEDULING=1
export VLLM_ASCEND_ENABLE_FLASHCOMM1=<0_OR_1>
export VLLM_ASCEND_ENABLE_MLAPO=<0_OR_1>

# v0.20.2 additional_config format
readonly ADDITIONAL_CONFIG='{"enable_balance_scheduling": true, "enable_flashcomm1": <bool>, "enable_mlapo": <bool>}'

echo "============================================"
echo "[INFO] <Model> <Quant> — Deployment"
echo "[INFO] Model: $MODEL_PATH"
echo "[INFO] TP=$TP PP=$PP DP=$DP PORT=$PORT"
echo "[INFO] MAX_MODEL_LEN=$MAX_MODEL_LEN MAX_NUM_SEQS=$MAX_NUM_SEQS"
echo "[INFO] GPU_MEM_UTIL=$GPU_MEM_UTIL"
echo "============================================"

vllm serve "$MODEL_PATH" \
    --host "$HOST" \
    --port "$PORT" \
    --served-model-name "<api-name>" \
    --trust-remote-code \
    --dtype bfloat16 \
    --tensor-parallel-size "$TP" \
    --pipeline-parallel-size "$PP" \
    --data-parallel-size "$DP" \
    --distributed-executor-backend ray \
    --quantization ascend \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --max-num-batched-tokens <N> \
    --chat-template-content-format string \
    --enable-chunked-prefill \
    --enable-prefix-caching \
    --enforce-eager \
    --enable-expert-parallel \
    --enable-auto-tool-choice \
    --tool-call-parser <parser> \
    --reasoning-parser <parser> \        # GLM 系列需要
    --speculative-config '{"num_speculative_tokens": 3, "method": "mtp"}' \   # MTP 模型需要
    --language-model-only \               # Kimi 多模态纯文本场景
    --mm-encoder-tp-mode data \           # Kimi 多模态
    --allowed-local-media-path /home/jianzhnie/llmtuner/ \   # Kimi 多模态
    --additional-config "$ADDITIONAL_CONFIG" \
    --seed 1024 \
    "$@"
```

### 模型特定参数替换表

| 模型 | PORT | TP 默认 | 量化 | FLASHCOMM1 | MLAPO | MTP | Parser |
|------|------|---------|------|------------|-------|-----|--------|
| GLM-5 | 8001 | 8 | W4A8 | 0 | 1 | ✓ | glm47/glm45 |
| GLM-5.1 | 8002 | 8 | W4A8 | 0 | 1 | ✓ | glm47/glm45 |
| Kimi-K2.6 | 8003 | 8 | W4A8 | 1 | 1 | ✗ | kimi_k2 |
| MiniMax-M2.7 | 8004 | 4 | W8A8 | 1 | N/A | ✗ | minimax_m2 |

---

## 3. `vllm_server.sh` 模板

```bash
#!/bin/bash
# =============================================================================
# <Model> <Quant> — Traditional wrapper deployment
# =============================================================================
# Calls scripts/vllm/vllm_model_server.sh to deploy <Model>.
# Architecture: <Arch>
#
# Usage:
#   ./vllm_server.sh
#   TENSOR_PARALLEL_SIZE=<tp> MAX_MODEL_LEN=<len> ./vllm_server.sh
#
# Reference:
#   <url>
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_DIR
readonly VLLM_SCRIPT="${SCRIPT_DIR}/../../../scripts/vllm/vllm_model_server.sh"

if [[ ! -f "$VLLM_SCRIPT" ]]; then
    echo "[ERROR] vLLM startup script not found: $VLLM_SCRIPT" >&2
    exit 1
fi

# ------------------------------------------------------------------------------
# Model path and base configuration
# ------------------------------------------------------------------------------
export MODEL_PATH="${MODEL_PATH:-/home/jianzhnie/llmtuner/hfhub/models/Eco-Tech/<MODEL>}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-<api-name>}"
export HOST="${HOST:-0.0.0.0}"
export PORT="${PORT:-<PORT>}"

# ------------------------------------------------------------------------------
# Huawei NPU environment variables
# ------------------------------------------------------------------------------
export HCCL_OP_EXPANSION_MODE="${HCCL_OP_EXPANSION_MODE:-AIV}"
export OMP_PROC_BIND="${OMP_PROC_BIND:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export HCCL_BUFFSIZE="${HCCL_BUFFSIZE:-<N>}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"
export VLLM_ASCEND_BALANCE_SCHEDULING="${VLLM_ASCEND_BALANCE_SCHEDULING:-1}"
export VLLM_ASCEND_ENABLE_FLASHCOMM1="${VLLM_ASCEND_ENABLE_FLASHCOMM1:-<0_OR_1>}"
export VLLM_ASCEND_ENABLE_MLAPO="${VLLM_ASCEND_ENABLE_MLAPO:-<0_OR_1>}"

# ------------------------------------------------------------------------------
# Parallel configuration
# ------------------------------------------------------------------------------
export TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-<TP>}"
export PIPELINE_PARALLEL_SIZE="${PIPELINE_PARALLEL_SIZE:-1}"
export ENABLE_EXPERT_PARALLEL="${ENABLE_EXPERT_PARALLEL:-1}"
export DATA_PARALLEL_SIZE="${DATA_PARALLEL_SIZE:-1}"

# ------------------------------------------------------------------------------
# Quantization and memory configuration
# ------------------------------------------------------------------------------
export DTYPE="${DTYPE:-bfloat16}"
export QUANTIZATION="${QUANTIZATION:-ascend}"
export LOAD_FORMAT="${LOAD_FORMAT:-auto}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-<0.XX>}"
export SWAP_SPACE="${SWAP_SPACE:-<N>}"

# ------------------------------------------------------------------------------
# Sequence scheduling
# ------------------------------------------------------------------------------
if [[ -z "${MAX_MODEL_LEN:-}" ]]; then
    if [[ "${TENSOR_PARALLEL_SIZE:-<TP>}" -ge <THRESHOLD> ]]; then
        export MAX_MODEL_LEN=<LARGE_LEN>
    else
        export MAX_MODEL_LEN=32768
    fi
fi
if [[ -z "${MAX_NUM_SEQS:-}" ]]; then
    export MAX_NUM_SEQS=<N>
fi
export ENABLE_CHUNKED_PREFILL="${ENABLE_CHUNKED_PREFILL:-1}"
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-<N>}"
export MAX_TOKENS_PER_SEQUENCE="${MAX_TOKENS_PER_SEQUENCE:-<N>}"
export CHAT_TEMPLATE_CONTENT_FORMAT="${CHAT_TEMPLATE_CONTENT_FORMAT:-string}"

# ------------------------------------------------------------------------------
# Acceleration features
# ------------------------------------------------------------------------------
export PREFIX_CACHING="${PREFIX_CACHING:-1}"
export ENFORCE_EAGER="${ENFORCE_EAGER:-1}"

# ------------------------------------------------------------------------------
# Speculative decoding (MTP) — enable for MTP models only
# ------------------------------------------------------------------------------
export SPECULATIVE_METHOD="${SPECULATIVE_METHOD:-mtp}"
export SPECULATIVE_NUM_TOKENS="${SPECULATIVE_NUM_TOKENS:-3}"

# ------------------------------------------------------------------------------
# NPU compilation optimization
# ------------------------------------------------------------------------------
export CUDAGRAPH_MODE="${CUDAGRAPH_MODE:-FULL_DECODE_ONLY}"
export ENABLE_NPUGRAPH_EX="${ENABLE_NPUGRAPH_EX:-true}"
export FUSE_MULS_ADD="${FUSE_MULS_ADD:-true}"
export MULTISTREAM_OVERLAP_SHARED_EXPERT="${MULTISTREAM_OVERLAP_SHARED_EXPERT:-true}"

# ------------------------------------------------------------------------------
# Tool calling
# ------------------------------------------------------------------------------
export ENABLE_TOOL_CALLING="${ENABLE_TOOL_CALLING:-1}"
export TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-<parser>}"

# ------------------------------------------------------------------------------
# Monitoring and logging
# ------------------------------------------------------------------------------
export ENABLE_METRICS="${ENABLE_METRICS:-1}"
export LOG_LEVEL="${LOG_LEVEL:-info}"
export MAX_RETRIES="${MAX_RETRIES:-3}"
export RETRY_DELAY="${RETRY_DELAY:-10}"

# ------------------------------------------------------------------------------
# Startup arguments
# ------------------------------------------------------------------------------
EXTRA_ARGS=(
    --seed 1024
    --trust-remote-code
)

if [[ "$SPECULATIVE_METHOD" == "mtp" ]]; then
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
# Startup banner
# ------------------------------------------------------------------------------
echo "[INFO] Starting <Model> <Quant> server"
echo "[INFO] Model:     ${MODEL_PATH}"
echo "[INFO] Hardware:  TP=$TENSOR_PARALLEL_SIZE, PP=$PIPELINE_PARALLEL_SIZE, DP=$DATA_PARALLEL_SIZE"
echo "[INFO] Quant:     <Quant> (ascend), dtype=$DTYPE"
echo "[INFO] Memory:    max_len=$MAX_MODEL_LEN, max_seqs=$MAX_NUM_SEQS, gpu_util=$GPU_MEMORY_UTILIZATION"
echo "[INFO] Features:  <feature list>"
echo "[INFO] HCCL:      OP_EXPANSION_MODE=$HCCL_OP_EXPANSION_MODE, BUFFSIZE=${HCCL_BUFFSIZE}MB"

exec bash "$VLLM_SCRIPT" "${EXTRA_ARGS[@]}" "$@"
```

### 关键注意事项

- `SCRIPT_DIR` 必须先赋值再 `readonly`
- 环境变量名使用全称（`TENSOR_PARALLEL_SIZE` 而非 `TP`），与 `vllm_model_server.sh` 对齐
- 所有导出变量支持 `${VAR:-default}` 覆盖

---

## 4. `curl_test.sh` 模板

```bash
#!/bin/bash
# =============================================================================
# <Model> <Quant> — API functional test script
# =============================================================================
# Targets localhost:<PORT> by default.
#
# Usage:
#   ./curl_test.sh
#   HOST=10.0.0.1 PORT=9000 ./curl_test.sh
#   MODEL_NAME=my-model ./curl_test.sh
# =============================================================================
set -euo pipefail

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
HOST="${HOST:-localhost}"
PORT="${PORT:-<PORT>}"
MODEL_NAME="${MODEL_NAME:-<api-name>}"
readonly TIMEOUT=300
readonly WAIT_INTERVAL=5
readonly BASE_URL="http://${HOST}:${PORT}"

# ------------------------------------------------------------------------------
# Logging helpers
# ------------------------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info()    { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error()   { echo -e "${RED}[ERROR]${NC} $1"; }

# ------------------------------------------------------------------------------
# Wait for the service to become ready
# ------------------------------------------------------------------------------
wait_for_service() {
    log_info "等待服务启动: ${BASE_URL} ..."
    local start_time elapsed
    start_time=$(date +%s)

    while true; do
        if curl -s "${BASE_URL}/health" >/dev/null 2>&1 || \
           curl -s "${BASE_URL}/v1/models" >/dev/null 2>&1; then
            log_success "服务已就绪!"
            return 0
        fi

        elapsed=$(( $(date +%s) - start_time ))
        if [[ "$elapsed" -ge "$TIMEOUT" ]]; then
            log_error "等待服务超时 (${TIMEOUT} 秒)!"
            return 1
        fi

        log_info "服务未就绪，等待 ${WAIT_INTERVAL}s... (已等待 ${elapsed}s)"
        sleep "$WAIT_INTERVAL"
    done
}

# ------------------------------------------------------------------------------
# Run a single JSON API test
# Args:
#   $1: test name
#   $2: endpoint URL
#   $3: JSON payload (empty for GET)
#   $4: optional extra header
#   $5: optional "quiet" flag
# ------------------------------------------------------------------------------
run_test() {
    local test_name="$1"
    local endpoint="$2"
    local payload="$3"
    local header="${4:-}"
    local quiet="${5:-}"
    local response curl_status=0
    local curl_opts=(-s "$endpoint" -H "Content-Type: application/json")

    [[ -n "$header" ]] && curl_opts+=(-H "$header")
    [[ -n "$payload" ]] && curl_opts+=(-d "$payload")

    echo ""
    log_info "测试: ${test_name}"
    log_info "Endpoint: ${endpoint}"

    if [[ "$quiet" == "quiet" ]]; then
        curl "${curl_opts[@]}" >/dev/null 2>&1 || curl_status=$?
    else
        response=$(curl "${curl_opts[@]}") || curl_status=$?
        echo "$response" | python3 -m json.tool 2>/dev/null || echo "$response"
    fi

    if [[ "$curl_status" -eq 0 ]]; then
        log_success "${test_name} 完成 (退出码: ${curl_status})"
    else
        log_warning "${test_name} 可能存在问题 (退出码: ${curl_status})"
    fi
    return "$curl_status"
}

# ------------------------------------------------------------------------------
# Run a streaming chat completion test
# ------------------------------------------------------------------------------
run_stream_test() {
    local test_name="$1"
    local payload="$2"

    echo ""
    log_info "测试: ${test_name}"
    curl -s "${BASE_URL}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "$payload" 2>&1 | head -10 || true
    log_success "${test_name} 完成"
}

# ------------------------------------------------------------------------------
# Main test sequence
# ------------------------------------------------------------------------------
echo "=========================================="
echo "  <Model> <Quant> API 功能测试"
echo "  目标地址: ${BASE_URL}"
echo "  模型名称: ${MODEL_NAME}"
echo "=========================================="

wait_for_service || exit 1

# 1. Model list (GET)
run_test "模型列表查询" "${BASE_URL}/v1/models" ""

# 2. Chat completion (English)
run_test "Chat Completion (英文)" \
    "${BASE_URL}/v1/chat/completions" \
    '{"model":"'"$MODEL_NAME"'","messages":[{"role":"user","content":"Hello, who are you?"}],"max_tokens":128,"temperature":0.7}'

# 3. Chat completion (Chinese)
run_test "Chat Completion (中文)" \
    "${BASE_URL}/v1/chat/completions" \
    '{"model":"'"$MODEL_NAME"'","messages":[{"role":"system","content":"You are a helpful assistant."},{"role":"user","content":"你好，请简单介绍一下你自己。"}],"max_tokens":128,"temperature":0.7}'

# 4. Tool calling
run_test "Tool Calling" \
    "${BASE_URL}/v1/chat/completions" \
    '{"model":"'"$MODEL_NAME"'","messages":[{"role":"user","content":"What is the weather like in Beijing?"}],"tools":[{"type":"function","function":{"name":"get_weather","description":"Get the current weather","parameters":{"type":"object","properties":{"city":{"type":"string","description":"The city to get weather for"}},"required":["city"]}}}],"tool_choice":"auto","max_tokens":100}'

# 5. Anthropic Messages API
run_test "Anthropic Messages API" \
    "${BASE_URL}/v1/messages" \
    '{"model":"'"$MODEL_NAME"'","max_tokens":100,"messages":[{"role":"user","content":"Hi there!"}]}' \
    "x-api-key: dummy"

# 6. Streaming chat completion
run_stream_test "流式 Chat Completion" \
    '{"model":"'"$MODEL_NAME"'","messages":[{"role":"user","content":"从1数到5"}],"max_tokens":100,"stream":true}'

# 7. Multimodal Vision (Kimi-K2.6 only)
run_test "多模态 Vision (图片 URL)" \
    "${BASE_URL}/v1/chat/completions" \
    '{"model":"'"$MODEL_NAME"'","messages":[{"role":"user","content":[{"type":"text","text":"Describe this image briefly."},{"type":"image_url","image_url":{"url":"https://example.com/test.png"}}]}],"max_tokens":128}'

echo ""
echo "=========================================="
log_success "<Model> <Quant> 所有测试完成!"
echo "=========================================="
```

### 测试项说明

| 编号 | 测试 | 适用模型 |
|------|------|----------|
| 1 | 模型列表 (GET) | 全部 |
| 2 | 英文 Chat Completion | 全部 |
| 3 | 中文 Chat Completion | 全部 |
| 4 | Tool Calling | 全部 |
| 5 | Anthropic Messages API | 全部 |
| 6 | 流式 Chat Completion | 全部 |
| 7 | 多模态 Vision | Kimi-K2.6 |

---

## 5. 高级特性脚本模板

### 5.1 `run_dynamic_chunked_pp.sh`

适用于支持 PP 的模型（Kimi-K2.6、MiniMax-M2.7）。GLM 系列不支持的，应输出说明并退出。

```bash
#!/bin/bash
# =============================================================================
# <Model> <Quant> — Dynamic Chunked Pipeline Parallel
# =============================================================================
# Purpose: Dynamic chunking strategy based on profiling.
# Architecture: <Arch> | supports PP > 1
#
# Requirements:
#   - pipeline_parallel_size > 1
#   - --enable-chunked-prefill must be enabled
#   - Not compatible with enable_balance_scheduling
#
# Usage:
#   TP=8 PP=2 MAX_MODEL_LEN=131072 bash run_dynamic_chunked_pp.sh
#
# Reference:
#   https://docs.vllm.ai/projects/ascend/zh-cn/releases-v0.20.2rc/tutorials/features/dynamic_chunked_pipeline_parallel.html
# =============================================================================
set -euo pipefail

# ... CANN load, base config ...

readonly PROFILING_CHUNK_CONFIG="${PROFILING_CHUNK_CONFIG:-{\"enabled\": true, \"smooth_factor\": 1.0, \"min_chunk\": 4096, \"need_timing\": true}}"

export VLLM_ASCEND_BALANCE_SCHEDULING=0

# ... NPU env ...

vllm serve "$MODEL_PATH" \
    ... \
    --pipeline-parallel-size "$PP" \
    --enable-chunked-prefill \
    --additional-config "{\"profiling_chunk_config\": $PROFILING_CHUNK_CONFIG}" \
    "$@"
```

### 5.2 `run_long_seq_cp.sh`

适用于支持 Context Parallelism 的模型。

```bash
#!/bin/bash
# =============================================================================
# <Model> <Quant> — Long Sequence Context Parallel
# =============================================================================
# Purpose: Break single-card sequence length limit via CP.
#
# Constraints:
#   - tp_size must be divisible by dcp_size
#   - Currently only Atlas A3 devices are supported
#
# Usage:
#   TP=16 DCP=2 MAX_MODEL_LEN=131072 bash run_long_seq_cp.sh
# =============================================================================
set -euo pipefail

# ... CANN load, base config ...

readonly PCP_SIZE="${PCP_SIZE:-2}"
readonly DCP_SIZE="${DCP_SIZE:-2}"

export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ASCEND_BALANCE_SCHEDULING=0

# ... NPU env ...

vllm serve "$MODEL_PATH" \
    ... \
    --prefill-context-parallel-size "$PCP_SIZE" \
    --decode-context-parallel-size "$DCP_SIZE" \
    --no-enable-prefix-caching \
    "$@"
```

### 5.3 `run_pd_colocated.sh`

```bash
#!/bin/bash
# =============================================================================
# <Model> <Quant> — PD Colocated with Mooncake
# =============================================================================
# Purpose: Prefill-decode colocation via Mooncake distributed KV Cache.
#
# Prerequisites:
#   1. Mooncake installed
#   2. Mooncake Master started: mooncake_master --port 50088
#   3. mooncake.json configured
#
# Usage:
#   MOONCAKE_CONFIG_PATH=/path/to/mooncake.json bash run_pd_colocated.sh
# =============================================================================
set -euo pipefail

# ... CANN load, base config ...

export MOONCAKE_CONFIG_PATH="${MOONCAKE_CONFIG_PATH:-./mooncake.json}"
export ASCEND_BUFFER_POOL="${ASCEND_BUFFER_POOL:-4:8}"

# ... NPU env ...

vllm serve "$MODEL_PATH" \
    ... \
    --kv-transfer-config '{
        "kv_connector": "MooncakeConnectorStoreV1",
        "kv_role": "kv_both",
        "kv_connector_extra_config": {
            "use_layerwise": false,
            "mooncake_rpc_port": "0",
            "load_async": true,
            "register_buffer": true
        }
    }' \
    "$@"
```

### 5.4 `run_pd_disaggregated.sh`

```bash
#!/bin/bash
# =============================================================================
# <Model> <Quant> — PD Disaggregation with Mooncake
# =============================================================================
# Purpose: Separate Prefill and Decode onto different nodes.
#
# Usage:
#   KV_ROLE=kv_producer KV_PORT=30000 ENGINE_ID=0 bash run_pd_disaggregated.sh
#   KV_ROLE=kv_consumer KV_PORT=30001 ENGINE_ID=1 PORT=810X bash run_pd_disaggregated.sh
# =============================================================================
set -euo pipefail

# ... CANN load, base config ...

readonly KV_ROLE="${KV_ROLE:-kv_producer}"
readonly KV_PORT="${KV_PORT:-30000}"
readonly ENGINE_ID="${ENGINE_ID:-0}"
readonly DATA_PARALLEL_SIZE="${DATA_PARALLEL_SIZE:-2}"
readonly DATA_PARALLEL_ADDRESS="${DATA_PARALLEL_ADDRESS:-}"

export MOONCAKE_CONFIG_PATH="${MOONCAKE_CONFIG_PATH:-./mooncake.json}"
export ASCEND_BUFFER_POOL="${ASCEND_BUFFER_POOL:-4:8}"

# ... NPU env ...

SERVE_ARGS=(
    --host "$HOST" --port "$PORT"
    --served-model-name "<api-name>"
    --trust-remote-code
    --distributed-executor-backend mp
    ...
)

if [[ "$DATA_PARALLEL_SIZE" -gt 1 ]]; then
    SERVE_ARGS+=(--data-parallel-size "$DATA_PARALLEL_SIZE")
fi
if [[ -n "$DATA_PARALLEL_ADDRESS" ]]; then
    SERVE_ARGS+=(--data-parallel-address "$DATA_PARALLEL_ADDRESS")
fi

SERVE_ARGS+=(
    --kv-transfer-config "{
        \"kv_connector\": \"MooncakeLayerwiseConnector\",
        \"kv_role\": \"$KV_ROLE\",
        \"kv_port\": \"$KV_PORT\",
        \"engine_id\": \"$ENGINE_ID\",
        \"kv_connector_module_path\": \"vllm_ascend.distributed.mooncake_layerwise_connector\",
        \"kv_connector_extra_config\": {
            \"prefill\": {\"dp_size\": $DATA_PARALLEL_SIZE, \"tp_size\": $TP},
            \"decode\": {\"dp_size\": 1, \"tp_size\": $TP}
        }
    }"
)

vllm serve "$MODEL_PATH" "${SERVE_ARGS[@]}" "$@"
```

---

## 6. 检查清单

新增模型示例脚本提交前必须确认：

- [ ] 4 个基础文件存在：`run_vllm.sh`、`vllm_server.sh`、`curl_test.sh`、`README.md`
- [ ] 高级特性脚本按需存在或明确标记不适用
- [ ] 所有脚本 `chmod +x` 可执行
- [ ] `bash -n <file>.sh` 全部通过
- [ ] `shellcheck <file>.sh` 无 warning/error（SC1091 info 除外）
- [ ] `MODEL_PATH` 默认值正确
- [ ] `PORT` 不与其他模型冲突
- [ ] `SERVED_MODEL_NAME` 与 `curl_test.sh` 中 `MODEL_NAME` 一致
- [ ] MoE 模型包含 `--enable-expert-parallel`
- [ ] MTP 模型包含 `--speculative-config '{"num_speculative_tokens": 3, "method": "mtp"}'`
- [ ] GLM 系列设置 `VLLM_ASCEND_ENABLE_FLASHCOMM1=0`
- [ ] 多模态模型包含 `--language-model-only`、`--mm-encoder-tp-mode data`
- [ ] `curl_test.sh` 无 `eval`
- [ ] `vllm_server.sh` 中 `SCRIPT_DIR` 先赋值再 `readonly`
