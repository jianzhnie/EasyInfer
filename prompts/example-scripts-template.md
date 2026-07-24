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

## 3. 检查清单

新增模型示例脚本提交前必须确认：

- [ ] 3 个基础文件存在：`run_vllm.sh`、`curl_test.sh`、`README.md`
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
