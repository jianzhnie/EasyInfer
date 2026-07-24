# EasyInfer 示例脚本模板与规范

本文件定义 `examples/<model>/vllm/` 目录下 Shell 脚本的统一格式。所有新模型示例脚本必须按此模板生成。

通用 Shell 规范（缩进、引号、函数命名等）见 CLAUDE.md，本文档仅描述模板特有的格式要求。

> **相关文档**：
> - [vllm_env_vars.md](vllm_env_vars.md) — 环境变量完整参考（脚本中所有可用变量及其含义）
> - [example-readme-template.md](example-readme-template.md) — 模型 README 模板（与脚本配套的文档模板）
> - [vllm-prompt.md](vllm-prompt.md) — 模型部署工作流（如何使用生成的脚本进行部署）

## 1. 文件结构

每个模型在 `examples/<model>/vllm/` 下至少包含以下 3 个文件：

```
examples/<model_dir>/vllm/
├── run_vllm.sh       ← 直接 vllm serve 部署
├── curl_test.sh      ← API 功能测试
└── README.md         ← 部署文档（见 prompts/example-readme-template.md）
```

### 1.1 文件头

每个直接执行的脚本必须以统一横幅开头：

```bash
#!/bin/bash
# =============================================================================
# <模型名> <量化> — <一句话用途>
# =============================================================================
# Architecture: <Arch> | <Experts> Experts | <MoE/MLA/Dense/...>
# Default: TP=<tp> PP=1 (single-node)
#
# Usage:
#   bash <script>.sh
#   VAR=value bash <script>.sh
#
# Reference:
#   <vLLM-Ascend 官方文档链接>
# =============================================================================
```

### 1.2 CANN 环境加载

CANN 环境变量必须在 `set +u` / `set -u` 之间加载：

```bash
set -euo pipefail

set +u
source /usr/local/Ascend/cann/set_env.sh
[[ -f /usr/local/Ascend/nnal/atb/set_env.sh ]] && source /usr/local/Ascend/nnal/atb/set_env.sh
set -u
```

### 1.3 变量声明规范

| 类型 | 命名 | 声明方式 | 示例 |
|------|------|----------|------|
| 可覆盖配置 | `UPPER_SNAKE_CASE` | `${VAR:-default}` | `TP="${TP:-8}"` |
| 本地常量 | `UPPER_SNAKE_CASE` | `readonly` | `readonly BASE_MODEL_PATH="..."` |
| 函数局部变量 | `snake_case` | `local` | `local elapsed` |

---

## 2. `run_vllm.sh` 模板

```bash
#!/bin/bash
# =============================================================================
# <Model> <Quant> — Direct vllm serve deployment
# =============================================================================
# Architecture: <Arch> | <Experts> Experts | <MoE/MLA/Dense/...>
# Default: TP=<tp> PP=1 (single-node)
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
readonly DP="${DP:-1}"
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

# Fallback variables (older versions)
export VLLM_ASCEND_BALANCE_SCHEDULING=1
export VLLM_ASCEND_ENABLE_FLASHCOMM1=<0_OR_1>
export VLLM_ASCEND_ENABLE_MLAPO=<0_OR_1>

# v0.20.2+ additional_config
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
    --seed 1024 \
    "$@"

# --- 模型特定参数（按需取消注释）---
# MoE 模型:
#   --enable-expert-parallel
# MTP 模型:
#   --speculative-config '{"num_speculative_tokens": 3, "method": "mtp"}'
# GLM 系列工具调用:
#   --enable-auto-tool-choice
#   --tool-call-parser glm47
#   --reasoning-parser glm45
# Kimi 系列工具调用:
#   --enable-auto-tool-choice
#   --tool-call-parser kimi_k2
# Kimi 多模态纯文本场景:
#   --language-model-only
#   --mm-encoder-tp-mode data
#   --allowed-local-media-path /home/jianzhnie/llmtuner/
# DP 场景:
#   --data-parallel-size "$DP"
```

### 2.1 模型特定参数指南

按模型架构/系列选择对应的参数组合：

| 特征 | 参数 | 适用模型系列 |
|------|------|-------------|
| MoE 架构 | `--enable-expert-parallel` | DeepSeek, LongCat, GLM, Kimi-K2.6, MiniMax |
| MTP 投机解码 | `--speculative-config '{"num_speculative_tokens": 3, "method": "mtp"}'` | GLM-5, GLM-5.1, DeepSeek-V4 |
| NPU 不兼容 FLASHCOMM1 | `VLLM_ASCEND_ENABLE_FLASHCOMM1=0` | GLM 系列 |
| MLA 算子优化 | `VLLM_ASCEND_ENABLE_MLAPO=1` | MLA 架构 (GLM, Kimi, DeepSeek) |
| 工具调用 (GLM) | `--tool-call-parser glm47 --reasoning-parser glm45 --enable-auto-tool-choice` | GLM-5, GLM-5.1 |
| 工具调用 (Kimi) | `--tool-call-parser kimi_k2 --enable-auto-tool-choice` | Kimi-K2 |
| 工具调用 (MiniMax) | `--tool-call-parser minimax_m2 --enable-auto-tool-choice` | MiniMax-M2 |
| 多模态 (Kimi) | `--language-model-only --mm-encoder-tp-mode data --allowed-local-media-path ...` | Kimi-K2 多模态 |

### 2.2 端口分配

| 模型 | 端口 |
|------|------|
| DeepSeek-V4-Flash | 8000 |
| GLM-5 / GLM-5-W4A8 | 8001 |
| GLM-5.1-W4A8 | 8002 |
| Kimi-K2.6-W4A8 | 8003 |
| MiniMax-M2.7 / W8A8 | 8004 |
| DeepSeek-V4-Pro | 8005 |
| MiniMax-M2.5 | 8006 |
| GLM-5.2-W8A8 | 8007 |
| LongCat | 8010 |
| GLM-5-W8A8 | 8011 |
| GLM-5.1-W8A8 | 8012 |
| Kimi-K2.7-Code-W4A8 | 8013 |
| MiniMax-M3-W8A8 | 8014 |
| Step-3.7-Flash-W8A8 | 8015 |
| Kimi-K2-Thinking | 8016 |
| Kimi-K2.5 | 8017 |
| Qwen3-235B | 8018 |
| LongCat-2Layer | 8300 |
| <新模型> | 按顺序递增，避免冲突 |

---

## 3. 检查清单

新增模型示例脚本提交前必须确认：

- [ ] 3 个基础文件存在：`run_vllm.sh`、`curl_test.sh`、`README.md`
- [ ] 所有脚本 `chmod +x` 可执行
- [ ] `bash -n <file>.sh` 全部通过
- [ ] `shellcheck <file>.sh` 无 warning/error（SC1091 info 除外）
- [ ] `MODEL_PATH` 默认值正确
- [ ] `PORT` 不与其他模型冲突（见端口分配表）
- [ ] `SERVED_MODEL_NAME` 与 `curl_test.sh` 中 `MODEL_NAME` 一致
- [ ] MoE 模型：包含 `--enable-expert-parallel`（MoE 但不需要 EP 的模型除外）
- [ ] MTP 模型：包含 `--speculative-config`
- [ ] GLM 系列：`VLLM_ASCEND_ENABLE_FLASHCOMM1=0`
- [ ] 多模态模型：包含 `--language-model-only`、`--mm-encoder-tp-mode data`
