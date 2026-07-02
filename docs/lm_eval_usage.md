# run_lmeval.sh 使用指南

## 概述

`tools/eval/run_lmeval.sh` 是 lm-evaluation-harness (lm_eval ≥ 0.4.x) 的 Bash 封装脚本，支持三种后端模式。

```
tools/eval/run_lmeval.sh [MODEL_PATH] [OPTIONS]
```

MODEL_PATH 可作为位置参数或 `--model-path VALUE` 传入。

## 前置依赖

```bash
conda activate llmeval
pip install lm-eval[api]          # API 后端需要 [api] 扩展
```

## 任务发现

```bash
# 列出所有可用任务
lm-eval ls tasks

# 列出任务分组
lm-eval ls groups

# 按关键词筛选
lm-eval ls tasks | grep -i mmlu
lm-eval ls tasks | grep -i gsm

# 验证任务配置是否正确
lm-eval validate --tasks mmlu
lm-eval validate --tasks gsm8k,hellaswag,arc_easy
```

## 三种后端模式

| 后端 | 参数 | 适用场景 |
|------|------|---------|
| `vllm` | `--backend vllm` | 直接加载模型，最快，需 GPU/NPU |
| `hf` | `--backend hf` | HuggingFace 原生加载，用于对比 |
| `api` | `--backend api` | 连接已部署的 vLLM/OpenAI 兼容服务 |

### 模式 1：vLLM 后端（直接加载）

绕过服务端，在当前进程加载模型。适用于单机测试。

```bash
run_lmeval.sh /path/to/model \
    --backend vllm \
    --tasks wikitext \
    -d 0 -t 1
```

### 模式 2：API 后端（连接服务端）⭐ 最常用

需要先部署模型服务，然后通过 OpenAI 兼容 API 评测。

```bash
# 步骤 1：部署服务
bash tools/serve/deploy_vllm.sh /path/to/model -d 0 -t 1

# 步骤 2：评测
run_lmeval.sh /path/to/model \
    --backend api \
    --model-name my-model \
    --port 8000 \
    --tasks mmlu \
    --fewshot 5 \
    --max-model-len 4096
```

`model-name` 和 `model-path` 可分离：
- `model-name`：发给 API 的模型 ID（出现在请求体中）
- `model-path`：本地 tokenizer 路径（用于本地分词）

两者不同时，脚本自动将 `model-name` 映射为 API 请求中的 `model` 字段，`model-path` 映射为本地 tokenizer。这在以下场景很有用：
- 远程 API（如 DeepSeek）模型名与本地 tokenizer 路径不同
- 部署服务的模型 ID 与本地 checkpoint 路径不同

### 模式 3：HF 后端（HuggingFace 原生）

```bash
run_lmeval.sh /path/to/model \
    --backend hf \
    --tasks wikitext \
    -d 0
```

## 完整参数列表

### 基础参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `MODEL_PATH` | 必填 | 位置参数或 `--model-path VALUE` |
| `--model-path VALUE` | — | 显式指定模型路径。API 模式下还可作为本地 tokenizer 路径 |
| `--backend` | `vllm` | 后端类型：`vllm` / `hf` / `api` |
| `--tasks` | `wikitext` | 任务列表，逗号或空格分隔 |
| `--fewshot` / `--num-fewshot` | `0` | Few-shot 示例数 |
| `--batch-size` | `auto` | 批次大小：`auto` / `auto:N` / 整数。API 后端内部默认 1，不建议修改 |
| `--max-batch-size` | — | auto 模式下的批次上限 |
| `--output-dir` | `outputs/benchmark/lmeval` | 结果输出目录 |
| `--limit` | — | 每任务限制样本数（整数或小数，如 `100` 或 `0.1`） |
| `--log-samples` | — | 保存模型输出到结果文件中 |
| `--seed` | — | 随机种子，如 `42` 或 `0,None,8,52`（4 值格式：`seed,numpy_seed,torch_seed,fewshot_seed`）。未指定时由 lm-eval 决定默认值 |

### 模型 / 硬件参数（vllm / hf）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `-d, --devices` | `0` | 设备 ID，如 `0,1,2,3` |
| `-t, --tp` | `1` | 张量并行度 |
| `--device` | 自动检测 | 覆盖设备类型：`cuda` / `npu` / `cpu`，也可带索引如 `cuda:0`、`npu:1` |
| `--max-model-len` | `4096` | 上下文总长度（prompt + 生成）。API 模式映射为 `max_length`；vLLM 映射为 `max_model_len`；HF 映射为 `max_length` |

### vLLM 专属

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--gpu-memory` | `0.8` | GPU/NPU 显存利用率 |
| `-q, --quantization [TYPE]` | — | 量化方法。不传 TYPE 时自动设为 `ascend`（NPU） |
| `-ep, --enable-expert-parallel` | — | MoE 模型专家并行 |
| `--compilation-config` | — | 编译配置 JSON（NPU 场景），如 `'{"cudagraph_mode":"FULL_DECODE_ONLY"}'` |
| `--enforce-eager` | — | 禁用 CUDA graph，使用 eager 模式 |
| `--hccl-port` | `60000` | NPU HCCL 通信端口（仅 NPU 有效） |

### HF 专属

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `-d, --devices` | `0` | 设备 ID |
| `-t, --tp` | `1` | 张量并行度（>1 时启用 `parallelize=True`） |

> HF 后端不支持 `max_gen_toks` model_arg，需使用 `--gen-kwargs` 替代。

### API 专属

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--port` | `8080` | 服务端口 |
| `--url` | `http://127.0.0.1:PORT/v1/completions` | 完整 API 地址（设置后覆盖 `--port`） |
| `--model-name` | `MODEL_PATH` | 发给 API 的模型名。与 MODEL_PATH 不同时，MODEL_PATH 用作本地 tokenizer 路径 |
| `--num-concurrent` | `1` | 并发请求数。**2-4x 推理加速，推荐设 4** |
| `--max-retries` | `3` | 请求失败重试次数（lm_eval 内部默认） |
| `--chat` | — | 使用 `/v1/chat/completions` + `local-chat-completions` 模型 |
| `--apply-chat-template [TEMPLATE]` | — | 应用 chat template。可不传值（使用模型默认）或指定模板名（如 `llama3`） |

> `--chat` + `--apply-chat-template` 适用于 `mmlu_generative` 等生成式任务。
> 注意：`mmlu`（loglikelihood）不能用这两个参数，需要改用 `mmlu_generative` 任务。

### 生成控制

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--max-gen-toks` | 不设（后端默认 256） | 每样本最大生成 token 数（vllm / api）。HF 不支持 |
| `--gen-kwargs` | — | 额外生成参数，逗号分隔：`'temperature=0.8,max_gen_toks=512'` |

### 网络 / 认证

| 参数 | 说明 |
|------|------|
| `--api-key-file PATH` | 从文件读取 API key 设置 `OPENAI_API_KEY`（推荐 chmod 600） |
| `--verify-certificate` | 启用 SSL 证书验证（默认） |
| `--no-verify-certificate` | 禁用 SSL 证书验证 |
| `--timeout SECONDS` | 请求超时秒数（lm_eval 内部默认 300） |

### 缓存 / 离线

| 参数 | 说明 |
|------|------|
| `--use-cache PATH` | 缓存模型响应到指定路径，避免重复推理 |
| `--offline` | 离线模式：自动设置 `HF_DATASETS_OFFLINE=1` 和 `TRANSFORMERS_OFFLINE=1` |

### 其他

| 参数 | 说明 |
|------|------|
| `--trust-remote-code` | 允许执行 HF Hub 远程代码（同时设置顶层 flag 和 model_arg） |
| `-h, --help` | 显示帮助 |

## 常用任务示例

### 基础评测

```bash
# wikitext 困惑度（快速验证）
run_lmeval.sh model --backend vllm --tasks wikitext -d 0

# 多任务批量评测
run_lmeval.sh model --backend vllm \
    --tasks arc_challenge,arc_easy,boolq,hellaswag,openbookqa,piqa,winogrande \
    -d 0,1 -t 2
```

### MMLU（知识评测，loglikelihood）

```bash
# 基础用法：5-shot，串行
run_lmeval.sh /path/to/model \
    --backend api --model-name my-model --port 8000 \
    --tasks mmlu --fewshot 5 --max-model-len 4096

# 推荐用法：加并发加速
run_lmeval.sh /path/to/model \
    --backend api --model-name my-model --port 8000 \
    --tasks mmlu --fewshot 5 --max-model-len 4096 \
    --num-concurrent 4

# 快速验证：限制样本数
run_lmeval.sh /path/to/model \
    --backend api --model-name my-model --port 8000 \
    --tasks mmlu --fewshot 5 --limit 50
```

### GSM8K（数学推理，需要生成）

```bash
run_lmeval.sh /path/to/model \
    --backend api --model-name my-model --port 8000 \
    --tasks gsm8k --fewshot 5 \
    --max-model-len 4096 --max-gen-toks 512
```

### 生成类任务 + Chat Template

```bash
# mmlu_generative 替代 mmlu（适用于 chat 模型）
run_lmeval.sh /path/to/model \
    --backend api --model-name my-chat-model --port 8000 \
    --tasks mmlu_generative --fewshot 5 \
    --max-model-len 4096 \
    --chat --apply-chat-template

# 使用指定模板名
run_lmeval.sh /path/to/model \
    --backend api --model-name my-chat-model --port 8000 \
    --tasks mmlu_generative --fewshot 5 \
    --chat --apply-chat-template llama3
```

### 远程 API

```bash
export OPENAI_API_KEY=sk-xxx
run_lmeval.sh deepseek-chat \
    --backend api \
    --url https://api.deepseek.com/v1/completions
```

### 离线 / 缓存

```bash
# 离线模式（所有数据集已预下载）
run_lmeval.sh model --backend api --port 8000 --tasks mmlu --offline

# 缓存响应（第二次跑跳过已缓存的推理）
run_lmeval.sh model --backend api --port 8000 --tasks mmlu \
    --use-cache .eval_cache/
```

### CEVAL（中文评测）

```bash
run_lmeval.sh /path/to/model \
    --backend api --model-name my-model --port 8000 \
    --tasks ceval-valid --fewshot 5 --max-model-len 4096
```

### 使用 API Key

```bash
run_lmeval.sh deepseek-chat \
    --backend api \
    --url https://api.deepseek.com/v1/completions \
    --api-key sk-xxx
```

## 示例脚本集成

项目提供预配置脚本，可直接运行或通过环境变量覆盖：

### `examples/lm_eval.sh`（通用模板）

```bash
bash examples/lm_eval.sh

# 环境变量覆盖
MODEL_PATH=/data/model TASKS=mmlu,gsm8k BACKEND=vllm bash examples/lm_eval.sh
```

### `examples/longcat/lm_eval.sh`（LongCat 专用）

```bash
bash examples/longcat/lm_eval.sh

# 环境变量覆盖
TASKS=gsm8k MAX_MODEL_LEN=8192 bash examples/longcat/lm_eval.sh
```

预置默认值：

| 变量 | 值 |
|------|-----|
| `MODEL_PATH` | `.../LongCat-Flash-Chat-combined` |
| `MODEL_NAME` | `longcat-flash` |
| `PORT` | `8000` |
| `TASKS` | `mmlu` |
| `FEWSHOT` | `5` |
| `BACKEND` | `api` |
| `MAX_MODEL_LEN` | `4096` |
| `NUM_CONCURRENT` | `4`（硬编码） |

## `max_model_len` 和 `max_gen_toks` 的兼容处理

### 核心公式

```
prompt_length + max_gen_toks ≤ max_model_len
```

### 各后端 `max_gen_toks` 默认值

| Backend | 默认 | 说明 |
|---------|------|------|
| API | 256 | `TemplateAPI` 参数 |
| vLLM | 256 | `VLLM` 参数 |
| HF | **不支持** | 使用 `--gen-kwargs 'max_gen_toks=N'` |

`run_lmeval.sh` 不设默认值，未指定 `--max-gen-toks` 时由各后端自行决定。

### 超长处理

lm_eval 自动左截断超长 prompt，**不会报错崩溃**。但截断过多会失真——看到 `"left truncated"` 警告时考虑增大 `max_model_len`。

### API 模式双层限制

```
服务端 MAX_MODEL_LEN（硬上限）
  └─ run_lmeval.sh --max-model-len（必须 ≤ 服务端）
       └─ prompt + max_gen_toks（实际使用）
```

**必须**：`--max-model-len` ≤ 服务端部署值，否则请求被拒绝。

### 按任务推荐配置

| 任务 | max_model_len | max_gen_toks | 说明 |
|------|--------------|-------------|------|
| wikitext | 4096 | — | loglikelihood，不生成 |
| MMLU (5-shot) | 4096 | — | loglikelihood，prompt ~1-2K |
| CEVAL (5-shot) | 4096 | — | 中文 MMLU 等价 |
| GSM8K (5-shot) | 4096 | 512 | CoT 需要输出空间 |
| humaneval | 4096 | 1024 | 代码生成 |
| RULER | 32768 | 128 | 长上下文 |

## 结果输出

结果保存在 `--output-dir` 下，格式 `{TASKS}_{TIMESTAMP}/`，每个子任务一个 JSON 文件：

```
outputs/benchmark/lmeval/mmlu_20260702_103847/
├── mmlu_abstract_algebra.json
├── mmlu_anatomy.json
├── ...
└── results.json              # 聚合结果
```

## 常见问题

### API 模式报 "Server not responding"

```bash
curl http://127.0.0.1:8000/health       # 检查服务是否存活
curl http://127.0.0.1:8000/v1/models    # 检查模型列表
```

### 推理太慢

1. **加并发**：`--num-concurrent 4`（2-4x 加速）
2. **限样本**：`--limit 100` 快速验证
3. **减少 fewshot**：`--fewshot 0` 最快但分数不准

### MMLU 全量耗时估算

| 模式 | 样本数 | 预计耗时 |
|------|--------|---------|
| API 串行 (`num_concurrent=1`) | ~14K | 1-3 小时 |
| API 并发 (`num_concurrent=4`) | ~14K | 20-40 分钟 |
| vLLM 直接 (`batch_size=auto`) | ~14K | 10-20 分钟 |

### HF 后端设置 max_gen_toks 无效

HF 后端（`HFLM`）没有 `max_gen_toks` 参数，改用：
```bash
--gen-kwargs 'max_gen_toks=512'
```

### 模型需要 trust_remote_code

```bash
run_lmeval.sh model --backend vllm --trust-remote-code --tasks wikitext
```

对于自定义模型（如 LongCat、GLM），此参数是必需的，否则 tokenizer 加载失败。

### 日志中出现 "left truncated"

prompt 超长被截断。增大 `--max-model-len` 或减少 `--fewshot`。
