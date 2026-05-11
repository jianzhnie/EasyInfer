# 在 Claude Code 中配置本地 vLLM 模型

本文介绍如何将 Claude Code 连接到本地部署的 vLLM 推理服务，使用自己的模型替代 Anthropic 云端 API。

## 原理

vLLM 实现了 **Anthropic Messages API**（与 Claude Code 使用的协议一致）。通过将 `ANTHROPIC_BASE_URL` 指向本地 vLLM 服务，Claude Code 会将请求发送到本地模型，而非 Anthropic 服务器。vLLM 负责将 Anthropic 格式的请求翻译为模型可处理的格式，并将响应转换回 Claude Code 期望的格式。

## 前置条件

- 已安装 [Claude Code](https://docs.anthropic.com/en/docs/claude-code)（`npm install -g @anthropic-ai/claude-code`）
- 已安装 vLLM（`pip install vllm`），版本建议 >= 0.17.1（包含 prefix caching 修复）
- 至少一块 NVIDIA GPU，显存 >= 16GB（取决于所用模型）
- 已下载支持**工具调用（Tool Calling）**的开源模型

## 第一步：启动 vLLM 服务

```bash
vllm serve Qwen/Qwen3-Coder-32B-Instruct \
  --served-model-name my-model \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --port 8000
```

关键参数说明：

| 参数 | 说明 |
|------|------|
| `--served-model-name` | 给模型取一个**不含 `/` 的名字**（Claude Code 不支持带 `/` 的模型名） |
| `--enable-auto-tool-choice` | 必须开启，启用工具调用自动选择 |
| `--tool-call-parser` | 根据模型选择对应的工具调用解析器 |
| `--port` | 服务端口，默认 8000 |
| `--host 0.0.0.0` | 如需远程访问时使用 |

### 常见模型的 tool-call-parser 对照

| 模型 | `--tool-call-parser` |
|------|---------------------|
| Qwen2.5 / Qwen3 / Qwen3-Coder | `hermes` |
| Llama 3.x / 4 | `llama` 或 `pythonic` |
| Mistral / Mistral-Large | `mistral` |
| DeepSeek V2/V3 | `deepseekv3` |
| GPT-OSS | `openai` |
| GLM-4-MoE | `glm4_moe` |
| InternLM2 | `internlm2` |
| Granite | `granite` |
| Hunyuan | `hunyuan_a13b` |

> 完整列表参见 vLLM 源码 `vllm/tool_parsers/` 目录，或查阅 [Tool Calling 文档](https://docs.vllm.ai/en/stable/features/tool_calling.html)。

### 多 GPU / 多节点部署

单机多卡：

```bash
vllm serve Qwen/Qwen3-Coder-32B-Instruct \
  --served-model-name my-model \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --tensor-parallel-size 4 \
  --port 8000
```

多节点部署：在首节点启动 ray cluster，其余节点加入后启动 vLLM，具体参见项目中 `scripts/` 目录下的集群部署脚本。

## 第二步：配置 Claude Code 连接

Claude Code 通过以下环境变量定位模型服务：

| 环境变量 | 说明 |
|----------|------|
| `ANTHROPIC_BASE_URL` | vLLM 服务地址（默认端口 8000） |
| `ANTHROPIC_API_KEY` | 任意非空值（vLLM 默认不鉴权） |
| `ANTHROPIC_AUTH_TOKEN` | 必填，任意非空值 |
| `ANTHROPIC_DEFAULT_SONNET_MODEL` | Sonnet 级别请求使用的模型名 |
| `ANTHROPIC_DEFAULT_HAIKU_MODEL` | Haiku 级别请求使用的模型名 |
| `ANTHROPIC_DEFAULT_OPUS_MODEL` | Opus 级别请求使用的模型名 |

> 三个模型变量建议都设为相同的 `--served-model-name` 值。

### 方式一：命令行临时使用

```bash
ANTHROPIC_BASE_URL=http://localhost:8000 \
ANTHROPIC_API_KEY=dummy \
ANTHROPIC_AUTH_TOKEN=dummy \
ANTHROPIC_DEFAULT_SONNET_MODEL=my-model \
ANTHROPIC_DEFAULT_HAIKU_MODEL=my-model \
ANTHROPIC_DEFAULT_OPUS_MODEL=my-model \
claude
```

### 方式二：写入 `~/.claude/settings.json`（推荐）

```json
{
  "env": {
    "ANTHROPIC_BASE_URL": "http://localhost:8000",
    "ANTHROPIC_API_KEY": "dummy",
    "ANTHROPIC_AUTH_TOKEN": "dummy",
    "ANTHROPIC_DEFAULT_SONNET_MODEL": "my-model",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL": "my-model",
    "ANTHROPIC_DEFAULT_OPUS_MODEL": "my-model"
  }
}
```

### 方式三：写入 shell 配置文件

在 `~/.zshrc` 或 `~/.bashrc` 中添加：

```bash
export ANTHROPIC_BASE_URL=http://localhost:8000
export ANTHROPIC_API_KEY=dummy
export ANTHROPIC_AUTH_TOKEN=dummy
export ANTHROPIC_DEFAULT_SONNET_MODEL=my-model
export ANTHROPIC_DEFAULT_HAIKU_MODEL=my-model
export ANTHROPIC_DEFAULT_OPUS_MODEL=my-model
```

修改后执行 `source ~/.zshrc` 生效。

## 第三步：验证连接

启动 Claude Code 后，输入一个简单的提示验证：

```
> hello, respond with "ok"
```

如果模型正常回复，说明配置成功。可以进一步测试工具调用能力：

```
> 列出当前目录下的文件
```

如果模型能正确调用工具并返回文件列表，说明 tool calling 配置正确。

## 性能优化

Claude Code 会在系统提示中注入每次请求的 hash，这会**破坏 vLLM 的 prefix caching**，导致推理速度下降约 90%。

- **vLLM > 0.17.1**：已自动修复，无需额外配置
- **vLLM <= 0.17.1**：在 `~/.claude/settings.json` 中添加 `"CLAUDE_CODE_ATTRIBUTION_HEADER": "0"`

```json
{
  "env": {
    "CLAUDE_CODE_ATTRIBUTION_HEADER": "0"
  }
}
```

其他优化建议：

- 启用 vLLM 的 `--enable-prefix-caching` 以加速重复前缀的请求
- 使用 `--gpu-memory-utilization 0.95` 最大化 GPU 显存利用率
- 对于长上下文场景，使用 `--max-model-len` 限制上下文长度以节省显存

## 替代方案：通过 LiteLLM 代理

如果模型不直接支持 Anthropic API 格式，或需要同时对接多个模型后端，可以使用 LiteLLM 作为协议转换层：

```
Claude Code → LiteLLM (端口 4000) → vLLM (端口 8000) → 本地 GPU
```

### LiteLLM 配置文件 `litellm_config.yaml`

```yaml
model_list:
  - model_name: claude-3-5-sonnet-20241022
    litellm_params:
      model: openai/my-model
      api_base: http://localhost:8000/v1
      api_key: dummy

litellm_settings:
  drop_params: true  # 丢弃 Anthropic 特有参数，避免不兼容报错
```

### 启动 LiteLLM

```bash
pip install 'litellm[proxy]'
litellm --config litellm_config.yaml --port 4000
```

### Claude Code 连接 LiteLLM

```bash
ANTHROPIC_BASE_URL=http://localhost:4000 \
ANTHROPIC_API_KEY=sk-anything \
ANTHROPIC_AUTH_TOKEN=sk-anything \
claude
```

## 推荐的本地模型

| 模型 | 参数量 | 最低显存 | 适用场景 |
|------|--------|---------|---------|
| Qwen3-Coder-32B | 32B | 24GB (BF16) / 16GB (INT4) | 编码 + 工具调用，综合推荐 |
| Qwen3-8B | 8B | 8GB+ (INT4) | 显存有限的轻量级场景 |
| DeepSeek-V3-0324 | 685B (MoE) | 多 GPU | 推理能力强，适合复杂任务 |
| GPT-OSS-120B | 120B | 多 GPU | 高质量代码生成 |
| Llama-4-Scout-17B | 17B (MoE) | 16GB+ | 平衡性能与资源消耗 |

## 常见问题排查

| 问题 | 原因 | 解决方案 |
|------|------|---------|
| `Connection refused` | vLLM 未运行或端口不匹配 | 确认 vLLM 进程存在：`curl http://localhost:8000/v1/models` |
| Tool calls 失败 | parser 配置错误 | 检查 `--tool-call-parser` 是否与模型匹配 |
| `Model not found` | 模型名不一致 | 环境变量中的名称必须与 `--served-model-name` 完全一致 |
| 推理速度极慢 | prefix caching 被破坏 | 设置 `CLAUDE_CODE_ATTRIBUTION_HEADER=0` 或升级 vLLM |
| OOM (显存不足) | 模型太大 | 使用量化版本（AWQ/GPTQ/GGUF）或减少 `--max-model-len` |
| 模型名含 `/` 报错 | Claude Code 限制 | 用 `--served-model-name` 起一个不含 `/` 的别名 |

## 参考来源

- [vLLM 官方文档 - Claude Code 集成](https://docs.vllm.ai/en/stable/serving/integrations/claude_code/)
- [How to Run Claude Code on a Local vLLM Model Using LiteLLM Proxy](https://www.roborhythms.com/how-to-run-claude-code-on-local-vllm-model/)
- [How to Run Local LLMs with Claude Code - Unsloth](https://unsloth.ai/docs/basics/claude-code)
- [Running Claude Code with Local LLMs via vLLM and LiteLLM](https://dev.to/dcruver/running-claude-code-with-local-llms-via-vllm-and-litellm-599b)
- [Claude Code on OpenShift with vLLM and Dev Spaces](https://piotrminkowski.com/2026/02/27/claude-code-on-openshift-with-vllm-and-dev-spaces/)
