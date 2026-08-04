# Qwen3-Dense (Qwen3-8B) 部署指南

> **vLLM-Ascend v0.23.0rc1-a3** | 端口: **8021**
> 架构: Qwen3 Dense | 8.2B total / 6.95B non-embedding | GQA (32H/8KVH) | BF16
> 已验证配置: **TP=1 PP=1 DP=1** (单节点 A2/A3 64G)
> 原生上下文: **32768** | YaRN 可扩展到 **131072**
> Reasoning ✅ | Tool Calling ✅ | 多模态 ❌ | MTP ❌
> 验证状态: ⚠️ 本次仅完成脚本/配置静态校验，未在本机启动 vLLM 实机回归

Qwen3 的稠密 8B 版本，单节点即可部署，适合本地 OpenAI API / Claude Code 接入。

## 模型简介

| 属性 | 值 |
|------|-----|
| **架构** | Qwen3 dense (Qwen3ForCausalLM, GQA) |
| **参数量** | 8.2B total / 6.95B non-embedding |
| **网络层数** | 36 |
| **注意力头** | 32 Q / 8 KV |
| **rope_theta** | 1,000,000 |
| **词表大小** | 151,936 |
| **原生上下文** | 32,768 |
| **长上下文** | 131,072 with YaRN |
| **量化方式** | BF16 本地权重；切换量化模型时可设 `QUANTIZATION=ascend` |
| **PP 支持** | ✅ 可选，默认不需要 |
| **多模态** | ❌ 纯文本 |
| **工具调用解析器** | hermes |
| **推理解析器** | qwen3 |

### 架构注意事项

- 这是 dense 模型，不需要 MoE / EP / MLA 相关配置。
- 默认 `TP=1`，单个 64G A2/A3 就够。
- `FLASHCOMM1` 和 `MLAPO` 默认保持 0；对这个 dense GQA 基线没有必要。
- `MAX_MODEL_LEN=32768` 是保守默认值；如果要做长上下文，建议显式开启 YaRN。

### 官方文档参考

- vLLM-Ascend Qwen3-Dense: https://docs.vllm.ai/projects/ascend/zh-cn/latest/tutorials/models/Qwen3-Dense.html
- Qwen3 vLLM guide: https://qwen.readthedocs.io/en/latest/deployment/vllm.html
- 本地模型路径: `/home/jianzhnie/llmtuner/hfhub/models/Qwen/Qwen3-8B`

### 硬件要求

| 硬件 | 配置 | 推荐上下文 | 备注 |
|------|------|------------|------|
| Atlas 800 A3 (64G × 16) | TP=1 | 32K | 默认配置 |
| Atlas 800 A2 (64G × 8) | TP=1 | 32K | 默认配置 |

## 快速开始

### 前置条件

```bash
# 启动容器
bash scripts/docker/manage_npuslim_containers.sh start --file /home/jianzhnie/llmtuner/llm/EasyInfer/node_list0.txt
```

### 部署

```bash
# 默认 BF16 单节点
bash examples/qwen3-dense/vllm/run_vllm.sh

# 长上下文
ENABLE_YARN=1 MAX_MODEL_LEN=131072 bash examples/qwen3-dense/vllm/run_vllm.sh

# 单节点多卡
TP=2 DISTRIBUTED_EXECUTOR_BACKEND=mp bash examples/qwen3-dense/vllm/run_vllm.sh

# 跨节点（Ray）
DISTRIBUTED_EXECUTOR_BACKEND=ray RAY_ADDRESS=<head>:6379 TP=2 bash examples/qwen3-dense/vllm/run_vllm.sh
```

> 若你显式切到 `ray` 后端，可配合 `RAY_ADDRESS=<head>:6379` 使用；这个 8B 基线不需要它。

### 验证

```bash
bash examples/qwen3-dense/vllm/curl_test.sh

curl http://localhost:8021/v1/models
curl http://localhost:8021/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3","messages":[{"role":"user","content":"Hello"}],"max_tokens":50}'
```

## 并行策略

| 场景 | TP | PP | DP | NPU | 上下文 | 状态 |
|------|----|----|----|-----|--------|------|
| 默认 | 1 | 1 | 1 | 1 | 32K | ✅ |
| 长上下文 | 1 | 1 | 1 | 1 | 131072 | ⚠️ 需 YaRN |
| 扩吞吐 | 2+ | 1 | 1 | 2+ | 32K | 可选 |

## 常见问题

### Q: 为什么默认 TP=1？

A: Qwen3-8B 的 BF16 权重在单个 64G A2/A3 上就能放下，TP=1 最简单。

### Q: 如何开启 131072 长上下文？

A: 使用 YaRN：

```bash
ENABLE_YARN=1 MAX_MODEL_LEN=131072 bash examples/qwen3-dense/vllm/run_vllm.sh
```

### Q: 需要 MLA / MTP 吗？

A: 不需要。这个模型是 dense GQA，不走 MLA 或 MTP 路径。

### Q: 为什么 `curl_test.sh` 不测 Vision？

A: 这是纯文本模型，Vision 默认跳过。

### Q: Claude Code 怎么接入？

A: 用本地 `qwen3` 服务名即可：

```bash
ANTHROPIC_BASE_URL=http://localhost:8021 \
ANTHROPIC_API_KEY=dummy \
ANTHROPIC_AUTH_TOKEN=dummy \
ANTHROPIC_DEFAULT_SONNET_MODEL=qwen3 \
ANTHROPIC_DEFAULT_HAIKU_MODEL=qwen3 \
ANTHROPIC_DEFAULT_OPUS_MODEL=qwen3 \
claude
```

## 验证记录

| 时间 | 镜像 | 节点 | 配置 | 结果 | 说明 |
|------|------|------|------|------|------|
| 2026-08-04 | `quay.io/ascend/vllm-ascend:v0.23.0rc1-a3` | 单节点 A2/A3 | TP=1 PP=1 DP=1 | ⚠️ | 脚本/路径静态校验完成，待实机回归 |
