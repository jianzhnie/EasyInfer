# Kimi-K2.5 W4A8 部署指南

> **vLLM-Ascend 0.20.2 + CANN 9.0.0** | 端口: **8017**
> 架构: KimiK25ForConditionalGeneration | 384 Experts | MoE | MLA | Vision (多模态) | W4A8 量化
> 文本骨干: DeepseekV3ForCausalLM | max_position_embeddings: 262,144
> Agent 优化版: Prefix Caching ✅ | max_num_seqs=16 | Tool Calling (kimi_k2) ✅ | Anthropic Messages API ✅

## 模型简介

| 属性 | 值 |
|------|-----|
| **架构** | KimiK25ForConditionalGeneration (Vision + Text) |
| **文本骨干** | DeepseekV3ForCausalLM |
| **路由专家** | 384 (每 Token 激活 8 专家) |
| **隐藏维度** | 7168 |
| **网络层数** | 61 |
| **MLA** | kv_lora_rank=512, q_lora_rank=1536, v_head_dim=128 |
| **原生上下文** | **262,144** |
| **量化方式** | W4A8 (4-bit 权重 + 8-bit 激活, compressed-tensors) |
| **MTP** | ❌ 不支持 (num_nextn_predict_layers=0) |
| **PP 支持** | ✅ **支持 Pipeline Parallelism** |
| **多模态** | ✅ Vision Transformer (mm_hidden_size=1152, patch_size=14) |
| **词表大小** | 163,840 |
| **工具调用解析器** | kimi_k2 |

### 架构注意事项

Kimi-K2.5 使用 `KimiK25ForConditionalGeneration` 架构，包含独立的 Vision Transformer 和 DeepseekV3ForCausalLM 文本骨干。通过 `--language-model-only` 可以跳过 Vision Encoder 加载，适合纯文本 Agent 场景。

### 工具调用解析器

Kimi-K2.5 的 tokenizer 使用自定义工具调用 token (`<tool>`, `</tool>` 等)，**必须使用 `kimi_k2` parser**，不能使用 `deepseek_v3`。

| Parser | 状态 | 说明 |
|--------|------|------|
| `deepseek_v3` | ❌ 不兼容 | 报错: "could not locate tool call start/end tokens" |
| `kimi_k2` | ✅ 正确 | `KimiK2ToolParser`, 适配 Kimi 的 token 格式 |

### 官方文档参考

- vLLM 官方文档: https://docs.vllm.ai/en/stable/
- vLLM-Ascend 模型文档: https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/index.html

## 快速开始

### 前置条件

模型路径: `/home/jianzhnie/llmtuner/hfhub/models/moonshotai/Kimi-K2.5`

```bash
# 1. 启动 NPU Docker 容器
bash scripts/docker/manage_npuslim_containers.sh start --file node_list.txt

# 2. 启动 Ray 集群
bash scripts/ray_cluster/start_npuslim_ray_cluster.sh start --file node_list.txt
```

### 部署

```bash
# 单节点 (32K 上下文, TP=8, 文本模式)
bash examples/kimi-k2.5/vllm/run_vllm.sh

# 2 节点 PP (大上下文)
TP=8 PP=2 MAX_MODEL_LEN=131072 bash examples/kimi-k2.5/vllm/run_vllm.sh

# 后台运行
nohup bash examples/kimi-k2.5/vllm/run_vllm.sh > kimi_k25_vllm.log 2>&1 &
```

### 验证

```bash
# 运行测试脚本
bash examples/kimi-k2.5/vllm/curl_test.sh

# 手动验证
curl http://localhost:8017/v1/models
curl http://localhost:8017/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"kimi-k2.5","messages":[{"role":"user","content":"Hello"}],"max_tokens":50}'
```

## 并行策略

| 场景 | TP | PP | DP | NPU | 上下文 | 状态 |
|------|-----|-----|-----|-----|--------|------|
| 单节点 | 8 | 1 | 1 | 8 | 32K | ⚠️ 待验证 |
| 2 节点 PP | 8 | 2 | 1 | 16 | 131K | ⚠️ 待验证 |
| 多节点扩展 | 8 | 4 | 2 | 64 | 131K | ⚠️ 待验证 |

> Kimi-K2.5 **支持 Pipeline Parallelism**，适合多节点扩展。

## 环境变量

> 完整环境变量说明见 [prompts/vllm_env_vars.md](../../../prompts/vllm_env_vars.md)。
> Claude Code 集成方式见 [prompts/vllm-prompt.md](../../../prompts/vllm-prompt.md)。
## 功能验证清单

### 基础功能

| 功能 | 状态 | 脚本 |
|------|------|------|
| 基础 Chat Completion | ⚠️ 待验证 | `run_vllm.sh` |
| Tool Calling (kimi_k2) | ⚠️ 待验证 | `curl_test.sh` |
| Anthropic Messages API | ⚠️ 待验证 | `curl_test.sh` |
| 多模态 Vision | ⚠️ 待验证 | `curl_test.sh` |
| MTP 投机解码 | ❌ 不支持 | 模型无 MTP 模块 |

## 常见问题

### Q: Kimi-K2.5 和 Kimi-K2.6 有什么区别？

A: 两者均基于 KimiK25ForConditionalGeneration 架构，支持多模态 (Vision + Text)。K2.5 是上一代版本，K2.6 是最新版本。部署脚本通用，仅 `MODEL_PATH` 和 `SERVED_MODEL_NAME` 不同。

### Q: --language-model-only 是什么？

A: 仅加载语言模型部分，跳过 Vision Encoder，适合纯文本场景和 Agent 使用，节省显存。

### Q: 多模态如何使用？

A: 通过 `/v1/chat/completions` 传入 image 类型的 content。视觉 token 占用上下文窗口，建议预留 20-30%。纯文本 Agent 使用时视觉组件不激活。

### Q: PP 如何工作？

A: Kimi-K2.5 支持 Pipeline Parallelism，每层分配到不同节点。TP=8 PP=2 表示 2 个 PP stage，每个 stage 在 8 张 NPU 上运行 TP。

### Q: 384 专家对部署有什么影响？

A: 专家数更多 (384 vs 256)，EP_SIZE 需能整除 384 (推荐 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 384)。384 专家的 MoE 层参数量更大，需要更大的 SWAP_SPACE。
