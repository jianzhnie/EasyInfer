# Kimi-K2-Thinking W4A8 部署指南

> **vLLM-Ascend 0.20.2 + CANN 9.0.0** | 端口: **8016**
> 架构: DeepseekV3ForCausalLM | 384 Experts | MoE | MLA | Thinking (文本推理) | W4A8 量化
> 已验证配置: TP=8 PP=1 (单节点) | 上下文: 32K | max_position_embeddings: 262,144
> Agent 优化版: Prefix Caching ✅ | max_num_seqs=16 | Tool Calling (kimi_k2) ✅ | Anthropic Messages API ✅

## 模型简介

| 属性 | 值 |
|------|-----|
| **架构** | DeepseekV3ForCausalLM |
| **路由专家** | 384 (每 Token 激活 8 专家) |
| **隐藏维度** | 7168 |
| **网络层数** | 61 |
| **MLA** | kv_lora_rank=512, q_lora_rank=1536, v_head_dim=128 |
| **原生上下文** | **262,144** |
| **量化方式** | W4A8 (4-bit 权重 + 8-bit 激活, compressed-tensors) |
| **MTP** | ❌ 不支持 (num_nextn_predict_layers=0) |
| **PP 支持** | ✅ **支持 Pipeline Parallelism** |
| **多模态** | ❌ 纯文本 (无 Vision) |
| **推理模式** | ✅ Thinking/Reasoning |
| **词表大小** | 163,840 |
| **工具调用解析器** | kimi_k2 |

### 工具调用解析器

Kimi-K2-Thinking 的 tokenizer 使用自定义工具调用 token (`<|tool_call_begin|>`, `<|tool_call_end|>` 等)，**必须使用 `kimi_k2` parser**，不能使用 `deepseek_v3`。

| Parser | 状态 | 说明 |
|--------|------|------|
| `deepseek_v3` | ❌ 不兼容 | 报错: "could not locate tool call start/end tokens" |
| `kimi_k2` | ✅ 正确 | `KimiK2ToolParser`, 适配 Kimi 的 token 格式 |

### 官方文档参考

- vLLM 官方文档: https://docs.vllm.ai/en/stable/
- vLLM-Ascend 模型文档: https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/index.html

## 快速开始

### 前置条件

模型路径: `/home/jianzhnie/llmtuner/hfhub/models/moonshotai/Kimi-K2-Thinking`

```bash
# 1. 启动 NPU Docker 容器
bash scripts/docker/manage_npuslim_containers.sh start --file node_list.txt

# 2. 启动 Ray 集群
bash scripts/ray_cluster/start_npuslim_ray_cluster.sh start --file node_list.txt
```

### 部署

```bash
# 单节点 (32K 上下文, TP=8)
bash examples/kimi-k2-thinking/vllm/run_vllm.sh

# 2 节点 PP (大上下文)
TP=8 PP=2 MAX_MODEL_LEN=131072 bash examples/kimi-k2-thinking/vllm/run_vllm.sh

# 后台运行
nohup bash examples/kimi-k2-thinking/vllm/run_vllm.sh > kimi_k2_thinking_vllm.log 2>&1 &

# 使用传统包装器部署
bash examples/kimi-k2-thinking/vllm/vllm_server.sh
```

### 验证

```bash
# 运行测试脚本
bash examples/kimi-k2-thinking/vllm/curl_test.sh

# 手动验证
curl http://localhost:8016/v1/models
curl http://localhost:8016/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"kimi-k2-thinking","messages":[{"role":"user","content":"Hello"}],"max_tokens":50}'
```

## 并行策略

| 场景 | TP | PP | DP | NPU | 上下文 | 状态 |
|------|-----|-----|-----|-----|--------|------|
| 单节点 | 8 | 1 | 1 | 8 | 32K | ⚠️ 待验证 |
| 2 节点 PP | 8 | 2 | 1 | 16 | 131K | ⚠️ 待验证 |
| 多节点扩展 | 8 | 4 | 2 | 64 | 131K | ⚠️ 待验证 |

> Kimi-K2-Thinking **支持 Pipeline Parallelism**，适合多节点扩展。

## 环境变量

> 完整环境变量说明见 [prompts/vllm_env_vars.md](../../../prompts/vllm_env_vars.md)。
> Claude Code 集成方式见 [prompts/vllm-prompt.md](../../../prompts/vllm-prompt.md)。
## 功能验证清单

### 基础功能

| 功能 | 状态 | 脚本 |
|------|------|------|
| 基础 Chat Completion | ⚠️ 待验证 | `run_vllm.sh` |
| Thinking/Reasoning | ⚠️ 待验证 | `curl_test.sh` |
| Tool Calling (kimi_k2) | ⚠️ 待验证 | `curl_test.sh` |
| Anthropic Messages API | ⚠️ 待验证 | `curl_test.sh` |
| MTP 投机解码 | ❌ 不支持 | 模型无 MTP 模块 |

## 常见问题

### Q: Kimi-K2-Thinking 和 Kimi-K2.6 有什么区别？

A: Kimi-K2-Thinking 是纯文本推理模型 (Thinking)，无 Vision Transformer，专注于推理能力。Kimi-K2.6 支持多模态 (Vision + Text)。文本骨干均基于 DeepSeek V3 架构 (384 专家)。

### Q: Kimi-K2-Thinking 支持 MTP 投机解码吗？

A: 不支持。模型 config 中 `num_nextn_predict_layers=0`，无 MTP 模块。

### Q: 384 专家对部署有什么影响？

A: 专家数更多 (384 vs 256)，EP_SIZE 需能整除 384 (推荐 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 384)。384 专家的 MoE 层参数量更大，需要更大的 SWAP_SPACE。
