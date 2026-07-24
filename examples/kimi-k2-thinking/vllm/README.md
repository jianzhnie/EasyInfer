# Kimi-K2-Thinking 部署指南

> **vLLM-Ascend 0.22.1rc1 + CANN 8.5.1** | 端口: **8016**
> 架构: DeepseekV3ForCausalLM | 384 Experts | MoE | MLA | Thinking (文本推理)
> 硬件: 1× Atlas 800 A3 (64G × 16) | TP=16 最低要求
> Prefix Caching 默认关闭（官方推荐）| Tool Calling (kimi_k2) ✅

## 模型简介

| 属性 | 值 |
|------|-----|
| **架构** | DeepseekV3ForCausalLM |
| **路由专家** | 384 (每 Token 激活 8 专家) |
| **隐藏维度** | 7168 |
| **网络层数** | 61 |
| **MLA** | kv_lora_rank=512, q_lora_rank=1536, v_head_dim=128 |
| **原生上下文** | **262,144** |
| **量化方式** | W4A8 (compressed-tensors) |
| **MTP** | ❌ 不支持 (num_nextn_predict_layers=0) |
| **工具调用解析器** | kimi_k2 |
| **词表大小** | 163,840 |

### 官方文档参考

- GLM-5.2 官方部署文档: https://docs.vllm.ai/projects/ascend/zh-cn/latest/tutorials/models/Kimi-K2-Thinking.html
- vLLM 官方文档: https://docs.vllm.ai/en/stable/

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
# A3 单节点 (32K 上下文, TP=16)
bash examples/kimi-k2-thinking/vllm/run_vllm.sh

# 大上下文
TP=16 MAX_MODEL_LEN=131072 bash examples/kimi-k2-thinking/vllm/run_vllm.sh

# 后台运行
nohup bash examples/kimi-k2-thinking/vllm/run_vllm.sh > kimi_k2_thinking_vllm.log 2>&1 &
```

### 验证

```bash
bash examples/kimi-k2-thinking/vllm/curl_test.sh
curl http://localhost:8016/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"kimi-k2-thinking","messages":[{"role":"user","content":"Hello"}],"max_tokens":50}'
```

## 并行策略

| 场景 | TP | PP | DP | NPU | 上下文 | 状态 |
|------|-----|-----|-----|-----|--------|------|
| 单节点 A3 | 16 | 1 | 1 | 16 | 32K | 官方推荐 |
| 长上下文 | 16 | 1 | 1 | 16 | 131K | 官方支持 |

> TP=16 最低要求，不支持 Pipeline Parallelism。

## 功能验证清单

| 功能 | 状态 | 脚本 |
|------|------|------|
| 基础 Chat Completion | ✅ | `run_vllm.sh` |
| Thinking/Reasoning | ✅ | `curl_test.sh` |
| Tool Calling (kimi_k2) | ✅ | `curl_test.sh` |
| Anthropic Messages API | ✅ | `curl_test.sh` |
| MTP 投机解码 | ❌ 不支持 | 模型无 MTP 模块 |
