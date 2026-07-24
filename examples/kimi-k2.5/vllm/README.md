# Kimi-K2.5 W4A8 部署指南

> **vLLM-Ascend 0.22.1rc1 + CANN 8.5.1** | 端口: **8017**
> 架构: KimiK25ForConditionalGeneration | 384 Experts | MoE | MLA | Vision (多模态)
> 官方推荐: **TP=4 DP=4** (A3 单节点) | Eagle3 投机解码 ✅
> Prefix Caching 默认关闭（官方推荐）| Tool Calling (kimi_k2) ✅

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
| **量化方式** | W4A8 (--quantization ascend) |
| **MTP** | ❌ (Eagle3 替代) |
| **投机解码** | ✅ Eagle3 (`lightseekorg/kimi-k2.5-eagle3`, 3 tokens) |
| **词表大小** | 163,840 |

### 官方文档参考

- 官方部署教程: https://docs.vllm.ai/projects/ascend/zh-cn/latest/tutorials/models/Kimi-K2.5.html
- vLLM 官方文档: https://docs.vllm.ai/en/stable/

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
# A3 单节点 (TP=4 DP=4, 官方推荐)
bash examples/kimi-k2.5/vllm/run_vllm.sh

# TP=8 DP=2 (高吞吐)
DP=2 TP=8 bash examples/kimi-k2.5/vllm/run_vllm.sh

# 长上下文 128K
TP=16 DP=1 MAX_MODEL_LEN=131072 bash examples/kimi-k2.5/vllm/run_vllm.sh

# 多节点 DP (A2 × 2)
TP=4 DP=4 RAY_ADDRESS=<head>:6379 bash examples/kimi-k2.5/vllm/run_vllm.sh
```

### 验证

```bash
bash examples/kimi-k2.5/vllm/curl_test.sh
curl http://localhost:8017/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"kimi-k2.5","messages":[{"role":"user","content":"Hello"}],"max_tokens":50}'
```

## 并行策略

| 场景 | TP | DP | NPU | 上下文 | 状态 |
|------|-----|-----|-----|--------|------|
| 单节点 A3 | 4 | 4 | 16 | 32K | 官方推荐 |
| 高吞吐 | 8 | 2 | 16 | 32K | 官方推荐 |
| 长上下文 | 16 | 1 | 16 | 128K | 官方推荐 |

> 官方推荐 dp4tp4 而非 dp2tp8：TP=4 时 MLA 维度对齐更好。
> 多节点 DP 支持 A2 64G × 2 节点。

## 功能验证清单

| 功能 | 状态 | 脚本 |
|------|------|------|
| 基础 Chat Completion | ✅ | `run_vllm.sh` |
| Tool Calling (kimi_k2) | ✅ | `curl_test.sh` |
| Reasoning Parser (kimi_k2) | ✅ | `run_vllm.sh` |
| Eagle3 投机解码 | ✅ | `run_vllm.sh` (内置) |
| 多模态 Vision | ✅ | `curl_test.sh` |
