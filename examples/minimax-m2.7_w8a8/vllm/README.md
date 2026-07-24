# MiniMax-M2.7 W8A8 QuaRot 部署指南

> **vLLM-Ascend 0.22.1rc1 + CANN 8.5.1** | 端口: **8004**
> 架构: MiniMaxM2ForCausalLM | 256 Experts | MoE | W8A8 QuaRot
> 官方推荐: **TP=4 DP=4** (A3) / TP=8 (A2) | Eagle3 投机解码 ✅
> BALANCE_SCHEDULING=0（官方推荐）| Async Scheduling ✅

## 模型简介

| 属性 | 值 |
|------|-----|
| **架构** | MiniMaxM2ForCausalLM (MoE) |
| **路由专家** | 256 |
| **隐藏维度** | 6144 |
| **网络层数** | 80 |
| **原生上下文** | 204,800 |
| **量化方式** | W8A8 QuaRot (--quantization ascend) |
| **投机解码** | ✅ Eagle3 (3 tokens, enforce_eager) |
| **工具调用解析器** | minimax_m2 / minimax_m2_append_think |
| **词表大小** | 100,672 |

### 官方文档参考

- 官方部署教程: https://docs.vllm.ai/projects/ascend/zh-cn/latest/tutorials/models/MiniMax-M2.html
- vLLM 官方文档: https://docs.vllm.ai/en/stable/

## 快速开始

### 前置条件

模型路径: `/home/jianzhnie/llmtuner/hfhub/models/Eco-Tech/MiniMax-M2.7-w8a8-QuaRot`

```bash
# 1. 启动 NPU Docker 容器
bash scripts/docker/manage_npuslim_containers.sh start --file node_list.txt

# 2. 启动 Ray 集群
bash scripts/ray_cluster/start_npuslim_ray_cluster.sh start --file node_list.txt
```

### 部署

```bash
# A3 单节点 (TP=4 DP=4, 官方推荐)
bash examples/minimax-m2.7_w8a8/vllm/run_vllm.sh

# A2 单节点 (TP=8)
TP=8 DP=1 bash examples/minimax-m2.7_w8a8/vllm/run_vllm.sh

# 高吞吐 (TP=8 DP=2)
TP=8 DP=2 bash examples/minimax-m2.7_w8a8/vllm/run_vllm.sh

# 长上下文 128K (DSA CP)
TP=8 DECODE_CP=2 MAX_MODEL_LEN=138000 bash examples/minimax-m2.7_w8a8/vllm/run_vllm.sh
```

### 验证

```bash
bash examples/minimax-m2.7_w8a8/vllm/curl_test.sh
curl http://localhost:8004/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"MiniMax-M2.7","messages":[{"role":"user","content":"Hello"}],"max_tokens":50}'
```

## 并行策略

| 场景 | TP | DP | NPU | 上下文 | 状态 |
|------|-----|-----|-----|--------|------|
| 单节点 A3 | 4 | 4 | 16 | 40K | 官方推荐 |
| 单节点 A2 | 8 | 1 | 8 | 32K | 可用 |
| 高吞吐 A3 | 8 | 2 | 16 | 32K | 官方推荐 |
| 长上下文 128K | 8 | 1 | 8 | 138K | DSA CP |

> A3: `additional_config` 含 enable_cpu_binding, enable_fused_mc2, enable_flashcomm1, weight_nz_mode。
> Eagle3 投机解码默认 3 tokens。

## 功能验证清单

| 功能 | 状态 | 脚本 |
|------|------|------|
| 基础 Chat Completion | ✅ | `run_vllm.sh` |
| Tool Calling (minimax_m2) | ✅ | `curl_test.sh` |
| Reasoning Parser | ✅ | `run_vllm.sh` |
| Eagle3 投机解码 | ✅ | `run_vllm.sh` (内置) |
| Async Scheduling | ✅ | `run_vllm.sh` (内置) |

## 验证记录

| 时间 | 镜像 | 节点 | 配置 | 结果 |
|------|------|------|------|------|
| 2026-07-20 | `quay.io/ascend/vllm-ascend:v0.22.1rc1-a3` | pair3: 10.42.11.200/201 | PORT=8004 | ✅ PASS |
| 2026-07-21 | 同上 | pair3: 10.42.11.200/201 | PORT=8004 | ✅ PASS (复测) |
