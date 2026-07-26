# DeepSeek-V4-Flash W8A8 MTP 部署指南

> **vLLM-Ascend v0.22.1rc1+** | 端口: **8000**
> 架构: DeepseekV4ForCausalLM | 256 Experts (+1 shared) | MLA | MTP=1 | W8A8
> 原生上下文: **1,048,576** (1M) | 单节点 A3 (128G×8) 或 A2 (64G×8)
> ✅ 已验证 PASS (2026-07-21 复测) | Hybrid KV Cache ✅ | Prefix Caching ✅

DeepSeek-V4-Flash 是 DeepSeek-V4 的轻量级变体：流形约束超连接 (mHC)、混合注意力 (Compress-4/128-Attention)、DeepSeekMoE (256 专家 + 1 共享)，每 Token 激活 6 专家。

## 模型简介

| 属性 | 值 |
|------|-----|
| **架构** | DeepseekV4ForCausalLM (MoE + MLA + mHC) |
| **路由专家** | 256 (+ 1 共享)，每 Token 激活 6 |
| **隐藏维度** | 4096 |
| **网络层数** | 43 |
| **注意力头** | 64 (GQA: 1 KV head) |
| **原生上下文** | **1,048,576** (1M) |
| **量化方式** | W8A8 (`--quantization ascend`) |
| **MTP** | ✅ 1 nextn layer (method=`mtp`, tokens=1) |
| **词表大小** | 129,280 |
| **Tokenizer** | deepseek_v4 |
| **工具调用解析器** | deepseek_v4 |

### 架构注意事项

- **Hybrid KV Cache Manager**: `--no-disable-hybrid-kv-cache-manager` 支持 Compress-4/128-Attention 混合注意力。
- **Block Size 128**: MLA attention kernel 固定值。4K prefix caching 需改为 32（实验性）。
- **Prefix Caching**: 官方推荐开启。PD 分离场景的解码节点需要关闭。
- **MTP=1**: enforce_eager=true，1 token 推测解码。

### 硬件要求

| 硬件 | 配置 | 上下文 |
|------|------|--------|
| Atlas 800I A3 (128G × 8) | W8A8, TP=4 DP=4 | 1M |
| Atlas 800I A2 (64G × 8) | W8A8, TP=8 DP=1 | 64K |

### 官方文档参考

- vLLM-Ascend 模型文档: https://docs.vllm.ai/projects/ascend/zh-cn/latest/tutorials/models/DeepSeek-V4-Flash.html

## 快速开始

### 前置条件

模型权重: https://www.modelscope.cn/models/Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp

```bash
bash scripts/docker/manage_npuslim_containers.sh start --file node_list.txt
```

### 部署

```bash
# 默认 (TP=4 DP=4, 1M context, MTP on)
bash examples/deepseek-v4-flash_w8a8/vllm/run_vllm.sh

# 禁用 MTP
ENABLE_MTP=0 bash examples/deepseek-v4-flash_w8a8/vllm/run_vllm.sh

# 后台运行
nohup bash examples/deepseek-v4-flash_w8a8/vllm/run_vllm.sh > dsv4_flash.log 2>&1 &
```

### 验证

```bash
bash examples/deepseek-v4-flash_w8a8/vllm/curl_test.sh

curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"dsv4","messages":[{"role":"user","content":"Who are you?"}],"max_tokens":256}'
```

## 并行策略

| 场景 | TP | DP | NPU | 上下文 | MTP |
|------|-----|-----|-----|--------|-----|
| 默认 (高吞吐) | 4 | 4 | 16 | 1M | 1 |
| 低延迟 | 8 | 1 | 8 | 64K | 1 |

> 官方推荐 TP=4 DP=4。TP=8 DP=1 适用于低延迟场景。

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `MODEL_PATH` | Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp | 模型路径 |
| `TP` | 4 | Tensor Parallel |
| `DP` | 4 | Data Parallel |
| `ENABLE_MTP` | 1 | MTP 推测解码 |
| `MAX_MODEL_LEN` | 1048576 | 最大上下文 (1M) |
| `MAX_NUM_SEQS` | 64 | 最大并发请求 |
| `MAX_NUM_BATCHED_TOKENS` | 10240 | 每 batch 最大 token |
| `GPU_MEM_UTIL` | 0.90 | 显存利用率 |
| `HCCL_BUFFSIZE` | 1024 | HCCL 缓冲区 (MB) |
| `NIC_NAME` | — | 多节点网卡名 |

## 常见问题

### Q: 为什么默认 TP=4 DP=4？

官方推荐高吞吐配置。DP+TP 分布 256 MoE 专家到 16 NPU 最大化吞吐。

### Q: block_size 为什么是 128？

MLA attention kernel 绑定值。4K prefix caching 需改为 32（实验性，显存增加）。

### Q: 和 DeepSeek V3 区别？

V4 Flash: hidden_size 4096 vs 7168，层数 43 vs 61。推理更快，显存更小。

## 验证记录

| 时间 | 镜像 | 配置 | 结果 |
|------|------|------|------|
| 2026-07-20 | v0.22.1rc1-a3 (CANN 8.5.1) | TP=8 PP=1, MAX_MODEL_LEN=65536 | ✅ PASS |
| 2026-07-21 | 同上 | TP=8 PP=1, PORT=8000 | ✅ PASS (复测) |
