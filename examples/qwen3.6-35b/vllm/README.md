# Qwen3.6-35B-A3B 部署指南

> **vLLM-Ascend v0.18.0rc1+** | 端口: **8020**
> 架构: Qwen3.6 MoE (sparse) | 35B total / 3B activated | Hybrid Attention | W8A8/BF16
> 首次支持: v0.18.0rc1 | 已验证版本: v0.22.1rc1
> 单节点: A3 (64G×16) 或 A2 (64G×8) | 上下文: 128K (可扩展至 262K)
> 前缀缓存 ✅ | MTP 推测解码 ✅ | Shared Expert Overlap ✅ | FlashComm1 ✅

紧凑型稀疏 MoE 模型，35B 总参数仅 3B 激活，单节点即可部署 BF16 或 W8A8，支持长上下文和高吞吐。

## 模型简介

| 属性 | 值 |
|------|-----|
| **架构** | Qwen3.6 MoE (sparse, Hybrid Attention) |
| **参数量** | 35B total / 3B activated |
| **路由专家** | 稀疏 MoE (每 Token 激活约 3B 参数) |
| **注意力机制** | Hybrid Attention (Qwen3.5 风格) |
| **原生上下文** | **262,144** |
| **量化方式** | W8A8 (`--quantization ascend`) |
| **MTP** | ✅ `qwen3_5_mtp` 方法 (3 tokens, `ENABLE_MTP=1` 打开) |
| **PP 支持** | ❌ 单节点即够（35B 模型无需 PP） |
| **多模态** | ❌ 纯文本 |

### 架构注意事项

- **紧凑 MoE**：35B 总参数仅 3B 激活，单节点 A2 或 A3 即可部署 BF16 和 W8A8。
- **混合注意力**：Qwen3.5 风格，支持长上下文推理。根据 KV 缓存容量调整上下文长度。
- **前缀缓存默认开启**：官方推荐开启（与大多数 MoE 模型不同）。
- **高吞吐**：利用 DP 在同一节点内增加本地 DP 组（TP=2 DP=4+）。

### 硬件要求

| 硬件 | 配置 | 量化 | 推荐上下文 |
|------|------|------|-----------|
| Atlas 800I A3 (64G × 16) | TP=2 | W8A8 / BF16 | 128K |
| Atlas 800I A2 (64G × 8) | TP=2 | W8A8 / BF16 | 128K |

### 官方文档参考

- vLLM-Ascend 模型文档: https://docs.vllm.ai/projects/ascend/zh-cn/latest/tutorials/models/Qwen3.6-35B-A3B.html
- vLLM 官方文档: https://docs.vllm.ai/en/stable/

## 快速开始

### 前置条件

模型权重下载:

| 版本 | 下载 |
|------|------|
| BF16 | https://modelscope.cn/models/Qwen/Qwen3.6-35B-A3B |
| W8A8 | https://www.modelscope.cn/models/Eco-Tech/Qwen3.6-35B-A3B-w8a8 |

```bash
# 1. 启动 NPU Docker 容器
bash scripts/docker/manage_npuslim_containers.sh start --file node_list.txt
```

### 部署

```bash
# 单节点 W8A8 (默认, TP=2, 128K context)
bash examples/qwen3.6-35b/vllm/run_vllm.sh

# 高吞吐模式 (TP=2 DP=4, prefix caching)
TP=2 DP=4 DP_LOCAL=4 HCCL_IF_IP=<ip> DP_ADDRESS=<ip> bash examples/qwen3.6-35b/vllm/run_vllm.sh

# 长上下文 (262K)
MAX_MODEL_LEN=262144 bash examples/qwen3.6-35b/vllm/run_vllm.sh

# MTP 推测解码
ENABLE_MTP=1 bash examples/qwen3.6-35b/vllm/run_vllm.sh

# BF16 全精度
QUANTIZATION=none bash examples/qwen3.6-35b/vllm/run_vllm.sh

# 后台运行
nohup bash examples/qwen3.6-35b/vllm/run_vllm.sh > qwen3.6_35b.log 2>&1 &
```

### 验证

```bash
bash examples/qwen3.6-35b/vllm/curl_test.sh

curl http://localhost:8020/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.6","messages":[{"role":"user","content":"Hello"}],"max_tokens":50}'
```

## 并行策略

| 场景 | TP | DP | NPU | 上下文 | 量化 | 前缀缓存 |
|------|-----|-----|-----|--------|------|----------|
| 默认 | 2 | 1 | 2+ | 128K | W8A8 | ✅ |
| 长上下文 | 2 | 1 | 2+ | 262K | W8A8 | ✅ |
| 高吞吐 | 2 | 4+ | 8+ | 64K | W8A8 | ✅ |
| 低延迟 | 2 | 1 | 2+ | 32K | W8A8 | 按需 |
| BF16 | 2 | 1 | 2+ | 128K | BF16 | ✅ |

> 紧凑 MoE，单节点即可覆盖所有场景。高吞吐利用本地 DP 扩展。

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `MODEL_PATH` | Eco-Tech/Qwen3.6-35B-A3B-w8a8 | 模型路径 |
| `TP` | 2 | Tensor Parallel |
| `DP` | 1 | Data Parallel (高吞吐设为 4+) |
| `QUANTIZATION` | ascend | 量化 (ascend / none) |
| `ENABLE_MTP` | 0 | MTP 推测解码 (qwen3_5_mtp) |
| `MAX_MODEL_LEN` | 131072 | 最大上下文 (可设 262144) |
| `MAX_NUM_SEQS` | 32 | 最大并发请求 |
| `GPU_MEM_UTIL` | 0.90 | 显存利用率 |
| `HCCL_BUFFSIZE` | 1024 | HCCL 缓冲区 (MB) |

## 常见问题

### Q: 硬件要求？

35B 稀疏 MoE，单节点 A3 或 A2 即可部署 BF16 和 W8A8。无需多节点。

### Q: 为什么默认开启前缀缓存？

Qwen3.6 官方推荐开启前缀缓存。对于重复前缀工作负载有明显收益。如果工作负载是随机提示或内存受限，可与 `--no-enable-prefix-caching` 对比。

### Q: 如何启用 MTP 推测解码？

`ENABLE_MTP=1 bash run_vllm.sh`。使用 `qwen3_5_mtp` 方法，3 tokens。验证工作负载的 TTFT、TPOT 和吞吐量后调整。

### Q: 长上下文如何配置？

设置 `MAX_MODEL_LEN=262144`。长上下文消耗大量 KV 缓存，如果 OOM 则降低 `--max-num-seqs` 或 `--gpu-memory-utilization`。

### Q: 如何提升吞吐量？

在同一节点内增加本地 DP 组：`TP=2 DP=4 DP_LOCAL=4`。同时调整 `--max-num-batched-tokens`（高吞吐用较大值，解码密集用较小值）。
