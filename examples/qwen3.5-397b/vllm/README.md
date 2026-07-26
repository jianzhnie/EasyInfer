# Qwen3.5-397B-A17B 部署指南

> **vLLM-Ascend v0.17.0rc1+** | 端口: **8019**
> 架构: Qwen3.5 MoE | 397B total / 17B activated | W8A8/BF16
> 首次支持: v0.17.0rc1 | 已验证版本: v0.22.1rc1
> 单节点 A3: W8A8 TP=16 (128K ctx) | 多节点 A2: W8A8 TP=8 DP=2
> MTP 推测解码 ✅ | FlashComm1 ✅ | Fused MC2 ✅ | Shared Expert Overlap ✅

大规模 Qwen3.5 MoE 模型，397B 总参数 / 17B 激活参数，支持 MTP 推测解码和 W8A8 量化部署。

## 模型简介

| 属性 | 值 |
|------|-----|
| **架构** | Qwen3.5 MoE |
| **参数量** | 397B total / 17B activated |
| **路由专家** | — (每 Token 激活部分专家) |
| **MTP** | num_nextn_predict_layers=1（`ENABLE_MTP=1` 打开；method=`qwen3_5_mtp`） |
| **原生上下文** | **131,072** |
| **量化方式** | W8A8 (`--quantization ascend`)，权重 ≈200G |
| **PP 支持** | ✅ 支持 Pipeline Parallelism |
| **多模态** | ❌ 纯文本 |
| **工具调用解析器** | qwen3.5（内置） |

### 架构注意事项

- **W8A8 量化**：单节点 A3 (64G×16) 可运行 W8A8 版本。BF16 需要 2× A3 或 4× A2。
- **MTP 推测解码**：使用 `qwen3_5_mtp` 方法，3 tokens。若延迟或稳定性下降，可减少 `num_speculative_tokens` 或移除。
- **Shared Expert Overlap**：推荐启用 `multistream_overlap_shared_expert: true` 以提升 MoE 吞吐量。
- **FlashComm1**：高并发 TP 场景最有效。若性能下降，与禁用状态对比。
- **Fused MC2**：高吞吐 TP8 DP2 开启，低延迟 TP16 关闭。多 DP 大 token 场景可能不适用。

### 硬件要求

| 硬件 | 配置 | 量化 | 推荐上下文 |
|------|------|------|-----------|
| Atlas 800I A3 (64G × 16) | TP=16 DP=1 | W8A8 | 128K |
| Atlas 800I A3 (64G × 16) × 2 | TP=8 DP=4 | BF16 | 128K |
| Atlas 800I A2 (64G × 8) × 2 | TP=8 DP=2 | W8A8 | 128K |
| Atlas 800I A2 (64G × 8) × 4 | TP=8 DP=4 | BF16 | 128K |

### 官方文档参考

- vLLM-Ascend 模型文档: https://docs.vllm.ai/projects/ascend/zh-cn/latest/tutorials/models/Qwen3.5-397B-A17B.html
- vLLM 官方文档: https://docs.vllm.ai/en/stable/

## 快速开始

### 前置条件

模型权重下载（推荐 W8A8 量化版）:

| 版本 | 下载 |
|------|------|
| BF16 | https://www.modelscope.cn/models/Qwen/Qwen3.5-397B-A17B |
| W8A8 | https://www.modelscope.cn/models/Eco-Tech/Qwen3.5-397B-A17B-w8a8-mtp |

```bash
# 1. 启动 NPU Docker 容器
bash scripts/docker/manage_npuslim_containers.sh start --file node_list.txt

# 2. 确认 NPU 内存（容器内执行）
npu-smi info | grep "HBM-Usage" | head -1
```

### 部署

```bash
# 单节点 W8A8 (A3, TP=16, 128K context)
bash examples/qwen3.5-397b/vllm/run_vllm.sh

# 高吞吐模式 (A3, TP=8 DP=2, Fused MC2)
TP=8 DP=2 DP_LOCAL=2 HCCL_IF_IP=<ip> DP_ADDRESS=<ip> bash examples/qwen3.5-397b/vllm/run_vllm.sh

# MTP 推测解码
ENABLE_MTP=1 bash examples/qwen3.5-397b/vllm/run_vllm.sh

# BF16 全精度 (多节点)
QUANTIZATION=none TP=8 DP=4 bash examples/qwen3.5-397b/vllm/run_vllm.sh

# 后台运行
nohup bash examples/qwen3.5-397b/vllm/run_vllm.sh > qwen3.5_397b.log 2>&1 &
```

### 验证

```bash
bash examples/qwen3.5-397b/vllm/curl_test.sh

curl http://localhost:8019/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.5","messages":[{"role":"user","content":"Hello"}],"max_tokens":50}'
```

## 并行策略

| 场景 | TP | DP | NPU | 量化 | FUSED_MC2 | MTP |
|------|-----|-----|-----|------|-----------|-----|
| 默认 (A3) | 16 | 1 | 16 | W8A8 | 0 | 0 |
| 高吞吐 (A3) | 8 | 2 | 16 | W8A8 | 1 | 0 |
| 低延迟 (A3) | 16 | 1 | 16 | W8A8 | 0 | 3 |
| 长上下文 | 8 | 2 | 16 | W8A8 | 1 | 0 |
| BF16 (A3×2) | 8 | 4 | 32 | BF16 | 1 | 0 |

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `MODEL_PATH` | Eco-Tech/Qwen3.5-397B-A17B-w8a8-mtp | 模型路径 |
| `TP` | 16 | Tensor Parallel |
| `DP` | 1 | Data Parallel |
| `QUANTIZATION` | ascend | 量化 (ascend / none) |
| `ENABLE_MTP` | 0 | MTP 推测解码 (qwen3_5_mtp) |
| `MAX_MODEL_LEN` | 131072 | 最大上下文长度 |
| `MAX_NUM_SEQS` | 128 | 最大并发请求 |
| `MAX_NUM_BATCHED_TOKENS` | 16384 | 每 batch 最大 token 数 |
| `GPU_MEM_UTIL` | 0.90 | 显存利用率 |
| `HCCL_BUFFSIZE` | 1024 | HCCL 缓冲区 (MB) |
| `NIC_NAME` | — | 多节点网卡名 |
| `HCCL_IF_IP` | — | 多节点 HCCL IP |

## 常见问题

### Q: 硬件要求？

W8A8: 1× A3 (64G×16) 或 2× A2 (64G×8)。BF16: 2× A3 或 4× A2。

### Q: 如何启用 MTP 推测解码？

`ENABLE_MTP=1 bash run_vllm.sh`。使用 `qwen3_5_mtp` 方法，3 tokens。如果延迟或稳定性下降，调整 `num_speculative_tokens` 或关闭。

### Q: 为什么有时启动时 OOM？

397B 模型对权重 + KV Cache 内存需求高。降低 `--max-model-len`、`--max-num-seqs` 或 `--gpu-memory-utilization`。保持 `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True`。

### Q: 多节点部署初始化时挂起？

确认 `HCCL_IF_IP`、网卡名、DP rank 和 RPC 端口在所有节点一致。先验证多节点通信。

### Q: PD 分离时前缀缓存为何关闭？

D 节点前缀缓存为已知限制（issue #7944），PD 分离使用 `--no-enable-prefix-caching`。单节点非 PD 服务可开启。

### Q: FlashComm1 或 Fused MC2 开启后性能下降？

这些优化依赖工作负载。FlashComm1 在高并发 TP 场景最有效。Fused MC2 在多 DP 大 token 场景可能不适用。与禁用状态对比后保留更优配置。
