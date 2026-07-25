# Qwen3-235B-A22B 部署指南

> **vLLM-Ascend v0.21.0+** | 端口: **8018**
> 架构: Qwen3MoeForCausalLM | 128 Experts | GQA (64H/4KVH) | W8A8/BF16
> 首次支持: v0.8.4rc2 | 已验证版本: v0.21.0
> 单节点: A3 (64G×16) 或 A2 (64G×8) | 上下文: 32K (可扩展至 135K)

## 模型简介

| 属性 | 值 |
|------|-----|
| **架构** | Qwen3MoeForCausalLM (MoE + GQA) |
| **总参数** | 235B (每 Token 激活 22B) |
| **路由专家** | 128 (每 Token 激活 8) |
| **隐藏维度** | 4096 |
| **FFN 维度** | 12288 |
| **MoE FFN 维度** | 1536 |
| **网络层数** | 94 |
| **注意力头数** | 64 |
| **KV 头数** | 4 (GQA) |
| **原生上下文** | **262,144** |
| **Head Dim** | 128 |
| **rope_theta** | 5,000,000 |
| **MLA** | ❌ 标准 GQA（非 MLA） |
| **MTP** | ❌ 不支持 |
| **多模态** | ❌ 纯文本 |

## 快速开始

### 前置条件

```bash
# 模型权重 (BF16 或 W8A8 量化版)
# BF16:    https://www.modelscope.cn/models/Qwen/Qwen3-235B-A22B
# W8A8:    https://modelers.cn/models/Modelers_Park/Qwen3-235B-A22B-w8a8

# 启动 NPU Docker 容器
bash scripts/docker/manage_npuslim_containers.sh start --file node_list.txt
```

### 部署

```bash
# 单节点 W8A8 (默认, TP=8)
bash examples/qwen3-235b/vllm/run_vllm.sh

# 高吞吐模式 (A3, TP=4 DP=4)
TP=4 DP=4 DP_LOCAL=4 HCCL_IF_IP=<ip> DP_ADDRESS=<ip> bash examples/qwen3-235b/vllm/run_vllm.sh

# 低延迟模式 (A3, TP=16)
TP=16 bash examples/qwen3-235b/vllm/run_vllm.sh

# 长上下文 (135K, yarn rope-scaling)
MAX_MODEL_LEN=135000 bash examples/qwen3-235b/vllm/run_vllm.sh

# BF16 全精度
QUANTIZATION=none bash examples/qwen3-235b/vllm/run_vllm.sh

# 后台运行
nohup bash examples/qwen3-235b/vllm/run_vllm.sh > qwen3_235b.log 2>&1 &
```

### 验证

```bash
bash examples/qwen3-235b/vllm/curl_test.sh

# 手动测试
curl http://localhost:8018/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3","messages":[{"role":"user","content":"Hello"}],"max_tokens":50}'
```

## 并行策略

| 场景 | TP | DP | NPU | 上下文 | 量化 | FUSED_MC2 |
|------|-----|-----|-----|--------|------|-----------|
| 默认单节点 | 8 | 1 | 8/16 | 32K | W8A8 | 0 |
| 高吞吐 (A3) | 4 | 4 | 16 | 32K | W8A8 | 1 |
| 低延迟 (A3) | 16 | 1 | 16 | 32K | W8A8 | 0 |
| 长上下文 | 8 | 1 | 8/16 | 135K | W8A8 | 1 |

> 官方推荐：EP 始终开启 (`--enable-expert-parallel`)。128 专家 MoE，EP 配合 TP 自动分配。

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `MODEL_PATH` | Qwen3-235B-A22B-Instruct-2507 | 模型路径 |
| `TP` | 8 | Tensor Parallel |
| `DP` | 1 | Data Parallel (高吞吐设为 4) |
| `QUANTIZATION` | ascend | 量化 (ascend / none) |
| `MAX_MODEL_LEN` | 32768 | 最大上下文 (可设 135000) |
| `MAX_NUM_SEQS` | 32 | 最大并发请求 |
| `MAX_NUM_BATCHED_TOKENS` | 8096 | 每 batch 最大 token 数 |
| `GPU_MEM_UTIL` | 0.95 | 显存利用率 |
| `HCCL_BUFFSIZE` | 512 | HCCL 缓冲区 (MB) |
| `NIC_NAME` | — | 多节点网卡名 |
| `HCCL_IF_IP` | — | 多节点 HCCL IP |
| `FUSED_MC2` | auto | Fused MC2 (高吞吐=1, 低延迟=0) |

## 官方参考

- 模型文档: https://docs.vllm.ai/projects/ascend/zh-cn/latest/tutorials/models/Qwen3-235B-A22B.html
- 环境变量: https://docs.vllm.ai/projects/ascend/zh-cn/latest/user_guide/configuration/env_vars.html
- 性能调优: https://docs.vllm.ai/projects/ascend/zh-cn/latest/developer_guide/performance_and_debug/optimization_and_tuning.html

## 常见问题

### Q: 硬件要求？

BF16 需要 1× A3 (64G×16) 或 1× A2 (64G×8)。W8A8 量化版硬件需求相同但显存占用更小。

### Q: Qwen3-235B 需要 MLA 优化吗？

不需要。Qwen3-235B 使用标准 GQA 注意力，非 MLA (Multi-head Latent Attention)。无需设置 VLLM_ASCEND_ENABLE_MLAPO。

### Q: EP 是否必须开启？

是。MoE 模型必须通过 `--enable-expert-parallel` 开启 EP，将 FFN 专家分布到多张 NPU 上以降低单卡计算量。

### Q: 如何扩展长上下文？

使用 yarn rope-scaling：`--hf-overrides '{"rope_parameters":{"rope_type":"yarn","rope_theta":1000000,"factor":4,"original_max_position_embeddings":32768}}'`。Qwen3-235B-A22B-Instruct-2507 原生支持长上下文，通常无需此参数。

### Q: 什么时候用 PD 分离 vs 单节点？

单节点部署更简单，适合模型能放入单节点的场景。PD 分离将 Prefill 和 Decode 分布到不同节点，适合大规模高吞吐服务。3 节点 A3 PD 分离可达单节点 ~3× 吞吐。
