# LongCat-Flash-Chat 2-Layer BF16 部署指南

> **vLLM-Ascend v0.23.0rc1** | 端口: **8300**
> 架构: LongcatFlashForCausalLM | 512 Routed + 256 Zero Experts | MoE + MLA | 2 层
> 已验证配置: **TP=2 EP=1** (单节点 2 NPU) | 上下文: 4096 | BF16 无量化
> 注意: 从原 28 层模型中提取 2 层用于调试；EP 模式必需（单卡无法容纳 512 专家）
> 验证状态: ⚠️ 推理阶段 CANN kernel 异常待解决

LongCat-Flash-Chat 的 2 层精简版，用于 EP 修复插件的快速迭代验证。

## 模型简介

| 属性 | 值 |
|------|-----|
| **架构** | LongcatFlashForCausalLM (MLA + MoE) |
| **路由专家** | 512 (每 Token 激活 12) |
| **Zero 专家** | 256 (Identity) |
| **隐藏维度** | 6144 |
| **网络层数** | **2** (从 28 层原模型中提取) |
| **KV LoRA Rank** | 512 |
| **精度** | BF16 (无量化) |
| **模型大小** | ≈80 GB |
| **MTP** | ❌ 不支持 |
| **PP 支持** | ❌ 不支持（仅 2 层） |
| **多模态** | ❌ 纯文本 |
| **工具调用解析器** | 不适用 |
| **推理解析器** | 不适用 |

### 架构注意事项

- 从 28 层 LongCat-Flash-Chat 中提取 2 层用于调试目的
- **MLA 注意力**仅支持 block_size=128
- **MC2 MoE comm** 与 Zero Expert 权重置零不兼容，EasyInfer 插件通过 `EASYINFER_MOE_COMM=allgather` 覆盖
- **Chunked Prefill** 与 EP token dispatch 冲突，默认禁用
- 推理阶段 CANN MLP kernel aicore 异常（fftsplus aivector）是 CANN 在 EP 模式下处理 512 专家 MoE 的 kernel 层限制

### 硬件要求

| 硬件 | 配置 | 推荐上下文 | 备注 |
|------|------|-----------|------|
| Atlas 800 A2/A3 (64G × 2) | BF16, TP=2, EP=1 | 4K | 单节点 2 卡，EP 必需 |

## 快速开始

### 前置条件

模型路径: `/home/jianzhnie/llmtuner/hfhub/models/meituan-longcat/expand/LongCat-Flash-Chat-2layer`

```bash
# 1. 启动 NPU Docker 容器
bash scripts/docker/manage_npuslim_containers.sh start --file node_list.txt

# 2. 启动 Ray 集群
bash scripts/ray_cluster/start_npuslim_ray_cluster.sh start --file node_list.txt
```

### 部署

```bash
# EP 模式 (TP=2, 1 节点)
EP=1 TP=2 EXECUTOR=mp bash examples/longcat-2layer/vllm/run_vllm.sh

# 标准模式
bash examples/longcat-2layer/vllm/run_vllm.sh
```

### 验证

```bash
bash examples/longcat-2layer/vllm/curl_test.sh
```

## 并行策略

| 场景 | TP | PP | EP | NPU | 上下文 | 量化 | 状态 |
|------|-----|-----|-----|-----|--------|------|------|
| EP 模式 | 2 | 1 | 1 | 2 | 4K | BF16 | ⚠️ |
| 标准模式 | 8 | 1 | — | 8 | 4K | BF16 | ⚠️ |

> 单卡无法容纳 512 专家模型（需 ≈41 GB 权重 + KV Cache），EP 模式使用 2 卡。

## 关键配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| TP | 8 | 张量并行度（2 卡调试推荐 TP=2） |
| EP | 0 | 专家并行开关（2 卡必须开启） |
| PORT | 8300 | 服务端口 |
| MAX_MODEL_LEN | 4096 | 最大序列长度 |
| MAX_NUM_SEQS | 32 | 最大并发序列数 |
| GPU_MEM_UTIL | 0.90 | 显存利用率 |
| BLOCK_SIZE | 128 | MLA 注意力 block size |
| HCCL_BUFFSIZE | 2048 | HCCL EP 缓冲区 |

## EasyInfer EP 修复插件

| 模块 | 路径 | 作用 |
|------|------|------|
| EP 零号专家 | `easyinfer/plugins/vllm_ascend/ops/fused_moe/fix_ep_zero_expert.py` | 修复 AssertionError / Token dispatch 越界 / 版本兼容 |
| EP forward_impl | `easyinfer/plugins/vllm_ascend/ops/fused_moe/zero_expert_fused_moe.py` | EP 路由覆盖（旧版本适配） |

> 插件通过 vLLM `general_plugins` 自动发现加载。

## 验证记录

| 阶段 | 状态 | 说明 |
|------|------|------|
| 插件加载 | ✅ | general_plugins 自动发现 |
| 模型加载 | ✅ | EP Rank 0/2, 256/512 experts, 38.87 GB/worker |
| Zero Expert | ✅ | AssertionError 已修复 |
| Token Dispatch | ✅ | ID sanitization 已修复 |
| KV Cache | ✅ | 3.87 GiB, 902K tokens |
| API 启动 | ✅ | Application startup complete |
| 推理 | ⚠️ | CANN MLP kernel aicore 异常 (fftsplus aivector) |

> 推理阶段的 aicore 异常是 CANN kernel 层限制，需 CANN 版本更新解决。插件层能修复的问题已全部修复。
