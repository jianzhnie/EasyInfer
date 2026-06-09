# Kimi-K2.6 W4A8 部署指南

本文档提供 Kimi-K2.6 W4A8 量化模型在华为昇腾 NPU 环境下的部署指南。

## 模型简介

| 属性 | 值 |
|------|-----|
| **架构** | Kimi K2.5 (DeepSeek V3 MoE 文本 + Vision Transformer) |
| **文本骨干** | DeepSeek V3 (MLA + MoE) |
| **路由专家** | 384 (+ 1 共享专家) |
| **每 Token 激活专家** | 8 |
| **隐藏维度** | 7168 |
| **网络层数** | 61 |
| **注意力头** | 64 (全 GQA) |
| **原生上下文** | 262,144 |
| **量化方式** | W4A8 (4-bit 权重 + 8-bit 激活) |
| **投机解码** | 不支持 (num_nextn_predict_layers=0) |
| **多模态** | 支持 (Vision Transformer, 27 层) |
| **词表大小** | 163,840 |

> **与 Kimi-K2 的区别**: Kimi-K2.6 增加视觉多模态能力 (Vision Transformer + unified vision chunk)，文本骨干基于 DeepSeek V3 架构 (384 专家)。

## 官方文档参考

- vLLM-Ascend 模型列表: https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/index.html
- vLLM 官方文档: https://docs.vllm.ai/en/stable/

## 模型权重

模型路径: `/home/jianzhnie/llmtuner/hfhub/models/Eco-Tech/Kimi-K2.6-w4a8`

> **注意**: 模型包含自定义代码 (`configuration_kimi_k25.py`, `modeling_kimi_k25.py`)，必须启用 `--trust-remote-code`。

## 硬件要求

### 单节点部署

| 硬件 | 配置 | 推荐上下文 |
|------|------|-----------|
| Atlas 800 A2 (64G × 8) | W4A8, TP=8 | 32k |
| Atlas 800 A3 (64G × 16) | W4A8, TP=16 | 131k |

### 多节点部署

| 节点数 | 配置 | 推荐上下文 |
|--------|------|-----------|
| 2 节点 × 8 NPU | TP=8, PP=2 | 64k |
| 4 节点 × 8 NPU | TP=8, PP=4 | 131k |
| 8 节点 × 8 NPU | TP=8, PP=8 | 262k |

## 快速开始

### 前置条件

```bash
# 1. 启动 NPU Docker 容器
bash scripts/docker/manage_npuslim_containers.sh start --file node_list.txt

# 2. 启动 Ray 集群
bash scripts/ray_cluster/start_npuslim_ray_cluster.sh start --file node_list.txt
```

### 单节点 A2 部署 (8 卡, 默认)

```bash
cd /home/jianzhnie/llmtuner/llm/EasyInfer
bash examples/kimi_k2_6_w4a8/vllm_server.sh
```

### 单节点 A3 部署 (16 卡, 131k 上下文)

```bash
TENSOR_PARALLEL_SIZE=16 MAX_MODEL_LEN=131072 MAX_NUM_SEQS=16 \
bash examples/kimi_k2_6_w4a8/vllm_server.sh
```

### 多节点部署 (8 节点 × 8 NPU)

```bash
PIPELINE_PARALLEL_SIZE=8 DATA_PARALLEL_SIZE=8 \
MAX_MODEL_LEN=131072 \
bash examples/kimi_k2_6_w4a8/vllm_server.sh
```

### 后台运行

```bash
nohup bash examples/kimi_k2_6_w4a8/vllm_server.sh > kimi_k2_6_w4a8_server.log 2>&1 &
```

## 环境变量说明

### 基础配置

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `MODEL_PATH` | `/home/jianzhnie/llmtuner/hfhub/models/Eco-Tech/Kimi-K2.6-w4a8` | 模型权重路径 |
| `SERVED_MODEL_NAME` | `kimi-k2.6` | API 中的模型名称 |
| `HOST` | `0.0.0.0` | 监听地址 |
| `PORT` | `8003` | 监听端口 |

### 并行配置

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `TENSOR_PARALLEL_SIZE` | `8` | 张量并行度 (A2=8, A3=16) |
| `PIPELINE_PARALLEL_SIZE` | `1` | 流水线并行度 |
| `ENABLE_EXPERT_PARALLEL` | `1` | 专家并行开关 (384 专家 MoE 必需) |
| `DATA_PARALLEL_SIZE` | `1` | 数据并行度 |

### 内存与量化

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `DTYPE` | `bfloat16` | 计算数据类型 |
| `QUANTIZATION` | `ascend` | W4A8 Ascend 量化 |
| `GPU_MEMORY_UTILIZATION` | `0.92` | NPU 显存利用率 |
| `SWAP_SPACE` | `32` | CPU 交换空间 (GiB, 384 专家需较大空间) |

### 序列调度

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `MAX_MODEL_LEN` | A2: 32768, A3: 131072 (自动) | 最大上下文长度 |
| `MAX_NUM_SEQS` | A2: 8, A3: 16 (自动) | 最大并发请求数 |
| `MAX_NUM_BATCHED_TOKENS` | `8192` | 每 step 最大 token 数 |
| `ENABLE_CHUNKED_PREFILL` | `1` | 分块预填充 |

### 华为 NPU 专用

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `HCCL_OP_EXPANSION_MODE` | `AIV` | HCCL 操作扩展模式 |
| `HCCL_BUFFSIZE` | `200` | HCCL 缓冲区大小 (MB) |
| `OMP_PROC_BIND` | `false` | 禁用 OpenMP 线程绑定 |
| `OMP_NUM_THREADS` | `1` | OpenMP 线程数 |
| `PYTORCH_NPU_ALLOC_CONF` | `expandable_segments:True` | NPU 内存分配 |
| `VLLM_ASCEND_BALANCE_SCHEDULING` | `1` | 负载均衡调度 |

### 加速特性

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `PREFIX_CACHING` | `1` | 前缀缓存 |
| `ENFORCE_EAGER` | `1` | 禁用 CUDA Graph (NPU 推荐) |
| `NUM_SCHEDULER_STEPS` | `8` | 多步调度步数 |
| `ENABLE_ASYNC_SCHEDULING` | `1` | 异步调度 (W4A8 推荐) |
| `CUDAGRAPH_MODE` | `FULL_DECODE_ONLY` | CUDA Graph 模式 |
| `ENABLE_NPUGRAPH_EX` | `true` | NPU Graph 扩展 |
| `FUSE_MULS_ADD` | `true` | 融合乘法加法 |
| `MULTISTREAM_OVERLAP_SHARED_EXPERT` | `true` | 多流共享专家重叠 |

### 工具调用

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `ENABLE_TOOL_CALLING` | `1` | 工具调用开关 |
| `TOOL_CALL_PARSER` | `deepseekv3` | 基于 DeepSeek V3 架构，使用 deepseekv3 parser |

## 并行策略推荐

### 8 节点 × 8 NPU (64 卡) 环境

```
场景               TP   PP   EP   DP   MAX_MODEL_LEN
─────────────────────────────────────────────────────
低延迟 (单节点)     8    1    8    1    32768
均衡 (2 节点)       8    2    8    1    65536
高吞吐 (4 节点)     8    4    8    1    131072
长上下文 (8 节点)   8    8    8    1    262144
```

> 注意: Kimi-K2.6 有 384 专家 (多于其他模型的 256)，EP 建议设置为 8, 12, 16, 24 等能整除 384 的值。

## 性能调优

### 低延迟场景
- 单节点 A3 部署 (TP=16)
- 减小 `MAX_NUM_SEQS` (如 4-8)
- 减小 `NUM_SCHEDULER_STEPS` (如 4)
- 启用 Prefix Caching

### 高吞吐场景
- 多节点 + 数据并行
- 增大 `MAX_NUM_SEQS` (如 16-32)
- 启用 Chunked Prefill + Prefix Caching
- 启用异步调度
- 增大 `NUM_SCHEDULER_STEPS` (如 8-16)

### 多模态场景
- Kimi-K2.6 支持视觉输入 (Vision Transformer)
- 视觉 token 会额外占用上下文窗口
- 建议预留 20-30% 上下文给视觉 token

### 长上下文场景
- 多节点扩展 PP (流水线并行)
- 增大 `MAX_MODEL_LEN`
- 提高 `GPU_MEMORY_UTILIZATION`
- 增大 `SWAP_SPACE`

## 功能验证

### 自动测试

```bash
bash examples/kimi_k2_6_w4a8/curl_test.sh
```

### 手动测试

```bash
# 检查服务
curl http://localhost:8003/v1/models

# Text Chat Completion
curl http://localhost:8003/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "kimi-k2.6",
    "messages": [{"role": "user", "content": "你好，请介绍一下自己"}],
    "max_tokens": 200
  }'

# 流式输出
curl http://localhost:8003/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "kimi-k2.6",
    "messages": [{"role": "user", "content": "写一首诗"}],
    "max_tokens": 200,
    "stream": true
  }'
```

## 常见问题

### Q: Kimi-K2.6 和 Kimi-K2 有什么区别？
A: Kimi-K2.6 增加了多模态 (Vision) 能力，包含 Vision Transformer (27 层)。文本骨干基于 DeepSeek V3 架构 (384 专家)。纯文本推理性能与 Kimi-K2 类似。

### Q: 为什么没有启用 MTP/投机解码？
A: Kimi-K2.6 的 config.json 中 `num_nextn_predict_layers=0`，表示模型不支持 Multi-Token Prediction。不需要配置投机解码参数。

### Q: 多模态功能如何使用？
A: 通过 vLLM 的 `/v1/chat/completions` 端点，传入 image 类型的 content 即可。具体 API 格式参考 vLLM 多模态文档。

### Q: 工具调用使用什么 parser？
A: Kimi-K2.6 基于 DeepSeek V3 架构，推荐使用 `deepseekv3` tool parser。

### Q: 384 专家对部署有什么影响？
A: 专家数更多 (384 vs 256)，EP_SIZE 需能整除 384 (推荐 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 384)。384 专家的 MoE 层参数量更大，需要更大的 SWAP_SPACE。
