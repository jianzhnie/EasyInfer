# GLM-5.1 W4A8 部署指南

> ✅ **部署验证**: 已在 vLLM-Ascend 0.18.0rc1 + CANN 8.5.1 环境成功部署。
> 配置: TP=16 PP=1 (2节点), Ray backend. 端口 8002.

本文档提供 GLM-5.1 W4A8 量化模型在华为昇腾 NPU 环境下的部署指南。

## 模型简介

| 属性 | 值 |
|------|-----|
| **架构** | GLM MoE DSA (MoE + DeepSeek-style Attention) |
| **路由专家** | 256 (+ 1 共享专家) |
| **每 Token 激活专家** | 8 |
| **隐藏维度** | 6144 |
| **网络层数** | 78 |
| **注意力头** | 64 (全 GQA) |
| **原生上下文** | 202,752 |
| **量化方式** | W4A8 (4-bit 权重 + 8-bit 激活) |
| **投机解码** | MTP (Multi-Token Prediction), 1 nextn layer |
| **词表大小** | 154,880 |

> **与 GLM-5 的区别**: GLM-5.1 是 GLM-5 的升级版，采用相同的模型架构 (GlmMoeDsaForCausalLM)，但改进了训练数据和后训练流程，具有更强的推理和 agent 能力。部署配置与 GLM-5 几乎相同。

## 官方文档参考

- GLM-5 官方部署文档: https://docs.vllm.ai/projects/ascend/en/v0.18.0/tutorials/models/GLM5.html

## 模型权重

模型路径: `/home/jianzhnie/llmtuner/hfhub/models/Eco-Tech/GLM-5.1-w4a8`

## 硬件要求

### 单节点部署

| 硬件 | 配置 | 推荐上下文 |
|------|------|-----------|
| Atlas 800 A2 (64G × 8) | W4A8, TP=8 | 32k |
| Atlas 800 A3 (64G × 16) | W4A8, TP=16 | 200k |

### 多节点部署

| 节点数 | 配置 | 推荐上下文 |
|--------|------|-----------|
| 2 节点 × 8 NPU | TP=16, DP=1 | 32k (大TP跨节点) |
| 8 节点 × 8 NPU | TP=64, DP=1 | 200k |

> **注意**: GLM-5.1 不支持 Pipeline Parallelism (PP)，多节点部署应使用更大的 TP 值跨节点。

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
bash examples/glm5_1_w4a8/vllm_server.sh
```

### 单节点 A3 部署 (16 卡, 200k 上下文)

```bash
TENSOR_PARALLEL_SIZE=16 MAX_MODEL_LEN=200000 MAX_NUM_SEQS=8 \
bash examples/glm5_1_w4a8/vllm_server.sh
```

### 多节点部署 (8 节点 × 8 NPU)

```bash
TENSOR_PARALLEL_SIZE=64 DATA_PARALLEL_SIZE=1 \
MAX_MODEL_LEN=131072 \
bash examples/glm5_1_w4a8/vllm_server.sh
```

### 后台运行

```bash
nohup bash examples/glm5_1_w4a8/vllm_server.sh > glm5_1_w4a8_server.log 2>&1 &
```

## 环境变量说明

### 基础配置

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `MODEL_PATH` | `/home/jianzhnie/llmtuner/hfhub/models/Eco-Tech/GLM-5.1-w4a8` | 模型权重路径 |
| `SERVED_MODEL_NAME` | `glm-5.1` | API 中的模型名称 |
| `HOST` | `0.0.0.0` | 监听地址 |
| `PORT` | `8002` | 监听端口 |

### 并行配置

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `TENSOR_PARALLEL_SIZE` | `8` | 张量并行度 (A2=8, A3=16) |
| `PIPELINE_PARALLEL_SIZE` | `1` | 流水线并行度 |
| `ENABLE_EXPERT_PARALLEL` | `1` | 专家并行开关 (MoE 必需) |
| `DATA_PARALLEL_SIZE` | `1` | 数据并行度 |

### 内存与量化

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `DTYPE` | `bfloat16` | 计算数据类型 |
| `QUANTIZATION` | `ascend` | W4A8 Ascend 量化 |
| `GPU_MEMORY_UTILIZATION` | `0.95` | NPU 显存利用率 |
| `SWAP_SPACE` | `16` | CPU 交换空间 (GiB) |

### 序列调度

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `MAX_MODEL_LEN` | A2: 32768, A3: 200000 (自动) | 最大上下文长度 |
| `MAX_NUM_SEQS` | A2: 2, A3: 8 (自动) | 最大并发请求数 |
| `MAX_NUM_BATCHED_TOKENS` | `4096` | 每 step 最大 token 数 |
| `ENABLE_CHUNKED_PREFILL` | `1` | 分块预填充 |

### 投机解码 (MTP)

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `SPECULATIVE_METHOD` | `deepseek_mtp` | 投机解码方法 |
| `SPECULATIVE_NUM_TOKENS` | `3` | 每次投机 token 数 |

### 华为 NPU 专用

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `HCCL_OP_EXPANSION_MODE` | `AIV` | HCCL 操作扩展模式 |
| `HCCL_BUFFSIZE` | `200` | HCCL 缓冲区大小 (MB) |
| `OMP_PROC_BIND` | `false` | 禁用 OpenMP 线程绑定 |
| `OMP_NUM_THREADS` | `1` | OpenMP 线程数 |
| `PYTORCH_NPU_ALLOC_CONF` | `expandable_segments:True` | NPU 内存分配 |
| `VLLM_ASCEND_BALANCE_SCHEDULING` | `1` | 负载均衡调度 |
| `VLLM_ASCEND_ENABLE_FLASHCOMM1` | `1` | 通信优化 |
| `VLLM_ASCEND_ENABLE_MLAPO` | `1` | 融合算子 (W8A8 必需, W4A8 推荐) |

### 加速特性

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `PREFIX_CACHING` | `1` | 前缀缓存 |
| `ENFORCE_EAGER` | `1` | 禁用 CUDA Graph |
| `NUM_SCHEDULER_STEPS` | `4` | 多步调度步数 |
| `ENABLE_ASYNC_SCHEDULING` | `1` | 异步调度 |
| `CUDAGRAPH_MODE` | `FULL_DECODE_ONLY` | CUDA Graph 模式 |
| `ENABLE_NPUGRAPH_EX` | `true` | NPU Graph 扩展 |
| `FUSE_MULS_ADD` | `true` | 融合乘法加法 |
| `MULTISTREAM_OVERLAP_SHARED_EXPERT` | `true` | 多流共享专家重叠 |

### 工具调用

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `ENABLE_TOOL_CALLING` | `1` | 工具调用开关 |
| `TOOL_CALL_PARSER` | `glm47` | GLM 系列工具调用解析器 |

## 并行策略推荐

### 8 节点 × 8 NPU (64 卡) 环境

```
场景               TP   PP   EP   DP   MAX_MODEL_LEN
─────────────────────────────────────────────────────
低延迟 (单节点)     8    1    8    1    32768
均衡 (2 节点)       16   1    16   1    65536
高吞吐 (4 节点)     32   1    32   1    65536
长上下文 (8 节点)   64   1    64   1    200000
```

> GLM-5.1 不支持 PP，多节点时使用大 TP 跨节点。

## 功能验证

### 自动测试

```bash
bash examples/glm5_1_w4a8/curl_test.sh
```

### 手动测试

```bash
# 检查服务
curl http://localhost:8002/v1/models

# Chat Completion
curl http://localhost:8002/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "glm-5.1",
    "messages": [{"role": "user", "content": "你好"}],
    "max_tokens": 200
  }'
```

## 常见问题

### Q: GLM-5.1 和 GLM-5 的部署配置是否通用？
A: 是的，两者架构完全相同 (GlmMoeDsaForCausalLM)，部署配置可以通用。只需修改 `MODEL_PATH` 和 `SERVED_MODEL_NAME`。

### Q: 工具调用使用什么 parser？
A: GLM 系列使用 `glm47` tool parser，与 GLM-5 相同。

### Q: W4A8 量化精度如何？
A: W4A8 在大多数场景下精度接近 BF16，推理速度更快，显存占用仅约 1/4。推荐用于生产部署。
