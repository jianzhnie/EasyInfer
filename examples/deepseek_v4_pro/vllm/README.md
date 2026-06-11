# DeepSeek-V4-Pro W4A8 MTP 部署指南

> ⚠️ **部署受阻** | vLLM-Ascend 0.20.2 + CANN 9.0.0 | 384 Experts | MoE | MTP
> **已知问题**: Eco-Tech 版模型权重含 `attn_sink` 参数，vLLM-Ascend 0.20.2 无法识别 (KeyError)
> **修复方法**: 需在 `deepseek_v4.py:1204` 添加 `if name not in params_dict: continue` 跳过未知 sink 权重

DeepSeek-V4-Pro 是 DeepSeek V4 系列的旗舰模型，384 路由专家 + 1 共享专家，支持 MTP 投机解码。
W4A8 量化版本在保持推理质量的同时大幅降低显存占用。

## 模型简介

| 属性 | 值 |
|------|-----|
| **架构** | DeepseekV4ForCausalLM (MoE + MLA) |
| **路由专家** | 384 (每 Token 激活 8 专家) |
| **隐藏维度** | 7168 |
| **网络层数** | 61 |
| **MLA** | kv_lora_rank=512, q_lora_rank=1536, v_head_dim=128 |
| **原生上下文** | **1,048,576** (1M tokens) |
| **量化方式** | W4A8 (4-bit 权重 + 8-bit 激活) |
| **投机解码** | MTP (1 nextn layer) |
| **词表大小** | 129,280 |

## 官方文档参考

- vLLM-Ascend 模型列表: https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/index.html
- vLLM 官方文档: https://docs.vllm.ai/en/stable/

## 模型权重

模型路径: `/home/jianzhnie/llmtuner/hfhub/models/Eco-Tech/DeepSeek-V4-Pro-w4a8-mtp`

## 硬件要求

### 单节点部署

| 硬件 | 配置 | 推荐上下文 |
|------|------|-----------|
| Atlas 800 A2 (64G × 8) | W4A8, TP=8 | 32k |
| Atlas 800 A3 (64G × 16) | W4A8, TP=16 | 64k-128k |

### 多节点部署

| 节点数 | 配置 | 推荐上下文 |
|--------|------|-----------|
| 2 节点 × 8 NPU | TP=8, PP=2 | 64k |
| 4 节点 × 8 NPU | TP=8, PP=4 | 128k |
| 8 节点 × 8 NPU | TP=8, PP=8 | 256k+ |

## 快速开始

### 前置条件

```bash
# 1. 启动 NPU Docker 容器 (所有节点)
bash scripts/docker/manage_npuslim_containers.sh start --file node_list.txt

# 2. 启动 Ray 集群 (所有节点)
bash scripts/ray_cluster/start_npuslim_ray_cluster.sh start --file node_list.txt
```

### 单节点部署 (默认)

```bash
# 在容器内执行
cd /home/jianzhnie/llmtuner/llm/EasyInfer
bash examples/deepseek_v4_pro/vllm/run_vllm.sh
```

### 多节点部署 (8 节点 × 8 NPU)

```bash
# 在 Head 节点容器内执行
PIPELINE_PARALLEL_SIZE=8 \
MAX_MODEL_LEN=131072 \
bash examples/deepseek_v4_pro/vllm/run_vllm.sh
```

### 后台运行

```bash
nohup bash examples/deepseek_v4_pro/vllm/run_vllm.sh > deepseek_v4_pro.log 2>&1 &
```

## 环境变量说明

### 基础配置

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `MODEL_PATH` | `/home/jianzhnie/llmtuner/hfhub/models/Eco-Tech/DeepSeek-V4-Pro-w4a8-mtp` | 模型权重路径 |
| `HOST` | `0.0.0.0` | 监听地址 |
| `PORT` | `8000` | 监听端口 |

### 并行配置

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `TP` | `8` | 张量并行度 (单节点 = 8) |
| `PP` | `1` | 流水线并行度 (多节点时增加) |
| `GPU_MEM_UTIL` | `0.92` | NPU 显存利用率 |

### 序列调度

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `MAX_MODEL_LEN` | `32768` | 最大上下文长度 |
| `MAX_NUM_SEQS` | `64` | 最大并发请求数 |

### 华为 NPU 专用

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `HCCL_OP_EXPANSION_MODE` | `AIV` | HCCL 操作扩展模式 |
| `HCCL_BUFFSIZE` | `400` | HCCL 缓冲区大小 (MB, 384 专家需更大) |
| `OMP_PROC_BIND` | `false` | 禁用 OpenMP 线程绑定 |
| `OMP_NUM_THREADS` | `1` | OpenMP 线程数 |
| `PYTORCH_NPU_ALLOC_CONF` | `expandable_segments:True` | NPU 内存分配 |
| `VLLM_ASCEND_BALANCE_SCHEDULING` | `1` | 负载均衡调度 |

## 并行策略推荐

### 8 节点 × 8 NPU (64 卡) 环境

```
场景               TP   PP   EP   MAX_MODEL_LEN
─────────────────────────────────────────────────
低延迟 (单节点)     8    1    8    32768
均衡 (2 节点)       8    2    8    65536
高吞吐 (4 节点)     8    4    8    131072
长上下文 (8 节点)   8    8    8    262144
```

## 性能调优

### 低延迟场景
- 单节点部署 (TP=8)
- 减小 `MAX_NUM_SEQS` (如 8-16)
- MTP tokens=1 (低延迟，减少投机开销)
- 减小 `MAX_NUM_BATCHED_TOKENS` (如 4096)

### 高吞吐场景
- 多节点部署，增大 PP
- 增大 `MAX_NUM_SEQS` (如 32-64)
- 启用 Chunked Prefill 和 Prefix Caching
- MTP tokens=3 (高吞吐投机)

### 长上下文场景
- 多节点扩展 PP
- 增大 `MAX_MODEL_LEN` (如 131072-262144)
- 提高 `GPU_MEM_UTIL` (如 0.95)

## 功能验证

### 基础测试

```bash
bash examples/deepseek_v4_pro/vllm/curl_test.sh
```

### 手动 API 测试

```bash
# 检查服务
curl http://localhost:8000/v1/models

# Chat Completion
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-v4-pro",
    "messages": [{"role": "user", "content": "你好，请介绍一下自己"}],
    "max_tokens": 200
  }'

# 流式输出
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-v4-pro",
    "messages": [{"role": "user", "content": "写一首诗"}],
    "max_tokens": 200,
    "stream": true
  }'

# Tool Calling (Claude Code 集成)
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-v4-pro",
    "messages": [{"role": "user", "content": "Weather in Beijing?"}],
    "tools": [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}],
    "max_tokens": 100
  }'
```

## Claude Code 集成

```bash
ANTHROPIC_BASE_URL=http://localhost:8000 \
ANTHROPIC_API_KEY=dummy \
ANTHROPIC_AUTH_TOKEN=dummy \
ANTHROPIC_DEFAULT_SONNET_MODEL=deepseek-v4-pro \
ANTHROPIC_DEFAULT_HAIKU_MODEL=deepseek-v4-pro \
ANTHROPIC_DEFAULT_OPUS_MODEL=deepseek-v4-pro \
claude
```

## 常见问题

### Q: DeepSeek-V4-Pro 和 DeepSeek-V4-Flash 有什么区别？
A: Pro 有 384 专家（vs 256），更强的推理能力，但显存占用更大。Flash 更轻量，推理速度更快。

### Q: W4A8 和 W8A8 有什么区别？
A: W4A8 显存占用约为 W8A8 的 50%，但精度略有下降。W4A8 适合单节点/少节点部署，W8A8 适合追求精度的场景。

### Q: MTP 是否必须启用？
A: 不是必须的，但推荐启用。MTP (Multi-Token Prediction) 可显著加速解码阶段。不启用时删除 `--speculative-config` 参数。

### Q: 如何调整上下文长度？
A: 通过 `MAX_MODEL_LEN` 环境变量。DeepSeek-V4-Pro 原生支持 1M 上下文，但实际可用长度受 NPU 显存限制。
