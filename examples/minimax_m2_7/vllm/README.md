# MiniMax-M2.7 W8A8 QuaRot MTP 部署指南

> ✅ **部署验证通过** | 2026-06-11 | vLLM-Ascend 0.20.2 + CANN 9.0.0
> **已验证配置**: TP=8 PP=2 (2节点) | **上下文**: 65,536 | Chat ✅
> **注意**: MTP 不支持 (vLLM-Ascend 0.20.2 不兼容 MiniMax mtp 方法) | W8A8 QuaRot

MiniMax-M2.7 基于 MiniMaxM2ForCausalLM 架构，256 路由专家 MoE，支持 MTP 投机解码。
W8A8 QuaRot 量化在精度和显存之间取得平衡，官方推荐 A2 环境使用 TP=4。

## 模型简介

| 属性 | 值 |
|------|-----|
| **架构** | MiniMaxM2ForCausalLM (MoE + Attention) |
| **路由专家** | 256 (每 Token 激活 8 专家) |
| **隐藏维度** | 3072 |
| **网络层数** | 62 |
| **原生上下文** | **204,800** (204K tokens) |
| **量化方式** | W8A8 QuaRot (Ascend 量化) |
| **投机解码** | MTP (1 nextn layer, 3 speculative tokens) |
| **词表大小** | 待查 |

## 官方文档参考

- vLLM-Ascend MiniMax-M2.5 文档: https://docs.vllm.ai/projects/ascend/zh-cn/releases-v0.20.2rc/tutorials/models/MiniMax-M2.5.html
- vLLM 官方文档: https://docs.vllm.ai/en/stable/

## 模型权重

模型路径: `/home/jianzhnie/llmtuner/hfhub/models/Eco-Tech/MiniMax-M2.7-w8a8-QuaRot`

## 硬件要求

### 单节点部署

| 硬件 | 配置 | 推荐上下文 |
|------|------|-----------|
| Atlas 800 A2 (64G × 8) | W8A8, TP=4 (官方推荐) | 32k |
| Atlas 800 A3 (64G × 16) | W8A8, TP=8 | 64k-128k |

### 多节点部署

| 节点数 | 配置 | 推荐上下文 |
|--------|------|-----------|
| 2 节点 × 8 NPU | TP=8, PP=2 | 64k |
| 4 节点 × 8 NPU | TP=8, PP=4 | 128k |
| 8 节点 × 8 NPU | TP=8, PP=8 | 204k |

> **注意**: MiniMax-M2.7 官方推荐 A2 环境使用 TP=4 而非 TP=8。
> 这是模型架构特性决定的，与 W8A8 量化相关。

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
bash examples/minimax_m2_7/vllm/run_vllm.sh
```

### 多节点部署

```bash
# A3 16 卡或 2 节点 A2
TP=8 PP=2 MAX_MODEL_LEN=65536 bash examples/minimax_m2_7/vllm/run_vllm.sh
```

### 后台运行

```bash
nohup bash examples/minimax_m2_7/vllm/run_vllm.sh > minimax_m2_7.log 2>&1 &
```

## 环境变量说明

### 基础配置

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `MODEL_PATH` | `/home/jianzhnie/llmtuner/hfhub/models/Eco-Tech/MiniMax-M2.7-w8a8-QuaRot` | 模型权重路径 |
| `SERVED_MODEL_NAME` | `minimax-m2.7` | API 中的模型名称 |
| `HOST` | `0.0.0.0` | 监听地址 |
| `PORT` | `8004` | 监听端口 |

### 并行配置

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `TP` | `4` | 张量并行度 (官方推荐 A2=4) |
| `PP` | `1` | 流水线并行度 |
| `ENABLE_EXPERT_PARALLEL` | `1` | 专家并行开关 (MoE 必需) |

### 内存与量化

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `DTYPE` | `bfloat16` | 计算数据类型 |
| `QUANTIZATION` | `ascend` | W8A8 QuaRot Ascend 量化 |
| `GPU_MEM_UTIL` | `0.85` | NPU 显存利用率 |
| `SWAP_SPACE` | `32` | CPU 交换空间 (GiB) |

### 序列调度

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `MAX_MODEL_LEN` | `32768` | 最大上下文长度 |
| `MAX_NUM_SEQS` | `16` | 最大并发请求数 |
| `MAX_NUM_BATCHED_TOKENS` | `8192` | 每 step 最大 token 数 |
| `ENABLE_CHUNKED_PREFILL` | `1` | 分块预填充 |

### 投机解码 (MTP)

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `SPECULATIVE_METHOD` | `mtp` | 投机解码方法 |
| `SPECULATIVE_NUM_TOKENS` | `3` | 每次投机 token 数 |

### 华为 NPU 专用

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `HCCL_OP_EXPANSION_MODE` | `AIV` | HCCL 操作扩展模式 |
| `HCCL_BUFFSIZE` | `1024` | HCCL 缓冲区大小 (MB, MiniMax 推荐 1024) |
| `OMP_PROC_BIND` | `false` | 禁用 OpenMP 线程绑定 |
| `OMP_NUM_THREADS` | `1` | OpenMP 线程数 |
| `PYTORCH_NPU_ALLOC_CONF` | `expandable_segments:True` | NPU 内存分配 |
| `TASK_QUEUE_ENABLE` | `1` | 任务队列优化 |
| `VLLM_ASCEND_ENABLE_FUSED_MC2` | `1` | 融合 MC2 通信 |
| `VLLM_ASCEND_ENABLE_FLASHCOMM1` | `1` | FlashComm 通信优化 |
| `VLLM_ASCEND_BALANCE_SCHEDULING` | `1` | 负载均衡调度 |

### 加速特性

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `PREFIX_CACHING` | `1` | 前缀缓存 (Agent 优化) |
| `ENFORCE_EAGER` | `1` | 禁用 CUDA Graph (NPU 推荐) |
| `CUDAGRAPH_MODE` | `FULL_DECODE_ONLY` | CUDA Graph 模式 |
| `ENABLE_NPUGRAPH_EX` | `true` | NPU Graph 扩展 |
| `FUSE_MULS_ADD` | `true` | 融合乘法加法 |
| `MULTISTREAM_OVERLAP_SHARED_EXPERT` | `true` | 多流共享专家重叠 |

### 工具调用

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `ENABLE_TOOL_CALLING` | `1` | 工具调用开关 |
| `TOOL_CALL_PARSER` | `minimax_m2` | MiniMax 工具调用解析器 |

## Agent 优化

```bash
--enable-prefix-caching          # Claude Code 系统提示缓存复用
--enable-chunked-prefill         # 长上下文分块预填充
--tool-call-parser minimax_m2    # MiniMax 工具调用解析器
--speculative-config '{"num_speculative_tokens": 3, "method": "mtp"}'
--max-num-seqs 16                # 高并发
--max-num-batched-tokens 8192    # 预填充吞吐量
```

## 并行策略推荐

```
场景               TP   PP   上下文    说明
──────────────────────────────────────────────
低延迟 (单节点)     4    1    32k      A2 官方推荐
均衡 (2 节点)       8    2    64k      标准多节点
高吞吐 (4 节点)     8    4    128k     大数据量
长上下文 (8 节点)   8    8    204k      全上下文
```

## 性能调优

### 低延迟场景
- 单节点 TP=4 (官方推荐)
- 减小 `MAX_NUM_SEQS` (如 4-8)
- 减小 `MAX_NUM_BATCHED_TOKENS` (如 4096)
- MTP tokens=1 (降低投机开销)

### 高吞吐场景
- 增大 TP 或增加节点
- 增大 `MAX_NUM_SEQS` (如 16-32)
- 启用 Chunked Prefill + Prefix Caching
- MTP tokens=3 (高吞吐投机)

### 长上下文场景
- 多节点扩展 PP
- 增大 `MAX_MODEL_LEN`
- 提高 `GPU_MEM_UTIL` (如 0.90-0.92)

## 功能验证

### 基础测试

```bash
bash examples/minimax_m2_7/vllm/curl_test.sh
```

### 手动 API 测试

```bash
# 检查服务
curl http://localhost:8004/v1/models

# Chat Completion
curl http://localhost:8004/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "minimax-m2.7",
    "messages": [{"role": "user", "content": "你好，请介绍一下自己"}],
    "max_tokens": 200
  }'

# 流式输出
curl http://localhost:8004/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "minimax-m2.7",
    "messages": [{"role": "user", "content": "写一首诗"}],
    "max_tokens": 200,
    "stream": true
  }'

# Tool Calling
curl http://localhost:8004/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "minimax-m2.7",
    "messages": [{"role": "user", "content": "Weather in Paris?"}],
    "tools": [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}],
    "max_tokens": 100
  }'
```

## Claude Code 集成

```bash
ANTHROPIC_BASE_URL=http://localhost:8004 \
ANTHROPIC_API_KEY=dummy \
ANTHROPIC_AUTH_TOKEN=dummy \
ANTHROPIC_DEFAULT_SONNET_MODEL=minimax-m2.7 \
ANTHROPIC_DEFAULT_HAIKU_MODEL=minimax-m2.7 \
ANTHROPIC_DEFAULT_OPUS_MODEL=minimax-m2.7 \
claude
```

## 常见问题

### Q: 为什么 TP=4 而不是 8？
A: MiniMax-M2.7 官方推荐 A2 环境使用 TP=4。这是模型架构和 W8A8 量化决定的，TP=8 可能导致显存不足或性能下降。A3 环境可尝试 TP=8。

### Q: W8A8 QuaRot 和其他量化有什么区别？
A: QuaRot 是 MiniMax 专有的量化方案，在 W8A8 精度下通过旋转矩阵优化激活分布，精度损失更小。

### Q: MTP 是否必须启用？
A: 推荐启用。MiniMax-M2.7 内置 3 个 MTP 模块 (`num_mtp_modules=3`)，投机解码可显著加速。不启用时删除 `--speculative-config`。

### Q: 如何调整上下文长度？
A: 通过 `MAX_MODEL_LEN`。MiniMax-M2.7 原生支持 204K，W8A8 量化下建议单节点 ≤32k，多节点扩展。

### Q: HCCL_BUFFSIZE 为什么是 1024？
A: MiniMax 官方推荐值。相比 DeepSeek (200-400) 和 GLM (200)，MiniMax 需要更大的 HCCL 缓冲以支持其通信模式。
