# Kimi-K2.6 W4A8 部署指南

> **vLLM-Ascend 0.20.2 + CANN 9.0.0** 

> 架构: KimiK25ForConditionalGeneration | 384 Experts | MoE | MLA | Vision (多模态) | W4A8 量化

> **已验证配置**: TP=8 PP=2 (2节点: 40+153) | **上下文**: 262,144 (max_position_embeddings)
> Agent 优化版: Prefix Caching ✅ | max_num_seqs=16 | Tool Calling (kimi_k2) ✅ | Anthropic Messages API ✅


## 模型简介

| 属性 | 值 |
|------|-----|
| **架构** | KimiK25ForConditionalGeneration → DeepseekV3ForCausalLM |
| **路由专家** | 384 (每 Token 激活 8 专家) |
| **隐藏维度** | 7168 |
| **网络层数** | 61 |
| **MLA** | kv_lora_rank=512, q_lora_rank=1536, v_head_dim=128 |
| **原生上下文** | **262,144** |
| **量化方式** | W4A8 (4-bit 权重 + 8-bit 激活) |
| **MTP** | ❌ 不支持 (num_nextn_predict_layers=0) |
| **PP 支持** | ✅ **支持 Pipeline Parallelism** |
| **多模态** | ✅ Vision Transformer (27 层) |
| **词表大小** | 163,840 |

> **与 Kimi-K2 的区别**: Kimi-K2.6 增加视觉多模态能力 (Vision Transformer + unified vision chunk)，文本骨干基于 DeepSeek V3 架构 (384 专家)。

### 注意事项

Kimi-K2.6 使用 `DeepseekV3ForCausalLM` 注意力路径，不走 GLM 的 SFA/DSA 路径。
因此**不需要** `VLLM_ASCEND_ENABLE_FLASHCOMM1=0`，也**需要不同**的 HCCL 配置。

### ⚠️ 重要: 工具调用解析器

Kimi-K2.6 的 tokenizer 使用自定义工具调用 token (`<|tool_call_begin|>`, `<|tool_call_end|>` 等)，
**必须使用 `kimi_k2` parser**，不能使用 `deepseek_v3`。

| Parser | 状态 | 说明 |
|--------|------|------|
| `deepseek_v3` | ❌ 不兼容 | 报错: "could not locate tool call start/end tokens" |
| `kimi_k2` | ✅ 正确 | `KimiK2ToolParser`, 适配 Kimi 的 token 格式 |

## 快速开始

### 前置条件

模型路径: `/home/jianzhnie/llmtuner/hfhub/models/Eco-Tech/Kimi-K2.6-w4a8`

```bash
# 1. 启动 NPU Docker 容器
bash scripts/docker/manage_npuslim_containers.sh start --file node_list.txt

# 2. 启动 Ray 集群
bash scripts/ray_cluster/start_npuslim_ray_cluster.sh start --file node_list.txt
```

### 部署

```bash
# 单节点 (32K 上下文, TP=8)
bash examples/kimi_k2_6_w4a8/vllm/run_vllm.sh

# 2 节点 PP (大上下文)
TP=8 PP=2 MAX_MODEL_LEN=131072 bash examples/kimi_k2_6_w4a8/vllm/run_vllm.sh

# 后台运行
nohup bash examples/kimi_k2_6_w4a8/vllm/run_vllm.sh > kimi_k26_vllm.log 2>&1 &

# 使用传统包装器部署
bash examples/kimi_k2_6_w4a8/vllm/vllm_server.sh
```

### 验证

```bash
# 运行测试脚本
bash examples/kimi_k2_6_w4a8/vllm/curl_test.sh

# 手动验证
curl http://localhost:8003/v1/models
curl http://localhost:8003/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"kimi-k2.6","messages":[{"role":"user","content":"Hello"}],"max_tokens":50}'
```

### 并行策略

| 场景 | TP | PP | DP | NPU | 上下文 |
|------|-----|-----|-----|-----|--------|
| 单节点 | 8 | 1 | 1 | 8 | 32K |
| 2 节点 PP | 8 | 2 | 1 | 16 | 131K |
| 多节点扩展 | 8 | 4 | 2 | 64 | 131K |

> Kimi-K2.6 **支持 Pipeline Parallelism**，适合多节点扩展。


## vLLM 参数配置

### 基础配置

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `MODEL_PATH` | `Eco-Tech/Kimi-K2.6-w4a8` | 模型权重路径 |
| `SERVED_MODEL_NAME` | `kimi-k2.6` | API 中的模型名称 |
| `HOST` | `0.0.0.0` | 监听地址 |
| `PORT` | `8003` | 监听端口 |

### 并行配置

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `TENSOR_PARALLEL_SIZE` | `8` | 张量并行度 (A2=8, A3=16) |
| `PIPELINE_PARALLEL_SIZE` | `1` | 流水线并行度 |
| `ENABLE_EXPERT_PARALLEL` | `1` | 专家并行开关 (384 专家 MoE 必需) |
| `DATA_PARALLEL_SIZE` | `4` | 数据并行度 (官方推荐 dp4 tp4) |

### 内存与量化

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `DTYPE` | `bfloat16` | 计算数据类型 |
| `QUANTIZATION` | `ascend` | W4A8 Ascend 量化 |
| `GPU_MEMORY_UTILIZATION` | `0.9` | NPU 显存利用率 |
| `SWAP_SPACE` | `32` | CPU 交换空间 (GiB, 384 专家需较大空间) |

### 序列调度

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `MAX_MODEL_LEN` | `32768` | 最大上下文长度 |
| `MAX_NUM_SEQS` | `64` | 最大并发请求数 |
| `MAX_NUM_BATCHED_TOKENS` | `16384` | 每 step 最大 token 数 |
| `ENABLE_CHUNKED_PREFILL` | `1` | 分块预填充 |


### 加速特性

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `PREFIX_CACHING` | `1` | 前缀缓存 (Agent 优化) |
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
| `TOOL_CALL_PARSER` | `kimi_k2` | Kimi 工具调用解析器 (基于 DeepSeek V3 但适配 Kimi tokenizer) |


### Agent 优化
```bash
--enable-prefix-caching          # 前缀缓存 (无 MTP 开销，效果更好)
--enable-chunked-prefill         # 长上下文分块预填充
--enable-auto-tool-choice        # Anthropic API tool_use 必需
--tool-call-parser kimi_k2       # ⚠️ 必须是 kimi_k2，不是 deepseek_v3
--max-num-seqs 16                # 高并发 (无 MTP 内存开销)
--max-num-batched-tokens 16384
```

### 多模态
```bash
--allowed-local-media-path /     # 本地媒体文件路径
--mm-encoder-tp-mode data        # 视觉编码器 TP 模式
```


## Claude Code 集成

```bash
ANTHROPIC_BASE_URL=http://localhost:8003 \
ANTHROPIC_API_KEY=dummy \
ANTHROPIC_AUTH_TOKEN=dummy \
ANTHROPIC_DEFAULT_SONNET_MODEL=kimi-k2.6 \
ANTHROPIC_DEFAULT_HAIKU_MODEL=kimi-k2.6 \
ANTHROPIC_DEFAULT_OPUS_MODEL=kimi-k2.6 \
claude
```

## 常见问题

### Q: Kimi-K2.6 和 Kimi-K2 有什么区别？
A: Kimi-K2.6 增加了多模态 (Vision) 能力，包含 Vision Transformer (27 层)。文本骨干基于 DeepSeek V3 架构 (384 专家)。纯文本推理性能与 Kimi-K2 类似。

### Q: Kimi-K2.6 支持 MTP 投机解码吗？
A: 不支持。模型 config 中 `num_nextn_predict_layers=0`，无 MTP 模块。

### Q: --language-model-only 是什么？
A: 仅加载语言模型部分，跳过 Vision Encoder，适合纯文本场景和 Agent 使用，节省显存。

### Q: 多模态如何使用？
A: 通过 `/v1/chat/completions` 传入 image 类型的 content。视觉 token 占用上下文窗口，建议预留 20-30%。纯文本 Agent 使用时视觉组件不激活。

### Q: PP 如何工作？
A: Kimi-K2.6 支持 Pipeline Parallelism，每层分配到不同节点。TP=8 PP=2 表示 2 个 PP stage，每个 stage 在 8 张 NPU 上运行 TP。

### Q: 384 专家对部署有什么影响？
A: 专家数更多 (384 vs 256)，EP_SIZE 需能整除 384 (推荐 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 384)。384 专家的 MoE 层参数量更大，需要更大的 SWAP_SPACE。