# Kimi-K2.6 W4A8 部署指南

> ✅ **部署验证通过** | 2026-06-09 | vLLM-Ascend 0.18.0rc1 + CANN 8.5.1
> **已验证配置**: TP=8 PP=2 (2节点: 40+153) | **上下文**: 262,144 (max_position_embeddings)
> Agent 优化版: Prefix Caching ✅ | max_num_seqs=16 | Tool Calling (kimi_k2) ✅ | Anthropic Messages API ✅

Kimi-K2.6 是部署的三个模型中**唯一支持 Pipeline Parallelism** 且**上下文最大 (262K)** 的模型，
同时支持多模态 (Vision Transformer)，推荐作为 Claude Code 的主模型。

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

### 注意力路径

Kimi-K2.6 使用 `DeepseekV3ForCausalLM` 注意力路径，不走 GLM 的 SFA/DSA 路径。
因此**不需要** `VLLM_ASCEND_ENABLE_FLASHCOMM1=0`，也**需要不同**的 HCCL 配置。

## ⚠️ 重要: 工具调用解析器

Kimi-K2.6 的 tokenizer 使用自定义工具调用 token (`<|tool_call_begin|>`, `<|tool_call_end|>` 等)，
**必须使用 `kimi_k2` parser**，不能使用 `deepseek_v3`。

| Parser | 状态 | 说明 |
|--------|------|------|
| `deepseek_v3` | ❌ 不兼容 | 报错: "could not locate tool call start/end tokens" |
| `kimi_k2` | ✅ 正确 | `KimiK2ToolParser`, 适配 Kimi 的 token 格式 |


## 部署

### 官方文档参考

- vLLM-Ascend 模型列表: https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/index.html
- vLLM 官方文档: https://docs.vllm.ai/en/stable/

### 硬件要求

#### 单节点部署

| 硬件 | 配置 | 推荐上下文 |
|------|------|-----------|
| Atlas 800 A2 (64G × 8) | W4A8, TP=8 | 32k |
| Atlas 800 A3 (64G × 16) | W4A8, TP=16 | 131k |

#### 多节点部署

| 节点数 | 配置 | 推荐上下文 |
|--------|------|-----------|
| 2 节点 × 8 NPU | TP=8, PP=2 | 64k |
| 4 节点 × 8 NPU | TP=8, PP=4 | 131k |
| 8 节点 × 8 NPU | TP=8, PP=8 | 262k |

### 已验证部署方案

#### 推荐: 2 节点 × 8 NPU (已验证)

```bash
MAX_MODEL_LEN=262144 TP=8 PP=2 DP=1 PORT=8003 bash run_vllm.sh
```

| 参数 | 值 | 说明 |
|------|-----|------|
| TP × PP | 8 × 2 | 均衡跨 2 节点 |
| max_model_len | **262,144** | 模型原生最大上下文 |
| max_num_seqs | 16 | 无 MTP 开销，高并发 |
| GPU 利用率 | 0.92 | 预留视觉组件空间 |
| 加载时间 | ~15 分钟 | 126 shards(比 GLM 多 26 个), 含 warmup |


## 快速开始

### 前置条件

模型路径: `/home/jianzhnie/llmtuner/hfhub/models/Eco-Tech/Kimi-K2.6-w4a8`

> **注意**: 模型包含自定义代码 (`configuration_kimi_k25.py`, `modeling_kimi_k25.py`)，必须启用 `--trust-remote-code`。

基于下面的脚本启动 NPU 容器和 Ray 集群：

```bash
# 1. 启动 NPU Docker 容器
bash scripts/docker/manage_npuslim_containers.sh start --file node_list.txt

# 2. 启动 Ray 集群
bash scripts/ray_cluster/start_npuslim_ray_cluster.sh start --file node_list.txt
```

### 部署 (2 节点, 256 K 全上下文)


```bash
# 部署 (2 节点, 262K)
# 1. 确保在 NPU 容器中执行以下命令
docker exec npuslim-env bash

# 2. 进入项目目录
cd /home/jianzhnie/llmtuner/llm/EasyInfer/examples/kimi_k2_6_w4a8

# 3. 部署模型
MAX_MODEL_LEN=262144 TP=8 PP=2 DP=1 PORT=8003 nohup bash run_vllm.sh >> vllm_kimi.log 2>&1 &

# 4. 验证模型部署 (~15 分钟后)
curl http://10.16.201.40:8003/v1/models
# 预期: model=kimi-k2.6, max_model_len=262144
```


## NPU 环境变量

### 性能优化

```bash
HCCL_OP_EXPANSION_MODE=AIV
HCCL_BUFFSIZE=800                # 384 专家需要更大的 HCCL 缓冲 (GLM 用 200)
OMP_PROC_BIND=false
OMP_NUM_THREADS=1
PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
VLLM_ASCEND_BALANCE_SCHEDULING=1
TASK_QUEUE_ENABLE=1              # Kimi 性能优化
```


## vLLM 参数配置

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
| `DATA_PARALLEL_SIZE` | `4` | 数据并行度 (官方推荐 dp4tp4) |

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


## API 验证

### Chat Completion
```bash
curl http://10.16.201.40:8003/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"kimi-k2.6","messages":[{"role":"user","content":"Hello"}],"max_tokens":50}'
```

### Tool Calling
```bash
curl http://10.16.201.40:8003/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"kimi-k2.6","messages":[{"role":"user","content":"Weather in Beijing?"}],"tools":[{"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}],"max_tokens":100}'
```

### Anthropic Messages (Claude Code 兼容 + tool_use)
```bash
curl http://10.16.201.40:8003/v1/messages \
  -H "Content-Type: application/json" -H "x-api-key: dummy" \
  -d '{
    "model":"kimi-k2.6",
    "max_tokens":100,
    "messages":[{"role":"user","content":"Read /tmp/test.txt"}],
    "tools":[{"name":"read_file","description":"Read a file","input_schema":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}]
  }'
# 预期返回: type=message, stop_reason=tool_use, tool_use name=read_file
```

## Claude Code 集成 (推荐)

Kimi-K2.6 是 Claude Code 推荐的模型，原因:
- **最大上下文 (262K)**: 比其他模型多 60K
- **无 MTP 开销**: max_num_seqs=16 (vs GLM 的 8)
- **多模态支持**: 可处理图像文件
- **更多专家 (384)**: 推理能力更强
- **kimi_k2 parser**: 工具调用 token 格式更清晰

```bash
ANTHROPIC_BASE_URL=http://10.16.201.40:8003 \
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

### Q: 为什么用 kimi_k2 而不是 deepseek_v3 parser？
A: Kimi-K2.6 tokenizer 使用自定义工具 token (`<|tool_call_begin|>` 等)，`deepseek_v3` parser 寻找 DeepSeek 专用分隔符 (`"éri"`)，不兼容。`kimi_k2` parser 专为 Kimi tokenizer 设计。这是一个已验证的 bug (Bug 2)。

### Q: 为什么没有 MTP？
A: config 中 `num_nextn_predict_layers=0`，表示模型不支持 Multi-Token Prediction，不需要配置投机解码参数。这也意味着没有 MTP 的内存开销，可以设置更高的 max_num_seqs。

### Q: 多模态如何使用？
A: 通过 `/v1/chat/completions` 传入 image 类型的 content。视觉 token 占用上下文窗口，建议预留 20-30%。纯文本 Agent 使用时视觉组件不激活。

### Q: PP 如何工作？
A: Kimi-K2.6 支持 Pipeline Parallelism，每层分配到不同节点。TP=8 PP=2 表示 2 个 PP stage，每个 stage 在 8 张 NPU 上运行 TP。

### Q: 384 专家对部署有什么影响？
A: 专家数更多 (384 vs 256)，EP_SIZE 需能整除 384 (推荐 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 384)。384 专家的 MoE 层参数量更大，需要更大的 SWAP_SPACE。