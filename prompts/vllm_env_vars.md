## 环境变量

### 基础配置

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `MODEL_PATH` | `.../<MODEL>` | 模型权重路径 |
| `SERVED_MODEL_NAME` | `<api-name>` | API 中的模型名称 |
| `HOST` | `0.0.0.0` | 监听地址 |
| `PORT` | `<PORT>` | 监听端口 |

### 并行配置

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `TP` / `TENSOR_PARALLEL_SIZE` | `<TP>` | 张量并行度 |
| `PP` / `PIPELINE_PARALLEL_SIZE` | `<PP>` | 流水线并行度 |
| `ENABLE_EXPERT_PARALLEL` | `1` | 专家并行开关 (MoE 必需) |
| `DATA_PARALLEL_SIZE` | `<DP>` | 数据并行度 |

### 内存与量化

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `DTYPE` | `bfloat16` | 计算数据类型 |
| `QUANTIZATION` | `ascend` | Ascend 量化 |
| `GPU_MEM_UTIL` / `GPU_MEMORY_UTILIZATION` | `<0.XX>` | NPU 显存利用率 |
| `SWAP_SPACE` | `<N>` | CPU 交换空间 (GiB) |

### 序列调度

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `MAX_MODEL_LEN` | `<N>` | 最大上下文长度 |
| `MAX_NUM_SEQS` | `<N>` | 最大并发请求数 |
| `MAX_NUM_BATCHED_TOKENS` | `<N>` | 每 step 最大 token 数 |
| `ENABLE_CHUNKED_PREFILL` | `1` | 分块预填充 |
| `CHAT_TEMPLATE_CONTENT_FORMAT` | `string` | Chat Template 内容格式 |

### NPU 专用

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `HCCL_OP_EXPANSION_MODE` | `AIV` | HCCL 操作扩展模式 |
| `HCCL_BUFFSIZE` | `<N>` | HCCL 缓冲区大小 (MB) |
| `OMP_PROC_BIND` | `false` | 禁用 OpenMP 线程绑定 |
| `OMP_NUM_THREADS` | `1` | OpenMP 线程数 |
| `PYTORCH_NPU_ALLOC_CONF` | `expandable_segments:True` | NPU 内存分配 |
| `VLLM_ASCEND_ENABLE_FLASHCOMM1` | `<0/1>` | FlashComm 通信优化 |
| `VLLM_ASCEND_ENABLE_MLAPO` | `<0/1>` | MLA 算子融合优化 |
| `VLLM_ASCEND_BALANCE_SCHEDULING` | `1` | 负载均衡调度 |

### 加速特性

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `PREFIX_CACHING` | `1` | 前缀缓存 |
| `ENFORCE_EAGER` | `1` | 禁用 CUDA Graph |
| `CUDAGRAPH_MODE` | `FULL_DECODE_ONLY` | CUDA Graph 模式 |
| `ENABLE_NPUGRAPH_EX` | `true` | NPU Graph 扩展 |
| `FUSE_MULS_ADD` | `true` | 融合乘法加法 |
| `MULTISTREAM_OVERLAP_SHARED_EXPERT` | `true` | 多流共享专家重叠 |
| `NUM_SCHEDULER_STEPS` | `<N>` | 多步调度步数 |
| `ENABLE_ASYNC_SCHEDULING` | `1` | 异步调度 |

### 工具调用

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `ENABLE_TOOL_CALLING` | `1` | 工具调用开关 |
| `TOOL_CALL_PARSER` | `<parser>` | 工具调用解析器 |

### 投机解码 (MTP)

> 仅 MTP 模型保留本节；非 MTP 模型删除本节，并在功能验证清单中标注 ❌ 不支持。

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `SPECULATIVE_METHOD` | `mtp` | 投机解码方法 |
| `SPECULATIVE_NUM_TOKENS` | `3` | 每次投机 token 数 |

### 多模态 (如适用)

> 仅多模态模型保留本节；纯文本模型删除本节。

| 参数 | 值 | 说明 |
|------|-----|------|
| `--allowed-local-media-path` | `/home/jianzhnie/llmtuner/` | 本地媒体文件路径 |
| `--mm-encoder-tp-mode` | `data` | 视觉编码器 TP 模式 |
| `--language-model-only` | 默认启用 | 纯文本 Agent 场景跳过 Vision Encoder |

### Agent 优化参数

```bash
--enable-prefix-caching          # Claude Code 系统提示缓存复用
--enable-chunked-prefill         # 长上下文分块预填充
--enable-auto-tool-choice        # Anthropic API tool_use 必需
--tool-call-parser <parser>     # 工具调用解析器
--max-num-seqs <N>              # 最大并发请求数
--max-num-batched-tokens <N>    # 预填充吞吐量
```

## Claude Code 集成

```bash
ANTHROPIC_BASE_URL=http://localhost:<PORT> \
ANTHROPIC_API_KEY=dummy \
ANTHROPIC_AUTH_TOKEN=dummy \
ANTHROPIC_DEFAULT_SONNET_MODEL=<api-name> \
ANTHROPIC_DEFAULT_HAIKU_MODEL=<api-name> \
ANTHROPIC_DEFAULT_OPUS_MODEL=<api-name> \
claude
```