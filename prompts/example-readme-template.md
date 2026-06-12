# EasyInfer 示例 README 模板

本文件为 `examples/<model>/vllm/README.md` 的统一模板。新增模型时，复制本模板并替换 `<占位符>`。

---

```markdown
# <模型名> <量化> 部署指南

> **vLLM-Ascend 0.20.2 + CANN 9.0.0** | 端口: **<PORT>**
> 架构: <Arch> | <Experts> Experts | <MoE/MLA/...> | <量化> 量化
> 已验证配置: <TP> PP=<PP> (<节点描述>) | 上下文: <MAX_LEN> | <关键特性>

## 模型简介

| 属性 | 值 |
|------|-----|
| **架构** | <Arch> (<备注>) |
| **路由专家** | <N> (<每 token 激活数>) |
| **隐藏维度** | <N> |
| **网络层数** | <N> |
| **MLA** | <kv_lora_rank>, <q_lora_rank>, <head_dim> |
| **原生上下文** | **<MAX_POSITION_EMBEDDINGS>** |
| **量化方式** | <Quant> |
| **MTP** | <支持/不支持> (num_nextn_predict_layers=<N>) |
| **PP 支持** | ✅/❌ 支持 Pipeline Parallelism |
| **多模态** | ✅/❌ <Vision/...> |
| **词表大小** | <N> |
| **工具调用解析器** | <parser> |
| **推理解析器** | <parser> / 不适用 |

### 架构注意事项

<关键兼容性说明，例如 FLASHCOMM1 必须为 0、必须使用特定 tool parser、DSA CP 路径不兼容等。>

### 官方文档参考

<如模型无特定 vLLM-Ascend 文档，可省略本节，仅保留 vLLM 官方文档。>

- vLLM-Ascend 模型文档: <url>
- vLLM 官方文档: https://docs.vllm.ai/en/stable/

## 快速开始

### 前置条件

模型路径: `/home/jianzhnie/llmtuner/hfhub/models/Eco-Tech/<MODEL_REL_PATH>`

```bash
# 1. 启动 NPU Docker 容器
bash scripts/docker/manage_npuslim_containers.sh start --file node_list.txt

# 2. 启动 Ray 集群
bash scripts/ray_cluster/start_npuslim_ray_cluster.sh start --file node_list.txt
```

### 部署

```bash
# 单节点 (<默认上下文>, TP=<TP>)
bash examples/<model_dir>/vllm/run_vllm.sh

# 多节点 (<大上下文>)
TP=<TP> PP=<PP> MAX_MODEL_LEN=<LEN> bash examples/<model_dir>/vllm/run_vllm.sh

# 后台运行
nohup bash examples/<model_dir>/vllm/run_vllm.sh > <log_file>.log 2>&1 &

# 使用传统包装器部署
bash examples/<model_dir>/vllm/vllm_server.sh
```

### 验证

```bash
# 运行测试脚本
bash examples/<model_dir>/vllm/curl_test.sh

# 手动验证
curl http://localhost:<PORT>/v1/models
curl http://localhost:<PORT>/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"<api-name>","messages":[{"role":"user","content":"Hello"}],"max_tokens":50}'
```

## 并行策略

| 场景 | TP | PP | DP | NPU | 上下文 | 状态 |
|------|-----|-----|-----|-----|--------|------|
| 单节点 | <TP> | 1 | 1 | <N> | <32K> | ✅/⚠️ |
| 多节点 | <TP> | <PP> | <DP> | <N> | <LEN> | ✅/⚠️ |

> 模型特定约束说明，例如"不支持 PP，多节点必须使用大 TP"。>

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

## 功能验证清单

### 基础功能

| 功能 | 状态 | 脚本 |
|------|------|------|
| 基础 Chat Completion | ✅/⚠️ | `run_vllm.sh` |
| Tool Calling (<parser>) | ✅/⚠️ | `curl_test.sh` |
| Anthropic Messages API | ✅/⚠️ | `curl_test.sh` |
| MTP 投机解码 | ✅/❌ | `run_vllm.sh` |

### 高级功能

| 功能 | 状态 | 脚本 | 硬件要求 |
|------|------|------|----------|
| 基于 Mooncake 多实例 PD 共置部署 | 📋/❌ | `run_pd_colocated.sh` | 多节点 + Mooncake + RoCE |
| 预填充-解码分离部署 | 📋/⚠️/❌ | `run_pd_disaggregated.sh` | 多节点 + Mooncake |
| 长序列上下文并行 | 📋/⚠️/❌ | `run_long_seq_cp.sh` | Atlas A3 |
| 动态分块流水线并行 | 📋/❌ | `run_dynamic_chunked_pp.sh` | PP ≥ 2 |

## 常见问题

### Q: <问题 1>?

A: <回答>

### Q: <问题 2>?

A: <回答>
```

---

## 填写检查清单

复制模板后，逐项确认：

- [ ] H1 模型名与量化正确
- [ ] 顶部 banner 包含 vLLM 版本、端口、已验证配置、架构
- [ ] 模型简介属性表完整（架构/专家/隐藏维度/层数/MLA/上下文/量化/MTP/PP/多模态/词表/parser/推理解析器）
- [ ] 非 MTP 模型已删除"投机解码 (MTP)"环境变量表
- [ ] 非多模态模型已删除"多模态"参数表
- [ ] 快速开始中的 `MODEL_PATH` 和脚本路径正确
- [ ] 并行策略表包含 TP/PP/DP/NPU/上下文/状态六列
- [ ] 环境变量按基础/并行/内存/序列/NPU/加速/工具/MTP/多模态分组
- [ ] Claude Code 集成中的端口和模型名正确
- [ ] 功能验证清单状态符号统一（✅ ⚠️ ❌ 📋）
- [ ] 常见问题覆盖模型特异性问题
- [ ] 无重复标题
- [ ] 所有代码块语法正确
