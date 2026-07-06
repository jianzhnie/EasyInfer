# GLM-5 部署指南

> **vLLM-Ascend 0.20.2 + CANN 9.0.0** | 端口: **8001**
> 架构: GlmMoeDsaForCausalLM | 256 Experts | MoE | MLA | MTP=1
> 已验证配置: TP=8 PP=1 (单节点) | 上下文: 32K | max_position_embeddings: 202,752
> Agent 优化版: Prefix Caching ✅ | MTP 投机解码 ✅ | Tool Calling (glm47) ✅ | Anthropic Messages API ✅

## 模型简介

| 属性 | 值 |
|------|-----|
| **架构** | GlmMoeDsaForCausalLM (MoE + DSA + MLA) |
| **路由专家** | 256 (每 Token 激活 8 专家) |
| **隐藏维度** | 6144 |
| **网络层数** | 78 |
| **MLA** | kv_lora_rank=512, q_lora_rank=2048, head_dim=64 |
| **原生上下文** | **202,752** |
| **MTP** | num_nextn_predict_layers=1 |
| **PP 支持** | ❌ 不支持 Pipeline Parallelism |
| **工具调用解析器** | glm47 |
| **推理解析器** | glm45 |
| **词表大小** | 154,880 |

### 架构注意事项

GLM-5 的 config.json 包含 `index_topk: 2048`，导致 vLLM-Ascend 识别为 DeepSeek V3.2，触发 DSA CP 路径。W4A8 量化下 CP 路径不兼容，**必须设置 `VLLM_ASCEND_ENABLE_FLASHCOMM1=0`**。

### 官方文档参考

- GLM-5 官方部署文档: https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/GLM5.html
- vLLM 官方文档: https://docs.vllm.ai/en/stable/

## 快速开始

### 前置条件

模型路径: `/home/jianzhnie/llmtuner/hfhub/models/ZhipuAI/GLM-5`

```bash
# 1. 启动 NPU Docker 容器
bash scripts/docker/manage_npuslim_containers.sh start --file node_list.txt

# 2. 启动 Ray 集群
bash scripts/ray_cluster/start_npuslim_ray_cluster.sh start --file node_list.txt
```

### 部署

```bash
# 单节点 (32K 上下文, TP=8)
bash examples/glm5/vllm/run_vllm.sh

# 2 节点大 TP (202K 上下文)
TP=16 MAX_MODEL_LEN=202752 bash examples/glm5/vllm/run_vllm.sh

# 后台运行
nohup bash examples/glm5/vllm/run_vllm.sh > glm5_vllm.log 2>&1 &
```

### 验证

```bash
# 运行测试脚本
bash examples/glm5/vllm/curl_test.sh

# 手动验证
curl http://localhost:8001/v1/models
curl http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"glm-5","messages":[{"role":"user","content":"Hello"}],"max_tokens":50}'
```

## 并行策略

| 场景 | TP | PP | DP | NPU | 上下文 | 状态 |
|------|-----|-----|-----|-----|--------|------|
| 单节点轻量 | 8 | 1 | 1 | 8 | 32K | ✅ |
| 2 节点全量 | 16 | 1 | 1 | 16 | **202K** | ⚠️ 待验证 |
| 4 节点大规模 | 32 | 1 | 1 | 32 | 202K | ⚠️ 待验证 |

> GLM-5 **不支持 Pipeline Parallelism**，多节点必须使用大 TP。

## 环境变量

### 基础配置

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `MODEL_PATH` | `/home/jianzhnie/llmtuner/hfhub/models/ZhipuAI/GLM-5` | 模型权重路径 |
| `SERVED_MODEL_NAME` | `glm-5` | API 中的模型名称 |
| `HOST` | `0.0.0.0` | 监听地址 |
| `PORT` | `8001` | 监听端口 |

### 并行配置

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `TP` / `TENSOR_PARALLEL_SIZE` | `8` | 张量并行度 |
| `PP` / `PIPELINE_PARALLEL_SIZE` | `1` | 流水线并行度 |
| `ENABLE_EXPERT_PARALLEL` | `1` | 专家并行开关 (MoE 必需) |
| `DATA_PARALLEL_SIZE` | `1` | 数据并行度 |

### 内存与量化

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `DTYPE` | `bfloat16` | 计算数据类型 |
| `QUANTIZATION` | `ascend` | Ascend W4A8 量化 |
| `GPU_MEM_UTIL` / `GPU_MEMORY_UTILIZATION` | `0.94` | NPU 显存利用率 |
| `SWAP_SPACE` | `16` | CPU 交换空间 (GiB) |

### 序列调度

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `MAX_MODEL_LEN` | `32768` | 最大上下文长度 |
| `MAX_NUM_SEQS` | `8` | 最大并发请求数 |
| `MAX_NUM_BATCHED_TOKENS` | `16384` | 每 step 最大 token 数 |
| `ENABLE_CHUNKED_PREFILL` | `1` | 分块预填充 |
| `CHAT_TEMPLATE_CONTENT_FORMAT` | `string` | Chat Template 内容格式 |

### NPU 专用

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `HCCL_OP_EXPANSION_MODE` | `AIV` | HCCL 操作扩展模式 |
| `HCCL_BUFFSIZE` | `200` | HCCL 缓冲区大小 (MB) |
| `OMP_PROC_BIND` | `false` | 禁用 OpenMP 线程绑定 |
| `OMP_NUM_THREADS` | `1` | OpenMP 线程数 |
| `PYTORCH_NPU_ALLOC_CONF` | `expandable_segments:True` | NPU 内存分配 |
| `VLLM_ASCEND_ENABLE_FLASHCOMM1` | `0` | 通信优化 (GLM 必须为 0) |
| `VLLM_ASCEND_ENABLE_MLAPO` | `1` | MLA 算子融合优化 |
| `VLLM_ASCEND_BALANCE_SCHEDULING` | `1` | 负载均衡调度 |

### 投机解码 (MTP)

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `SPECULATIVE_METHOD` | `mtp` | 投机解码方法 |
| `SPECULATIVE_NUM_TOKENS` | `3` | 每次投机 token 数 |

## Claude Code 集成

```bash
ANTHROPIC_BASE_URL=http://localhost:8001 \
ANTHROPIC_API_KEY=dummy \
ANTHROPIC_AUTH_TOKEN=dummy \
ANTHROPIC_DEFAULT_SONNET_MODEL=glm-5 \
ANTHROPIC_DEFAULT_HAIKU_MODEL=glm-5 \
ANTHROPIC_DEFAULT_OPUS_MODEL=glm-5 \
claude
```

## 功能验证清单

### 基础功能

| 功能 | 状态 | 脚本 |
|------|------|------|
| 基础 Chat Completion | ⚠️ 待验证 | `run_vllm.sh` |
| Tool Calling (glm47) | ⚠️ 待验证 | `curl_test.sh` |
| Anthropic Messages API | ⚠️ 待验证 | `curl_test.sh` |
| MTP 投机解码 | ⚠️ 待验证 | `run_vllm.sh` (内置) |

## 常见问题

### Q: 为什么必须设置 FLASHCOMM1=0？

A: GLM-5 的 `index_topk: 2048` 触发 DSA CP 路径，W4A8 下缺少 `aclnn_input_scale` 属性导致 crash。

### Q: MTP 投机解码对内存有什么影响？

A: MTP 加载第二份模型权重，减少 KV cache 可用空间。TP=8 单节点时 max_model_len 从 64K 降至 ~32K。

### Q: 为什么不用 PP？

A: GLM-5 架构不支持 Pipeline Parallelism (`SupportsPP` 接口缺失)。多节点必须使用大 TP。
