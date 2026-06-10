# GLM-5.1 W4A8 部署指南

> ✅ **部署验证通过** | 2026-06-09 | vLLM-Ascend 0.18.0rc1 + CANN 8.5.1
> **已验证配置**: TP=16 PP=1 (2节点: 40+153) | **上下文**: 202,752 (max_position_embeddings)
> Agent 优化版: Prefix Caching ✅ | MTP 投机解码 ✅ | Tool Calling (glm47) ✅ | Anthropic Messages API ✅

GLM-5.1 是 GLM-5 的升级版，架构完全相同 (GlmMoeDsaForCausalLM)，改进了训练数据和后训练流程。
部署配置与 GLM-5 通用，仅需修改 `MODEL_PATH` 和 `SERVED_MODEL_NAME`。

## 模型简介

| 属性 | 值 |
|------|-----|
| **架构** | GlmMoeDsaForCausalLM (MoE + DSA + MLA) |
| **路由专家** | 256 (每 Token 激活 8 专家) |
| **隐藏维度** | 6144 |
| **网络层数** | 78 |
| **MLA** | kv_lora_rank=512, q_lora_rank=2048, head_dim=64 |
| **原生上下文** | **202,752** |
| **量化方式** | W4A8 (4-bit 权重 + 8-bit 激活) |
| **MTP** | num_nextn_predict_layers=1, deepseek_mtp |
| **PP 支持** | ❌ 不支持 Pipeline Parallelism |
| **词表大小** | 154,880 |

### 架构注意事项

GLM-5 的 config.json 包含 `index_topk: 2048`，导致 vLLM-Ascend 将其识别为 DeepSeek V3.2。
这会触发 DSA CP (Context Parallelism) 路径。W4A8 量化环境下 CP 路径不兼容，**必须设置 `VLLM_ASCEND_ENABLE_FLASHCOMM1=0`** 禁用。


## 部署

### 官方文档参考

- GLM-5 官方部署文档: https://docs.vllm.ai/projects/ascend/en/v0.18.0/tutorials/models/GLM5.html

### 硬件要求

#### 单节点部署

| 硬件 | 配置 | 推荐上下文 |
|------|------|-----------|
| Atlas 800 A2 (64G × 8) | W4A8, TP=8 | 32k |
| Atlas 800 A3 (64G × 16) | W4A8, TP=16 | 200k |

#### 多节点部署  

| 节点数 | 配置 | 推荐上下文 |
|--------|------|-----------|
| 2 节点 × 8 NPU | TP=16, DP=1 | 200k (大TP跨节点) |
| 8 节点 × 8 NPU | TP=64, DP=1 | 200k |

> **注意**: GLM-5.1 不支持 Pipeline Parallelism (PP)，多节点部署应使用更大的 TP 值跨节点。


### 已验证部署方案

### 方案 A: 2 节点 × 8 NPU (推荐，已验证)

```bash
# 节点: 10.16.201.40 + 10.16.201.153
# 总 NPU: 16 × 64GB
# 配置: TP=16 PP=1, 使用大 TP 跨节点

MAX_MODEL_LEN=202752 TP=16 PP=1 PORT=8002 bash run_vllm.sh
```

| 参数 | 值 | 说明 |
|------|-----|------|
| TP × PP | 16 × 1 | 大 TP 跨 2 节点 |
| max_model_len | **202,752** | 模型原生最大上下文 |
| max_num_seqs | 8 | MTP 占用额外内存 |
| GPU 利用率 | 0.94 | W4A8 高利用率 |
| MTP | ✅ | 3 tokens, deepseek_mtp |
| 加载时间 | ~12 分钟 | 含权重加载 + warmup |

### 方案 B: 单节点无 MTP (内存优化)

```bash
MAX_MODEL_LEN=32768 TP=8 PP=1 PORT=8002 bash run_vllm_nomtp.sh
```

| 参数 | 值 | 说明 |
|------|-----|------|
| TP × PP | 8 × 1 | 单节点 |
| max_model_len | 32,768 | MTP 内存限制 |
| max_num_seqs | 4 | 单节点内存紧张 |

禁用 MTP 后单节点可达 32K 上下文。使用 `run_vllm_nomtp.sh` (移除 `--speculative-config`)。

## 快速开始

### 前置条件

模型路径: `/home/jianzhnie/llmtuner/hfhub/models/Eco-Tech/GLM-5.1-w4a8`

基于下面的脚本启动 NPU 容器和 Ray 集群：

```bash
# 1. 启动 NPU Docker 容器
bash scripts/docker/manage_npuslim_containers.sh start --file node_list.txt

# 2. 启动 Ray 集群
bash scripts/ray_cluster/start_npuslim_ray_cluster.sh start --file node_list.txt
```

### 部署 (2 节点, 202K 全上下文)

```bash
# 部署 (2 节点, 202K)
# 1. 确保在 NPU 容器中执行以下命令
docker exec npuslim-env bash

# 2. 进入项目目录
cd /home/jianzhnie/llmtuner/llm/EasyInfer/examples/glm5_1_w4a8

# 3. 部署模型
MAX_MODEL_LEN=202752 TP=16 PP=1 PORT=8002 nohup bash run_vllm.sh > vllm_glm51.log 2>&1 &

# 4. 验证模型部署
# 等待 ~12 分钟后验证
curl http://localhost:8002/v1/models
# 预期: model=glm-5.1, max_model_len=202752
```

## NPU 环境变量 (与 GLM-5 完全相同)

### 必须设置
```bash
VLLM_ASCEND_ENABLE_FLASHCOMM1=0  # ⚠️ 必须为 0！防止 DSA CP crash
VLLM_ASCEND_ENABLE_MLAPO=1       # MLA 算子融合优化
```

### 性能优化
```bash
HCCL_OP_EXPANSION_MODE=AIV
HCCL_BUFFSIZE=200                # 256 专家 HCCL 缓冲
OMP_PROC_BIND=false
OMP_NUM_THREADS=1
PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
VLLM_ASCEND_BALANCE_SCHEDULING=1
```

## vLLM 参数配置

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

### Agent 优化参数配置

```bash
--enable-prefix-caching          # Claude Code 系统提示缓存复用 (~90% KV cache hit)
--enable-chunked-prefill         # 长上下文分块预填充
--enable-auto-tool-choice        # Anthropic API tool_use 必需
--tool-call-parser glm47         # GLM 系列工具调用解析器
--speculative-config '{"num_speculative_tokens": 3, "method": "deepseek_mtp"}'
--max-num-seqs 8                  # MTP 内存限制下的最大并发请求数
--max-num-batched-tokens 16384    # 预填充吞吐量
```

### ⚠️ 禁用的参数 (vLLM-Ascend 0.18.0rc1 不支持)
- `--num-scheduler-steps` — 当前版本不支持
- `--async-scheduling` — Ray backend 不支持 (仅 mp/external_launcher 支持)
- `--enable-npugraph-ex` — 与 `--enforce-eager` 冲突
- `VLLM_ASCEND_ENABLE_FLASHCOMM1=1` — GLM W4A8 下会触发 DSA CP crash


## 并行策略

| 场景 | TP | PP | NPU | 上下文 | 状态 |
|------|-----|-----|-----|--------|------|
| 单节点轻量 | 8 | 1 | 8 | 32K | ✅ |
| 2 节点全量 | 16 | 1 | 16 | **202K** | ✅ 已验证 |
| 4 节点大规模 | 32 | 1 | 32 | 202K | ⚠️ TP>16 设备映射问题 |

> GLM-5 **不支持 Pipeline Parallelism**，多节点必须使用大 TP。


## API 验证

### Chat Completion
```bash
curl http://10.16.201.40:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"glm-5","messages":[{"role":"user","content":"Hello"}],"max_tokens":50}'
```

### Tool Calling
```bash
curl http://10.16.201.40:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"glm-5","messages":[{"role":"user","content":"Weather in Paris?"}],"tools":[{"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}],"max_tokens":100}'
```

### Anthropic Messages API (Claude Code 兼容)
```bash
curl http://10.16.201.40:8001/v1/messages \
  -H "Content-Type: application/json" -H "x-api-key: dummy" \
  -d '{"model":"glm-5","messages":[{"role":"user","content":"Hi"}],"max_tokens":30}'
```

## Claude Code 集成

```bash
ANTHROPIC_BASE_URL=http://localhost:8002 \
ANTHROPIC_API_KEY=dummy \
ANTHROPIC_AUTH_TOKEN=dummy \
ANTHROPIC_DEFAULT_SONNET_MODEL=glm-5.1 \
ANTHROPIC_DEFAULT_HAIKU_MODEL=glm-5.1 \
ANTHROPIC_DEFAULT_OPUS_MODEL=glm-5.1 \
claude
```

## 常见问题

### Q: 为什么必须设置 FLASHCOMM1=0？
A: GLM-5 的 `index_topk: 2048` 触发 DSA CP 路径，W4A8 下缺少 `aclnn_input_scale` 属性导致 crash。详见 Bug 3。

### Q: MTP 投机解码对内存有什么影响？
A: MTP 会加载第二份模型权重，显著减少 KV cache 可用空间。TP=8 单节点时 MTP 导致 max_model_len 从 64K 降至 ~10K。TP=16 时可在 2 节点上达到 202K。

### Q: 为什么不用 PP？
A: GLM-5/5.1 架构不支持 Pipeline Parallelism (`SupportsPP` 接口缺失)。多节点必须使用大 TP。
