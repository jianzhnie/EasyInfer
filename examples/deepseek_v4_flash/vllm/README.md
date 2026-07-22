# DeepSeek-V4-Flash W8A8 MTP 部署指南

> ✅ **已验证 PASS** | vLLM-Ascend 0.22.1rc1 + CANN 8.5.1 | 端口: **8000**
> 已验证配置: TP=8 PP=1 (单节点), MAX_MODEL_LEN=65536；2026-07-21 复测通过
> 历史问题: vLLM-Ascend 0.18.0rc1 不支持 `DeepseekV4ForCausalLM`（0.22.1 起原生支持）

本文档提供 DeepSeek-V4-Flash W8A8 MTP 模型在华为昇腾 NPU 环境下的部署指南。

## 模型简介

| 属性 | 值 |
|------|-----|
| **架构** | DeepSeek V4 Flash (MoE + MLA) |
| **参数量** | DeepSeek V4 Flash 版本 (精简高效) |
| **路由专家** | 256 (+ 1 共享专家) |
| **每 Token 激活专家** | 6 |
| **隐藏维度** | 4096 |
| **网络层数** | 43 |
| **注意力头** | 64 (GQA: 1 KV head) |
| **原生上下文** | 1,048,576 (1M tokens) |
| **量化方式** | W8A8 (8-bit 权重 + 8-bit 激活) |
| **投机解码** | MTP (Multi-Token Prediction), 1 nextn layer |
| **词表大小** | 129,280 |

## 官方文档参考

- vLLM-Ascend 模型列表: https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/index.html
- vLLM 官方文档: https://docs.vllm.ai/en/stable/

## 模型权重

模型路径: `/home/jianzhnie/llmtuner/hfhub/models/Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp`

## 硬件要求

### 单节点部署

| 硬件 | 配置 | 推荐上下文 |
|------|------|-----------|
| Atlas 800 A2 (64G × 8) | W8A8, TP=8 | 32k |
| Atlas 800 A3 (64G × 16) | W8A8, TP=16 | 64k-128k |

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
bash examples/deepseek_v4_flash/vllm_server.sh
```

### 多节点部署 (8 节点 × 8 NPU)

```bash
# 在 Head 节点容器内执行
PIPELINE_PARALLEL_SIZE=8 \
MAX_MODEL_LEN=131072 \
bash examples/deepseek_v4_flash/vllm_server.sh
```

### 后台运行

```bash
nohup bash examples/deepseek_v4_flash/vllm_server.sh > deepseek_v4_flash_server.log 2>&1 &
```

## 环境变量说明

### 基础配置

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `MODEL_PATH` | `/home/jianzhnie/llmtuner/hfhub/models/Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp` | 模型权重路径 |
| `SERVED_MODEL_NAME` | `deepseek-v4-flash` | API 中的模型名称 |
| `HOST` | `0.0.0.0` | 监听地址 |
| `PORT` | `8000` | 监听端口 |

### 并行配置

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `TENSOR_PARALLEL_SIZE` | `8` | 张量并行度 (建议 = 单节点 NPU 数) |
| `PIPELINE_PARALLEL_SIZE` | `1` | 流水线并行度 (多节点时设为节点数) |
| `ENABLE_EXPERT_PARALLEL` | `1` | 专家并行开关 (MoE 模型必需) |
| `DATA_PARALLEL_SIZE` | `1` | 数据并行度 |

### 内存与量化

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `DTYPE` | `bfloat16` | 计算数据类型 |
| `QUANTIZATION` | `ascend` | 量化方式 (W8A8 使用 Ascend 量化) |
| `GPU_MEMORY_UTILIZATION` | `0.90` | NPU 显存利用率 |
| `SWAP_SPACE` | `32` | CPU 交换空间 (GiB) |

### 序列调度

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `MAX_MODEL_LEN` | `65536` | 最大上下文长度 |
| `MAX_NUM_SEQS` | `16` | 最大并发请求数 |
| `MAX_NUM_BATCHED_TOKENS` | `8192` | 每 step 最大处理 token 数 |
| `ENABLE_CHUNKED_PREFILL` | `1` | 分块预填充开关 |

### 投机解码 (MTP)

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `SPECULATIVE_METHOD` | `mtp` | 投机解码方法 (A2: `mtp`, A3/PD: `deepseek_mtp`) |
| `SPECULATIVE_NUM_TOKENS` | `1` | 每次投机生成的 token 数 (A2 推荐 1) |

### 华为 NPU 专用

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `HCCL_OP_EXPANSION_MODE` | `AIV` | HCCL 操作扩展模式 |
| `HCCL_BUFFSIZE` | `200` | HCCL 缓冲区大小 (MB) |
| `OMP_PROC_BIND` | `false` | 禁用 OpenMP 线程绑定 |
| `OMP_NUM_THREADS` | `8` | OpenMP 线程数 (A2: 8, A3: 10) |
| `PYTORCH_NPU_ALLOC_CONF` | `expandable_segments:True` | NPU 内存分配配置 |
| `VLLM_ASCEND_BALANCE_SCHEDULING` | `1` | 负载均衡调度 |
| `USE_MULTI_GROUPS_KV_CACHE` | `1` | KV Cache 分组 |
| `USE_MULTI_BLOCK_POOL` | `1` | 多 Block Pool |
| `ACL_OP_INIT_MODE` | `1` | 算子初始化模式 |
| `VLLM_ASCEND_ENABLE_FLASHCOMM1` | `1` | 通信优化 |
| `LD_PRELOAD` | `/usr/lib/aarch64-linux-gnu/libjemalloc.so.2` | 内存分配器 |

### 加速特性

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `PREFIX_CACHING` | `1` | 前缀缓存 |
| `ENFORCE_EAGER` | `1` | 禁用 CUDA Graph (NPU 推荐) |
| `NUM_SCHEDULER_STEPS` | `8` | 多步调度步数 |
| `ENABLE_ASYNC_SCHEDULING` | `1` | 异步调度 |

### 工具调用

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `ENABLE_TOOL_CALLING` | `1` | 工具调用开关 |
| `TOOL_CALL_PARSER` | `deepseek_v4` | 工具调用解析器 (需同时设置 `--tokenizer-mode deepseek_v4` 和 `--reasoning-parser deepseek_v4`) |

## 并行策略推荐

### 8 节点 × 8 NPU (64 卡) 环境

```
场景               TP   PP   EP   DP   MAX_MODEL_LEN
─────────────────────────────────────────────────────
低延迟 (单节点)     8    1    8    1    65536
均衡 (2 节点)       8    2    8    1    131072
高吞吐 (4 节点)     8    4    8    1    131072
长上下文 (8 节点)   8    8    8    1    262144
```

## 性能调优

### 低延迟场景
- 单节点部署 (TP=8)
- 减小 `MAX_NUM_SEQS` (如 4-8)
- 启用投机解码 (MTP tokens=3)
- 减小 `MAX_NUM_BATCHED_TOKENS` (如 4096)

### 高吞吐场景
- 多节点部署，增大 `DATA_PARALLEL_SIZE`
- 增大 `MAX_NUM_SEQS` (如 16-32)
- 启用 Chunked Prefill 和 Prefix Caching
- 启用异步调度
- 增大 `NUM_SCHEDULER_STEPS` (如 8-16)

### 长上下文场景
- 多节点扩展 PP
- 增大 `MAX_MODEL_LEN` (如 131072-262144)
- 提高 `GPU_MEMORY_UTILIZATION` (如 0.95)
- 增大 `SWAP_SPACE` (如 64-128)

## 功能验证

### 基础测试

```bash
bash examples/deepseek_v4_flash/curl_test.sh
```

### 手动 API 测试

```bash
# 检查服务
curl http://localhost:8000/v1/models

# Chat Completion
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-v4-flash",
    "messages": [{"role": "user", "content": "你好，请介绍一下自己"}],
    "max_tokens": 200
  }'

# 流式输出
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-v4-flash",
    "messages": [{"role": "user", "content": "写一首诗"}],
    "max_tokens": 200,
    "stream": true
  }'
```

## 常见问题

### Q: W8A8 和 W4A8 有什么区别？
A: W8A8 精度更高但显存占用更大 (约 2× W4A8)。W8A8 需要更多 NPU 或更大显存。

### Q: MTP 是否必须启用？
A: 不是必须的，但推荐启用。MTP (Multi-Token Prediction) 可显著加速解码阶段 (1.5-2× 加速)。不启用 MTP 时删除 `--speculative-config` 参数即可。

### Q: 如何调整上下文长度？
A: 通过 `MAX_MODEL_LEN` 环境变量。DeepSeek-V4-Flash 原生支持 1M 上下文，但实际可用长度受 NPU 显存限制。W8A8 量化下建议: 单节点 ≤64k, 多节点可按比例扩展。

### Q: DeepSeek V4 Flash 和 DeepSeek V3 的区别？
A: DeepSeek V4 Flash 是更高效的版本，hidden_size 更小 (4096 vs 7168)，层数更少 (43 vs 61)，专家数相同 (256)。推理速度更快，显存占用更小。

## 验证记录

| 时间 | 镜像 | 节点 | 配置 | 结果 | 日志 |
|------|------|------|------|------|------|
| 2026-07-20 | `quay.io/ascend/vllm-ascend:v0.22.1rc1-a3` (CANN 8.5.1) | pair0: 10.42.11.194/195 | TP=8 PP=1, MAX_MODEL_LEN=65536, PORT=8000 | ✅ PASS | `logs/parallel_deploy_v022_rerun/deepseek-v4-flash_*.log` |
| 2026-07-21 | 同上 | pair0: 10.42.11.194/195 | TP=8 PP=1, PORT=8000 | ✅ PASS (复测) | `logs/dsv4_flash_reverify_vllm.log` |

- 基础 Chat Completion、Tool Calling、流式输出测试均通过。
- 流式测试曾因 `set -euo pipefail` 下 `head` 触发 `SIGPIPE` 被修正为 `|| true`。
