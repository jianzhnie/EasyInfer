# DeepSeek-V4-Pro W4A8 MTP 部署指南

> ✅ **已验证 PASS** | vLLM-Ascend 0.22.1rc1 + CANN 8.5.1 | 384 Experts | MoE | MTP
> 已验证配置: TP=8 PP=2 (2 节点), **MAX_MODEL_LEN=4096**（更大值 KV cache 不足） | 端口: **8005**
> 历史问题: vLLM-Ascend 0.20.2 无法识别 Eco-Tech 版权重的 `attn_sink` 参数（0.22.1 已修复）

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
bash examples/deepseek-v4-pro_w4a8/vllm/run_vllm.sh
```

### 多节点部署 (8 节点 × 8 NPU)

```bash
# 在 Head 节点容器内执行
PIPELINE_PARALLEL_SIZE=8 \
MAX_MODEL_LEN=131072 \
bash examples/deepseek-v4-pro_w4a8/vllm/run_vllm.sh
```

### 后台运行

```bash
nohup bash examples/deepseek-v4-pro_w4a8/vllm/run_vllm.sh > deepseek_v4_pro.log 2>&1 &
```

## 环境变量

> 完整环境变量说明见 [prompts/vllm_env_vars.md](../../../prompts/vllm_env_vars.md)。
> Claude Code 集成方式见 [prompts/vllm-prompt.md](../../../prompts/vllm-prompt.md)。

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
bash examples/deepseek-v4-pro_w4a8/vllm/curl_test.sh
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

## 常见问题

### Q: DeepSeek-V4-Pro 和 DeepSeek-V4-Flash 有什么区别？
A: Pro 有 384 专家（vs 256），更强的推理能力，但显存占用更大。Flash 更轻量，推理速度更快。

### Q: W4A8 和 W8A8 有什么区别？
A: W4A8 显存占用约为 W8A8 的 50%，但精度略有下降。W4A8 适合单节点/少节点部署，W8A8 适合追求精度的场景。

### Q: MTP 是否必须启用？
A: 不是必须的，但推荐启用。MTP (Multi-Token Prediction) 可显著加速解码阶段。不启用时删除 `--speculative-config` 参数。

### Q: 如何调整上下文长度？
A: 通过 `MAX_MODEL_LEN` 环境变量。DeepSeek-V4-Pro 原生支持 1M 上下文，但实际可用长度受 NPU 显存限制。

## 验证记录

| 时间 | 镜像 | 节点 | 配置 | 结果 | 日志 | 说明 |
|------|------|------|------|------|------|------|
| 2026-07-20 | `quay.io/ascend/vllm-ascend:v0.22.1rc1-a3` (CANN 8.5.1) | pair4: 10.42.11.202/203 | TP=8 PP=2, MAX_MODEL_LEN=8192, PORT=8005 | ❌ FAIL_SERVICE | `logs/parallel_deploy_v022_rerun/deepseek-v4-pro_*.log` | KV cache 不足：8192 需要 3.14 GiB，仅 2.33 GiB 可用 |
| 2026-07-20 | `quay.io/ascend/vllm-ascend:v0.22.1rc1-a3` (CANN 8.5.1) | pair0: 10.42.11.194/195 | TP=8 PP=2, MAX_MODEL_LEN=4096, PORT=8005 | ✅ PASS | `logs/parallel_deploy_remaining_v022/deepseek-v4-pro-retry_*.log` | 将 `MAX_MODEL_LEN` 降至 4096 后服务正常启动；模型列表、Chat、Tool Calling 测试通过 |

- 注意：`curl_test.sh` 原默认端口为 8000，已修正为 8005，避免测试脚本连错端口。
