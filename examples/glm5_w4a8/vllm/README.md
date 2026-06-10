# GLM-5 W4A8 部署指南

> ✅ **部署验证通过** | 2026-06-09 | vLLM-Ascend 0.18.0rc1 + CANN 8.5.1
> **已验证配置**: TP=16 PP=1 (2节点: 40+153) | **上下文**: 202,752 (max_position_embeddings)
> Agent 优化版: Prefix Caching ✅ | MTP 投机解码 ✅ | Tool Calling (glm47) ✅ | Anthropic Messages API ✅

本文档提供 GLM-5 W4A8 量化模型在华为昇腾 NPU 环境下的 Agent 优化部署指南。

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

## 已验证部署方案

### 方案 A: 2 节点 × 8 NPU (推荐，已验证)

```bash
# 节点: 10.16.201.40 + 10.16.201.153
# 总 NPU: 16 × 64GB
# 配置: TP=16 PP=1, 使用大 TP 跨节点
MAX_MODEL_LEN=202752 TP=16 PP=1 PORT=8001 bash run_vllm.sh
```

| 参数 | 值 | 说明 |
|------|-----|------|
| TP × PP | 16 × 1 | 大 TP 跨 2 节点 |
| max_model_len | **202,752** | 模型原生最大上下文 |
| max_num_seqs | 8 | MTP 占用额外内存，限制并发 |
| GPU 利用率 | 0.94 | W4A8 高利用率 |
| 加载时间 | ~12 分钟 | 含权重加载 + warmup |

### 方案 B: 单节点 (轻量测试)

```bash
MAX_MODEL_LEN=32768 TP=8 PP=1 PORT=8001 bash run_vllm.sh
```

| 参数 | 值 | 说明 |
|------|-----|------|
| TP × PP | 8 × 1 | 单节点 |
| max_model_len | 32,768 | MTP 内存限制 |
| max_num_seqs | 4 | 单节点内存紧张 |

## 快速开始

### 前置条件

模型路径: `/home/jianzhnie/llmtuner/hfhub/models/Eco-Tech/GLM-5-w4a8`

```bash
# 1. 启动容器
ssh 10.16.201.40 "docker restart npuslim-env"
ssh 10.16.201.153 "docker restart npuslim-env"
sleep 15

# 2. 启动 Ray 集群 (2 节点)
ssh 10.16.201.40 "docker exec npuslim-env bash -c 'source /usr/local/Ascend/cann/set_env.sh; ray start --head --port=6379 --resources='\''{\"NPU\": 8}'\'' --num-gpus=8'"
sleep 5
ssh 10.16.201.153 "docker exec npuslim-env bash -c 'source /usr/local/Ascend/cann/set_env.sh; ray start --address=10.16.201.40:6379 --resources='\''{\"NPU\": 8}'\'' --num-gpus=8'"
```

### 部署 (2 节点, 202K 全上下文)

```bash
ssh 10.16.201.40 'docker exec npuslim-env bash -c "
> /tmp/vllm_glm5.log 2>&1
cd /home/jianzhnie/llmtuner/llm/EasyInfer/examples/glm5_w4a8
MAX_MODEL_LEN=202752 TP=16 PP=1 PORT=8001 nohup bash run_vllm.sh >> /tmp/vllm_glm5.log 2>&1 &
echo PID: \$!
"'

# 等待 ~12 分钟后验证
curl http://10.16.201.40:8001/v1/models
# 预期: model=glm-5, max_model_len=202752
```

## NPU 环境变量

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

## vLLM 参数说明

### Agent 优化参数
```bash
--enable-prefix-caching          # Claude Code 系统提示缓存复用 (~90% KV cache hit)
--enable-chunked-prefill         # 长上下文分块预填充
--enable-auto-tool-choice        # Anthropic API tool_use 必需
--tool-call-parser glm47         # GLM 系列工具调用解析器
--speculative-config '{"num_speculative_tokens": 3, "method": "deepseek_mtp"}'  # MTP
--max-num-seqs 8                 # MTP 内存限制下的最大并发
--max-num-batched-tokens 16384   # 预填充吞吐量
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
ANTHROPIC_BASE_URL=http://10.16.201.40:8001 \
ANTHROPIC_API_KEY=dummy \
ANTHROPIC_AUTH_TOKEN=dummy \
ANTHROPIC_DEFAULT_SONNET_MODEL=glm-5 \
ANTHROPIC_DEFAULT_HAIKU_MODEL=glm-5 \
ANTHROPIC_DEFAULT_OPUS_MODEL=glm-5 \
claude
```

## 常见问题

### Q: 为什么必须设置 FLASHCOMM1=0？
A: GLM-5 的 `index_topk: 2048` 触发 DSA CP 路径，W4A8 下缺少 `aclnn_input_scale` 属性导致 crash。详见 Bug 3。

### Q: MTP 投机解码对内存有什么影响？
A: MTP 会加载第二份模型权重，显著减少 KV cache 可用空间。TP=8 单节点时 MTP 导致 max_model_len 从 64K 降至 ~10K。TP=16 时可在 2 节点上达到 202K。

### Q: 为什么不用 PP？
A: GLM-5/5.1 架构不支持 Pipeline Parallelism (`SupportsPP` 接口缺失)。多节点必须使用大 TP。
