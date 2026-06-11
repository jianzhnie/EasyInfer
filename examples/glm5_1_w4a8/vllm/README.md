# GLM-5.1 W4A8 部署指南

> ✅ **部署验证通过** | 2026-06-11 | vLLM-Ascend 0.20.2 + CANN 9.0.0
> **已验证配置**: TP=16 PP=1 (2节点: 229+40) | **上下文**: 131,072 | Chat ✅ Tool Calling ✅
> 与 GLM-5 W4A8 使用**完全相同**的部署配置 | 256 Experts | MoE | MTP

GLM-5.1 是 GLM-5 的升级版，架构完全相同 (GlmMoeDsaForCausalLM)，改进了训练数据和后训练流程。
部署配置与 GLM-5 W4A8 通用，仅需修改 `MODEL_PATH` 和 `SERVED_MODEL_NAME`。

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
| **MTP** | num_nextn_predict_layers=1, mtp |
| **PP 支持** | ❌ 不支持 Pipeline Parallelism |
| **词表大小** | 154,880 |

### 架构注意事项

GLM-5.1 的 config.json 包含 `index_topk: 2048`，导致 vLLM-Ascend 将其识别为 DeepSeek V3.2。
这会触发 DSA CP (Context Parallelism) 路径。W4A8 量化环境下 CP 路径不兼容，**必须设置 `VLLM_ASCEND_ENABLE_FLASHCOMM1=0`** 禁用。

## 官方文档参考

- GLM-5 官方部署文档: https://docs.vllm.ai/projects/ascend/en/v0.18.0/tutorials/models/GLM5.html
- vLLM 官方文档: https://docs.vllm.ai/en/stable/

## 模型权重

模型路径: `/home/jianzhnie/llmtuner/hfhub/models/Eco-Tech/GLM-5.1-w4a8`

> 💡 GLM-5 的模型路径: `/home/jianzhnie/llmtuner/hfhub/models/Eco-Tech/GLM-5-w4a8`
> 两个模型使用相同的部署脚本，仅需通过 `MODEL_PATH` 切换。

## 快速开始

### 前置条件

```bash
# 1. 启动 NPU Docker 容器 (所有节点)
bash scripts/docker/manage_npuslim_containers.sh start --file node_list.txt

# 2. 启动 Ray 集群 (所有节点)
bash scripts/ray_cluster/start_npuslim_ray_cluster.sh start --file node_list.txt
```

### 部署 (2 节点, 202K 全上下文)

```bash
# 在容器内执行
cd /home/jianzhnie/llmtuner/llm/EasyInfer

# 2 节点全量部署 (202K)
MAX_MODEL_LEN=202752 TP=16 PP=1 PORT=8002 bash examples/glm5_1_w4a8/vllm/run_vllm.sh
```

### 单节点部署

```bash
# 单节点 (32K 上下文)
TP=8 MAX_MODEL_LEN=32768 PORT=8002 bash examples/glm5_1_w4a8/vllm/run_vllm.sh
```

### 后台运行

```bash
nohup bash examples/glm5_1_w4a8/vllm/run_vllm.sh > glm5_1_vllm.log 2>&1 &
```

## 环境变量说明

与 GLM-5 W4A8 完全相同。参见 `examples/glm5_w4a8/vllm/README.md` 获取完整环境变量文档。

| 变量 | GLM-5.1 默认值 | GLM-5 默认值 | 说明 |
|------|---------------|-------------|------|
| `MODEL_PATH` | `.../GLM-5.1-w4a8` | `.../GLM-5-w4a8` | 模型权重路径 |
| `SERVED_MODEL_NAME` | `glm-5.1` | `glm-5` | API 模型名称 |
| `PORT` | `8002` | `8001` | 监听端口 |

其他所有环境变量（并行、量化、调度、NPU 等）**完全相同**。

## 并行策略

| 场景 | TP | PP | NPU | 上下文 |
|------|-----|-----|-----|--------|
| 单节点轻量 | 8 | 1 | 8 | 32K |
| 2 节点全量 | 16 | 1 | 16 | **202K** |
| 4 节点大规模 | 32 | 1 | 32 | 202K |

> GLM-5.1 **不支持 Pipeline Parallelism**，多节点必须使用大 TP。

## API 验证

### Chat Completion
```bash
curl http://localhost:8002/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"glm-5.1","messages":[{"role":"user","content":"Hello"}],"max_tokens":50}'
```

### Tool Calling
```bash
curl http://localhost:8002/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"glm-5.1","messages":[{"role":"user","content":"Weather in Paris?"}],"tools":[{"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}],"max_tokens":100}'
```

### 运行测试脚本
```bash
bash examples/glm5_1_w4a8/vllm/curl_test.sh
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

## GLM-5.1 vs GLM-5

| 特性 | GLM-5.1 | GLM-5 |
|------|---------|-------|
| **架构** | GlmMoeDsaForCausalLM | GlmMoeDsaForCausalLM |
| **专家数** | 256 | 256 |
| **上下文** | 202,752 | 202,752 |
| **量化** | W4A8 | W4A8 |
| **MTP** | ✅ | ✅ |
| **PP** | ❌ | ❌ |
| **部署配置** | 与 GLM-5 完全相同 | 与 GLM-5.1 完全相同 |
| **区别** | 改进训练数据 + 后训练 | 原始版本 |

## 常见问题

### Q: GLM-5.1 和 GLM-5 的部署配置有什么不同？
A: **完全相同**。仅 `MODEL_PATH` 和 `SERVED_MODEL_NAME` 不同。所有 NPU 环境变量、并行配置、量化参数通用。

### Q: 为什么必须设置 FLASHCOMM1=0？
A: GLM-5.1 的 `index_topk: 2048` 触发 DSA CP 路径，W4A8 下缺少 `aclnn_input_scale` 属性导致 crash。

### Q: MTP 投机解码对内存有什么影响？
A: MTP 会加载第二份模型权重，显著减少 KV cache 可用空间。TP=8 单节点时 MTP 导致 max_model_len 从 64K 降至 ~10K。

### Q: 为什么不用 PP？
A: GLM-5.1 架构不支持 Pipeline Parallelism (`SupportsPP` 接口缺失)。多节点必须使用大 TP。
