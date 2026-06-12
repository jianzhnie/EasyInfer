# GLM-5.1 W4A8 部署指南

> **vLLM-Ascend 0.20.2 + CANN 9.0.0** 

> **已验证配置**: TP=16 PP=1 (2节点: 8个NPU) | **上下文**: 131,072 | Chat ✅ Tool Calling ✅

> 架构: GlmMoeDsaForCausalLM | 256 Experts | MoE | MTP | W4A8 量化

> 与 GLM-5 W4A8 使用**完全相同**的部署配置，仅 MODEL_PATH 和端口不同

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
| **工具调用解析器** | glm47 |
| **推理解析器** | glm45 |
| **词表大小** | 154,880 |

### 架构注意事项

GLM-5.1 的 config.json 包含 `index_topk: 2048`，导致 vLLM-Ascend 识别为 DeepSeek V3.2，触发 DSA CP 路径。
W4A8 量化下 CP 路径不兼容，**必须设置 `VLLM_ASCEND_ENABLE_FLASHCOMM1=0`**。
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
# 1. 启动 NPU Docker 容器
bash scripts/docker/manage_npuslim_containers.sh start --file node_list.txt

# 2. 启动 Ray 集群
bash scripts/ray_cluster/start_npuslim_ray_cluster.sh start --file node_list.txt
```

### 部署

```bash
# 单节点 (32K 上下文, TP=8)
bash examples/glm5_1_w4a8/vllm/run_vllm.sh

# 2 节点大 TP (202K 上下文)
TP=16 MAX_MODEL_LEN=202752 bash examples/glm5_1_w4a8/vllm/run_vllm.sh

# 后台运行
nohup bash examples/glm5_1_w4a8/vllm/run_vllm.sh > glm5_1_vllm.log 2>&1 &

# 使用传统包装器部署
bash examples/glm5_1_w4a8/vllm/vllm_server.sh
```

### 验证

```bash
# 运行测试脚本
bash examples/glm5_1_w4a8/vllm/curl_test.sh

# 手动验证
curl http://localhost:8002/v1/models
curl http://localhost:8002/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"glm-5.1","messages":[{"role":"user","content":"Hello"}],"max_tokens":50}'
```

## 并行策略

| 场景 | TP | PP | NPU | 上下文 |
|------|-----|-----|-----|--------|
| 单节点轻量 | 8 | 1 | 8 | 32K |
| 2 节点全量 | 16 | 1 | 16 | **202K** |
| 4 节点大规模 | 32 | 1 | 32 | 202K |

> GLM-5.1 **不支持 Pipeline Parallelism**，多节点必须使用大 TP。

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `MODEL_PATH` | `.../GLM-5.1-w4a8` | 模型权重路径 |
| `PORT` | `8002` | 监听端口 |
| `TP` | `8` | 张量并行度 |
| `PP` | `1` | 流水线并行度 |
| `MAX_MODEL_LEN` | `32768` | 最大上下文长度 |
| `MAX_NUM_SEQS` | `8` | 最大并发请求数 |
| `GPU_MEM_UTIL` | `0.94` | NPU 显存利用率 |

## 关键 NPU 环境变量

```bash
VLLM_ASCEND_ENABLE_FLASHCOMM1=0  # ⚠️ 必须为 0！防止 DSA CP crash
VLLM_ASCEND_ENABLE_MLAPO=1       # MLA 算子融合优化
HCCL_OP_EXPANSION_MODE=AIV
HCCL_BUFFSIZE=200
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

## 功能验证清单

### 基础功能

| 功能 | 状态 | 脚本 |
|------|------|------|
| 基础 Chat Completion | ✅ | `run_vllm.sh` |
| Tool Calling (glm47) | ✅ | `curl_test.sh` |
| Anthropic Messages API | ✅ | `curl_test.sh` |
| MTP 投机解码 | ✅ | `run_vllm.sh` (内置) |

### 高级功能

| 功能 | 状态 | 脚本 | 硬件要求 |
|------|------|------|----------|
| 基于 Mooncake 多实例 PD 共置部署 | 📋 已配置 | `run_pd_colocated.sh` | 多节点 + Mooncake + RoCE |
| 预填充-解码分离部署 | ⚠️ 有限支持 | `run_pd_disaggregated.sh` | 多节点 (GLM-5.1 不支持 PP) |
| 长序列上下文并行 | ⚠️ 需 A3 | `run_long_seq_cp.sh` | Atlas A3 (A2 不支持 CP) |
| 动态分块流水线并行 | ❌ 不适用 | `run_dynamic_chunked_pp.sh` | GLM-5.1 不支持 PP |

## 常见问题

### Q: GLM-5.1 和 GLM-5 的部署配置有什么不同？
A: **完全相同**。仅 `MODEL_PATH` 和 `SERVED_MODEL_NAME` 不同。所有 NPU 环境变量、并行配置、量化参数通用。

### Q: 为什么必须设置 FLASHCOMM1=0？
A: GLM-5.1 的 `index_topk: 2048` 触发 DSA CP 路径，W4A8 下缺少 `aclnn_input_scale` 属性导致 crash。

### Q: MTP 投机解码对内存有什么影响？
A: MTP 加载第二份模型权重，减少 KV cache 可用空间。TP=8 单节点时 max_model_len 从 64K 降至 ~32K。

### Q: 为什么不用 PP？
A: GLM-5.1 架构不支持 Pipeline Parallelism (`SupportsPP` 接口缺失)。多节点必须使用大 TP。
