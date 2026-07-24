# Qwen3-235B-A22B 部署指南

> **vLLM-Ascend 0.20.2 + CANN 9.0.0** | 端口: **8018**
> 架构: Qwen3MoeForCausalLM | 128 Experts | MoE | BF16
> 已验证配置: TP=8 PP=1 (单节点) | 上下文: 32K | max_position_embeddings: 262,144
> Agent 优化版: Prefix Caching ✅ | max_num_seqs=16 | Tool Calling (hermes) ✅ | Anthropic Messages API ✅

## 模型简介

| 属性 | 值 |
|------|-----|
| **架构** | Qwen3MoeForCausalLM (MoE) |
| **总专家数** | 128 (每 Token 激活 8 专家) |
| **隐藏维度** | 4096 |
| **FFN 维度** | 12288 |
| **MoE FFN 维度** | 1536 |
| **网络层数** | 94 |
| **注意力头数** | 64 |
| **KV 头数** | 4 (GQA) |
| **原生上下文** | **262,144** |
| **Head Dim** | 128 |
| **rope_theta** | 5,000,000 |
| **MTP** | ❌ 不支持 |
| **多模态** | ❌ 纯文本 |
| **词表大小** | 151,936 |
| **工具调用解析器** | hermes |

### 架构注意事项

Qwen3-235B-A22B 是一个 235B 参数的巨型 MoE 模型，包含 128 个路由专家。由于参数量巨大，单节点 A2 (8 NPU × 64G) 部署必须使用 `--quantization ascend` 进行 W4A8 量化。多节点部署可以设置 `QUANTIZATION=none` 使用 BF16 全精度。

### 官方文档参考

- vLLM 官方文档: https://docs.vllm.ai/en/stable/
- vLLM-Ascend 模型文档: https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/index.html

## 快速开始

### 前置条件

模型路径: `/home/jianzhnie/llmtuner/hfhub/models/Qwen/Qwen3-235B-A22B-Instruct-2507`

```bash
# 1. 启动 NPU Docker 容器
bash scripts/docker/manage_npuslim_containers.sh start --file node_list.txt

# 2. 启动 Ray 集群
bash scripts/ray_cluster/start_npuslim_ray_cluster.sh start --file node_list.txt
```

### 部署

```bash
# 单节点 (32K 上下文, TP=8, W4A8)
bash examples/qwen3-235b-a22b-instruct-2507/vllm/run_vllm.sh

# 大 TP 扩展上下文
TP=16 MAX_MODEL_LEN=65536 bash examples/qwen3-235b-a22b-instruct-2507/vllm/run_vllm.sh

# 多节点 BF16 全精度 (不推荐单节点跑)
QUANTIZATION=none TP=16 bash examples/qwen3-235b-a22b-instruct-2507/vllm/run_vllm.sh

# 后台运行
nohup bash examples/qwen3-235b-a22b-instruct-2507/vllm/run_vllm.sh > qwen3_235b_vllm.log 2>&1 &

# 使用传统包装器部署
bash examples/qwen3-235b-a22b-instruct-2507/vllm/vllm_server.sh
```

### 验证

```bash
# 运行测试脚本
bash examples/qwen3-235b-a22b-instruct-2507/vllm/curl_test.sh

# 手动验证
curl http://localhost:8018/v1/models
curl http://localhost:8018/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3-235b-a22b","messages":[{"role":"user","content":"Hello"}],"max_tokens":50}'
```

## 并行策略

| 场景 | TP | PP | DP | NPU | 上下文 | 量化 | 状态 |
|------|-----|-----|-----|-----|--------|------|------|
| 单节点 | 8 | 1 | 1 | 8 | 32K | W4A8 | ⚠️ 待验证 |
| 2 节点 | 16 | 1 | 1 | 16 | 65K | W4A8 | ⚠️ 待验证 |
| 4 节点 | 16 | 1 | 2 | 32 | 65K | BF16 | ⚠️ 待验证 |

> 128 专家 MoE 推荐至少 2 节点部署，4 节点可支持 BF16 全精度。

## 环境变量

> 完整环境变量说明见 [prompts/vllm_env_vars.md](../../../prompts/vllm_env_vars.md)。
> Claude Code 集成方式见 [prompts/vllm-prompt.md](../../../prompts/vllm-prompt.md)。
## 功能验证清单

### 基础功能

| 功能 | 状态 | 脚本 |
|------|------|------|
| 基础 Chat Completion | ⚠️ 待验证 | `run_vllm.sh` |
| Tool Calling (hermes) | ⚠️ 待验证 | `curl_test.sh` |
| Anthropic Messages API | ⚠️ 待验证 | `curl_test.sh` |
| MTP 投机解码 | ❌ 不支持 | 模型无 MTP 模块 |

## 常见问题

### Q: 为什么单节点部署需要 W4A8 量化？

A: Qwen3-235B 总参数量 235B，BF16 格式需 ~470GB 显存，远超单节点 A2 的 512GB 总显存（还需预留 KV Cache）。W4A8 量化后将模型压缩至 ~120GB，可以在单节点运行。

### Q: 128 专家对部署有什么影响？

A: EP_SIZE 需能整除 128 (推荐 8, 16, 32, 64, 128)。128 专家的 all-to-all 通信开销较大，建议使用较大的 HCCL_BUFFSIZE (800MB)。

### Q: 和 Qwen3-32B 部署有什么区别？

A: Qwen3-32B 是密集模型 (无 MoE)，不需要 `--enable-expert-parallel`。Qwen3-235B 是 MoE 模型，需要 EP，且由于参数量大，单节点需要 W4A8 量化。
