# MiniMax-M2.5 W8A8 部署指南

> **vLLM-Ascend 0.20.2 + CANN 9.0.0** | 端口: **8006**
> 架构: MiniMaxM2ForCausalLM | 256 Experts | MoE | W8A8 量化
> 已验证配置: TP=8 PP=1 (单节点 A2) | 上下文: 32K
> 注意: MTP 在模型中配置 (num_mtp_modules=3)，但 vLLM-Ascend 0.20.2 暂不支持 MiniMax 架构

## 模型简介

| 属性 | 值 |
|------|-----|
| **架构** | MiniMaxM2ForCausalLM (MoE) |
| **路由专家** | 256 (每 Token 激活 8 专家) |
| **隐藏维度** | 3072 |
| **网络层数** | 62 |
| **原生上下文** | 196,608 |
| **量化方式** | W8A8 (8-bit 权重 + 8-bit 激活) |
| **MTP** | num_mtp_modules=3 (vLLM-Ascend 0.20.2 暂不支持) |
| **PP 支持** | ✅ 支持 Pipeline Parallelism |
| **工具调用解析器** | minimax_m2 |
| **词表大小** | 200,064 |

### 架构注意事项

模型 config 中包含 `num_mtp_modules=3`，但 vLLM-Ascend 0.20.2 的 `mtp` speculative method 暂不支持 MiniMax 架构。因此部署脚本中未启用 MTP 投机解码。待 vLLM-Ascend 更新后可添加 `--speculative-config` 参数。

### 官方文档参考

- MiniMax 官方文档: https://docs.vllm.ai/projects/ascend/zh-cn/releases-v0.20.2rc/tutorials/models/MiniMax-M2.5.html
- vLLM 官方文档: https://docs.vllm.ai/en/stable/

## 快速开始

### 前置条件

模型路径: `/home/jianzhnie/llmtuner/hfhub/models/MiniMax/MiniMax-M2.5`

```bash
# 1. 启动 NPU Docker 容器
bash scripts/docker/manage_npuslim_containers.sh start --file node_list.txt

# 2. 启动 Ray 集群
bash scripts/ray_cluster/start_npuslim_ray_cluster.sh start --file node_list.txt
```

### 部署

```bash
# 单节点 A2 (32K 上下文, TP=8)
bash examples/minimax-m2.5_w8a8/vllm/run_vllm.sh

# A3 16 卡 (更大上下文)
TP=8 MAX_MODEL_LEN=65536 bash examples/minimax-m2.5_w8a8/vllm/run_vllm.sh

# 多节点
TP=8 PP=2 bash examples/minimax-m2.5_w8a8/vllm/run_vllm.sh

# 后台运行
nohup bash examples/minimax-m2.5_w8a8/vllm/run_vllm.sh > minimax_m25_vllm.log 2>&1 &
```

### 验证

```bash
# 运行测试脚本
bash examples/minimax-m2.5_w8a8/vllm/curl_test.sh

# 手动验证
curl http://localhost:8006/v1/models
curl http://localhost:8006/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"minimax-m2.5","messages":[{"role":"user","content":"Hello"}],"max_tokens":50}'
```

## 并行策略

| 场景 | TP | PP | DP | NPU | 上下文 | 状态 |
|------|-----|-----|-----|-----|--------|------|
| 单节点 A2 | 8 | 1 | 1 | 8 | 32K | ⚠️ 待验证 |
| 单节点 A3 | 8 | 1 | 1 | 16 | 65K | ⚠️ 待验证 |
| 多节点 | 8 | 2 | 1 | 16 | 65K | ⚠️ 待验证 |

## 环境变量

> 完整环境变量说明见 [prompts/vllm_env_vars.md](../../../prompts/vllm_env_vars.md)。
> Claude Code 集成方式见 [prompts/vllm-prompt.md](../../../prompts/vllm-prompt.md)。
## 功能验证清单

### 基础功能

| 功能 | 状态 | 脚本 |
|------|------|------|
| 基础 Chat Completion | ⚠️ 待验证 | `run_vllm.sh` |
| Tool Calling (minimax_m2) | ⚠️ 待验证 | `curl_test.sh` |
| Anthropic Messages API | ⚠️ 待验证 | `curl_test.sh` |
| MTP 投机解码 | ❌ 暂不支持 | vLLM-Ascend 0.20.2 不兼容 MiniMax mtp |

## 常见问题

### Q: MTP 什么时候能支持？

A: 模型中已配置 `num_mtp_modules=3`，但 vLLM-Ascend 0.20.2 的 `mtp` speculative method 不兼容 MiniMax 架构。等待后续版本更新。

### Q: W8A8 和 FP8 有什么区别？

A: W8A8 使用 8-bit 整数权重 + 8-bit 激活，通过 `--quantization ascend` 启用。FP8 使用 8-bit 浮点格式。本模型采用 W8A8 量化。

### Q: VLLM_ASCEND_ENABLE_FUSED_MC2 是什么？

A: MiniMax 架构专用的融合 MC2 算子优化，提升 MoE 专家路由效率。官方推荐启用。
