# Kimi-K2.6 W4A8 部署指南

> **vLLM-Ascend 0.23.0rc1 + CANN 8.5.1** | 端口: **8003**
> 架构: KimiK25ForConditionalGeneration | 384 Experts | MoE | MLA | Vision (多模态) | W4A8 量化
> 已验证配置: TP=8 PP=2 (2节点) + **`FLASHCOMM1=0`** | 上下文: 262,144 (max_position_embeddings)
> Agent 优化版: Prefix Caching ✅ | max_num_seqs=16 | Tool Calling (kimi_k2) ✅ | Anthropic Messages API ✅

## 模型简介

| 属性 | 值 |
|------|-----|
| **架构** | KimiK25ForConditionalGeneration → DeepseekV3ForCausalLM |
| **路由专家** | 384 (每 Token 激活 8 专家) |
| **隐藏维度** | 7168 |
| **网络层数** | 61 |
| **MLA** | kv_lora_rank=512, q_lora_rank=1536, v_head_dim=128 |
| **原生上下文** | **262,144** |
| **量化方式** | W4A8 (4-bit 权重 + 8-bit 激活) |
| **MTP** | ❌ 不支持 (num_nextn_predict_layers=0) |
| **PP 支持** | ✅ **支持 Pipeline Parallelism** |
| **多模态** | ✅ Vision Transformer (27 层) |
| **词表大小** | 163,840 |
| **工具调用解析器** | kimi_k2 |

### 架构注意事项

Kimi-K2.6 使用 `DeepseekV3ForCausalLM` 注意力路径，不走 GLM 的 SFA/DSA 路径。
但 **W4A8 量化路径下必须设置 `VLLM_ASCEND_ENABLE_FLASHCOMM1=0`**：FLASHCOMM1 的
AOT 编译图会给 `QuantMatmul` 传入空 tensor，报 161002
（`AclNN_Parameter_Error(EZ1001): QuantMatmul not support to process empty tensor currently`）。
v0.22.1/v0.23.0 均复现，`FLASHCOMM1=0` 后服务与输出完全正常（含多模态 Vision）。

### 工具调用解析器

Kimi-K2.6 的 tokenizer 使用自定义工具调用 token (`<tool>`, `</tool>` 等)，**必须使用 `kimi_k2` parser**，不能使用 `deepseek_v3`。

| Parser | 状态 | 说明 |
|--------|------|------|
| `deepseek_v3` | ❌ 不兼容 | 报错: "could not locate tool call start/end tokens" |
| `kimi_k2` | ✅ 正确 | `KimiK2ToolParser`, 适配 Kimi 的 token 格式 |

### 官方文档参考

- vLLM 官方文档: https://docs.vllm.ai/en/stable/
- vLLM-Ascend 模型文档: https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/index.html

## 快速开始

### 前置条件

模型路径: `/home/jianzhnie/llmtuner/hfhub/models/moonshotai/Kimi-K2.6-w4a8`

```bash
# 1. 启动 NPU Docker 容器
bash scripts/docker/manage_npuslim_containers.sh start --file node_list.txt

# 2. 启动 Ray 集群
bash scripts/ray_cluster/start_npuslim_ray_cluster.sh start --file node_list.txt
```

### 部署

```bash
# 单节点 (32K 上下文, TP=8)
bash examples/kimi-k2.6_w4a8/vllm/run_vllm.sh

# 2 节点 PP (大上下文)
TP=8 PP=2 MAX_MODEL_LEN=131072 bash examples/kimi-k2.6_w4a8/vllm/run_vllm.sh

# 后台运行
nohup bash examples/kimi-k2.6_w4a8/vllm/run_vllm.sh > kimi_k26_vllm.log 2>&1 &
```

### 验证

```bash
# 运行测试脚本
bash examples/kimi-k2.6_w4a8/vllm/curl_test.sh

# 手动验证
curl http://localhost:8003/v1/models
curl http://localhost:8003/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"kimi-k2.6","messages":[{"role":"user","content":"Hello"}],"max_tokens":50}'
```

## 并行策略

| 场景 | TP | PP | DP | NPU | 上下文 | 状态 |
|------|-----|-----|-----|-----|--------|------|
| 单节点 | 8 | 1 | 1 | 8 | 32K | ✅ |
| 2 节点 PP | 8 | 2 | 1 | 16 | 131K | ✅ 已验证 |
| 多节点扩展 | 8 | 4 | 2 | 64 | 131K | ⚠️ 待验证 |

> Kimi-K2.6 **支持 Pipeline Parallelism**，适合多节点扩展。

## 环境变量

> 完整环境变量说明见 [prompts/vllm_env_vars.md](../../../prompts/vllm_env_vars.md)。
> Claude Code 集成方式见 [prompts/vllm-prompt.md](../../../prompts/vllm-prompt.md)。
## 功能验证清单

### 基础功能

| 功能 | 状态 | 脚本 |
|------|------|------|
| 基础 Chat Completion | ✅ | `run_vllm.sh` |
| Tool Calling (kimi_k2) | ✅ | `curl_test.sh` |
| Anthropic Messages API | ✅ | `curl_test.sh` |
| MTP 投机解码 | ❌ 不支持 | 模型无 MTP 模块 |

## 常见问题

### Q: Kimi-K2.6 和 Kimi-K2 有什么区别？

A: Kimi-K2.6 增加了多模态 (Vision) 能力，包含 Vision Transformer (27 层)。文本骨干基于 DeepSeek V3 架构 (384 专家)。纯文本推理性能与 Kimi-K2 类似。

### Q: Kimi-K2.6 支持 MTP 投机解码吗？

A: 不支持。模型 config 中 `num_nextn_predict_layers=0`，无 MTP 模块。

### Q: --language-model-only 是什么？

A: 仅加载语言模型部分，跳过 Vision Encoder，适合纯文本场景和 Agent 使用，节省显存。

### Q: 多模态如何使用？

A: 通过 `/v1/chat/completions` 传入 image 类型的 content。视觉 token 占用上下文窗口，建议预留 20-30%。纯文本 Agent 使用时视觉组件不激活。

### Q: PP 如何工作？

A: Kimi-K2.6 支持 Pipeline Parallelism，每层分配到不同节点。TP=8 PP=2 表示 2 个 PP stage，每个 stage 在 8 张 NPU 上运行 TP。

### Q: 384 专家对部署有什么影响？

A: 专家数更多 (384 vs 256)，EP_SIZE 需能整除 384 (推荐 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 384)。384 专家的 MoE 层参数量更大，需要更大的 SWAP_SPACE。

## 验证记录

| 时间 | 镜像 | 节点 | 配置 | 结果 | 日志 | 说明 |
|------|------|------|------|------|------|------|
| 2026-07-20 | `quay.io/ascend/vllm-ascend:v0.22.1rc1-a3` (CANN 8.5.1) | pair2: 10.42.11.198/199 | TP=8 PP=2, PORT=8003 | ❌ FAIL_SERVICE | `logs/parallel_deploy_remaining_v022/kimi-k2.6-w4a8_*.log` | `npu_quant_matmul` 算子错误 161002：`AclNN_Parameter_Error(EZ1001): QuantMatmul not support to process empty tensor currently` |

- 该错误与 Kimi-K2.7-Code-w4a8 相同，说明当前 CANN/vLLM-Ascend 版本对 Kimi 系列 W4A8 量化路径不支持，需等版本修复。
| 2026-07-22 | `quay.io/ascend/vllm-ascend:v0.23.0rc1-a3` (CANN 8.5.1) | pair2: 10.42.11.198/199 | TP=8 PP=2, FLASHCOMM1=0, PORT=8003 | ✅ PASS | `logs/kimi26_fc0_vllm.log` | curl 全项通过（含多模态 Vision），质量探针输出连贯 |

### 2026-07-22 结论：FLASHCOMM1=0 可规避 161002

- `VLLM_ASCEND_ENABLE_FLASHCOMM1=0` 后 161002（QuantMatmul 空 tensor）消失，
  v0.23.0 上验证通过。根因是 FLASHCOMM1 的 AOT 编译路径会给 QuantMatmul 传入空 tensor。
