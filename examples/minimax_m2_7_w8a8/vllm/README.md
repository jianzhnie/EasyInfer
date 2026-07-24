# MiniMax-M2.7 W8A8 QuaRot 部署指南

> **vLLM-Ascend 0.22.1rc1 + CANN 8.5.1** | 端口: **8004**
> 架构: MiniMaxM2ForCausalLM | 256 Experts | MoE | W8A8 QuaRot 量化
> 已验证配置: TP=4 PP=1 (单节点 A2) | 上下文: 32K | 官方推荐 A2 环境 TP=4
> 注意: MTP 在模型中配置 (num_mtp_modules=3)，但 vLLM-Ascend 0.22.1 的 `mtp` speculative method 尚不支持 MiniMax 架构

## 模型简介

| 属性 | 值 |
|------|-----|
| **架构** | MiniMaxM2ForCausalLM (MoE) |
| **路由专家** | 256 |
| **隐藏维度** | 6144 |
| **网络层数** | 80 |
| **原生上下文** | 204,800 |
| **量化方式** | W8A8 QuaRot (8-bit 权重 + 8-bit 激活) |
| **MTP** | num_mtp_modules=3 (vLLM-Ascend 0.22.1 尚不支持) |
| **PP 支持** | ✅ 支持 Pipeline Parallelism |
| **工具调用解析器** | minimax_m2 |
| **词表大小** | 100,672 |

### 架构注意事项

模型 config 中包含 `num_mtp_modules=3`，但 vLLM-Ascend 0.22.1 的 `mtp` speculative method 尚不支持 MiniMax 架构。因此部署脚本中未启用 MTP 投机解码。待 vLLM-Ascend 更新后可添加 `--speculative-config` 参数。

### 官方文档参考

- MiniMax 官方文档: https://docs.vllm.ai/projects/ascend/zh-cn/releases-v0.20.2rc/tutorials/models/MiniMax-M2.5.html
- vLLM 官方文档: https://docs.vllm.ai/en/stable/

## 快速开始

### 前置条件

模型路径: `/home/jianzhnie/llmtuner/hfhub/models/Eco-Tech/MiniMax-M2.7-w8a8-QuaRot`

```bash
# 1. 启动 NPU Docker 容器
bash scripts/docker/manage_npuslim_containers.sh start --file node_list.txt

# 2. 启动 Ray 集群
bash scripts/ray_cluster/start_npuslim_ray_cluster.sh start --file node_list.txt
```

### 部署

```bash
# 单节点 A2 (32K 上下文, TP=4 官方推荐)
bash examples/minimax_m2_7/vllm/run_vllm.sh

# A3 16 卡 (更大上下文)
TP=8 MAX_MODEL_LEN=65536 bash examples/minimax_m2_7/vllm/run_vllm.sh

# 多节点
TP=8 PP=2 bash examples/minimax_m2_7/vllm/run_vllm.sh

# 后台运行
nohup bash examples/minimax_m2_7/vllm/run_vllm.sh > minimax_m27_vllm.log 2>&1 &

# 使用传统包装器部署
bash examples/minimax_m2_7/vllm/vllm_server.sh
```

### 验证

```bash
# 运行测试脚本
bash examples/minimax_m2_7/vllm/curl_test.sh

# 手动验证
curl http://localhost:8004/v1/models
curl http://localhost:8004/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"minimax-m2.7","messages":[{"role":"user","content":"Hello"}],"max_tokens":50}'
```

## 并行策略

| 场景 | TP | PP | DP | NPU | 上下文 | 状态 |
|------|-----|-----|-----|-----|--------|------|
| 单节点 A2 | 4 | 1 | 1 | 8 | 32K | ✅ 已验证 (需 GPU_MEM_UTIL≥0.95) |
| 单节点 A3 | 8 | 1 | 1 | 16 | 65K | ⚠️ 待验证 |
| 多节点 | 8 | 2 | 1 | 16 | 65K | ⚠️ 待验证 |

> 官方推荐 A2 环境 TP=4 (W8A8 量化下最稳定)。注意：64GB 卡上模型权重占 54.16 GB，必须 `GPU_MEM_UTIL≥0.92` 才能加载，32K 上下文推荐 `GPU_MEM_UTIL=0.95`。

## 环境变量

> 完整环境变量说明见 [prompts/vllm_env_vars.md](../../../prompts/vllm_env_vars.md)。
> Claude Code 集成方式见 [prompts/vllm-prompt.md](../../../prompts/vllm-prompt.md)。
## 功能验证清单

### 基础功能

| 功能 | 状态 | 脚本 |
|------|------|------|
| 基础 Chat Completion | ✅ 已验证 (Ascend910 64GB, 16K) | `run_vllm.sh` |
| Tool Calling (minimax_m2) | ⚠️ 待验证 | `curl_test.sh` |
| Anthropic Messages API | ⚠️ 待验证 | `curl_test.sh` |
| MTP 投机解码 | ❌ 暂不支持 | vLLM-Ascend 0.22.1 不兼容 MiniMax mtp |

### 高级功能

| 功能 | 状态 | 脚本 | 硬件要求 |
|------|------|------|----------|
| 基于 Mooncake 多实例 PD 共置部署 | 📋 已配置 | `run_pd_colocated.sh` | 多节点 + Mooncake + RoCE |
| 预填充-解码分离部署 | 📋 已配置 | `run_pd_disaggregated.sh` | 2P1D 多节点 + Mooncake |
| 长序列上下文并行 | 📋 已配置 | `run_long_seq_cp.sh` | Atlas A3 (GQA 架构支持 CP) |
| 动态分块流水线并行 | 📋 已配置 | `run_dynamic_chunked_pp.sh` | PP ≥ 2 (支持 PP) |

## 常见问题

### Q: 为什么 TP 默认是 4 而不是 8？

A: MiniMax-M2.7 W8A8 QuaRot 在 A2 环境官方推荐 TP=4，W8A8 量化下更稳定。注意 TP=4 时模型权重占比高（约 54GB / 64GB），KV Cache 空间有限，需 `GPU_MEM_UTIL≥0.95`；TP=8 时单卡仅承担约 27GB 权重，KV Cache 更充裕，可使用更低的 `GPU_MEM_UTIL`。

### Q: MTP 什么时候能支持？

A: 模型中已配置 `num_mtp_modules=3`，但 vLLM-Ascend 0.22.1 的 `mtp` speculative method 不兼容 MiniMax 架构。等待后续版本更新。

### Q: W8A8 和 W4A8 有什么区别？

A: W8A8 QuaRot 使用 8-bit 权重和 8-bit 激活量化，精度更高但显存占用较大（TP=4 时模型权重约 54GB），因此 GPU_MEM_UTIL 默认 0.95。W4A8 使用 4-bit 权重更省显存。

### Q: VLLM_ASCEND_ENABLE_FUSED_MC2 是什么？

A: MiniMax 架构专用的融合 MC2 算子优化，提升 MoE 专家路由效率。官方推荐启用。

## 验证记录

| 时间 | 镜像 | 节点 | 配置 | 结果 | 日志 |
|------|------|------|------|------|------|
| 2026-07-20 | `quay.io/ascend/vllm-ascend:v0.22.1rc1-a3` (CANN 8.5.1) | pair3: 10.42.11.200/201 | PORT=8004 | ✅ PASS | `logs/parallel_deploy_v022_rerun/minimax-m2.7_*.log` |
| 2026-07-21 | 同上 | pair3: 10.42.11.200/201 | PORT=8004 | ✅ PASS (复测) | `logs/minimax_m27_reverify_vllm.log` |

- 模型列表、中英文 Chat Completion、Tool Calling、Anthropic Messages API、流式输出测试均通过。
