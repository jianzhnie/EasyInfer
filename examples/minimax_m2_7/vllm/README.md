# MiniMax-M2.7 W8A8 QuaRot 部署指南

> **vLLM-Ascend 0.20.2 + CANN 9.0.0** | 端口: **8004**
> 架构: MiniMaxM2ForCausalLM | 256 Experts | MoE | W8A8 QuaRot 量化
> 官方推荐 A2 环境 TP=4 | MTP 暂不支持 (vLLM-Ascend 0.20.2 兼容性问题)

## 模型简介

| 属性 | 值 |
|------|-----|
| **架构** | MiniMaxM2ForCausalLM (MoE) |
| **路由专家** | 256 |
| **原生上下文** | 204,800 |
| **量化方式** | W8A8 QuaRot (8-bit 权重 + 8-bit 激活) |
| **MTP** | num_mtp_modules=3 (模型中配置，但 vLLM-Ascend 0.20.2 暂不支持) |
| **PP 支持** | ✅ 支持 Pipeline Parallelism |
| **工具调用解析器** | minimax_m2 |

### MTP 兼容性说明

模型 config 中包含 `num_mtp_modules=3`，但 vLLM-Ascend 0.20.2 的 `mtp` speculative method 暂不支持 MiniMax 架构。
因此部署脚本中未启用 MTP 投机解码。待 vLLM-Ascend 更新后可添加 `--speculative-config` 参数。

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

| 场景 | TP | PP | NPU | 上下文 |
|------|-----|-----|-----|--------|
| 单节点 A2 | 4 | 1 | 8 | 32K |
| 单节点 A3 | 8 | 1 | 16 | 65K |
| 多节点 | 8 | 2 | 16 | 65K |

> 官方推荐 A2 环境 TP=4 (W8A8 量化下最稳定)。

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `MODEL_PATH` | `.../MiniMax-M2.7-w8a8-QuaRot` | 模型权重路径 |
| `PORT` | `8004` | 监听端口 |
| `TP` | `4` | 张量并行度 (A2 官方推荐) |
| `PP` | `1` | 流水线并行度 |
| `MAX_MODEL_LEN` | `32768` | 最大上下文长度 |
| `MAX_NUM_SEQS` | `16` | 最大并发请求数 |
| `GPU_MEM_UTIL` | `0.85` | NPU 显存利用率 (W8A8 较低) |

## 关键 NPU 环境变量

```bash
HCCL_BUFFSIZE=1024                  # 256 专家大缓冲
HCCL_OP_EXPANSION_MODE=AIV
VLLM_ASCEND_ENABLE_FUSED_MC2=1      # MiniMax 专用融合 MC2 算子
VLLM_ASCEND_ENABLE_FLASHCOMM1=1     # FlashComm 通信优化
TASK_QUEUE_ENABLE=1                 # 任务队列优化
```

## Claude Code 集成

```bash
ANTHROPIC_BASE_URL=http://localhost:8004 \
ANTHROPIC_API_KEY=dummy \
ANTHROPIC_AUTH_TOKEN=dummy \
ANTHROPIC_DEFAULT_SONNET_MODEL=minimax-m2.7 \
ANTHROPIC_DEFAULT_HAIKU_MODEL=minimax-m2.7 \
ANTHROPIC_DEFAULT_OPUS_MODEL=minimax-m2.7 \
claude
```

## 功能验证清单

## 功能验证清单

### 基础功能

| 功能 | 状态 | 脚本 |
|------|------|------|
| 基础 Chat Completion | 待验证 | `run_vllm.sh` |
| Tool Calling (minimax_m2) | 待验证 | `curl_test.sh` |
| Anthropic Messages API | 待验证 | `curl_test.sh` |
| MTP 投机解码 | ❌ 暂不支持 | vLLM-Ascend 0.20.2 不兼容 MiniMax mtp |

### 高级功能

| 功能 | 状态 | 脚本 | 硬件要求 |
|------|------|------|----------|
| 基于 Mooncake 多实例 PD 共置部署 | 📋 已配置 | `run_pd_colocated.sh` | 多节点 + Mooncake + RoCE |
| 预填充-解码分离部署 | 📋 已配置 | `run_pd_disaggregated.sh` | 2P1D 多节点 + Mooncake |
| 长序列上下文并行 | 📋 已配置 | `run_long_seq_cp.sh` | Atlas A3 (GQA 架构支持 CP) |
| 动态分块流水线并行 | 📋 已配置 | `run_dynamic_chunked_pp.sh` | PP ≥ 2 (支持 PP) |

## 常见问题

### Q: 为什么 TP 默认是 4 而不是 8？
A: MiniMax-M2.7 W8A8 QuaRot 在 A2 环境官方推荐 TP=4，W8A8 量化下更稳定，同时保留更多显存给 KV Cache。

### Q: MTP 什么时候能支持？
A: 模型中已配置 `num_mtp_modules=3`，但 vLLM-Ascend 0.20.2 的 `mtp` speculative method 不兼容 MiniMax 架构。等待后续版本更新。

### Q: W8A8 和 W4A8 有什么区别？
A: W8A8 QuaRot 使用 8-bit 权重和 8-bit 激活量化，精度更高但显存占用较大，因此 GPU_MEM_UTIL 默认 0.85。W4A8 使用 4-bit 权重更省显存。

### Q: VLLM_ASCEND_ENABLE_FUSED_MC2 是什么？
A: MiniMax 架构专用的融合 MC2 算子优化，提升 MoE 专家路由效率。官方推荐启用。
