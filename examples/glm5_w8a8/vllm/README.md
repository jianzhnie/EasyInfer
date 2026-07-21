# GLM-5 W8A8 部署指南

> **vLLM-Ascend 0.22.1rc1 + CANN 8.5.1** | 端口: **8011**
> 架构: GlmMoeDsaForCausalLM | 256 Experts | MoE | MTP | W8A8 量化
> 目标配置: TP=16 PP=1 (2 节点，权重 ~718G 单节点无法容纳) | 上下文: 32K（可扩展至 202,752）
> 验证状态: 见文末「验证记录」

## 模型简介

| 属性 | 值 |
|------|-----|
| **架构** | GlmMoeDsaForCausalLM (MoE + DSA + MLA) |
| **路由专家** | 256 (每 Token 激活 8 专家) |
| **隐藏维度** | 6144 |
| **网络层数** | 78 |
| **MLA** | kv_lora_rank=512, head_dim=64 |
| **原生上下文** | **202,752** |
| **量化方式** | W8A8 (8-bit 权重 + 8-bit 激活)，权重 ~718G |
| **MTP** | num_nextn_predict_layers=1（TP=16 下默认关闭以省显存，`ENABLE_MTP=1` 打开） |
| **PP 支持** | ❌ 不支持 Pipeline Parallelism |
| **工具调用解析器** | glm47 |
| **推理解析器** | glm45 |

### 架构注意事项

- GLM-5 的 config.json 包含 `index_topk`，触发 DSA 路径，**必须设置 `VLLM_ASCEND_ENABLE_FLASHCOMM1=0`**。
- W8A8 权重 ~718G，A2 (64G × 8 = 512G/节点) 单节点放不下，**最低需要 2 节点 TP=16**。

## 快速开始

### 前置条件

模型路径: `/home/jianzhnie/llmtuner/hfhub/models/Eco-Tech/GLM-5-w8a8`

```bash
# 1. 启动 NPU Docker 容器（所有节点）
bash scripts/docker/manage_npuslim_containers.sh start --file node_list3.txt

# 2. 启动 Ray 集群（Head + Worker，至少 2 节点）
bash scripts/ray_cluster/start_npuslim_ray_cluster.sh start --file node_list3.txt
```

### 部署（在 Ray Head 节点容器内执行）

```bash
# 2 节点 TP=16（默认）
bash examples/glm5_w8a8/vllm/run_vllm.sh

# 大上下文
TP=16 MAX_MODEL_LEN=131072 bash examples/glm5_w8a8/vllm/run_vllm.sh

# 打开 MTP 投机解码
ENABLE_MTP=1 bash examples/glm5_w8a8/vllm/run_vllm.sh

# 后台运行
nohup bash examples/glm5_w8a8/vllm/run_vllm.sh > glm5_w8a8_vllm.log 2>&1 &
```

### 验证

```bash
bash examples/glm5_w8a8/vllm/curl_test.sh

# 手动验证
curl http://localhost:8011/v1/models
curl http://localhost:8011/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"glm-5","messages":[{"role":"user","content":"Hello"}],"max_tokens":50}'
```

## 并行策略

| 场景 | TP | PP | NPU | 上下文 | 状态 |
|------|-----|-----|-----|--------|------|
| 2 节点 | 16 | 1 | 16 | 32K–131K | 目标配置 |
| 4 节点 | 32 | 1 | 32 | 202K | 待验证 |

> GLM-5 **不支持 Pipeline Parallelism**，多节点必须使用大 TP。

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `MODEL_PATH` | `.../Eco-Tech/GLM-5-w8a8` | 模型权重路径 |
| `PORT` | `8011` | 监听端口 |
| `TP` | `16` | 张量并行度 |
| `MAX_MODEL_LEN` | `32768` | 最大上下文 |
| `MAX_NUM_SEQS` | `48` | 最大并发序列 |
| `GPU_MEM_UTIL` | `0.95` | 显存利用率 |
| `ENABLE_MTP` | `0` | MTP 投机解码开关 |
| `NIC_NAME` | 空 | 多节点高速网卡名（HCCL/GLOO 绑定） |
| `RAY_ADDRESS` | 空 | Ray head 地址（如 `10.42.11.194:6379`） |

## 验证记录

| 日期 | 环境 | 配置 | 结果 |
|------|------|------|------|
| 待填写 | vLLM-Ascend 0.22.1rc1 + CANN 8.5.1 | TP=16 2 节点 | 待验证 |

## 验证记录

| 时间 | 镜像 | 节点 | 配置 | 结果 | 日志 |
|------|------|------|------|------|------|
| 2026-07-20 | `quay.io/ascend/vllm-ascend:v0.22.1rc1-a3` (CANN 8.5.1) | pair5: 10.42.11.204/205 | TP=16 PP=1, PORT=8011 | ✅ PASS | `logs/parallel_deploy_v022_rerun/glm5-w8a8_*.log` |

- 缓存目录从 `/dev/shm` 调整到项目共享路径 `.cache/glm5-w8a8`，避免 worker 节点 `TMPDIR` 缺失导致 Triton 编译失败。
- 模型列表、中英文 Chat Completion、Tool Calling、Anthropic Messages API、流式输出测试均通过。
