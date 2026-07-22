# GLM-5.1 W8A8 部署指南

> **vLLM-Ascend 0.23.0rc1 + CANN 8.5.1** | 端口: **8012**
> 架构: GlmMoeDsaForCausalLM | 256 Experts | MoE | MTP | W8A8 量化
> 已验证配置: **TP=8 PP=2** (2 节点 A2 64G) | 上下文: 32K（可扩展至 202,752）
> ⚠️ **勿用 TP=16**：该 checkpoint 在 TP=16 下输出乱码（详见文末验证记录）
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
| **量化方式** | W8A8 (8-bit 权重 + 8-bit 激活)，权重 ~714G |
| **MTP** | num_nextn_predict_layers=1（默认关闭，`ENABLE_MTP=1` 打开；PP>1 时不可用） |
| **PP 支持** | ✅ PP=2 已验证（TP=8 PP=2 为推荐配置）；注意 PP>1 与 MTP 互斥 |
| **工具调用解析器** | glm47 |
| **推理解析器** | glm45 |

### 架构注意事项

- GLM-5 的 config.json 包含 `index_topk`，触发 DSA 路径，**必须设置 `VLLM_ASCEND_ENABLE_FLASHCOMM1=0`**（脚本已内置）。
- W8A8 权重 ~714G，A2 (64G × 8 = 512G/节点) 单节点放不下，需 2 节点。
- ⚠️ **TP=16 数值异常**：vllm-ascend v0.22.1/v0.23.0 上 TP=16（eager/cudagraph 均）输出乱码，
  根因为静态 W8A8 量化的 TP=16 DSA 路径缺陷；**TP=8 PP=2 输出正常，为默认配置**。

## 快速开始

### 前置条件

模型路径: `/home/jianzhnie/llmtuner/hfhub/models/Eco-Tech/GLM-5.1-w8a8`

```bash
# 1. 启动 NPU Docker 容器（所有节点）
bash scripts/docker/manage_npuslim_containers.sh start --file node_list3.txt

# 2. 启动 Ray 集群（Head + Worker，至少 2 节点）
bash scripts/ray_cluster/start_npuslim_ray_cluster.sh start --file node_list3.txt
```

### 部署（在 Ray Head 节点容器内执行）

```bash
# 2 节点 TP=8 PP=2（默认，已验证）
bash examples/glm5_1_w8a8/vllm/run_vllm.sh

# 大上下文
TP=8 PP=2 MAX_MODEL_LEN=131072 bash examples/glm5_1_w8a8/vllm/run_vllm.sh

# 打开 MTP 投机解码（仅 PP=1 时可用）
ENABLE_MTP=1 bash examples/glm5_1_w8a8/vllm/run_vllm.sh

# 后台运行
nohup bash examples/glm5_1_w8a8/vllm/run_vllm.sh > glm5_1_w8a8_vllm.log 2>&1 &
```

### 验证

```bash
bash examples/glm5_1_w8a8/vllm/curl_test.sh

# 手动验证
curl http://localhost:8012/v1/models
curl http://localhost:8012/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"glm-5.1","messages":[{"role":"user","content":"Hello"}],"max_tokens":50}'
```

## 并行策略

| 场景 | TP | PP | NPU | 上下文 | 状态 |
|------|-----|-----|-----|--------|------|
| 2 节点 | 8 | 2 | 16 | 32K–131K | ✅ 已验证（推荐） |
| 2 节点 | 16 | 1 | 16 | 32K–131K | ❌ 输出乱码，勿用 |
| 4 节点 | 32 | 1 | 32 | 202K | 待验证（注意 TP≥16 的乱码风险） |

> GLM-5.1 支持 PP=2（已验证）；**PP>1 与 MTP 互斥**（v0.23.0 明确拒绝）。

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `MODEL_PATH` | `.../Eco-Tech/GLM-5.1-w8a8` | 模型权重路径 |
| `PORT` | `8012` | 监听端口 |
| `TP` | `8` | 张量并行度（**勿用 16，输出乱码**） |
| `PP` | `2` | 流水线并行度（PP>1 时需 `ENABLE_MTP=0`） |
| `MAX_MODEL_LEN` | `31744` | 最大上下文 |
| `MAX_NUM_SEQS` | `8` | 最大并发序列 |
| `GPU_MEM_UTIL` | `0.95` | 显存利用率 |
| `ENABLE_MTP` | `0` | MTP 投机解码开关 |
| `ENFORCE_EAGER` | `0` | =1 时用 `--enforce-eager` 替代 cudagraph 编译 |
| `NIC_NAME` | 空 | 多节点高速网卡名（HCCL/GLOO 绑定） |
| `RAY_ADDRESS` | 空 | Ray head 地址（如 `10.42.11.194:6379`） |

## 验证记录

| 时间 | 镜像 | 节点 | 配置 | 结果 | 日志 | 说明 |
|------|------|------|------|------|------|------|
| 2026-07-20 | `quay.io/ascend/vllm-ascend:v0.22.1rc1-a3` (CANN 8.5.1) | pair6: 10.42.11.206/207 | TP=16 PP=1, PORT=8012 | ❌ FAIL_SERVICE | `logs/parallel_deploy_v022_rerun/glm5.1-w8a8_*.log` | 权重文件损坏：`quant_model_weights-00071-of-00179.safetensors` 文件头不完整 |
| 2026-07-21 | 同上 | pair6 | TP=16 PP=1 | ⚠️ 服务可起但**输出乱码** | `logs/glm51_w8a8_retry_vllm.log` | API 退出码全 0，但第 2-3 token 起多语言混杂；`ENFORCE_EAGER=1` 同样乱码 |
| 2026-07-22 | `quay.io/ascend/vllm-ascend:v0.23.0rc1-a3` (CANN 8.5.1) | pair6: 10.42.11.206/207 | **TP=8 PP=2**, PORT=8012 | ✅ PASS | `logs/glm51_w8a8_tp8pp2_vllm.log` | curl 全项通过，质量探针输出连贯（事实/算术正确） |

- 07-20 的 shard 71 损坏问题已消失（07-21 全量 179 shard 结构校验 + ModelScope sha256 校验均通过）。
- 缓存目录已调整到项目共享路径 `.cache/glm5.1-w8a8`。

### 2026-07-22 结论：TP=16 乱码，TP=8 PP=2 正常

- **TP=16 下输出乱码**（第 2-3 个 token 起多语言混杂）：v0.22.1 和 v0.23.0 均复现，
  `ENFORCE_EAGER=1` 也无法解决 → 排除 cudagraph，根因是该静态 W8A8 checkpoint 在
  vllm-ascend 的 **TP=16 量化 DSA 路径**数值异常（静默错误）。
- **TP=8 PP=2 输出完全正常**（与 GLM-5.2-w8a8 相同形态）。`run_vllm.sh` 默认值已改为
  TP=8 PP=2；PP>1 与 MTP 互斥（`ENABLE_MTP` 默认关）。
