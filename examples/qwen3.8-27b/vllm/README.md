# Qwen3.8-27B-W8A8 部署指南

> **vLLM-Ascend v0.23.0rc1-a3** | 端口: **8022**
> 架构: Qwen3_5ForConditionalGeneration | Dense Hybrid Attention | Vision | MTP=1 | W8A8
> 推荐配置: **TP=2 PP=1 DP=1** (单节点 Atlas 800 A2/A3, 8 NPU)
> 原生上下文: **262,144** | 默认服务上下文: **131,072**
> 验证状态: ⚠️ 已完成模型文件、官方参数和脚本静态校验；当前主机无可用 vLLM 容器/NPU，实机 API 回归需在目标节点执行

Qwen3.8-27B 是 270 亿参数的稠密视觉语言模型，采用 Qwen3.5 系列的混合注意力主干，适合文本、视觉和长上下文 Agent 服务。

## 模型简介

| 属性 | 值 |
|------|-----|
| **架构** | `Qwen3_5ForConditionalGeneration` |
| **参数量** | 27B Dense；本地 W8A8 权重约 29.9 GiB |
| **网络层数** | 64 (48 Linear/Gated DeltaNet + 16 Full Attention) |
| **隐藏维度** | 5120 |
| **注意力头** | 24 Q / 4 KV；Head Dim 256 |
| **线性注意力** | 16 Key Heads / 48 Value Heads；Head Dim 128 |
| **原生上下文** | **262,144** |
| **RoPE** | `rope_theta=10000000`，`partial_rotary_factor=0.25` |
| **词表大小** | 248,320 |
| **量化方式** | W8A8 Dynamic per-token activation + per-channel weight；`--quantization ascend` |
| **MTP** | `mtp_num_hidden_layers=1`；默认开启 `qwen3_5_mtp`、3 tokens |
| **PP 支持** | ✅ 支持；`PP>1` 时必须配置 `RAY_ADDRESS` |
| **多模态** | ✅ Vision/Video；默认启用视觉编码器，纯文本 Agent 可设 `LANGUAGE_MODEL_ONLY=1` |
| **工具调用解析器** | 未强制指定；模型 chat template 使用自定义 XML，需在目标 vLLM-Ascend 镜像确认 parser 后再启用自动工具调用 |
| **推理解析器** | 未强制指定；模型 chat template 原生使用 `<think>` |
| **MoE / EP** | ❌ Dense，无专家并行 |
| **MLA / MLAPO** | ❌ 非 MLA；`MLAPO=0` |

### 架构注意事项

- Qwen3.8-27B 在 vLLM-Ascend 0.23.0 中首次支持，使用 `qwen3_5_mtp` 兼容其内置 MTP 草稿头。
- A2/A3 W8A8 官方示例使用 `TP=2`、`GPU_MEM_UTIL=0.85`、`MAX_MODEL_LEN=131072`。出现 OOM 时先降低上下文或关闭 MTP (`ENABLE_MTP=0`)。
- 该模型是 Hybrid Attention，不是 MoE，也不是 MLA；不要添加 `--enable-expert-parallel` 或启用 MLAPO。
- FLASHCOMM1 官方示例未开启，脚本默认 `FLASHCOMM1=0`；可在目标镜像上显式对比 `FLASHCOMM1=1` 的性能。
- `HCCL_BUFFSIZE=512` 是官方推荐值。跨节点时额外设置 `NIC_NAME`、`HCCL_IF_IP`。

### 官方参考

- vLLM-Ascend Qwen3.8-27B: https://docs.vllm.ai/projects/ascend/zh-cn/latest/tutorials/models/Qwen3.8-27B.html
- Qwen3.8-27B 原始模型: https://www.modelscope.cn/models/Qwen/Qwen3.8-27B
- 本地量化模型说明: `/home/jianzhnie/llmtuner/hfhub/models/Eco-Tech/Qwen3.8-27B-w8a8/README.md`

## 硬件与并行策略

| 场景 | TP | PP | DP | NPU | 上下文 | 状态 |
|------|----|----|----|-----|--------|------|
| A2/A3 单节点 W8A8 | 2 | 1 | 1 | 8 | 131K | ✅ 官方推荐 |
| 单节点低显存 | 2 | 1 | 1 | 8 | 32K | ✅ 建议排障配置 |
| 多节点 | 2 | 2+ | 1 | 每节点 8 | 131K | ⚠️ 需 Ray 和共享权重 |
| MTP 关闭 | 2 | 1 | 1 | 8 | 131K | ✅ 牺牲投机解码换显存 |

TP=2 是官方 A2/A3 示例值；PP 需要能合理切分 64 层（例如 2/4/8），并且跨节点部署时必须显式指定 Ray 地址。

## 快速开始

### 前置条件

模型路径已确认存在：

```text
/home/jianzhnie/llmtuner/hfhub/models/Eco-Tech/Qwen3.8-27B-w8a8
```

目标环境约定：

```text
镜像: quay.io/ascend/vllm-ascend:v0.23.0rc1-a3
容器: vllm-ascend-env
挂载: /home/jianzhnie/llmtuner -> /home/jianzhnie/llmtuner
节点: /home/jianzhnie/llmtuner/llm/EasyInfer/node_list.txt
```

### 1. 清理并重启容器

```bash
HEAD=10.16.201.229
WORKER="${WORKER:-}"
for ip in $HEAD $WORKER; do
    [[ -n "$ip" ]] && ssh "$ip" "docker restart vllm-ascend-env"
done
sleep 15
```

### 2. 启动容器群和 Ray

```bash
bash scripts/docker/manage_npuslim_containers.sh start \
  --file /home/jianzhnie/llmtuner/llm/EasyInfer/node_list.txt
bash scripts/ray_cluster/start_npuslim_ray_cluster.sh start \
  --file /home/jianzhnie/llmtuner/llm/EasyInfer/node_list.txt
ssh "$HEAD" "docker exec vllm-ascend-env ray status | grep -E 'NPU|Active'"
```

### 3. 部署模型

在容器内执行：

```bash
docker exec -it vllm-ascend-env /bin/bash
cd /home/jianzhnie/llmtuner/llm/EasyInfer
bash examples/qwen3.8-27b/vllm/run_vllm.sh
```

默认是单节点 TP=2、131K 上下文、MTP 开启。显存不足时：

```bash
ENABLE_MTP=0 MAX_MODEL_LEN=32768 bash examples/qwen3.8-27b/vllm/run_vllm.sh
```

多节点 PP 部署前获取 Ray 地址并导出：

```bash
docker exec vllm-ascend-env python3 -c \
  "import ray; ray.init(address='auto', ignore_reinit_error=True); print(ray.get_runtime_context().gcs_address)"
RAY_ADDRESS=<head-ip>:6379 PP=2 \
  bash examples/qwen3.8-27b/vllm/run_vllm.sh
```

多节点高速网卡可追加 `NIC_NAME=<网卡名> HCCL_IF_IP=<本节点IP>`。服务 API 模型名默认为 `qwen3.8`，端口默认为 `8022`，均可用环境变量覆盖。

### 4. API 功能测试

```bash
bash examples/qwen3.8-27b/vllm/curl_test.sh
```

测试脚本覆盖健康检查、模型列表、中英文对话、数学、代码、流式、工具调用和 Anthropic Messages API。若镜像未提供该自定义 XML 的 tool parser，工具调用项会按测试库约定标记为 WARN，不影响基础文本 API。多模态图片请求默认关闭，目标节点网络可用时执行：

```bash
ENABLE_VISION=1 bash examples/qwen3.8-27b/vllm/curl_test.sh
```

也可手动验证：

```bash
curl http://localhost:8022/v1/models
curl http://localhost:8022/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8","messages":[{"role":"user","content":"请用一句话介绍你自己。"}],"max_tokens":50}'
```

预期 `/v1/models` 和对话请求均返回 HTTP 200，响应中的 `model` 为 `qwen3.8` 且 `choices[0]` 有非空内容。

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `MODEL_PATH` | `/home/jianzhnie/llmtuner/hfhub/models/Eco-Tech/Qwen3.8-27B-w8a8` | 本地权重目录或 ModelScope ID |
| `PORT` / `SERVED_MODEL_NAME` | `8022` / `qwen3.8` | API 端口和模型名 |
| `TP` / `PP` / `DP` | `2` / `1` / `1` | 并行度；PP>1 或 TP>8 需要 Ray |
| `RAY_ADDRESS` | 空 | 多节点 Ray GCS 地址，如 `10.16.201.229:6379` |
| `QUANTIZATION` / `DTYPE` | `ascend` / `bfloat16` | W8A8 后端与计算类型 |
| `MAX_MODEL_LEN` | `131072` | 最大上下文；排障可降至 32768 |
| `MAX_NUM_SEQS` / `MAX_NUM_BATCHED_TOKENS` | `32` / `16384` | 并发和预填充步长 |
| `GPU_MEM_UTIL` | `0.85` | NPU HBM 利用率上限 |
| `ENABLE_MTP` | `1` | `qwen3_5_mtp`、3 speculative tokens |
| `ENABLE_PREFIX_CACHING` | `1` | 重复系统提示时建议开启 |
| `LANGUAGE_MODEL_ONLY` | `0` | 设为 `1` 跳过 Vision Encoder，适合纯文本 Agent |
| `FLASHCOMM1` / `MLAPO` | `0` / `0` | 本模型默认关闭；非 MLA |
| `HCCL_BUFFSIZE` | `512` | HCCL 共享缓冲区 MB |
| `NIC_NAME` / `HCCL_IF_IP` | 空 | 多节点网卡名和本节点 HCCL IP |

## Claude Code 接入

```bash
ANTHROPIC_BASE_URL=http://localhost:8022 \
ANTHROPIC_API_KEY=dummy \
ANTHROPIC_AUTH_TOKEN=dummy \
ANTHROPIC_DEFAULT_SONNET_MODEL=qwen3.8 \
ANTHROPIC_DEFAULT_HAIKU_MODEL=qwen3.8 \
ANTHROPIC_DEFAULT_OPUS_MODEL=qwen3.8 \
claude
```

## 常见问题

### Q: 为什么默认 TP=2？

A: vLLM-Ascend 对 Atlas 800 A2/A3 的 Qwen3.8-27B-W8A8 官方示例使用 TP=2。这样为模型运行时、视觉模块、KV Cache 和编译缓存留出余量。

### Q: 为什么默认 MTP 开启？

A: 模型配置包含 1 层 MTP 草稿头，vLLM-Ascend 对 Qwen3.8 使用 `qwen3_5_mtp`。若显存或稳定性不足，设置 `ENABLE_MTP=0`。

### Q: 这个模型需要 EP 或 MLAPO 吗？

A: 不需要。它是 Dense Hybrid Attention，不是 MoE，也不使用 MLA；脚本默认 `MLAPO=0` 且不传 `--enable-expert-parallel`。

### Q: 多节点启动卡在 Ray placement group 怎么办？

A: 确认每个节点容器已启动、Ray 状态有 Active NPU，并在部署命令中设置 `RAY_ADDRESS=<head>:6379`。PP>1 未设置时脚本会直接报错。

### Q: 如何启用视觉测试？

A: 部署保持默认多模态模型，运行 `ENABLE_VISION=1 bash examples/qwen3.8-27b/vllm/curl_test.sh`。测试库使用公开图片 URL；内网无外网时可设置 `VISION_URL` 为可访问地址。

纯文本 Agent 可用 `LANGUAGE_MODEL_ONLY=1 bash examples/qwen3.8-27b/vllm/run_vllm.sh` 跳过视觉编码器并节省显存。

## 验证记录

| 时间 | 镜像 | 节点 | 配置 | 结果 | 说明 |
|------|------|------|------|------|------|
| 2026-08-25 | `quay.io/ascend/vllm-ascend:v0.23.0rc1-a3` | 当前开发主机 | 模型文件、官方参数、shell 静态校验 | ⚠️ | 本机 Docker daemon/NPU 服务不可用，待目标 Atlas 节点执行 `curl_test.sh` |
