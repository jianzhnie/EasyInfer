# MiniMax-M3 W8A8 部署指南

> ⚠️ **兼容性警告**: vLLM 0.22.1 (当前容器) **不支持** `MiniMaxM3SparseForConditionalGeneration` 架构，
> 模型注册表中仅有 MiniMaxM2 及更早版本。部署预计失败，需等待 vLLM/vLLM-Ascend 升级。
> **vLLM-Ascend 0.22.1rc1 + CANN 8.5.1** | 端口: **8014**
> 架构: MiniMaxM3SparseForConditionalGeneration (minimax_m3_vl) | MoE | VL | W8A8 量化
> 目标配置: TP=8 PP=1 (单节点，权重 ~418G → ~52G/NPU，显存紧张)

## 模型简介

| 属性 | 值 |
|------|-----|
| **架构** | MiniMaxM3SparseForConditionalGeneration (文本骨干 + VL) |
| **文本骨干** | 60 层 / hidden 6144 / 64 头 (GQA kv_heads=4) |
| **原生上下文** | 1,048,576 (1M tokens) |
| **量化方式** | W8A8，权重 ~418G |
| **工具调用解析器** | minimax_m2（待官方确认 M3 专用解析器） |

### 兼容性说明

当前容器 vLLM 0.22.1 注册表中的 MiniMax 系列：
`MiniMaxForCausalLM`、`MiniMaxM1ForCausalLM`、`MiniMaxM2ForCausalLM`、`MiniMaxText01ForCausalLM`、`MiniMaxVL01ForConditionalGeneration`
— 均不匹配 M3 的 `MiniMaxM3SparseForConditionalGeneration`。

脚本按已验证的 MiniMax-M2.7 配方编写，待支持落地后可直接使用。

## 快速开始

### 前置条件

模型路径: `/home/jianzhnie/llmtuner/hfhub/models/Eco-Tech/MiniMax-M3-w8a8`

```bash
bash scripts/docker/manage_npuslim_containers.sh start --file node_list3.txt
bash scripts/ray_cluster/start_npuslim_ray_cluster.sh start --file node_list3.txt
```

### 部署（容器内执行）

```bash
# 单节点 TP=8（默认）
bash examples/minimax_m3_w8a8/vllm/run_vllm.sh

# 2 节点 TP=16（更大 KV 余量）
TP=16 bash examples/minimax_m3_w8a8/vllm/run_vllm.sh

# 后台运行
nohup bash examples/minimax_m3_w8a8/vllm/run_vllm.sh > minimax_m3_vllm.log 2>&1 &
```

### 验证

```bash
bash examples/minimax_m3_w8a8/vllm/curl_test.sh
```

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `MODEL_PATH` | `.../Eco-Tech/MiniMax-M3-w8a8` | 模型权重路径 |
| `PORT` | `8014` | 监听端口 |
| `TP` / `PP` | `8` / `1` | 并行度 |
| `MAX_MODEL_LEN` | `32768` | 最大上下文 |
| `GPU_MEM_UTIL` | `0.95` | 显存利用率 |

## 验证记录

| 日期 | 环境 | 配置 | 结果 |
|------|------|------|------|
| 待填写 | vLLM-Ascend 0.22.1rc1 + CANN 8.5.1 | TP=8 单节点 | 预计失败：架构不支持 |

## 验证记录

| 时间 | 镜像 | 节点 | 配置 | 结果 | 日志 | 说明 |
|------|------|------|------|------|------|------|
| 2026-07-20 | `quay.io/ascend/vllm-ascend:v0.22.1rc1-a3` (CANN 8.5.1) | pair3: 10.42.11.200/201 | TP=8 PP=1, PORT=8014 | ❌ FAIL_SERVICE | `logs/minimax_m3_retry_v022/*.log` | `MiniMaxM3SparseForConditionalGeneration` 不在 vLLM 0.22.1 支持架构列表中 |

- 初次部署因脚本中包含当前版本不支持的 `--swap-space 32` 参数而直接退出，已移除该参数。
- 移除后服务仍无法启动，核心原因为 vLLM 0.22.1 registry 未注册 `MiniMaxM3SparseForConditionalGeneration`，需等后续版本支持。
