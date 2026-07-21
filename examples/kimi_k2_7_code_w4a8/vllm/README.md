# Kimi-K2.7-Code W4A8 部署指南

> **vLLM-Ascend 0.22.1rc1 + CANN 8.5.1** | 端口: **8013**
> 架构: KimiK25ForConditionalGeneration | 384 Experts | MoE | MLA | Vision (多模态) | W4A8 量化
> 目标配置: TP=8 PP=2 (2 节点，权重 ~500G) | 上下文: 32K（可扩展）
> 代码特化版本，部署配置与 Kimi-K2.6 相同，仅路径/端口/模型名不同
> 验证状态: 见文末「验证记录」

## 模型简介

| 属性 | 值 |
|------|-----|
| **架构** | KimiK25ForConditionalGeneration (DeepseekV3 文本骨干 + Vision) |
| **路由专家** | 384 |
| **量化方式** | W4A8，权重 ~500G |
| **PP 支持** | ✅ 支持 Pipeline Parallelism |
| **多模态** | Vision（部署时用 `--language-model-only` 关闭视觉塔） |
| **工具调用解析器** | kimi_k2 |

### 架构注意事项

- 权重 ~500G，A2 单节点 (512G HBM) 无法同时容纳权重 + KV Cache，**需 2 节点 (TP=8 PP=2 或 TP=16)**。
- 纯文本场景使用 `--language-model-only` 跳过视觉编码器，节省显存。

## 快速开始

### 前置条件

模型路径: `/home/jianzhnie/llmtuner/hfhub/models/Eco-Tech/Kimi-K2.7-Code-w4a8`

```bash
# 1. 启动 NPU Docker 容器（所有节点）
bash scripts/docker/manage_npuslim_containers.sh start --file node_list3.txt

# 2. 启动 Ray 集群（至少 2 节点）
bash scripts/ray_cluster/start_npuslim_ray_cluster.sh start --file node_list3.txt
```

### 部署（在 Ray Head 节点容器内执行）

```bash
# 2 节点 TP=8 PP=2（默认）
bash examples/kimi_k2_7_code_w4a8/vllm/run_vllm.sh

# 2 节点大 TP
TP=16 PP=1 bash examples/kimi_k2_7_code_w4a8/vllm/run_vllm.sh

# 后台运行
nohup bash examples/kimi_k2_7_code_w4a8/vllm/run_vllm.sh > kimi_k2_7_code_vllm.log 2>&1 &
```

### 验证

```bash
bash examples/kimi_k2_7_code_w4a8/vllm/curl_test.sh

# 手动验证
curl http://localhost:8013/v1/models
curl http://localhost:8013/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"kimi-k2.7-code","messages":[{"role":"user","content":"Hello"}],"max_tokens":50}'
```

## 并行策略

| 场景 | TP | PP | NPU | 上下文 | 状态 |
|------|-----|-----|-----|--------|------|
| 2 节点 PP | 8 | 2 | 16 | 32K–131K | 目标配置（K2.6 同构已验证） |
| 2 节点大 TP | 16 | 1 | 16 | 32K–131K | 备选 |

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `MODEL_PATH` | `.../Eco-Tech/Kimi-K2.7-Code-w4a8` | 模型权重路径 |
| `PORT` | `8013` | 监听端口 |
| `TP` / `PP` | `8` / `2` | 并行度 |
| `MAX_MODEL_LEN` | `32768` | 最大上下文 |
| `MAX_NUM_SEQS` | `16` | 最大并发序列 |
| `GPU_MEM_UTIL` | `0.92` | 显存利用率 |

## 验证记录

| 日期 | 环境 | 配置 | 结果 |
|------|------|------|------|
| 待填写 | vLLM-Ascend 0.22.1rc1 + CANN 8.5.1 | TP=8 PP=2 2 节点 | 待验证 |

## 验证记录

| 时间 | 镜像 | 节点 | 配置 | 结果 | 日志 | 说明 |
|------|------|------|------|------|------|------|
| 2026-07-20 | `quay.io/ascend/vllm-ascend:v0.22.1rc1-a3` (CANN 8.5.1) | pair7: 10.42.11.208/209 | TP=8 PP=2, PORT=8013 | ❌ FAIL_SERVICE | `logs/parallel_deploy_v022_rerun/kimi-k2.7-code_*.log` | `npu_quant_matmul` 算子错误 161002：`AclNN_Parameter_Error(EZ1001): QuantMatmul not support to process empty tensor currently` |

- 该错误与 Kimi-K2.6-w4a8 相同，属于当前 CANN/vLLM-Ascend 版本对 Kimi W4A8 量化路径不支持，需等版本修复。
