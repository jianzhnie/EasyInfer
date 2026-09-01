# LongCat-Flash-Chat 2-Layer BF16 部署指南

> **vLLM-Ascend v0.23.0rc1-a3** | 端口: **8300**
> 架构: LongcatFlashForCausalLM | 512 Routed + 256 Zero Experts | MoE + MLA | 2 层
> 推荐验证配置: **TP=2 EP=1** (单节点 2 NPU) | 上下文: 4096 | BF16 无量化 | 默认 CUDA graph
> 注意: 从原 28 层模型中提取 2 层用于调试；EP 模式必需（单卡无法容纳 512 专家）
> 验证状态: ✅ graph/eager 均可启动并通过 API 测试；Chunked Prefill 已验证（2026-09-01）

LongCat-Flash-Chat 的 2 层精简版，用于 EP 修复插件的快速迭代验证。

## 模型简介

| 属性 | 值 |
|------|-----|
| **架构** | LongcatFlashForCausalLM (MLA + MoE) |
| **路由专家** | 512 (每 Token 激活 12) |
| **Zero 专家** | 256 (Identity) |
| **隐藏维度** | 6144 |
| **网络层数** | **2** (从 28 层原模型中提取) |
| **KV LoRA Rank** | 512 |
| **精度** | BF16 (无量化) |
| **模型大小** | ≈80 GB |
| **MTP** | ❌ 不支持 |
| **PP 支持** | ❌ 不支持（仅 2 层） |
| **多模态** | ❌ 纯文本 |
| **工具调用解析器** | 不适用 |
| **推理解析器** | 不适用 |

### 架构注意事项

- 从 28 层 LongCat-Flash-Chat 中提取 2 层用于调试目的
- **MLA 注意力**仅支持 block_size=128
- **MC2 MoE comm** 与 Zero Expert 权重置零不兼容，EasyInfer 插件通过 `EASYINFER_MOE_COMM=allgather` 覆盖
- **Chunked Prefill** 在 EP token dispatch 场景存在兼容性风险，默认禁用；本两层模型在当前镜像上已验证可用
- 推理阶段 CANN MLP kernel aicore 异常（fftsplus aivector）是 CANN 在 EP 模式下处理 512 专家 MoE 的 kernel 层限制

### 硬件要求

| 硬件 | 配置 | 推荐上下文 | 备注 |
|------|------|-----------|------|
| Atlas 800 A2/A3 (64G × 2) | BF16, TP=2, EP=1 | 4K | 单节点 2 卡，EP 必需 |

## 快速开始

### 前置条件

模型路径: `/home/jianzhnie/llmtuner/hfhub/models/meituan-longcat/LongCat-Flash-Chat/expand/LongCat-Flash-Chat-2layer`

```bash
# 1. 启动 NPU Docker 容器
bash scripts/docker/manage_npuslim_containers.sh start --file node_list.txt

# 2. 启动 Ray 集群
bash scripts/ray_cluster/start_npuslim_ray_cluster.sh start --file node_list.txt
```

### 部署

```bash
# EP 模式 (TP=2, 1 节点)
EP=1 TP=2 EXECUTOR=mp MAX_MODEL_LEN=4096 MAX_NUM_BATCHED_TOKENS=4096 \
  MAX_NUM_SEQS=4 ENFORCE_EAGER=0 bash examples/longcat-2layer/vllm/run_vllm.sh

# 标准模式
bash examples/longcat-2layer/vllm/run_vllm.sh

# Chunked Prefill 验证模式（当前镜像可用；吞吐结论见下文）
CHUNKED_PREFILL=1 MAX_MODEL_LEN=4096 MAX_NUM_BATCHED_TOKENS=4096 \
  MAX_NUM_SEQS=4 ENFORCE_EAGER=0 bash examples/longcat-2layer/vllm/run_vllm.sh
```

### 验证

```bash
bash examples/longcat-2layer/vllm/curl_test.sh
```

## 并行策略

| 场景 | TP | PP | EP | NPU | 上下文 | 量化 | 状态 |
|------|-----|-----|-----|-----|--------|------|------|
| EP + CUDA graph | 2 | 1 | 1 | 2 | 4K | BF16 | ✅ |
| EP + eager 排障 | 2 | 1 | 1 | 2 | 4K | BF16 | ✅ |

> 单卡无法容纳 512 专家模型（需 ≈41 GB 权重 + KV Cache），EP 模式使用 2 卡。

## 环境变量

> 完整环境变量说明见 [prompts/vllm_env_vars.md](../../../prompts/vllm_env_vars.md)。

2 层脚本只保留常用部署变量：`MODEL_PATH`、`PORT`、`TP`、`PP`、`DP`、`EP`、`EXECUTOR`、
`MAX_MODEL_LEN`、`MAX_NUM_SEQS`、`MAX_NUM_BATCHED_TOKENS`、`GPU_MEM_UTIL`、`ENFORCE_EAGER`、
`CHUNKED_PREFILL` 和 `PREFIX_CACHING`。
当 `CHUNKED_PREFILL=0` 时，必须满足 `MAX_NUM_BATCHED_TOKENS >= MAX_MODEL_LEN`；脚本会在启动前检查。
默认 `MAX_MODEL_LEN=4096`、`MAX_NUM_BATCHED_TOKENS=4096`，避免 vLLM SchedulerConfig 拒绝配置。

图模式 (`ENFORCE_EAGER=0`, 默认) 会自动关闭 FlashComm1、关闭 `fuse_allreduce_rms`，并生成 TP 对齐的
`cudagraph_capture_sizes`。若图捕获或推理阶段出现 CANN kernel 错误，可用 `ENFORCE_EAGER=1` 回退。

### 实测吞吐

在本节点、vLLM-Ascend `v0.23.0rc1-a3`、TP=2/EP=1、输入 64 token、输出 64 token、8 请求、并发 4 条件下：

| 模式 | 输出吞吐 | 总吞吐 | P99 TTFT | P99 TPOT |
|------|---------:|---------:|---------:|---------:|
| `ENFORCE_EAGER=0` graph | **614.85 tok/s** | 1335.37 tok/s | 66.42 ms | 6.05 ms |
| `ENFORCE_EAGER=1` eager | 242.22 tok/s | 526.08 tok/s | 77.89 ms | 16.19 ms |

### Chunked Prefill 实测

2026-09-01 在相同节点、TP=2/EP=1、Graph、输入 64 token、输出 64 token、8 请求、并发 4 条件下，
启用 `CHUNKED_PREFILL=1` 后服务成功启动，完整 API 测试通过，基准结果如下：

| 模式 | 输出吞吐 | 总吞吐 | P99 TTFT | P99 TPOT |
|------|---------:|---------:|---------:|---------:|
| Graph + Chunked Prefill | 592.63 tok/s | 1287.13 tok/s | 80.12 ms | 6.20 ms |

该短输入基准下 Chunked Prefill 未带来吞吐提升（相对关闭时约低 3.6%，存在测试波动）。因此本模型以
最大化短请求吞吐为目标时保持默认关闭；长输入或需要平滑 Prefill 内存占用时可显式开启，并应重新压测。

### Ascend fused MoE gating 实测

图模式与 Ascend fused gating 相互独立。保持 `ENFORCE_EAGER=0`，将
`VLLM_LONGCAT_DISABLE_FUSED_GATING` 设为 `0` 即可让 `experts_selector` 进入
`DeviceOperator.moe_gating_top_k`（A3 上对应 Ascend `moe_gating_top_k` 算子）：

```bash
VLLM_LONGCAT_PATCH=1 VLLM_LONGCAT_DISABLE_FUSED_GATING=0 \
  CHUNKED_PREFILL=0 ENFORCE_EAGER=0 TP=2 EP=1 EXECUTOR=mp \
  MAX_MODEL_LEN=4096 MAX_NUM_BATCHED_TOKENS=4096 MAX_NUM_SEQS=4 \
  bash examples/longcat-2layer/vllm/run_vllm.sh
```

本次复测（重启容器后）服务初始化、Graph、EP 和完整 API 测试均通过。相同 64 token 输入、64 token
输出、8 请求、并发 4 的单次基准结果为：输出吞吐 `602.02 tok/s`、总吞吐 `1307.51 tok/s`、P99
TTFT `71.39 ms`、P99 TPOT `6.16 ms`。

该开关只控制 MoE **gating**，不等同于 `VLLM_ASCEND_ENABLE_FUSED_MC2`（通信融合）。当前两层模型的
短请求结果没有超过 native selector 基线 `614.85 tok/s`，因此默认仍保留
`VLLM_LONGCAT_DISABLE_FUSED_GATING=1`；切换为 `0` 前应在目标输入长度和并发下重新压测，并保留
Graph 失败时的回退配置。

基准命令（需服务已启动）：

```bash
docker exec vllm-ascend-env vllm bench serve --backend openai-chat \
  --base-url http://127.0.0.1:8300 --endpoint /v1/chat/completions \
  --model longcat-flash-2layer \
  --tokenizer /home/jianzhnie/llmtuner/hfhub/models/meituan-longcat/LongCat-Flash-Chat/expand/LongCat-Flash-Chat-2layer \
  --trust-remote-code --dataset-name random --random-input-len 64 \
  --random-output-len 64 --num-prompts 8 --max-concurrency 4 \
  --request-rate inf --no-stream
```

## EasyInfer EP 修复插件

| 模块 | 路径 | 作用 |
|------|------|------|
| EP 零号专家 | `easyinfer/plugins/vllm_ascend/ops/fused_moe/fix_ep_zero_expert.py` | 修复 AssertionError / Token dispatch 越界 / 版本兼容 |
| EP forward_impl | `easyinfer/plugins/vllm_ascend/ops/fused_moe/zero_expert_fused_moe.py` | EP 路由覆盖（旧版本适配） |

> 插件通过 vLLM `general_plugins` 自动发现加载。

## 验证记录

| 阶段 | 状态 | 说明 |
|------|------|------|
| 插件加载 | ✅ | general_plugins 自动发现 |
| 模型加载 | ✅ | EP Rank 0/2, 256/512 experts, 每 worker 38.87 GiB |
| Zero Expert | ✅ | AssertionError 已修复 |
| Token Dispatch | ✅ | ID sanitization 已修复 |
| KV Cache | ✅ | 3.87 GiB, 902K tokens |
| API 启动 | ✅ | graph/eager 均出现 Application startup complete |
| 推理 | ✅ | graph/eager 的 curl_test.sh 均通过；2 层模型输出仅用于链路验证 |

> 2 层抽取模型不是质量评测模型，输出可能退化；此测试只验证模型加载、算子链路和 API 服务。
