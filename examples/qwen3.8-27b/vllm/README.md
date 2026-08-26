# Qwen3.8-27B-W8A8 部署指南

> **vLLM-Ascend v0.23.0rc1-a3** | 端口: **8022**
> 架构: Qwen3_5ForConditionalGeneration | Dense Hybrid Attention | Vision | MTP=1 | W8A8
> 吞吐配置: **TP=2 PP=1 DP=4**，`MAX_NUM_SEQS=64`，`MAX_NUM_BATCHED_TOKENS=32768`
> 原生上下文: **262,144** | 默认服务上下文: **131,072** | YaRN 扩展档: **1,000,000**
> 验证状态: ✅ 吞吐档已实测 **1955.05 tok/s**；1M 档已实测 900K 检索及 **4.22 tok/s 生成吞吐**

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
| **扩展上下文** | **1,000,000**，静态 YaRN 4x；需使用独立部署档 |
| **RoPE** | `rope_theta=10000000`，`partial_rotary_factor=0.25` |
| **词表大小** | 248,320 |
| **量化方式** | W8A8 Dynamic：权重/激活为 INT8（per-channel / per-token）；未量化参数和主计算为 BF16；`--quantization ascend` |
| **MTP** | `mtp_num_hidden_layers=1`；默认开启 `qwen3_5_mtp`、3 tokens |
| **PP 支持** | ✅ 支持；`PP>1` 时必须配置 `RAY_ADDRESS` |
| **多模态** | ✅ Vision/Video；默认启用视觉编码器，纯文本 Agent 可设 `LANGUAGE_MODEL_ONLY=1` |
| **工具调用解析器** | 未强制指定；模型 chat template 使用自定义 XML，需在目标 vLLM-Ascend 镜像确认 parser 后再启用自动工具调用 |
| **推理解析器** | 未强制指定；模型 chat template 原生使用 `<think>` |
| **推理强度** | `xhigh`（默认）/ `medium` / `low`，支持服务默认和逐请求覆盖 |
| **MoE / EP** | ❌ Dense，无专家并行 |
| **MLA / MLAPO** | ❌ 非 MLA；`MLAPO=0` |

### 架构注意事项

- Qwen3.8-27B 在 vLLM-Ascend 0.23.0 中首次支持，使用 `qwen3_5_mtp` 兼容其内置 MTP 草稿头。
- A2/A3 W8A8 官方示例使用 `TP=2`、`GPU_MEM_UTIL=0.85`、`MAX_MODEL_LEN=131072`。本机在此基础上使用 4 个 TP2 数据并行副本占满 8 卡。
- 该模型是 Hybrid Attention，不是 MoE，也不是 MLA；不要添加 `--enable-expert-parallel` 或启用 MLAPO。
- **必须保持 `ENABLE_BALANCE_SCHEDULING=0`**。在该镜像中设为 1 会选择 MoE 专用的 `DPEngineCoreProc`，Dense Qwen3.8 启动时报 `DPEngineCoreProc should only be used for MoE models`。
- FLASHCOMM1 官方示例未开启，脚本默认 `FLASHCOMM1=0`；模型不是 MLA，保持 `MLAPO=0`。
- `HCCL_BUFFSIZE=512` 是官方推荐值。跨节点时额外设置 `NIC_NAME`、`HCCL_IF_IP`。
- 超过 262K 必须启用静态 YaRN。静态缩放可能降低短文本质量和性能，因此 1M 服务不应替代默认吞吐服务。
- 本地 W8A8 checkpoint 启动前会检查 `quant_model_description.json` 和
  `quant_model_weights.safetensors.index.json`；缺少任一文件会直接失败，避免把错误目录交给量化后端。
- 脚本默认 `VLLM_USE_V1=1`，并使用 `FULL_DECODE_ONLY`；这与官方第 9 节的 Chunked Prefill、SplitFuse 和全 Decode ACL Graph 调优建议一致。`CUDAGRAPH_MODE=PIECEWISE` 仅用于排障或重新评测。
- 启动前会校验端口、并行度、显存比例、YaRN 原始长度和 Ray/DP 后端；校验失败会在加载模型前退出，不会占用 NPU。
- 官方第 9 节当前只给出通用调优方向，尚未发布此模型的完整性能验证数据；下文吞吐数字均为本环境实测，不外推到其他输入/输出长度。

### 官方参考

- vLLM-Ascend Qwen3.8-27B 第 9 节: https://docs.vllm.ai/projects/ascend/zh-cn/main/tutorials/models/Qwen3.8-27B.html#9
- vLLM Optimization and Tuning: https://docs.vllm.ai/en/latest/configuration/optimization/
- vLLM-Ascend 公共性能调优: https://docs.vllm.ai/projects/ascend/zh-cn/main/developer_guide/performance_and_debug/optimization_and_tuning.html
- Qwen3.8-27B 原始模型: https://www.modelscope.cn/models/Qwen/Qwen3.8-27B
- 本地量化模型说明: `/home/jianzhnie/llmtuner/hfhub/models/Eco-Tech/Qwen3.8-27B-w8a8/README.md`

## W8A8 与 BF16 dtype

该目录是 ModelSlim 生成的 `W8A8_DYNAMIC` checkpoint。`quant_model_description.json`
记录了 per-token INT8 activation 和 per-channel INT8 weight；LayerNorm、Embedding、
部分线性注意力参数等仍然是浮点参数。

因此脚本使用下面这组参数：

```text
--quantization ascend
--dtype bfloat16
```

这里的 `--dtype bfloat16` 指非量化参数、量化线性层的输出以及主干计算使用 BF16，
并不会把 INT8 权重重新加载成 BF16，也不会关闭动态 INT8 激活量化。vLLM-Ascend
量化线性层会根据输入激活 dtype 产生 BF16 输出，所以这是该 checkpoint 的推荐设置。
不要将 `--dtype` 设成 `int8`；INT8 不是 vLLM 的主计算 dtype。`DTYPE=auto` 也可
跟随模型配置中的 BF16，但显式写 `bfloat16` 更容易审计且对该模型更稳定。

若需要完整 BF16 权重，必须换用未量化的 Qwen3.8-27B 模型目录，并使用
`QUANTIZATION=none`；不能对本目录（或 `Eco-Tech/...-w8a8` 远程模型 ID）通过
`QUANTIZATION=none` 做 BF16 回退，脚本会在启动前拒绝这种组合。

## 硬件与并行策略

| 场景 | TP | PP | DP | DP_LOCAL | NPU | batched tokens | 状态 |
|------|----|----|----|----------|-----|----------------|------|
| 最大聚合吞吐 | 2 | 1 | 4 | 4 | 8 | 32768 | ✅ 实测，客户端并发 128 |
| 吞吐/时延折中 | 2 | 1 | 4 | 4 | 8 | 32768 | ✅ 实测，客户端并发 64 |
| 低负载回退 | 2 | 1 | 1 | 1 | 2 | 16384 | ✅ 实测基线 |
| 1M 长上下文 | 8 | 1 | 1 | 1 | 8 | 32768 | ✅ 实测，900K 检索通过 |
| 多节点 PP | 2 | 2+ | 1 | 1 | 每节点 8 | 16384 | ⚠️ 未在本轮验证，需 Ray |

`DP=4` 表示启动 4 个独立 TP2 推理副本，正好使用单节点 8 张 NPU。Qwen3.8 是 Dense 模型，不使用 EP。PP 需要合理切分 64 层，并且跨节点部署时必须显式指定 Ray 地址。

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

### 1. 确认容器和 NPU

```bash
docker ps --filter name=vllm-ascend-env
docker exec vllm-ascend-env npu-smi info
```

单节点默认配置使用 multiprocessing，不依赖 Ray。只有容器未启动或需要多节点时，才执行管理脚本：

```bash
bash scripts/docker/manage_npuslim_containers.sh start \
    --file /home/jianzhnie/llmtuner/llm/EasyInfer/node_list.txt
```

### 2. 部署模型

当前已验证节点直接在容器内执行：

```bash
docker exec -it vllm-ascend-env /bin/bash
cd /home/jianzhnie/llmtuner/llm/EasyInfer
bash examples/qwen3.8-27b/vllm/run_vllm.sh
```

默认值即实测最大吞吐服务端配置：TP2/DP4、4 个本地副本、131K 上下文、32K batched tokens、MTP 开启。脚本会自动启用 vLLM V1；显式环境变量会覆盖 profile 默认值。其他节点必须覆盖本机地址：

```bash
HCCL_IF_IP=<本节点IP> DP_ADDRESS=<本节点IP> \
    bash examples/qwen3.8-27b/vllm/run_vllm.sh
```

显存压力或低请求量场景可回退到单个 TP2 副本：

```bash
DP=1 DP_LOCAL=1 MAX_NUM_SEQS=32 MAX_NUM_BATCHED_TOKENS=16384 \
    bash examples/qwen3.8-27b/vllm/run_vllm.sh
```

部署前只打印最终命令、不占用 NPU：

```bash
DRY_RUN=1 bash examples/qwen3.8-27b/vllm/run_vllm.sh
```

多节点 PP 部署先启动 Ray，再获取地址并导出：

```bash
bash scripts/ray_cluster/start_npuslim_ray_cluster.sh start \
    --file /home/jianzhnie/llmtuner/llm/EasyInfer/node_list.txt
docker exec vllm-ascend-env python3 -c \
  "import ray; ray.init(address='auto', ignore_reinit_error=True); print(ray.get_runtime_context().gcs_address)"
RAY_ADDRESS=<head-ip>:6379 DISTRIBUTED_EXECUTOR_BACKEND=ray \
    TP=2 PP=2 DP=1 DP_LOCAL=1 \
    bash examples/qwen3.8-27b/vllm/run_vllm.sh
```

多节点高速网卡可追加 `NIC_NAME=<网卡名> HCCL_IF_IP=<本节点IP>`；`RAY_ADDRESS` 必须指向已启动 Ray 集群的 head。当前实机只验证了单节点 DP4；多节点 PP 目前只做了脚本约束检查，未在本轮进行实机启动/吞吐验证，跨节点 DP 的 rank 编排需单独验证。服务 API 模型名默认为 `qwen3.8`，端口默认为 `8022`，均可用环境变量覆盖。

### 3. API 功能测试

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

## 推理强度配置

Qwen3.8 的 `reasoning_effort` 由 chat template 转换成系统指令。它控制思考倾向，不是严格的 reasoning token budget。模型模板只接受 `xhigh`、`medium`、`low`。

### 服务端默认值

脚本默认显式设置 `xhigh`。修改默认值需要重启服务：

```bash
DEFAULT_REASONING_EFFORT=medium \
    bash examples/qwen3.8-27b/vllm/run_vllm.sh
```

脚本通过以下 vLLM 参数传递该值：

```text
--default-chat-template-kwargs '{"reasoning_effort":"medium"}'
```

### 逐请求覆盖

OpenAI Chat Completions API 推荐使用顶层字段；请求值会覆盖服务默认值：

```bash
curl http://localhost:8022/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{
      "model":"qwen3.8",
      "messages":[{"role":"user","content":"分析这个问题并给出结论。"}],
      "reasoning_effort":"low",
      "max_tokens":1024
    }'
```

也可以通过模板参数传递：

```json
{"chat_template_kwargs":{"reasoning_effort":"medium"}}
```

完全关闭思考模式使用：

```json
{"chat_template_kwargs":{"enable_thinking":false}}
```

多轮 Agent 若不希望保留历史思考块，可设置：

```json
{"chat_template_kwargs":{"preserve_thinking":false}}
```

## 1M 上下文

### 原理与默认值

官方模型配置的原生上限是 262,144。`long-context-1m` 配置档按照 Qwen 官方模型卡启用静态 YaRN：

| 参数 | 值 |
|------|----|
| `rope_type` | `yarn` |
| `factor` | `4.0` |
| `original_max_position_embeddings` | `262144` |
| `MAX_MODEL_LEN` | `1000000` |
| `TP` / `DP` | `8` / `1` |
| `MAX_NUM_SEQS` | `1` |
| `LANGUAGE_MODEL_ONLY` | `1` |
| `ENABLE_MTP` | `0`，优先给 KV Cache 留出 HBM |

当前吞吐档的每个 TP2 副本实测只有 `808,493` tokens KV Cache，低于 1M，不能仅把 `MAX_MODEL_LEN` 改为 1000000。长上下文档使用全部 8 张卡构成一个 TP8 副本，并默认关闭视觉编码器和 MTP。实测 TP8/DP1 可分配 `2,977,845` tokens KV Cache，1M 单请求容量通过。

### 启动和检查

先检查生成的完整参数：

```bash
DEPLOY_PROFILE=long-context-1m DRY_RUN=1 \
    bash examples/qwen3.8-27b/vllm/run_vllm.sh
```

同一节点无法同时运行占满 8 卡的吞吐档和 1M 档。停止原服务后启动：

```bash
DEPLOY_PROFILE=long-context-1m \
    bash examples/qwen3.8-27b/vllm/run_vllm.sh
```

启动日志必须显示 KV Cache 容量不少于 1,000,000 tokens，且模型列表返回 1M。本节点实测日志为 `2,977,845 tokens`，模型列表为 `max_model_len=1000000`：

```bash
grep 'GPU KV cache size' /path/to/vllm.log
curl -s http://localhost:8022/v1/models | \
    jq '.data[0] | {id, max_model_len}'
```

若 KV Cache 容量不足，不要继续发送 1M 请求。可增加节点/并行资源，或回退到 524K 与 YaRN 2x：

```bash
DEPLOY_PROFILE=long-context-1m \
YARN_FACTOR=2.0 \
MAX_MODEL_LEN=524288 \
    bash examples/qwen3.8-27b/vllm/run_vllm.sh
```

### 长上下文检索验证

服务成功启动后执行约 900K tokens 的大海捞针测试。测试脚本会从 `/v1/models` 自动读取上下文上限，不再用 131K 固定阈值：

```bash
ENABLE_LONG_CONTEXT=1 \
LONG_CONTEXT_CASES=8 \
TARGET_TOKENS=900000 \
MIN_ACCEPT_TOKENS=850000 \
LONG_CONTEXT_TIMEOUT=7200 \
    bash examples/qwen3.8-27b/vllm/curl_test.sh
```

静态 YaRN 的缩放因子不随输入长度变化。以短文本和高吞吐为主时继续使用 `DEPLOY_PROFILE=throughput`；典型上下文约 524K 时优先使用 `YARN_FACTOR=2.0`。

本次实测结果：`prompt=900025`，命中魔法数字 `7391842`，completion 63 tokens，长上下文矩阵 PASS。

## 性能验证

### 测试方法

每组测试使用同一服务镜像、模型和随机数据集，预热请求由 `vllm bench serve` 自动执行。固定输入 256 tokens、输出 128 tokens、`temperature=0`、`request-rate=inf`、`ignore-eos`：

```bash
vllm bench serve \
    --backend openai-chat \
    --base-url http://127.0.0.1:8022 \
    --endpoint /v1/chat/completions \
    --model /home/jianzhnie/llmtuner/hfhub/models/Eco-Tech/Qwen3.8-27B-w8a8 \
    --served-model-name qwen3.8 \
    --dataset-name random \
    --random-input-len 256 \
    --random-output-len 128 \
    --num-prompts 512 \
    --max-concurrency 128 \
    --request-rate inf \
    --ignore-eos \
    --temperature 0 \
    --save-result --result-dir /tmp/qwen38-bench \
    --result-filename throughput-256x128-c128.json \
    --ready-check-timeout-sec 30
```

### 实测结果

| 服务端配置 | 客户端并发 / 请求 | 成功 | 输出吞吐 | 总 token 吞吐 | P99 TTFT | P99 TPOT |
|------------|------------------|------|----------|----------------|----------|----------|
| TP2/DP1, 16K, seq32 | 16 / 64 | 64/64 | 292.18 tok/s | 996.10 tok/s | 5643.74 ms | 75.49 ms |
| TP2/DP4, 16K, seq64 | 16 / 64 | 64/64 | 757.93 tok/s | 2583.92 tok/s | 646.90 ms | 25.50 ms |
| TP2/DP4, 16K, seq64 | 64 / 256 | 256/256 | 1509.16 tok/s | 5145.95 tok/s | 3607.55 ms | 54.82 ms |
| TP2/DP4, 16K, seq64 | 128 / 512 | 512/512 | 1777.98 tok/s | 6061.97 tok/s | 4370.05 ms | 103.87 ms |
| **TP2/DP4, 32K, seq64** | **64 / 256** | **256/256** | **1664.77 tok/s** | **5676.56 tok/s** | **1145.00 ms** | **54.19 ms** |
| **TP2/DP4, 32K, seq64** | **128 / 512** | **512/512** | **1955.05 tok/s** | **6665.69 tok/s** | **4510.24 ms** | **95.59 ms** |

DP1 基线运行时未固定 `temperature=0`，因此它只用于说明 6 张卡闲置时的容量损失；16K 与 32K 的 DP4 组使用相同参数，可以直接比较。32K 相比 16K 在并发 64 和 128 下的输出吞吐分别提高约 **10.3%** 和 **10.0%**。

### 配置选择

- **最大聚合吞吐**：保持服务端默认值，调用端允许并发 128；实测 1955.05 输出 tok/s，但 TTFT 长尾达到 4.51 秒。
- **吞吐/时延折中**：保持同一服务端配置，在网关将并发限制为 64；实测 1664.77 输出 tok/s，P99 TTFT 1.15 秒。
- **低负载或显存回退**：使用 DP1/16K。它只使用 2 张卡，不能最大化单节点吞吐。
- benchmark JSON 保存在容器临时目录 `/tmp/qwen38-bench/`，容器重建后会丢失；上表已记录关键指标。

### 1M 档吞吐实测

1M 档使用 TP8/DP1、`MAX_NUM_SEQS=1`、`MAX_NUM_BATCHED_TOKENS=32768`、静态 YaRN 4x、关闭 MTP 和视觉编码器。以下结果在本节点 `10.16.201.229`、8 张 NPU、镜像 `quay.io/ascend/vllm-ascend:v0.23.0rc1-a3` 上直接对运行中的 `8022` 服务测得：

| 工作负载 | 请求数 / 并发 | 成功 | 请求耗时 | 输出吞吐 | 总 token 吞吐 | TTFT | TPOT |
|----------|---------------|------|----------|----------|----------------|------|------|
| 随机短请求，输入 256 / 输出 128 | 16 / 1 | 16/16 | 31.57 s | **64.88 tok/s** | 221.22 tok/s | 227.65 ms | 13.74 ms |
| 随机长请求，输入约 900K / 输出 128 | 1 / 1 | 1/1 | 30.33 s | **4.22 tok/s** | **29,683.71 tok/s** | 5.97 s | 191.76 ms |

长请求的总 token 吞吐主要由 900K token 预填充贡献，不能与短请求的生成吞吐直接比较；若关注 Agent 输出速度，应使用“输出吞吐”和 TPOT。900K 请求成功返回，且与前述大海捞针测试共同证明 1M 配置可用。由于该档固定单序列，`MAX_NUM_SEQS=1` 下提高客户端并发只会排队，不能据此推断多路并发吞吐。

复现实测命令（在容器内执行）：

```bash
mkdir -p /tmp/qwen38-bench

# 短请求生成基线
vllm bench serve --backend openai-chat \
    --base-url http://127.0.0.1:8022 --endpoint /v1/chat/completions \
    --model /home/jianzhnie/llmtuner/hfhub/models/Eco-Tech/Qwen3.8-27B-w8a8 \
    --served-model-name qwen3.8 --dataset-name random \
    --random-input-len 256 --random-output-len 128 --num-prompts 16 \
    --max-concurrency 1 --request-rate inf --ignore-eos --temperature 0 \
    --save-result --result-dir /tmp/qwen38-bench \
    --result-filename tp8-1m-short-c1.json --ready-check-timeout-sec 30

# 900K 长上下文端到端吞吐
vllm bench serve --backend openai-chat \
    --base-url http://127.0.0.1:8022 --endpoint /v1/chat/completions \
    --model /home/jianzhnie/llmtuner/hfhub/models/Eco-Tech/Qwen3.8-27B-w8a8 \
    --served-model-name qwen3.8 --dataset-name random \
    --random-input-len 900000 --random-output-len 128 --num-prompts 1 \
    --max-concurrency 1 --request-rate inf --ignore-eos --temperature 0 \
    --save-result --result-dir /tmp/qwen38-bench \
    --result-filename tp8-1m-long-900k.json --ready-check-timeout-sec 30
```

JSON 结果位于容器内 `/tmp/qwen38-bench/`。长上下文 benchmark 可能出现 tokenizer 的 `262144` 原生长度提示；只要服务端 `/v1/models` 返回 `max_model_len=1000000` 且请求成功，该提示不代表服务端上限仍为 262K。

## 环境变量

下表默认值以 `DEPLOY_PROFILE=throughput` 为准；`long-context-1m` 会按上文配置档自动覆盖并行度、上下文、MTP、视觉和 YaRN 参数。

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `MODEL_PATH` | `/home/jianzhnie/llmtuner/hfhub/models/Eco-Tech/Qwen3.8-27B-w8a8` | 本地权重目录或 ModelScope ID |
| `VLLM_USE_MODELSCOPE` | `False` | 本地路径保持 `False`；使用 ModelScope ID 时设为 `True` |
| `HOST` / `PORT` / `SERVED_MODEL_NAME` | `0.0.0.0` / `8022` / `qwen3.8` | 监听地址、API 端口和模型名 |
| `DEPLOY_PROFILE` | `throughput` | `throughput` 或 `long-context-1m` |
| `TP` / `PP` / `DP` | `2` / `1` / `4` | 4 个 TP2 副本占满 8 NPU；PP>1 或 TP>8 需要 Ray |
| `DP_LOCAL` / `DP_RANK_START` | `4` / `0` | 本节点 DP 副本数和起始 rank |
| `DP_ADDRESS` / `DP_RPC_PORT` | `10.16.201.229` / `13390` | DP 协调地址和端口；换节点必须覆盖地址 |
| `DP_BACKEND` | `mp` | vLLM 数据并行后端 |
| `RAY_ADDRESS` | 空 | 多节点 Ray GCS 地址，如 `10.16.201.229:6379` |
| `DISTRIBUTED_EXECUTOR_BACKEND` | 自动 | 有 `RAY_ADDRESS` 时自动为 `ray`，本地并行自动为 `mp`；显式使用 `ray` 必须同时设置 `RAY_ADDRESS`；跨节点 PP/TP 必须为 `ray` |
| `QUANTIZATION` / `DTYPE` | `ascend` / `bfloat16` | 仅支持 `ascend` / `none`；当前目录必须用 `ascend` 读取 W8A8_DYNAMIC；`bfloat16` 是非量化参数、激活输出和主干计算 dtype，不是把量化权重转成 BF16 |
| `MAX_MODEL_LEN` | `131072` | 最大上下文；1M 档自动设为 `1000000` |
| `MAX_NUM_SEQS` / `MAX_NUM_BATCHED_TOKENS` | `64` / `32768` | 每副本调度容量和每 step 最大 token 数 |
| `GPU_MEM_UTIL` | `0.85` | NPU HBM 利用率上限 |
| `ENABLE_MTP` | `1` | `qwen3_5_mtp`、3 speculative tokens；1M 档默认关闭 |
| `DEFAULT_REASONING_EFFORT` | `xhigh` | 服务默认推理强度；仅支持 `xhigh` / `medium` / `low` |
| `ENABLE_PREFIX_CACHING` | `1` | 重复系统提示时建议开启 |
| `ENABLE_CHUNKED_PREFILL` / `ENABLE_CPU_BINDING` | `1` / `1` | 长上下文分块预填充和 CPU worker 绑定；1M 档不可关闭前者 |
| `LANGUAGE_MODEL_ONLY` | `0` | 设为 `1` 跳过 Vision Encoder；1M 档默认开启 |
| `FLASHCOMM1` / `MLAPO` | `0` / `0` | 本模型默认关闭；非 MLA |
| `ENABLE_BALANCE_SCHEDULING` | `0` | Dense 模型必须关闭；设为 1 会进入 MoE-only DP 路径 |
| `ENABLE_YARN` / `YARN_FACTOR` | `0` / `4.0` | 1M 档自动设为 `1` / `4.0` |
| `YARN_ORIGINAL_MAX_MODEL_LEN` | `262144` | YaRN 原始上下文长度 |
| `HF_OVERRIDES` | 官方 YaRN JSON | 覆盖自动生成的 RoPE 配置；仅建议高级排障使用 |
| `CUDAGRAPH_MODE` | `FULL_DECODE_ONLY` | 官方推荐的全 Decode ACL Graph；排障时可设 `PIECEWISE` |
| `VLLM_USE_V1` | `1` | 默认使用 V1 调度器，启用 Chunked Prefill/SplitFuse 路径；改为 `0` 时必须同时关闭 Chunked Prefill |
| `VLLM_ENGINE_READY_TIMEOUT_S` | `1800` | 长上下文和多卡启动的 engine 就绪超时（秒） |
| `LOCAL_NPU_COUNT` / `DRY_RUN` | `8` / `0` | 本地 NPU 约束和只打印命令开关 |
| `HCCL_BUFFSIZE` | `512` | HCCL 共享缓冲区 MB |
| `NIC_NAME` / `HCCL_IF_IP` | 空 / `10.16.201.229` | 网卡名和本节点 HCCL IP；换节点必须覆盖 IP |

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

Claude Code 会把 effort 映射到 Anthropic `output_config.effort`。Qwen3.8 只接受 `low` 和 `medium` 这两个 Claude Code 可直接传递的值；要使用模型默认 `xhigh`，不要设置 Claude Code effort：

```bash
# Qwen xhigh：由服务端默认值提供
unset CLAUDE_CODE_EFFORT_LEVEL
claude

# 逐会话降低推理强度
CLAUDE_CODE_EFFORT_LEVEL=medium claude
CLAUDE_CODE_EFFORT_LEVEL=low claude
```

不要设置 `CLAUDE_CODE_EFFORT_LEVEL=high` 或 `max`。vLLM 会将其原样传给模型模板，Qwen3.8 随后返回参数错误。

## 常见问题

### Q: W8A8 模型为什么还要设置 `dtype=bfloat16`？

A: 这是正确的组合。W8A8_DYNAMIC 的权重和动态激活量化 dtype 是 INT8，由
`--quantization ascend` 选择；`--dtype bfloat16` 只指定未量化参数、量化线性层输出和
主干计算 dtype。不要设置 `dtype=int8`。脚本会检查本地量化描述文件，并拒绝对该
W8A8 目录或 `...-w8a8` 模型 ID 使用 `QUANTIZATION=none`；需要 BF16 全量权重时应换用
原始未量化模型目录。

### Q: 为什么默认 TP=2？

A: vLLM-Ascend 对 Atlas 800 A2/A3 的官方示例使用 TP=2。本部署复制 4 个 TP2 副本，既保持官方单副本拓扑，又用满 8 张卡。

### Q: 为什么不能开启 Balance Scheduling？

A: 该开关在 `v0.23.0rc1-a3` 的 DP 路径中实例化 MoE 专用 `DPEngineCoreProc`。Qwen3.8-27B 是 Dense 模型，实测会触发断言并启动失败。它不是此模型的通用 DP 负载均衡开关。

### Q: 16K 和 32K batched tokens 如何选择？

A: 当前 256/128 workload 下，32K 在并发 64/128 均比 16K 高约 10%，所以设为默认。若更长 prompt 导致 HBM 压力或 OOM，先回退 `MAX_NUM_BATCHED_TOKENS=16384`，再降低 `MAX_MODEL_LEN` 或关闭 MTP。

### Q: 为什么默认 MTP 开启？

A: 模型配置包含 1 层 MTP 草稿头，vLLM-Ascend 对 Qwen3.8 使用 `qwen3_5_mtp`。若显存或稳定性不足，设置 `ENABLE_MTP=0`。

### Q: 为什么 1M 档默认关闭 MTP 和视觉？

A: 1M 的首要约束是单请求 KV Cache 容量。关闭 MTP 和视觉编码器可减少 HBM 压力，先建立可启动、可检索的长上下文基线；容量确认后可分别设置 `ENABLE_MTP=1` 或 `LANGUAGE_MODEL_ONLY=0` 复测，但不能沿用本文的未验证结论。

### Q: reasoning_effort 为什么不支持 high 或 max？

A: vLLM 的通用协议接受更多 effort 名称，但本模型的 chat template 只定义 `xhigh`、`medium`、`low`。协议接收不等于模型模板支持，其他值最终会在模板渲染阶段报错。

### Q: 这个模型需要 EP 或 MLAPO 吗？

A: 不需要。它是 Dense Hybrid Attention，不是 MoE，也不使用 MLA；脚本默认 `MLAPO=0` 且不传 `--enable-expert-parallel`。

### Q: 多节点启动卡在 Ray placement group 怎么办？

A: 确认每个节点容器已启动、Ray 状态有 Active NPU，并在部署命令中设置 `RAY_ADDRESS=<head>:6379 DISTRIBUTED_EXECUTOR_BACKEND=ray`。PP>1 或 TP>8 未设置 Ray 时脚本会直接报错；跨节点 DP 还需要正确规划 `DP_LOCAL` 和 `DP_RANK_START`，本示例未对该模式做吞吐承诺。

### Q: 为什么脚本默认 `VLLM_USE_V1=1` 和 `FULL_DECODE_ONLY`？

A: Qwen3.8 的官方调优章节以 V1 Chunked Prefill/SplitFuse 和全 Decode ACL Graph 为基线。脚本保留 `CUDAGRAPH_MODE=PIECEWISE` 覆盖入口，但切换后必须重新验证吞吐和显存。

### Q: 如何启用视觉测试？

A: 部署保持默认多模态模型，运行 `ENABLE_VISION=1 bash examples/qwen3.8-27b/vllm/curl_test.sh`。测试库使用公开图片 URL；内网无外网时可设置 `VISION_URL` 为可访问地址。

纯文本 Agent 可用 `LANGUAGE_MODEL_ONLY=1 bash examples/qwen3.8-27b/vllm/run_vllm.sh` 跳过视觉编码器并节省显存。

## 验证记录

| 时间 | 镜像 | 节点 | 配置 | 结果 | 说明 |
|------|------|------|------|------|------|
| 2026-08-25 | `quay.io/ascend/vllm-ascend:v0.23.0rc1-a3` | `10.16.201.229`, 8 NPU | TP2/DP1, 16K, seq32 | ✅ | 64/64 benchmark 成功，输出 292.18 tok/s |
| 2026-08-25 | `quay.io/ascend/vllm-ascend:v0.23.0rc1-a3` | `10.16.201.229`, 8 NPU | TP2/DP4, 16K, seq64 | ✅ | 并发 16/64/128 均成功，最高 1777.98 tok/s |
| 2026-08-25 | `quay.io/ascend/vllm-ascend:v0.23.0rc1-a3` | `10.16.201.229`, 8 NPU | TP2/DP4, 32K, seq64 | ✅ | 并发 64/128 均成功，最高 1955.05 tok/s |
| 2026-08-25 | `quay.io/ascend/vllm-ascend:v0.23.0rc1-a3` | `10.16.201.229`, 8 NPU | `curl_test.sh` API 回归 | ✅ | 健康、模型列表、文本、流式、代码、数学、Anthropic PASS；工具调用 WARN |
| 2026-08-25 | `quay.io/ascend/vllm-ascend:v0.23.0rc1-a3` | `10.16.201.229`, 8 NPU | reasoning effort API | ✅ | `low`/`medium` 返回 200；不支持的 `high` 按预期返回 400 |
| 2026-08-25 | `quay.io/ascend/vllm-ascend:v0.23.0rc1-a3` | `10.16.201.229`, 8 NPU | 1M YaRN TP8/DP1 | ✅ | KV Cache 2,977,845 tokens，模型列表 1M，900K 检索命中 |
| 2026-08-25 | `quay.io/ascend/vllm-ascend:v0.23.0rc1-a3` | `10.16.201.229`, 8 NPU | `reasoning_effort` 实际请求 | ✅ | `xhigh`/`medium`/`low` 均 HTTP 200，`high` HTTP 400 |
| 2026-08-25 | `quay.io/ascend/vllm-ascend:v0.23.0rc1-a3` | `10.16.201.229`, 8 NPU | 1M `curl_test.sh` 回归 | ✅ | 基础 API 与 900K 长上下文矩阵 PASS；工具调用 WARN，多模态 SKIP |
| 2026-08-25 | `quay.io/ascend/vllm-ascend:v0.23.0rc1-a3` | `10.16.201.229`, 8 NPU | 1M TP8/DP1 吞吐 benchmark | ✅ | 256/128 单并发输出 64.88 tok/s；900K/128 输出 4.22 tok/s、总吞吐 29,683.71 tok/s |

### 2026-08-25 结论

单节点最大吞吐实践是 4 个 TP2 数据并行副本、32K batched tokens、MTP、Prefix Cache、Chunked Prefill、FULL decode graph 和 CPU binding。并发 128 追求峰值吞吐，并发 64 获得更好的长尾时延。Balance Scheduling 必须关闭。

1M 长上下文实践是 TP8/DP1、静态 YaRN 4x、MTP 关闭、纯文本模式；实测 KV Cache 为 2,977,845 tokens，900K tokens 检索成功。1M 档和吞吐档都占满 8 张卡，不能同时运行。
