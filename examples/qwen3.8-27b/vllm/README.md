# Qwen3.8-27B-W8A8 部署指南

> **vLLM-Ascend v0.23.0rc1-a3** | 单业务端口: **8022**
> 架构: Qwen3_5ForConditionalGeneration | Dense Hybrid Attention | Vision | MTP=1 | W8A8
> 默认部署: **TP=4 PP=1 DP=1**，1M 上下文，`MAX_NUM_SEQS=16`，`MAX_NUM_BATCHED_TOKENS=16384`
> 原生上下文: **262,144** | 默认服务上下文: **1,000,000**（静态 YaRN 4x）
> 验证状态: 历史 TP4/DP2、seq1、MTP off 已通过 900K 检索；当前 DP1 默认 `seq16 + MTP3` 需重新验证

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
| **扩展上下文** | **1,000,000**，静态 YaRN 4x；当前脚本默认值 |
| **RoPE** | `rope_theta=10000000`，`partial_rotary_factor=0.25` |
| **词表大小** | 248,320 |
| **量化方式** | W8A8 Dynamic：权重/激活为 INT8（per-channel / per-token）；未量化参数和主计算为 BF16；`--quantization ascend` |
| **MTP** | `mtp_num_hidden_layers=1`；默认开启 `qwen3_5_mtp`、3 tokens |
| **部署拓扑** | 当前脚本固定单机 `TP4/PP1/DP1`，multiprocessing backend |
| **多模态** | 模型支持 Vision/Video；当前 1M Agent 部署固定 `--language-model-only` |
| **工具调用解析器** | 固定 `qwen3_xml` 并开启自动工具选择；当前镜像已验证 OpenAI/Anthropic/Claude Code 工具调用 |
| **推理解析器** | 未强制指定；模型 chat template 原生使用 `<think>` |
| **推理强度** | `medium`（默认）/ `xhigh` / `low`，支持服务默认和逐请求覆盖 |
| **MoE / EP** | ❌ Dense，无专家并行 |
| **MLA / MLAPO** | ❌ 非 MLA；`MLAPO=0` |

### 架构注意事项

- Qwen3.8-27B 在 vLLM-Ascend 0.23.0 中首次支持，使用 `qwen3_5_mtp` 兼容其内置 MTP 草稿头。
- A2/A3 W8A8 官方示例使用 `TP=2`、`GPU_MEM_UTIL=0.85`、`MAX_MODEL_LEN=131072`。当前脚本面向 1M Agent 服务，使用一个 TP4 副本；它只占用 4 张卡，另 4 张卡保持空闲。历史 TP4/DP2 和 TP2/DP4 吞吐结果仅作对照。
- 该模型是 Hybrid Attention，不是 MoE，也不是 MLA；不要添加 `--enable-expert-parallel` 或启用 MLAPO。
- **必须保持 `VLLM_ASCEND_BALANCE_SCHEDULING=0`**。在该镜像中设为 1 会选择 MoE 专用的 `DPEngineCoreProc`，Dense Qwen3.8 启动时报 `DPEngineCoreProc should only be used for MoE models`。
- 当前脚本固定关闭未经该 1M 组合验证的 FlashComm1/Reduce Sample，并使用 `HCCL_BUFFSIZE=512`。
- 模型不是 MLA，始终保持 `MLAPO=0`；跨节点时还需在容器/集群层正确设置 `NIC_NAME` 和 `HCCL_IF_IP`。
- 超过 262K 必须启用静态 YaRN。静态缩放可能影响短文本质量和性能，因此下方历史 131K 吞吐结果不能直接代表当前 1M 服务。
- 本地 W8A8 checkpoint 启动前会检查 `quant_model_description.json` 和
  `quant_model_weights.safetensors.index.json`；缺少任一文件会直接失败，避免把错误目录交给量化后端。
- 脚本固定使用 `VLLM_USE_V1=1`、Chunked Prefill、SplitFuse 和 `FULL_DECODE_ONLY`；这与官方第 9 节的全 Decode ACL Graph 调优建议一致。
- 启动前会校验模型目录、W8A8 元数据、API/DP 端口、推理强度和 dry-run 值；失败会在加载模型前退出，不占用 NPU。
- 模型有 24 个注意力头和 16 个线性注意力 key heads；固定 TP4 可同时整除两者，并在每个 DP 副本上使用 4 张 NPU。
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

若需要完整 BF16 权重，必须换用未量化的 Qwen3.8-27B 模型目录，并使用另一份
明确配置 `--quantization` 的启动脚本。本脚本针对该 W8A8 目录固定使用
`--quantization ascend`，不提供 `QUANTIZATION=none` 回退路径。

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

当前脚本使用单节点 multiprocessing，不依赖 Ray。容器未启动时执行管理脚本：

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

默认命令在本节点 `10.16.201.229` 使用一个 TP4/DP1 副本：1M 上下文、YaRN 4x、seq16、16K batched tokens、MTP3、纯文本模式和工具调用。它只占用 4 张 NPU，对外只暴露一个 OpenAI/Anthropic 兼容 API 端口；DP=1 不需要 `13390` 内部协调端口。

脚本只允许通过环境变量覆盖模型路径、API 端口、服务名、默认 reasoning effort 和 dry-run。迁移到其他节点或改变 TP/DP、上下文、MTP 等部署形态时，应复制脚本并整体复测，不要依赖零散环境变量拼装新拓扑。

部署前只打印最终命令、不占用 NPU：

```bash
DRY_RUN=1 bash examples/qwen3.8-27b/vllm/run_vllm.sh
```

服务 API 模型名默认为 `qwen3.8`，端口默认为 `8022`，均可用环境变量覆盖。该精简脚本不支持 Ray 或多节点 PP；多节点部署应使用单独脚本维护网络地址、rank 和 Ray 生命周期。

### 3. API 功能测试

```bash
bash examples/qwen3.8-27b/vllm/curl_test.sh
```

测试脚本覆盖健康检查、模型列表、中英文对话、数学、代码、流式、结构化工具调用、Anthropic Messages API，以及 `xhigh/medium/low` 三档 reasoning effort。工具调用不是宽松 WARN：未返回结构化 `tool_call` 时脚本以非零状态退出。`MODEL_NAME` 未设置时会自动继承启动脚本的 `SERVED_MODEL_NAME`。

多模态图片请求默认关闭。目标节点网络可用时执行；显式启用后，图片请求失败或响应为空也会让测试失败：

```bash
ENABLE_VISION=1 bash examples/qwen3.8-27b/vllm/curl_test.sh
```

只做基础 API 回归、不执行三档 effort 矩阵时可设置 `SKIP_REASONING=1`。

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

脚本默认显式设置 `medium`。修改默认值需要重启服务：

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

官方模型配置的原生上限是 262,144。当前脚本按照 Qwen 官方模型卡固定启用静态 YaRN：

| 参数 | 值 |
|------|----|
| `rope_type` | `yarn` |
| `factor` | `4.0` |
| `original_max_position_embeddings` | `262144` |
| `max-model-len` | `1000000` |
| `TP` / `DP` | `4` / `1` |
| `max-num-seqs` | `16` |
| 视觉编码器 | `--language-model-only`，不加载 |
| MTP | `qwen3_5_mtp`，3 speculative tokens，固定开启 |

历史吞吐档的每个 TP2 副本实测只有 `808,493` tokens KV Cache，低于 1M，不能仅把上下文参数改为 1000000。当前部署使用一个 TP4 数据并行副本（占 4 张卡），并关闭视觉编码器。历史 TP4/DP2 保守基线在 seq1/MTP off 时，DP0/DP1 各分配到 `2,731,003` tokens KV Cache，1M 单请求容量通过；当前 DP1 的 seq16/MTP3 尚未启动验证，必须重新确认日志中的 KV 容量。

`MAX_NUM_SEQS=16` 是每个 DP 副本的调度上限，不会预留 `16 × 1M` KV Cache，也不保证 16 个 1M 请求可同时运行。它主要改善多个中短 Agent 请求在 1M 服务上的合批能力；超长请求的实际并发仍由日志中的 KV Cache tokens 决定。

### 启动和检查

先检查生成的完整参数：

```bash
DRY_RUN=1 bash examples/qwen3.8-27b/vllm/run_vllm.sh
```

脚本使用本节点 4 张 NPU；启动前确认这 4 张卡和业务端口未被其他服务占用：

```bash
bash examples/qwen3.8-27b/vllm/run_vllm.sh
```

以上命令使用新的推荐候选 `seq16 + MTP3`。为保持生产入口简单，脚本不再提供 seq/MTP/TP/DP 环境变量覆盖。需要复现 `seq1 + MTP off` 保守基线时，应复制脚本后同时修改 `--max-num-seqs` 和 `--speculative-config`，保留独立文件并重新验证，避免临时环境变量污染默认部署。

启动日志必须显示 DP0 的 KV Cache 容量不少于 1,000,000 tokens，且模型列表返回 1M。历史 TP4/DP2 保守基线实测 DP0/DP1 均为 `2,731,003 tokens`；当前 DP1 启动后应重新确认，模型列表应为 `max_model_len=1000000`：

```bash
grep 'GPU KV cache size' /path/to/vllm.log
curl -s http://localhost:8022/v1/models | \
    jq '.data[0] | {id, max_model_len}'
```

若 KV Cache 容量不足，不要继续发送 1M 请求，可增加资源或使用独立的保守基线脚本。当前脚本将 YaRN factor 固定为 4.0；不要只修改 `--max-model-len` 来构造 524K 服务，因为这不会同步改成 YaRN 2x。需要 524K 时应复制脚本，同时修改上下文上限和 `hf_overrides`，再重新做长上下文质量验证。

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

静态 YaRN 的缩放因子不随输入长度变化。以短文本和极限吞吐为主时，应维护独立的 131K 脚本并复现下方 TP2/DP4 历史配置；当前脚本固定为 YaRN 4x/1M。

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
    --max-concurrency 256 \
    --request-rate inf \
    --ignore-eos \
    --temperature 0 \
    --save-result --result-dir /tmp/qwen38-bench \
    --result-filename throughput-256x128-c256.json \
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
| **TP2/DP4, 32K, seq64, warm cache** | **128 / 512** | **512/512** | **2196.09 tok/s** | **7487.49 tok/s** | **4373.25 ms** | **87.21 ms** |
| **TP2/DP4, 32K, seq64, warm cache** | **256 / 512** | **512/512** | **2579.94 tok/s** | **8796.21 tok/s** | **10047.30 ms** | **83.09 ms** |
| TP2/DP4, 48K, seq64 | 128 / 512 | 512/512 | 1941.76 tok/s | 6620.38 tok/s | 7741.48 ms | 94.10 ms |
| TP2/DP4, 32K, MTP off | 128 / 512 | 512/512 | 1929.32 tok/s | 6577.96 tok/s | 5787.87 ms | 77.28 ms |

历史 TP2/DP1 基线运行时未固定 `temperature=0`，因此它只用于说明 6 张卡闲置时的容量损失；16K 与 32K 的 DP4 组使用相同参数，可以直接比较。32K 相比 16K 在并发 64 和 128 下的历史输出吞吐分别提高约 **10.3%** 和 **10.0%**。本轮暖机后复测显示 32K/DP4 是最佳点：48K 降至 1941.76 tok/s，关闭 MTP 降至 1929.32 tok/s；并发从 128 提高到 256 后输出吞吐增至 2579.94 tok/s，但 P99 TTFT 增至 10.05 秒。

### 配置选择

- **历史最大聚合吞吐**：使用 TP2/DP4、seq64 的 131K 服务，调用端并发 256；暖机后的实测输出吞吐为 **2579.94 tok/s**，但 P99 TTFT 达到 10.05 秒。该结果不是当前 1M 脚本的默认值。
- **吞吐/时延折中**：同一服务端配置将网关并发限制为 128；暖机后的实测输出吞吐为 **2196.09 tok/s**，P99 TTFT 4.37 秒。并发 64 时历史实测 1664.77 tok/s、P99 TTFT 1.15 秒。
- **低负载或显存回退**：历史 TP2/DP1/16K 配置只使用 2 张卡，不能最大化单节点吞吐；当前脚本是 TP4/DP1。
- benchmark JSON 保存在容器临时目录 `/tmp/qwen38-bench/`，容器重建后会丢失；上表已记录关键指标。

### 1M 档吞吐实测

以下是旧版 TP8/DP1 的历史结果，不是当前 TP4/DP1 或新默认值的吞吐结果。该配置使用 `MAX_NUM_SEQS=1`、`MAX_NUM_BATCHED_TOKENS=32768`、静态 YaRN 4x，并关闭 MTP 和视觉编码器。结果在本节点 `10.16.201.229`、8 张 NPU、镜像 `quay.io/ascend/vllm-ascend:v0.23.0rc1-a3` 上直接测得：

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

### 1M 吞吐 A/B 顺序

每组都使用同一批 prompt、并发、输出长度和预热次数，先确认 `/health`、模型上限和 KV Cache，再记录输出吞吐、TTFT、TPOT 与错误率：

| 组 | 配置 | 目的 | 状态 |
|----|------|------|------|
| A | TP4/DP1、seq1、MTP off、实验优化 off | 当前拓扑容量/功能基线，补测同口径吞吐 | ⚠️ 待验证 |
| B | seq16、MTP3、实验优化 off | 验证调度并发和 MTP 收益 | ⚠️ 待测，当前推荐候选 |
| C | seq16、MTP3、实验优化 on | 验证通信与服务端整组优化 | ⚠️ 待测 |
| D | B/C 胜出项，batched tokens 16K/32K/48K | 搜索该 workload 的合批点 | ⚠️ 待测 |

当前 `run_vllm.sh` 固定为 B 组。A/C/D 属于调优实验，不再通过生产脚本的环境变量切换；复测时应为每组复制独立脚本，只改表中单一变量并保留日志，避免启动入口随 shell 环境漂移。

单个活跃 Agent 可额外对比 TP8/DP1；如果需要用满 8 张卡，再单独验证 TP4/DP2。当前默认 TP4/DP1 只使用 4 张卡，不能把历史 TP4/DP2 或 TP2/DP4 吞吐直接套用到当前入口；不同拓扑必须用真实 Agent 轨迹 A/B。

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `MODEL_PATH` | `/home/jianzhnie/llmtuner/hfhub/models/Eco-Tech/Qwen3.8-27B-w8a8` | 本地 W8A8 权重目录 |
| `PORT` / `SERVED_MODEL_NAME` | `8022` / `qwen3.8` | API 端口和模型名；监听地址固定为 `0.0.0.0` |
| `DEFAULT_REASONING_EFFORT` | `medium` | 服务默认推理强度；仅支持 `xhigh` / `medium` / `low` |
| `DRY_RUN` | `0` | 只打印最终命令，不占用 NPU |

其余配置直接写在 `vllm serve` 参数中：TP4/PP1/DP1、1M、seq16、16K batched tokens、MTP3、YaRN 4x、W8A8/BF16、纯文本模式、`qwen3_xml`、Prefix Caching、Chunked Prefill、CPU binding、V1 和 FULL Decode Graph。Ascend 环境固定使用 `HCCL_BUFFSIZE=512`、`MLAPO=0` 和 Dense 模型所需的 Balance Scheduling=0。当前 DP1 组合仍需在目标节点完成启动、KV Cache、900K 检索和吞吐复测。

脚本使用 `--gpu-memory-utilization 0.90`，主要决定权重加载后留给 KV Cache 的 HBM，不能直接提高算子计算吞吐。上调前必须同时观察启动余量、峰值 HBM 和长请求 OOM；如果 MTP/seq16 组合发生 OOM，应先降低该值或回退 MTP，而不是修改多个变量。

## Claude Code 接入

推荐直接加载仓库内的环境脚本；它会覆盖外部残留的模型、地址和 `max` effort，使用本地 Qwen3.8 服务：

```bash
cd /home/jianzhnie/llmtuner/llm/EasyInfer
source ./claude_env_qwen.sh
claude --model qwen3.8 --effort medium
```

脚本默认设置 `CLAUDE_CODE_MAX_CONTEXT_TOKENS=1000000`，并把 `QWEN_REASONING_EFFORT` 映射为 Qwen 支持的 `xhigh`、`medium` 或 `low`：

```bash
QWEN_REASONING_EFFORT=low source ./claude_env_qwen.sh
claude --model qwen3.8 --effort low
```

当前服务已使用 `--enable-auto-tool-choice --tool-call-parser qwen3_xml`，Claude Code 的工具请求可被 Qwen XML parser 转换为 Anthropic `tool_use`。服务端与 Claude Code 共用唯一业务端口 `8022`；DP/HCCL 的内部通信端口不作为 Claude 接入地址。

```bash
ANTHROPIC_BASE_URL=http://localhost:8022 \
ANTHROPIC_API_KEY=dummy \
ANTHROPIC_AUTH_TOKEN=dummy \
ANTHROPIC_DEFAULT_SONNET_MODEL=qwen3.8 \
ANTHROPIC_DEFAULT_HAIKU_MODEL=qwen3.8 \
ANTHROPIC_DEFAULT_OPUS_MODEL=qwen3.8 \
claude
```

Claude Code 会把 effort 映射到 Anthropic `output_config.effort`。当前 Qwen3.8 模板接受 `xhigh`、`medium` 和 `low`；`high`/`max` 不可用。通过环境脚本可直接选择三种值：

```bash
# 逐会话选择推理强度
QWEN_REASONING_EFFORT=xhigh source ./claude_env_qwen.sh
claude --effort xhigh
CLAUDE_CODE_EFFORT_LEVEL=medium claude
CLAUDE_CODE_EFFORT_LEVEL=low claude
```

不要设置 `CLAUDE_CODE_EFFORT_LEVEL=high` 或 `max`。vLLM 会将其原样传给模型模板，Qwen3.8 随后返回参数错误。

Agent 以吞吐优先时，可降低思考强度并在 200K 左右压缩会话，减少后续轮次的重复 prefill：

```bash
claude --model qwen3.8 --effort low --autocompact 200k
```

Prefix Cache 已默认开启，对重复 system prompt 和工具定义有效；历史运行日志观察到约 64%/81% 的两个副本命中率。900K 输入的客户端 tokenizer 时间也可能成为瓶颈，`fastokens>=0.2` 是后续镜像候选，但已验证的 `v0.23.0rc1-a3` 容器未安装，脚本不会依赖它。

## 常见问题

### Q: W8A8 模型为什么还要设置 `dtype=bfloat16`？

A: 这是正确的组合。W8A8_DYNAMIC 的权重和动态激活量化 dtype 是 INT8，由
`--quantization ascend` 选择；`--dtype bfloat16` 只指定未量化参数、量化线性层输出和
主干计算 dtype。不要设置 `dtype=int8`。脚本会检查本地量化描述文件，并固定使用
`--quantization ascend`；需要 BF16 全量权重时应换用原始未量化模型目录和对应启动脚本。

### Q: 为什么默认 TP=4、DP=1？

A: 1M 上下文需要每个副本有足够 KV Cache。TP4 让单个副本使用 4 张卡，DP1 保证只启动一个副本并避免误用 DP 内部 RPC；历史 TP4/DP2 seq1/MTP off 基线中，DP0/DP1 各有 `2,731,003` KV Cache tokens，当前 DP1 仍需复测确认。TP2/DP4 的短文本峰值更高，但每副本历史 KV Cache 只有 `808,493` tokens，无法满足单请求 1M。若需要用满 8 卡，应单独验证 TP4/DP2 配置。

### Q: 为什么不能开启 Balance Scheduling？

A: 该开关在 `v0.23.0rc1-a3` 的 DP 路径中实例化 MoE 专用 `DPEngineCoreProc`。Qwen3.8-27B 是 Dense 模型，实测会触发断言并启动失败。它不是此模型的通用 DP 负载均衡开关。

### Q: 16K 和 32K batched tokens 如何选择？

A: 历史 256/128 workload 下，32K 在并发 64/128 均比 16K 高约 10%；当前 1M 脚本固定 16K，优先保证长上下文余量。若短请求吞吐优先且 HBM 余量充足，应复制脚本单独验证 32K，不要通过生产环境变量临时改动多个参数。

### Q: 为什么默认开启 MTP？

A: 模型配置包含 1 层 MTP 草稿头，脚本使用官方 `qwen3_5_mtp` 和 3 个 speculative tokens。历史吞吐配置中 MTP on 优于 off；当前 1M 配置的收益和容量影响尚待 A/B，因此启动后仍需核对 KV Cache。回退测试应复制脚本并删除 `--speculative-config`，不要改变默认生产入口。

### Q: 为什么 1M 档默认 seq16，但仍关闭视觉？

A: seq16 允许多个中短 Agent 请求合批，不代表 16 个 1M 请求可同时驻留；真正的并发受 KV Cache 容量约束。1M 服务面向纯文本 Claude Agent，所以固定 `--language-model-only` 释放视觉编码器资源。需要视觉时应使用独立脚本，并重新做 HBM、1M 和多模态功能验证。

### Q: 为什么不再提供实验调优开关？

A: FlashComm1、Reduce Sample、HCCL buffer 和 HTTP 参数会同时改变通信、采样与服务开销，而当前 1M/MTP3 组合没有对应实测。生产脚本固定保守值；实验时使用独立脚本，每次只改一项并记录吞吐、TTFT、TPOT、错误率和 KV Cache。

### Q: reasoning_effort 为什么不支持 high 或 max？

A: vLLM 的通用协议接受更多 effort 名称，但本模型的 chat template 只定义 `xhigh`、`medium`、`low`。协议接收不等于模型模板支持，其他值最终会在模板渲染阶段报错。

### Q: 这个模型需要 EP 或 MLAPO 吗？

A: 不需要。它是 Dense Hybrid Attention，不是 MoE，也不使用 MLA；脚本默认 `MLAPO=0` 且不传 `--enable-expert-parallel`。

### Q: 这个脚本支持 Ray 多节点吗？

A: 不支持。它专用于 `10.16.201.229` 单节点 8 NPU，并固定 multiprocessing backend。Ray 多节点需要单独脚本显式维护节点地址、rank、PP/TP 拓扑和集群生命周期。

### Q: 为什么脚本默认 `VLLM_USE_V1=1` 和 `FULL_DECODE_ONLY`？

A: Qwen3.8 的官方调优章节以 V1 Chunked Prefill/SplitFuse 和全 Decode ACL Graph 为基线，精简脚本固定采用这组参数。

### Q: 如何启用视觉测试？

A: 当前服务使用 `--language-model-only`，因此 `curl_test.sh` 默认跳过视觉测试。需要视觉时先用独立部署脚本移除该参数，确认 HBM/KV Cache 仍满足目标上下文，再设置 `ENABLE_VISION=1` 执行测试。

## 验证记录

| 时间 | 镜像 | 节点 | 配置 | 结果 | 说明 |
|------|------|------|------|------|------|
| 2026-08-25 | `quay.io/ascend/vllm-ascend:v0.23.0rc1-a3` | `10.16.201.229`, 8 NPU | TP2/DP1, 16K, seq32 | ✅ | 64/64 benchmark 成功，输出 292.18 tok/s |
| 2026-08-25 | `quay.io/ascend/vllm-ascend:v0.23.0rc1-a3` | `10.16.201.229`, 8 NPU | TP2/DP4, 16K, seq64 | ✅ | 并发 16/64/128 均成功，最高 1777.98 tok/s |
| 2026-08-25 | `quay.io/ascend/vllm-ascend:v0.23.0rc1-a3` | `10.16.201.229`, 8 NPU | TP2/DP4, 32K, seq64 | ✅ | 并发 64/128 均成功，最高 1955.05 tok/s |
| 2026-08-25 | `quay.io/ascend/vllm-ascend:v0.23.0rc1-a3` | `10.16.201.229`, 8 NPU | `curl_test.sh` API 回归 | ✅ | 健康、模型列表、文本、流式、代码、数学、Anthropic PASS；旧配置工具调用 WARN |
| 2026-08-25 | `quay.io/ascend/vllm-ascend:v0.23.0rc1-a3` | `10.16.201.229`, 8 NPU | reasoning effort API | ✅ | `low`/`medium` 返回 200；不支持的 `high` 按预期返回 400 |
| 2026-08-25 | `quay.io/ascend/vllm-ascend:v0.23.0rc1-a3` | `10.16.201.229`, 8 NPU | 旧版 1M YaRN TP8/DP1 | ✅ | KV Cache 2,977,845 tokens，模型列表 1M，900K 检索命中 |
| 2026-08-25 | `quay.io/ascend/vllm-ascend:v0.23.0rc1-a3` | `10.16.201.229`, 8 NPU | `reasoning_effort` 实际请求 | ✅ | `xhigh`/`medium`/`low` 均 HTTP 200，`high` HTTP 400 |
| 2026-08-25 | `quay.io/ascend/vllm-ascend:v0.23.0rc1-a3` | `10.16.201.229`, 8 NPU | 旧版 1M `curl_test.sh` 回归 | ✅ | 基础 API 与 900K 长上下文矩阵 PASS；工具调用 WARN，多模态 SKIP |
| 2026-08-25 | `quay.io/ascend/vllm-ascend:v0.23.0rc1-a3` | `10.16.201.229`, 8 NPU | 旧版 1M TP8/DP1 吞吐 benchmark | ✅ | 256/128 单并发输出 64.88 tok/s；900K/128 输出 4.22 tok/s、总吞吐 29,683.71 tok/s |
| 2026-08-26 | `quay.io/ascend/vllm-ascend:v0.23.0rc1-a3` | `10.16.201.229`, 8 NPU | TP2/DP4, 32K, MTP on, warm cache | ✅ | 并发 128 输出 2196.09 tok/s；并发 256 输出 2579.94 tok/s，P99 TTFT 10.05 s |
| 2026-08-26 | `quay.io/ascend/vllm-ascend:v0.23.0rc1-a3` | `10.16.201.229`, 8 NPU | TP2/DP4 参数对比 | ✅ | 48K 输出 1941.76 tok/s；32K 且 MTP off 输出 1929.32 tok/s，均低于最佳配置 |
| 2026-08-26 | `quay.io/ascend/vllm-ascend:v0.23.0rc1-a3` | `10.16.201.229`, 8 NPU | 历史 1M TP4/DP2 + `qwen3_xml` | ✅ | 两副本 KV Cache 各 2,731,003 tokens；Anthropic tool_use 与 Claude Code 请求通过 |
| 2026-08-27 | `quay.io/ascend/vllm-ascend:v0.23.0rc1-a3` | 精简脚本静态检查 | 固定 1M TP4/DP1 参数/reasoning/dry-run | ✅ | shell 语法、三档 reasoning、固定 TP4/DP1/MTP3/YaRN 参数展开通过；未启动容器 |

### 2026-08-26 调参结论

历史 131K 单节点峰值吞吐配置为 `TP=2 PP=1 DP=4`、`MAX_NUM_BATCHED_TOKENS=32768`、`MAX_NUM_SEQS=64`、MTP、Prefix Cache、Chunked Prefill、FULL decode graph 和 CPU binding。客户端并发 256 时测得 **2579.94 output tok/s**；若需要较短排队延迟，使用并发 128，测得 **2196.09 output tok/s**。48K batched tokens 和关闭 MTP 均未带来收益。Balance Scheduling 必须关闭。

当前 1M 脚本固定 TP4/DP1、seq16、MTP3，并关闭未经验证的通信/采样实验优化。该组合是基于现有排队、Prefix Cache 和历史 MTP 结果提出的待测配置，不是已验证吞吐结论；已验证回退点仍是历史 TP4/DP2、seq1、MTP off。若需要用满 8 张卡，应复制脚本并独立验证 TP4/DP2 或 TP8/DP1，不能把历史吞吐直接套用到当前入口。

### 2026-08-25 结论

单节点最大吞吐实践是 4 个 TP2 数据并行副本、32K batched tokens、MTP、Prefix Cache、Chunked Prefill、FULL decode graph 和 CPU binding。并发 128 追求峰值吞吐，并发 64 获得更好的长尾时延。Balance Scheduling 必须关闭。

历史 1M 已验证实践是 TP4/DP2、静态 YaRN 4x、seq1、MTP 关闭和纯文本模式；两个副本各实测 KV Cache 为 2,731,003 tokens，900K tokens 检索成功。当前脚本改为 TP4/DP1，只启动一个副本并使用 4 张卡；其 `seq16 + MTP3` 默认候选仍需重新做容量和吞吐验证。
