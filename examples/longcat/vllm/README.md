# LongCat-Flash-Chat BF16 部署指南

> **vLLM-Ascend v0.23.0rc1** | 端口: **8010**
> 架构: LongcatFlashForCausalLM | 512 Routed + 256 Zero Experts | MoE + MLA
> 已验证配置: **TP=64 EP=64 PP=1** (8 节点) / **PP=4 TP=32 EP=32** (16 节点) | 上下文: 4096 ~ 131072 | BF16 无量化
> 注意: 需要 EasyInfer 插件注册 EP 修复；MC2 MoE comm 与 Zero Expert 权重置零不兼容
> 插件清单及各插件作用见 [docs/longcat_plugins.md](../../../docs/longcat_plugins.md)
> 验证状态: ✅ 已验证 (2026-07-27, EP + ALLGATHER, PP=2/PP=4)

超大规模 MoE 模型（约 560B 参数，512 路由专家，TopK=12），最小需要 64 张 NPU 部署。

## 模型简介

| 属性 | 值 |
|------|-----|
| **架构** | LongcatFlashForCausalLM (MLA + MoE) |
| **路由专家** | 512 (每 Token 激活 12, routed_scaling_factor=6.0) |
| **Zero 专家** | 256 (Identity) |
| **隐藏维度** | 6144 |
| **网络层数** | 28 |
| **KV LoRA Rank** | 512 |
| **rope_theta** | 10000000.0 |
| **原生上下文** | **131072** |
| **量化方式** | BF16 (无量化)，权重 ≈1.1T (75 个 safetensors 分片) |
| **MTP** | ❌ 不支持 |
| **PP 支持** | ✅ 支持 Pipeline Parallelism |
| **多模态** | ❌ 纯文本 |
| **词表大小** | 131072 |
| **工具调用解析器** | 不适用 |
| **推理解析器** | 不适用 |

### 架构注意事项

- **MLA 注意力**仅支持 block_size=128，可通过 `BLOCK_SIZE` 覆盖
- **MC2 MoE comm** 与 Zero Expert 权重置零不兼容（MoeDistributeCombineV2 shape check 失败 → collective hang），EasyInfer 插件通过 `EASYINFER_MOE_COMM=allgather` 覆盖 comm 为 ALLGATHER
- **Chunked Prefill** 与 EP token dispatch 冲突，默认禁用
- 模型包含 256 个 Zero (Identity) 专家，vLLM ≥ 0.23 下启用原生零号专家路径（`fix_ep_zero_expert.py`）

### 官方文档参考

- vLLM-Ascend 文档: https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/index.html

### 硬件要求

| 硬件 | 配置 | 推荐上下文 | 备注 |
|------|------|-----------|------|
| Atlas 800 A2/A3 (64G × 64) | BF16, TP=64 或 PP=2 TP=32 | 4K | 8 节点 × 8 卡最小配置 |
| Atlas 800 A2/A3 (64G × 128) | BF16, PP=4 TP=32 | 4K ~ 128K | 16 节点 × 8 卡，长上下文推荐 |

## 快速开始

### 前置条件

模型路径: `/home/jianzhnie/llmtuner/hfhub/models/meituan-longcat/LongCat-Flash-Chat`

```bash
# 1. 启动 NPU Docker 容器
bash scripts/docker/manage_npuslim_containers.sh start --file node_list1.txt

# 2. 启动 Ray 集群
bash scripts/ray_cluster/manage_npuslim_ray_cluster.sh start --file node_list1.txt

# 验证: 确认 8 节点、64 NPU 全部就绪
ssh 10.42.11.130 "docker exec vllm-ascend-env ray status | grep -E 'NPU|Active'"
```

### 部署

```bash
# EP 模式 (专家并行, 已验证; 必须 ALLGATHER comm, 脚本默认已带)
EP=1 EASYINFER_MOE_COMM=allgather bash examples/longcat/vllm/run_vllm.sh

# PP 模式 (PP=2, 8 节点 / PP=4, 16 节点)
PP=2 TP=32 EP=1 bash examples/longcat/vllm/run_vllm.sh
PP=4 TP=32 EP=1 bash examples/longcat/vllm/run_vllm.sh   # 需 node_list.txt 全部 16 节点

# 纯 TP 模式 (EP=0)
EP=0 bash examples/longcat/vllm/run_vllm.sh

# 自定义上下文
TP=64 MAX_MODEL_LEN=8192 MAX_NUM_SEQS=64 bash examples/longcat/vllm/run_vllm.sh

# 最大上下文模式 (131072 / 128K, 参考 vllm-ascend GLM-5.2 1M 教程裁剪)
PP=4 TP=32 EP=1 bash examples/longcat/vllm/run_vllm_long-context.sh

# 最大吞吐 (FP8 KV cache + 高并发)
KV_CACHE_DTYPE=fp8 MAX_NUM_SEQS=64 MAX_NUM_BATCHED_TOKENS=32768 bash examples/longcat/vllm/run_vllm_long-context.sh

# 图模式有问题时回退 eager
ENFORCE_EAGER=1 bash examples/longcat/vllm/run_vllm_long-context.sh
```

> 注意: 不要从 EasyInfer 根目录运行，避免插件冲突。在容器内切换到一个非 EasyInfer 目录后执行。

### 验证

```bash
bash examples/longcat/vllm/curl_test.sh

# 长上下文实测 (大海捞针, ~130K token prompt; 需先用 run_vllm_long-context.sh 部署)
ENABLE_LONG_CONTEXT=1 bash examples/longcat/vllm/curl_test.sh

# 手动验证
curl -s http://localhost:8010/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"longcat-flash","messages":[{"role":"user","content":"你好"}],"max_tokens":50}' \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'])"
```

## 推荐配置 (吞吐/显存平衡)

> 依据 2026-07-27 实测: 权重 PP=4 下仅 8.6G/rank, MLA latent KV ~1.05G/rank/128K seq,
> KV 池 2.7M tokens (20.7×128K 并发)。**显存不是瓶颈, MAX_NUM_SEQS 和 PP 结构才是。**

| 场景 | 配置 | 说明 |
|------|------|------|
| **全能型 (推荐)** | 16 节点, PP=4 TP=32 EP=32, `run_vllm_long-context.sh` | 默认 MAX_NUM_SEQS=32, ENFORCE_EAGER=0 (CUDA graph), 上下文 4K~128K 通吃; TP=32 all-reduce 只跨 4 节点 (比 TP=64 跨 8 节点省通信), EP=32 每 rank 16 专家计算密度好, PP=4 权重减半腾出 KV 空间 |
| 省资源型 | 8 节点, PP=2 TP=32 EP=32, `run_vllm.sh` | 64 NPU 即可跑 128K (KV 2.1G/rank/seq); PP 层级少单请求延迟略优; 让出 8 节点给其他任务 |
| 低延迟短上下文 | 8 节点, TP=64 EP=64 PP=1, `run_vllm.sh` | 无流水线气泡, 4K 上下文 decode ~23 tok/s |
| 极限并发 | 16 节点, PP=4 TP=32 EP=32 + FP8 KV cache | `KV_CACHE_DTYPE=fp8 MAX_NUM_SEQS=64`, KV cache 容量翻倍 (~5.4M tokens), ~41 个 128K 并发 |

其他关键参数: `CHUNKED_PREFILL=1` + `MAX_NUM_BATCHED_TOKENS=16384` (长上下文必须, 整吞 OOM; 显存充足可提升到 32768 加速 prefill)、
`GPU_MEM_UTIL=0.92` (可尝试 0.95 极限挤压)、`EASYINFER_MOE_COMM=allgather` (EP 必须)、
`HCCL_BUFFSIZE=1024` (从 768 上调, 提升多节点通信带宽利用率)。

### 吞吐优化参数

| 参数 | 默认值 | 说明 | 吞吐影响 |
|------|--------|------|----------|
| `ENFORCE_EAGER` | **0** | CUDA graph FULL_DECODE_ONLY, decode 阶段消除逐 step 编译开销 | 🔥 **最大提升** 20-50% |
| `MAX_NUM_BATCHED_TOKENS` | 16384 | Prefill 分块大小, 显存充足可调到 32768~49152 | 加速 prefill 完成 |
| `KV_CACHE_DTYPE` | (bf16) | 设为 `fp8` 后 KV cache 容量翻倍 (2.7M→5.4M tokens) | 并发翻倍 |
| `MAX_NUM_SEQS` | 32 | 纯长上下文 16~32, 混合长短 64~128 | 平衡并发与 KV 碎片 |
| `PREFIX_CACHING` | 0 | 设为 `1` 开启 APCache, 共享前缀场景减少重复 prefill | 评测/多轮场景显著 |
| `HCCL_BUFFSIZE` | 1024 | 多节点通信 buffer, 可继续上调到 2048 | 通信密集型场景 |

### 图模式 (ENFORCE_EAGER=0, 默认开启)

`run_vllm_long-context.sh` 默认 `ENFORCE_EAGER=0` 开启 `FULL_DECODE_ONLY` CUDA graph。
而 `run_vllm.sh` 默认 `ENFORCE_EAGER=1` (eager 模式)，因为短上下文下兼容性优先。

图模式的两条硬性要求 (2026-07-27 排查结论):
① `cudagraph_capture_sizes` 必须含 TP 的倍数——`run_vllm.sh` 已自动生成 (TP 步进、覆盖
   MAX_NUM_SEQS、最多 4 档, 可用 `CUDAGRAPH_CAPTURE_SIZES` 覆盖);
② 必须 `VLLM_ASCEND_ENABLE_FLASHCOMM1=0`——FlashComm1 会激活
   `SequenceParallelismPass` 编译 pass, 其在 FX 图里直接插入
   `npu_add_rms_norm_bias` 调用, 绕过 fix_layernorm_dtype 的 dtype 保护,
   float32 激活直送 ACLNN 报 EZ1001 (vllm-ascend 的 pass 缺陷, 待上游修复)。

回退 eager 模式 (图模式有问题时):
```bash
ENFORCE_EAGER=1 bash run_vllm_long-context.sh
```

图模式排障:
```bash
ENFORCE_EAGER=0 VLLM_DEBUG_DUMP=/tmp/dump bash run_vllm_long-context.sh
```

## 并行策略

| 场景 | TP | EP | PP | NPU | 上下文 | 量化 | 状态 |
|------|-----|-----|-----|-----|--------|------|------|
| EP | 64 | 64 | 1 | 64 | 4K | BF16 | ✅ |
| PP+EP | 32 | 32 | 2 | 64 | 4K | BF16 | ✅ |
| PP+EP (16 节点) | 32 | 32 | 4 | 128 | 4K | BF16 | ✅ |
| 长上下文 (16 节点) | 32 | 32 | 4 | 128 | **128K** | BF16 | ✅ |
| 纯 TP | 64 | — | 1 | 64 | 4K | BF16 | ✅ |

> EP=1 模式下必须 ALLGATHER comm（`EASYINFER_MOE_COMM=allgather`，脚本默认）避免 MC2 冲突。模型加载约需 11-13 分钟（128 卡更久）。
> PP>1 时 28 层按 stage 均分（PP=2 每 stage 14 层，PP=4 每 stage 7 层），命令示例：`PP=2 TP=32 EP=1 bash run_vllm.sh`。
> 长上下文用 `run_vllm_long-context.sh`：`MAX_MODEL_LEN=131072`、`CHUNKED_PREFILL=1` + `MAX_NUM_BATCHED_TOKENS=16384`（整吞 128K 会 OOM，须分块喂入）、`MAX_NUM_SEQS=32`、`GPU_MEM_UTIL=0.92`、`ENFORCE_EAGER=0`（CUDA graph 默认开启）。完整参数说明见脚本头部注释。

## 环境变量

> 完整环境变量说明见 [prompts/vllm_env_vars.md](../../../prompts/vllm_env_vars.md)。

## 常见问题

### Q: 为什么不能使用 Chunked Prefill?

A: Chunked Prefill 与 EP token dispatch 的冲突仅存在于 **MC2 comm** 下，因此 `run_vllm.sh` 默认禁用 (`CHUNKED_PREFILL=0`) 以保持保守。ALLGATHER comm 下已实测正常——`run_vllm_long-context.sh` 默认开启 (`CHUNKED_PREFILL=1`)，curl_test 回归与 130K 大海捞针均通过。

### Q: EP 模式为什么需要覆盖 MoE Comm?

A: MC2 MoE comm 在处理 Zero Expert 权重置零时触发 MoeDistributeCombineV2 shape check 失败，导致 collective hang。EasyInfer 插件将 comm 覆盖为 ALLGATHER 规避此问题。

### Q: 为什么模型加载需要约 11 分钟?

A: 模型权重约 1.1T（75 个 safetensors 分片）+ 64 卡 HCCL 初始化，加载时间较长。

### Q: 长上下文 (128K) 部署与默认配置有什么区别?

A: 核心差异：① `MAX_MODEL_LEN=131072`（模型原生上限）；② `CHUNKED_PREFILL=1` + `MAX_NUM_BATCHED_TOKENS=16384` 分块喂入——整吞方案（batched_tokens=132096）会因 ALLGATHER comm 持久缓冲（~30G）+ 单步 prefill 尖峰（18.1G）在 64G 卡上 OOM；③ `MAX_NUM_SEQS=32`（默认，平衡并发与 KV 碎片，纯长上下文可降到 16，混合流量可提到 128）；④ `GPU_MEM_UTIL=0.92` 给 KV cache 让空间（实测 KV cache 271 万 tokens，131072 单请求 20.73x 并发容量）；⑤ `ENFORCE_EAGER=0` 默认开启 CUDA graph，decode 吞吐提升 20-50%（`run_vllm.sh` 默认 eager 模式）。额外吞吐优化：`KV_CACHE_DTYPE=fp8` 容量翻倍、`PREFIX_CACHING=1` 开启 APCache、`HCCL_BUFFSIZE=1024` 提升通信效率。GLM-5.2 1M 方案中的 MTP/DSA/PCP/DCP 对 LongCat 不适用（无 MTP、无稀疏注意力、MLA latent KV 小，128K 无需上下文并行）。实测用 `ENABLE_LONG_CONTEXT=1 bash curl_test.sh`（大海捞针矩阵，含针位置/多针/中文/多轮用例，`LONG_CONTEXT_CASES` 可选择）。

### Q: 部署时为什么提示 "failed to map segment from shared object"?

A: 编译缓存损坏。清理缓存后重启：
```bash
docker exec vllm-ascend-env bash -c 'rm -rf /root/.cache/vllm/*'
```

### Q: 服务为什么起在 8200 而不是 8010?

A: 容器镜像预设了 `PORT=8200` 环境变量，`run_vllm.sh` 的 `PORT` 默认读取环境。部署命令显式带 `PORT=8010` 即可。

### Q: 启动秒退, 报 "Ray 集群只有 X NPU, 当前配置需要 Y"?

A: 这是 `run_vllm.sh` 的前置校验：并行配置 (TP×PP×DP) 超过了 Ray 集群实际 NPU 数——通常是容器/集群重启后节点变少，或节点文件用错（8 节点用 `node_list1.txt`/`node_list2.txt`，16 节点用 `node_list.txt`）。以前这种情况 vllm 会无报错挂死在 placement group 等待上，现在 fail-fast。同时脚本还会校验 PP 必须整除 28 层。

## 验证记录

| 时间 | 镜像 | 节点 | 配置 | 结果 | 日志 | 说明 |
|------|------|------|------|------|------|------|
| 2026-07-27 | v0.23.0rc1-a3 | 8×8 NPU | TP=64 EP=64, ALLGATHER | ✅ | `logs/vllm_longcat_20260727_031627.log` | curl_test 全部 PASS，54 个 patch 全部生效，无刷屏告警/无 EZ1001 |
| 2026-07-27 | v0.23.0rc1-a3 | 8×8 NPU | PP=2 TP=32 EP=32, ALLGATHER | ✅ | `logs/vllm_longcat_20260727_034459.log` | curl_test 全部 PASS，PP stage 均分 14 层，无报错 |
| 2026-07-27 | v0.23.0rc1-a3 | 16×8 NPU | PP=4 TP=32 EP=32, ALLGATHER | ✅ | `logs/vllm_longcat_20260727_040323.log` | 冒烟 PASS，PP stage 均分 7 层，128 卡加载 13 分钟，无报错 |
| 2026-07-27 | v0.23.0rc1-a3 | 16×8 NPU | PP=4 TP=32 EP=32, 128K, chunked prefill | ✅ | `logs/vllm_longcat_20260727_043057.log` | KV cache 271万 tokens (131072 单请求 20.73x)；curl_test 回归 PASS；130040 tokens 大海捞针命中，8s 完成。注: 整吞方案 (batched=132096) OOM，须 chunked prefill |
