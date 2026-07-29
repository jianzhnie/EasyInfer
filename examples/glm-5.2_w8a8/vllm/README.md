# GLM-5.2 W8A8 部署指南

> **vLLM-Ascend 0.23.0rc1 + CANN 8.5.1** | 端口: **8007**
> 架构: GlmMoeDsaForCausalLM | 256 Experts | MoE | MTP | W8A8 量化
> 已验证: **32K — TP=8 PP=2 (2× A2)** | **1M — DP=8 TP=8 EP=64 DCP=8 (8× A2, `vllm/dp1m/`)** | 原生 DP=2 EP=16 (2× A2, ≤20K)
> GLM-5.2 与 GLM-5/5.1 共享相同架构，上下文窗口扩展至 1M
> PP=2 已验证可用（v0.23.0）；v0.23.0rc1 共部署 PP>1+MTP 未支持（修复 #11076 仅在 main）

## 模型简介

| 属性 | 值 |
|------|-----|
| **架构** | GlmMoeDsaForCausalLM (MoE + DSA + MLA) |
| **路由专家** | 256 (每 Token 激活 8 专家) |
| **隐藏维度** | 6144 |
| **网络层数** | 78 |
| **MLA** | kv_lora_rank=512, q_lora_rank=2048, qk_head_dim=256, v_head_dim=256 |
| **原生上下文** | **1,048,576** (1M) |
| **量化方式** | W8A8 (8-bit 权重 + 8-bit 激活) |
| **MTP** | num_nextn_predict_layers=1（默认关，`ENABLE_MTP=1` 打开；v0.23.0rc1 共部署 PP>1 时不可用） |
| **PP 支持** | ✅ PP=2 已验证（A2 64G 保底配置；推荐 TP=8 DP=2 EP=16）；共部署 PP>1+MTP 未进 v0.23.0rc1（#11076 仅在 main） |
| **工具调用解析器** | glm47 |
| **推理解析器** | glm45 |
| **词表大小** | 154,880 |

### 架构注意事项

GLM-5.2 的 config.json 包含 `index_topk: 2048` 和 `index_topk_freq: 4`（indexer 仅存在于
部分层：0,1,2,6,10,…）。**vLLM-Ascend 0.23.0 原生支持该层模式**（`index_skip_topk_offset`），
0.22.1 需手工 patch（仅旧版需要，建议直接升级 v0.23.0）。
DSA 路径不兼容 FLASHCOMM1，**必须设置 `VLLM_ASCEND_ENABLE_FLASHCOMM1=0`**（脚本已内置）。

### 官方文档参考

- GLM-5.2 官方部署文档: https://docs.vllm.ai/projects/ascend/en/main/tutorials/models/GLM5.2.html
- vLLM 官方文档: https://docs.vllm.ai/en/stable/

## 快速开始

### 前置条件

模型路径: `/home/jianzhnie/llmtuner/hfhub/models/Eco-Tech/GLM-5.2-w8a8`

**硬件要求**:
- **A3 (128GB/NPU)**: ✅ TP=8 单节点
- **A2 (64GB/NPU)**: 推荐 **TP=8 DP=2 EP=16 两节点**（PP=1 故 MTP 可用，待实测）；保底 **TP=8 PP=2**（已验证）；
  TP=8 单节点 OOM（权重 ~60.4GB/卡）；TP=16 两节点亦可（每卡 ~30GB，TP 跨节点，待 v0.23.0 复测）

```bash
# 确认 NPU 内存（容器内执行）
npu-smi info | grep "HBM-Usage" | head -1
# 65536 MB = 64GB (A2) | 131072 MB = 128GB (A3)
```

> 以下部署步骤适用于 **A2 (64GB) 与 A3 (128GB)**：A2 推荐 `TP=8 DP=2`（保底 `PP=2`，均两节点），A3 单节点直接起。

```bash
# 1. 启动 NPU Docker 容器
bash scripts/docker/manage_npuslim_containers.sh start --file nodes/node_list.txt

# 2. 启动 Ray 集群
bash scripts/ray_cluster/manage_npuslim_ray_cluster.sh start --file nodes/node_list.txt
```

### 部署

```bash
# A3 单节点 (32K 上下文, TP=8)
bash examples/glm-5.2_w8a8/vllm/run_vllm.sh

# A2 两节点 · DP=2 EP=16（已验证：MTP 不可用、上下文 ≤ ~20K，原生逐节点启动）
#   node0:
NIC_NAME=<nic> HCCL_IF_IP=<node0> DP=2 DP_ADDRESS=<node0> DP_START_RANK=0 \
  FLASHCOMM1=1 MAX_MODEL_LEN=20480 MAX_NUM_BATCHED_TOKENS=8192 \
  bash examples/glm-5.2_w8a8/vllm/run_vllm.sh
#   node1:
NIC_NAME=<nic> HCCL_IF_IP=<node1> DP=2 DP_ADDRESS=<node0> DP_START_RANK=1 HEADLESS=1 \
  FLASHCOMM1=1 MAX_MODEL_LEN=20480 MAX_NUM_BATCHED_TOKENS=8192 \
  bash examples/glm-5.2_w8a8/vllm/run_vllm.sh

# A2 两节点 · 保底（已验证 32K 配置, TP=8 PP=2, 需先起跨节点 Ray 集群）
RAY_ADDRESS=<head>:6379 PP=2 bash examples/glm-5.2_w8a8/vllm/run_vllm.sh

# 1M 超长上下文
#   A3 单节点 (TP=16 DSA CP):
bash examples/glm-5.2_w8a8/vllm/run_vllm_1m.sh
#   8×A2 共部署（已验证，原生 DP=8, 见 vllm/dp1m/README.md）:
bash examples/glm-5.2_w8a8/vllm/dp1m/remote_deploy.sh deploy

# 后台运行
nohup bash examples/glm-5.2_w8a8/vllm/run_vllm.sh > glm5_2_vllm.log 2>&1 &

# 打开 MTP 投机解码（仅 PP=1 可用；v0.23.0rc1 共部署 PP>1+MTP 未支持）
ENABLE_MTP=1 bash examples/glm-5.2_w8a8/vllm/run_vllm.sh
```

### 多节点 DP 部署方式（实测结论）

- **单次 `vllm serve` 启动 DP>1 不可行**：所有 DP engine core 都会落在启动节点（设备按
  `local_dp_rank × TP×PP` 本机枚举，超界即 IndexError），`--distributed-executor-backend ray`
  也一样——Ray 只负责单个 engine 内部的 worker 放置
- **跨节点 DP 必须原生逐节点启动**：每节点一个 `vllm serve`，rank0 带 API、其余 `--headless`
  （脚本旋钮：`DP_ADDRESS` / `DP_START_RANK` / `HEADLESS` / `DP_SIZE_LOCAL` / `DP_RPC_PORT`）
- **`--data-parallel-size-local` 必须显式给**：缺省时 headless 节点按全局 rank 切设备（IndexError）
- Ray 仅用于 **DP=1 且 TP/PP 跨节点** 的场景（如 PP=2 两节点、TP=16 单实例、TP=32 PP=4 单实例 16 节点）：

```bash
# TP=32 PP=4 DP=1 单实例（16 节点 128 卡，MTP 关；未实测，先小规模 TP=16 PP=2 起步验证）
RAY_ADDRESS=<head>:6379 NIC_NAME=<nic> HCCL_IF_IP=<head_ip> \
  TP=32 PP=4 bash examples/glm-5.2_w8a8/vllm/run_vllm.sh
```

```bash
# 获取 Ray 集群地址
docker exec vllm-ascend-env python3 -c "
import ray; ray.init(address='auto', ignore_reinit_error=True)
print(ray.get_runtime_context().gcs_address)
"

# 部署时导出
RAY_ADDRESS=10.42.11.130:6379 PP=2 bash run_vllm.sh
```

### vLLM-Ascend 版本要求

请直接使用最新的 vllm-ascend 版本（≥ **v0.23.0**）。旧版 v0.22.1 不原生支持 GLM-5.2
的 indexer 层模式（`index_skip_topk_offset`/`index_topk_freq`），加载 W8A8 权重时会报
`KeyError: model.layers.N.self_attn.indexer.wq_b.weight`；v0.23.0 起已原生支持，无需任何手工修复。

### 验证

```bash
# 运行测试脚本
bash examples/glm-5.2_w8a8/vllm/curl_test.sh

# 手动验证
curl http://localhost:8007/v1/models
curl http://localhost:8007/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"glm-5.2","messages":[{"role":"user","content":"Hello"}],"max_tokens":50}'
```

## 接入 Claude Code / Agent 工具

vLLM 提供 **Anthropic Messages API**，部署脚本已内置 Agent 必需参数
（`--enable-auto-tool-choice --tool-call-parser glm47 --reasoning-parser glm45`
及 `--enable-prefix-caching`），Claude Code 可直接接入，无需修改服务端：

```bash
# 一键加载环境并启动（含服务健康检查）
source examples/glm-5.2_w8a8/vllm/agent_api.sh && claude

# 连接远程服务
HOST=10.42.11.196 source examples/glm-5.2_w8a8/vllm/agent_api.sh && claude
```

| 环境变量 | 值 | 说明 |
|----------|-----|------|
| `ANTHROPIC_BASE_URL` | `http://<host>:8007` | vLLM 服务地址 |
| `ANTHROPIC_API_KEY` / `ANTHROPIC_AUTH_TOKEN` | `dummy` | vLLM 默认不鉴权，任意非空值 |
| `ANTHROPIC_DEFAULT_{SONNET,HAIKU,OPUS}_MODEL` | `glm-5.2` | 必须与 `--served-model-name` 一致，且不含 `/` |

也可写入 `~/.claude/settings.json` 的 `env` 字段持久化，详见
[docs/claude-code-vllm-setup.md](../../../docs/claude-code-vllm-setup.md)。

验证：

```bash
# Anthropic Messages API 与工具调用（curl_test.sh 已包含这些测试项）
bash examples/glm-5.2_w8a8/vllm/curl_test.sh

# Claude Code 内验证工具调用：
> 列出当前目录下的文件
```

Agent 场景要点：

- **并发**：Claude Code 主线程 + 子 agent 会并发请求，按实际并发度调整 `MAX_NUM_SEQS`
- **长上下文**：多轮工具调用上下文膨胀快，需要 1M 时改用 `run_vllm_1m.sh` 部署
- **prefix caching**：脚本已开启；若推理明显变慢，在 `agent_api.sh` 中取消
  `CLAUDE_CODE_ATTRIBUTION_HEADER=0` 的注释
- 其他类 Claude Code 的 Agent 工具只要支持 Anthropic Messages API（或 OpenAI 兼容 API），
  同样指向 `http://<host>:8007` 即可

## 并行策略

| 场景 | TP | PP | DP | NPU | 上下文 | 状态 |
|------|-----|-----|-----|-----|--------|------|
| 单节点 A3 (64G×16) | 8 | 1 | 2 | 16 | 32K | 官方推荐配置（dp2tp8） |
| 单节点 A3 (64G×16) 低延迟 | 16 | 1 | 1 | 16 | 32K | 官方建议 dp1tp16 关 EP（`ENABLE_EP=0`，单节点 `VLLM_ASCEND_ENABLE_FUSED_MC2=0`） |
| 单节点 A3 (128G×8) | 8 | 1 | 1 | 8 | 32K | ✅ 官方支持（w8a8 最低硬件） |
| 单节点 A2 (64G) | 8 | 1 | 1 | 8 | 4K+ | ❌ OOM（权重 ~60.4GB/卡） |
| **2 节点 A2 (64G)** | **8** | **2** | 1 | 16 | 32K | ✅ **已验证 PASS（07-22，32K 最实用）** |
| 2 节点 A2 (64G) 原生 DP | 8 | 1 | 2 | 16 | **≤20K** | ✅ 已验证 PASS（07-28）：EP=16 权重实测 ~33G/卡，**MTP 开不起来**（差 ~5G）、KV 池仅 ~2G；需 `FLASHCOMM1=1 + HCCL_BUFFSIZE=768 + batched 8192` |
| 2 节点 A2 (64G) | 16 | 1 | 1 | 16 | 1M | ❌ 两轮实证：EP=16 权重+MTP 后无 KV（util 0.95 仍失败） |
| 8 节点 A2 (64G) 共部署 1M | 8 | 1 | 8 | 64 | **1M** | ✅ **已验证 PASS（07-28，`vllm/dp1m/`）**：EP=64（4 专家/卡）+ DCP=8 + MTP |
| 4 节点 A2 (64G) | 16 | 2 | 1 | 32 | 32K | ⏳ 同 TP=16（维度可行，待复测；PP>1 时 MTP 不可用） |
| 16 节点 A2 (64G) 单实例 Ray | 32 | 4 | 1 | 128 | 32K~1M | ⏳ 未实测（机制同 PP=2：DP=1+Ray 跨节点放 worker）。维度可整除（64/32=2、indexer 32/32=1）；PP=4 每卡 ~20 层 ≈15G；TP 组跨 4 节点 + 3 个跨节点 PP 边界，通信开销大；需 `FLASHCOMM1=1 + FUSED_MC2=0 + ENABLE_MTP=0` |

> **关键结论**：
> - GLM-5.2 W8A8 在 64GB A2 NPU 上 TP=8 单节点 OOM（权重固定消耗 ~60.4GB/卡）
> - **2 节点 A2 首选 TP=8 PP=2（32K 已验证）**；原生 DP=2 EP=16 也已验证但有硬约束：
>   权重实测 ~33G/卡（16 专家），**MTP 开不起来**（KV 差 ~5G），上下文 ≤ ~20K，KV 池仅 ~2G
> - **1M on A2 只走多节点共部署 `vllm/dp1m/`（8 节点 DP=8 EP=64 DCP=8，已验证）**：
>   TP=16 共部署两轮实证内存不可行；PD 分离被两个上游 bug 阻断（见 `pd-separation/README.md` 已知问题）
> - **PP=2 已在 v0.23.0 上验证可用**；**v0.23.0rc1 共部署 PP>1+MTP 未支持**
>   （修复 #11076 仅在 main，未进 release；PD 分离 P 节点自 v0.22.1rc1 起支持，#10199），
>   脚本 `ENABLE_MTP` 默认关并在 PP>1 时自动禁用
> - TP=16 维度上可行：注意力头 64/16=4、indexer 头 32/16=2 均可整除，官方 1M 单节点配置即 TP=16。
>   早前"576 不可整除"的论断有误（576=kv_lora_rank 512+qk_rope 64，576/16=36）；
>   v0.22.1 的失败是分片长度计算 bug（length 704 vs 576，v0.23.0 已自动修正，启动日志可见）
>
> **W8A8 on A2 避坑清单（07-28 十轮实测）**：
> - `HCCL_BUFFSIZE`：跨节点 MoE A2A 窗口必须 **768**（400 报 `Get WinSize failed` 561002）
> - `FUSED_MC2`：W8A8 **跨节点 EP 必崩**（`aclnnDispatchFFNCombine`），默认 0；节点内 EP 可开
> - `FLASHCOMM1`：DCP>1 强制 =1（`DSA CP requires SP`）；SP 关闭时 A2A 走 flashinfer_all2allv 后端也会失败
> - `enable_dsa_cp` 会联动 DCP（需与 FLASHCOMM1 一致），32K 脚本已按 FLASHCOMM1 联动
> - 崩溃后重试前用 `remote_deploy.sh clean`（重启容器）——`pkill` 不能回收驱动级 NPU 显存残留

### 内存分析（W8A8 @ 64GB A2）

| 组件 | 消耗 (TP=8) | 说明 |
|------|-------------|------|
| 模型权重 (256 专家) | ≈60.4 GiB | 256 专家 MoE 权重，每卡 32 专家 |
| KV Cache (32K ctx) | ≈2-4 GiB | 随 max_model_len 变化 |
| 编译缓存 + 临时 | ≈1-2 GiB | CUDA Graph、triton、算子缓存 |
| **总计** | **≈64 GiB** | **超出 A2 64GB 上限** |

降低 `max_model_len`、`max_num_seqs`、禁用 MTP/CUDA Graph 均无法改变权重的固定消耗。

### 多节点 1M 部署（A2，07-28 实测结论）

> ⚠️ 官方 1M 配置仅覆盖 A3 (64G×16) + W4A8C8；A2 + W8A8 的 1M 为本仓库自建方案。

| 方案 | 全局并行 | 状态 | 说明 |
|------|----------|------|------|
| **8 节点共部署（`vllm/dp1m/`）** | DP=8 TP=8 EP=64 PCP1 DCP=8, MTP 3 | ✅ **已验证 PASS** | 唯一已验证的 A2 1M 方案；EP=64 → 4 专家/卡，KV ~40G/卡，单 rank ~3 条全 1M 并发，原生 DP 逐节点启动 |
| TP=16 共部署（2 节点） | TP=16 DP=1 DCP=16 | ❌ 两轮实证不可行 | EP=16 权重 ~33G/卡 + MTP，util 0.95 仍无 KV |
| TP=16 DP=4（原"方案 C"） | TP=16 DP=4 DCP=16 | ❌ 机制不可行 | vLLM DP>1 要求 rank 的 TP×PP ≤ 单节点卡数（engine core 本机枚举设备） |
| PD 分离（4P+4D，`pd-separation/`） | P/D 均 DP4 TP8 DCP8 | ⚠️ P 就绪 / D 被上游 bug 阻断 | MooncakeConnectorV1 DSA cache bug（#12863，不在 v0.23.0rc1）+ kv_consumer graph capture 量化 bug，详见 `pd-separation/README.md` |

```bash
# 已验证的 1M 部署（8 节点原生 DP，入口 master:8007）
bash examples/glm-5.2_w8a8/vllm/dp1m/remote_deploy.sh deploy
```

要点：

- **TP=8 保持在节点内**（8 卡高速互联）；跨节点 DP 必须原生逐节点启动（见上文「多节点 DP 部署方式」）。
- **DCP=8=TP**：1M KV 按 DCP 组分片（MLA 压缩后 ~11.8 GiB/条/卡）。
- **16 节点 (128 卡) 扩展**：同构放到 DP=16 EP=128（每 rank 1 节点），或起 2 个 DP=8 实例 + 负载均衡。
- EP all-to-all 横跨多节点，依赖 RoCE 与 `HCCL_BUFFSIZE=768`（A2A 窗口，见避坑清单）。

## 环境变量

> 完整环境变量说明见 [docs/prompts/vllm_env_vars.md](../../../docs/prompts/vllm_env_vars.md)。
> 部署工作流见 [docs/prompts/vllm-prompt.md](../../../docs/prompts/vllm-prompt.md)。

## 功能验证清单

### 基础功能

| 功能 | 状态 | 脚本 |
|------|------|------|
| 基础 Chat Completion (32K, PP=2) | ✅（2026-07-22 验证） | `run_vllm.sh` |
| 1M 超长上下文（8 节点共部署 DP=8 DCP=8） | ✅（2026-07-28 验证，curl 全项 PASS） | `vllm/dp1m/` |
| 原生多节点 DP（2 节点 TP=8 DP=2 EP=16, 20K） | ✅（2026-07-28 验证） | `run_vllm.sh` (DP_ADDRESS 模式) |
| Tool Calling (glm47) | ✅ | `curl_test.sh` |
| Anthropic Messages API | ✅ | `curl_test.sh` |
| Claude Code / Agent 接入 | ⏳ 待实测（Anthropic API 已验证） | `agent_api.sh` |
| MTP 投机解码 | ✅（dp1m 内置 3 tokens） | `vllm/dp1m/conode.sh` |
| 多节点 PP=2 | ✅（2026-07-22 验证） | `run_vllm.sh` (PP=2) |
| PD 分离 1M | ⚠️ P 就绪 / D 被上游 bug 阻断（#12863） | `pd-separation/` |

## 精度与性能评估

对齐[官方教程](https://docs.vllm.ai/projects/ascend/zh-cn/latest/tutorials/models/GLM5.2.html)的评估方法：

- **精度评估**：使用 AISBench，详见官方文档「使用 AISBench」章节；lm_eval 官方对该模型尚未验证，本仓库用法见 [docs/lm_eval_usage.md](../../../docs/lm_eval_usage.md)
- **性能评估**：AISBench 性能评估或 `vllm bench`（vllm bench serve / latency / throughput）
- 注意：`max-model-len` 与 `max-num-seqs` 需按实际业务场景设置后再压测，其余参数沿用部署章节配置

```bash
# 示例：vllm bench 吞吐压测（服务启动后执行）
vllm bench serve --model glm-5.2 --host localhost --port 8007 \
  --dataset-name random --num-prompts 100 --random-input-len 1024 --random-output-len 1024
```

## 常见问题

### Q: GLM-5.2 和 GLM-5/5.1 的部署配置有什么不同？

A: 架构相同 (GlmMoeDsaForCausalLM)，主要区别：GLM-5.2 原生上下文扩展至 **1M**，head_dim 从 64 增至 192，新增 qk_head_dim=256、v_head_dim=256。NPU 环境变量、并行配置、量化参数通用。W8A8 比 W4A8 精度更高但占用更多显存。

### Q: 为什么必须设置 FLASHCOMM1=0？

A: GLM-5.2 的 `index_topk: 2048` 触发 DSA CP 路径，W8A8 下缺少 `aclnn_input_scale` 属性导致 crash。

### Q: W8A8 和 W4A8 有什么区别？

A: W8A8 使用 8-bit 权重 + 8-bit 激活，精度更高（MMLU 损失 < 0.5%）；W4A8 使用 4-bit 权重 + 8-bit 激活，显存占用更少但精度略低。W8A8 在昇腾上通过 `--quantization ascend` 启用。

### Q: MTP 投机解码对内存有什么影响？

A: MTP 加载第二份模型权重，减少 KV cache 可用空间。TP=8 单节点时 max_model_len 从 64K 降至 ~32K。

### Q: A2 两节点用 PP=2 还是 DP=2？

A: **32K 实用首选 TP=8 PP=2**（07-22 已验证，MTP 因 PP 互斥不可用但配置宽松）。**TP=8 DP=2 EP=16 也已验证**（07-28），但实测有硬约束：权重 ~33G/卡 → **MTP 开不起来**（KV 差 ~5G）、上下文 ≤ ~20K、KV 池 ~2G，且需 `FLASHCOMM1=1 + HCCL_BUFFSIZE=768 + batched 8192 + 原生逐节点启动`（见「多节点 DP 部署方式」）。要 1M 用 8 节点共部署（`vllm/dp1m/`）。PP>1 时 MTP 在 v0.23.0rc1 共部署下不可用（#11076 已在 main 支持，待新镜像）。

### Q: 启动时报 "failed to map segment from shared object" 错误？

A: Triton/TorchInductor 编译缓存损坏或版本不兼容。清理缓存后重启：
```bash
docker exec vllm-ascend-env bash -c 'rm -rf /dev/shm/glm52-cache/triton/* /dev/shm/glm52-cache/torchinductor/*'
# 然后重新启动 run_vllm.sh
```

### Q: 如何配置多节点网络？

A: 设置 `NIC_NAME` 和 `HCCL_IF_IP` 环境变量绑定高速网卡：
```bash
NIC_NAME=enp66s0f0 HCCL_IF_IP=10.42.11.130 RAY_ADDRESS=10.42.11.130:6379 PP=2 bash run_vllm.sh
```
单节点无需设置（留空自动探测）。

### Q: 缓存目录在哪里？如何修改？

A: 缓存分为两类：
- **不可执行缓存** (`CACHE_ROOT`, 默认项目共享路径 `.cache/glm52-w8a8`): tmp、home、vllm、ascend-log（不要用 `/dev/shm`：worker 节点 clang 编译会因 TMPDIR 缺失失败）
- **可执行缓存** (`EXEC_CACHE_ROOT`, 默认 `/root/.cache/glm52-cache`): triton、torchinductor 编译的 `.so` 文件，需要可执行文件系统（容器 `/dev/shm` 通常挂载 `noexec`）

可通过环境变量分别覆盖路径。

### Q: enforce_eager 去哪里了？

A: 已从顶层 `--enforce-eager` 标志移至 `--speculative-config` 内的 `"enforce_eager": true`（对齐官方脚本）。主模型使用 CUDA Graph (`FULL_DECODE_ONLY`)，仅 MTP 草稿模型使用 eager 模式，确保兼容性与性能兼顾。

### Q: 部署时一直卡在 "Waiting for creating a placement group"？

A: Engine Core 子进程默认启动了本地 Ray 实例（只有 8 NPU），未连接到集群。设置 `RAY_ADDRESS` 环境变量指向 Ray Head 节点：

```bash
# 获取地址
docker exec vllm-ascend-env python3 -c "
import ray; ray.init(address='auto', ignore_reinit_error=True)
print(ray.get_runtime_context().gcs_address)
"

# 部署（A2 已验证配置：TP=8 PP=2 两节点）
RAY_ADDRESS=10.42.11.130:6379 PP=2 bash run_vllm.sh
```

### Q: A2 (64GB) 上能用 TP=8 部署吗？

A: **单节点不能**（权重固定消耗 ≈60.4 GiB/卡，仅剩 ~3.6GB 余量），但 **TP=8 PP=2 两节点已验证可用**（2026-07-22 PASS，每卡 ~30 GiB）。A3 (128GB) 可单节点 TP=8。PP>1 时 MTP 不可用（脚本会自动禁用）。

### Q: TP=16 能用吗？

A: **可以**。注意力头 64/16=4、indexer 头 32/16=2 均可整除，官方 GLM-5.2 1M 单节点配置即 TP=16（DCP16）。早前"num_kv_heads=3 × head_dim=192=576 不可整除"的论断有误：config 中 `num_key_value_heads=64`（MLA），"3"实为 `index_skip_topk_offset`，576=`kv_lora_rank`(512)+`qk_rope_head_dim`(64)，576/16=36。v0.22.1 上的报错 `start(0) + length(704) > 576` 是分片长度计算 bug（704=512+`qk_nope_head_dim` 192），非维度限制。A2 + W8A8 + v0.23.0 组合建议实测确认。

## 验证记录

| 时间 | 镜像 | 节点 | 配置 | 结果 | 说明 |
|------|------|------|------|------|------|
| 2026-07-22 | `vllm-ascend:v0.23.0rc1-a3` (CANN 8.5.1) | pair1: 196/197 | TP=8 PP=2, 32K | ✅ PASS | curl 全项通过 |
| 2026-07-28 | 同上 | 194-201 (8节点) | **DP=8 TP=8 EP=64 DCP=8, MTP, 1M**（`vllm/dp1m/`） | ✅ **PASS** | `/v1/models` max_model_len=1024000；chat/工具调用/流式/Anthropic API 全项 PASS |
| 2026-07-28 | 同上 | 131/132 | TP=8 DP=2 EP=16 原生 DP, MTP off, 20K | ✅ PASS（有约束） | 原生逐节点启动；BUFFSIZE=768/batched 8192/FLASHCOMM1=1 |
| 2026-07-28 | 同上 | 131/132 | 同上单次启动 + Ray | ❌ | 实证：单次启动 DP>1 全部引擎落头节点（IndexError [8,16)），Ray 无法跨节点放 engine |
| 2026-07-28 | 同上 | 194/195 | TP=16 DP=1 共部署 1M（两轮） | ❌ | EP=16 权重 ~33G/卡 + MTP，util 0.95 仍无 KV 空间 |
| 2026-07-28 | 同上 | 202-205 (P) | PD 分离 P 侧 DP4 TP8 DCP8 | ✅ 就绪 | FUSED_MC2=0 + util 0.93 |
| 2026-07-28 | 同上 | 206-209 (D) | PD 分离 D 侧 | ⚠️ 阻断 | MooncakeConnectorV1 DSA cache bug（上游 #12863，不在 v0.23.0rc1）；另 kv_consumer+FULL_DECODE capture 量化 bug（规避：enforce-eager） |

### 2026-07-22 结论：v0.23.0 原生支持

- v0.23.0 的 `deepseek_v2.py` 原生实现 `index_skip_topk_offset`/`index_topk_freq`，
  与 checkpoint 的 indexer 层模式（0,1,2,6,10,…）完全吻合，无需任何手工修复，
  直接使用最新 vllm-ascend 版本即可。
- TP=8 单节点在 A2 64G 上 OOM（~60.4GiB/卡）→ **TP=8 PP=2**（两节点 ~30GiB/卡）。
- PP>1+MTP 在 v0.23.0rc1 共部署被拒绝（mixed 支持 #11076 未进 release，仅 main），脚本新增 `ENABLE_MTP` 开关（默认关，PP>1 自动禁用）。
- `CACHE_ROOT` 改到项目共享路径（worker 节点 `/dev/shm` 下 clang 编译失败）。

### 2026-07-28 结论：1M on A2 = 多节点共部署（dp1m）

- 1M 唯一已验证路径：8 节点原生 DP（DP=8 TP=8 EP=64 DCP=8 + MTP），详见 `vllm/dp1m/README.md`。
- 跨节点 DP 的工程结论：原生逐节点启动 + 显式 `--data-parallel-size-local`；单次启动 DP>1 不可行（Ray 无关）。
- W8A8 on A2 的 MoE A2A/量化坑全部记录在「关键结论 · 避坑清单」；PD 分离待上游 #12863 镜像。
