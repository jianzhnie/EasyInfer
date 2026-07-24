# GLM-5.2 W8A8 部署指南

> **vLLM-Ascend 0.23.0rc1 + CANN 8.5.1** | 端口: **8007**
> 架构: GlmMoeDsaForCausalLM | 256 Experts | MoE | MTP | W8A8 量化
> 已验证配置: **TP=8 PP=2 (2× A2 64G)** | 上下文: 32K | Chat ✅ Tool Calling ✅
> GLM-5.2 与 GLM-5/5.1 共享相同架构，上下文窗口扩展至 1M
> PP=2 已验证可用（v0.23.0）；PP>1 与 MTP 互斥

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
| **MTP** | num_nextn_predict_layers=1（默认关，`ENABLE_MTP=1` 打开；PP>1 时不可用） |
| **PP 支持** | ✅ PP=2 已验证（A2 64G 的推荐配置）；PP>1 与 MTP 互斥 |
| **工具调用解析器** | glm47 |
| **推理解析器** | glm45 |
| **词表大小** | 154,880 |

### 架构注意事项

GLM-5.2 的 config.json 包含 `index_topk: 2048` 和 `index_topk_freq: 4`（indexer 仅存在于
部分层：0,1,2,6,10,…）。**vLLM-Ascend 0.23.0 原生支持该层模式**（`index_skip_topk_offset`），
0.22.1 需手工 patch（存档于 `container_patch/`，仅旧版有效）。
DSA 路径不兼容 FLASHCOMM1，**必须设置 `VLLM_ASCEND_ENABLE_FLASHCOMM1=0`**（脚本已内置）。

### 官方文档参考

- GLM-5.2 官方部署文档: https://docs.vllm.ai/projects/ascend/en/main/tutorials/models/GLM5.2.html
- vLLM 官方文档: https://docs.vllm.ai/en/stable/

## 快速开始

### 前置条件

模型路径: `/home/jianzhnie/llmtuner/hfhub/models/Eco-Tech/GLM-5.2-w8a8`

**硬件要求**:
- **A3 (128GB/NPU)**: ✅ TP=8 单节点
- **A2 (64GB/NPU)**: ✅ **TP=8 PP=2 两节点**（已验证）；TP=8 单节点 OOM（权重 ~60.4GB/卡），
  TP=16 因 MLA 维度不可整除不可用

```bash
# 确认 NPU 内存（容器内执行）
npu-smi info | grep "HBM-Usage" | head -1
# 65536 MB = 64GB (A2) | 131072 MB = 128GB (A3)
```

> 以下部署步骤适用于 **A2 (64GB) 与 A3 (128GB)**：A2 用 `PP=2`（两节点），A3 单节点直接起。

```bash
# 1. 启动 NPU Docker 容器
bash scripts/docker/manage_npuslim_containers.sh start --file node_list.txt

# 2. 启动 Ray 集群
bash scripts/ray_cluster/start_npuslim_ray_cluster.sh start --file node_list.txt
```

### 部署

```bash
# A3 单节点 (32K 上下文, TP=8)
bash examples/glm5_2_w8a8/vllm/run_vllm.sh

# A2 两节点（已验证配置, TP=8 PP=2, 需先起跨节点 Ray 集群）
RAY_ADDRESS=<head>:6379 PP=2 bash examples/glm5_2_w8a8/vllm/run_vllm.sh

# 1M 超长上下文 (A3, TP=16, DSA CP)
bash examples/glm5_2_w8a8/vllm/run_vllm_1m.sh

# 后台运行
nohup bash examples/glm5_2_w8a8/vllm/run_vllm.sh > glm5_2_vllm.log 2>&1 &

# 打开 MTP 投机解码（仅 PP=1 可用，PP>1 与 MTP 互斥）
ENABLE_MTP=1 bash examples/glm5_2_w8a8/vllm/run_vllm.sh
```

### 多节点部署前提（TP>8 或 PP>1）

跨节点部署时，**必须设置 `RAY_ADDRESS`**，否则 Engine Core 子进程无法连接到 Ray 集群：

```bash
# 获取 Ray 集群地址
docker exec vllm-ascend-env python3 -c "
import ray; ray.init(address='auto', ignore_reinit_error=True)
print(ray.get_runtime_context().gcs_address)
"

# 部署时导出
RAY_ADDRESS=10.42.11.130:6379 PP=2 bash run_vllm.sh
```

### 量化文件修复（仅 v0.22.1 需要）

> ⚠️ **v0.23.0 起不再需要**：v0.23.0 原生支持 `index_skip_topk_offset`/`index_topk_freq`
> 的层模式，KeyError 已不存在。以下修复仅适用于 v0.22.1（更推荐直接用 v0.23.0；
> v0.22.1 的 vLLM 代码 patch 存档于 `container_patch/`）。

如果遇到 `KeyError: model.layers.N.self_attn.indexer.wq_b.weight`，需要修复 `quant_model_description.json`：

```bash
cd /path/to/GLM-5.2-w8a8
python3 -c "
import json, copy
with open('quant_model_description.json') as f:
    desc = json.load(f)
src_idx = {k: v for k, v in desc.items() if 'layers.6.self_attn.indexer' in k}
for layer in range(78):
    if not any(f'layers.{layer}.self_attn.indexer' in k for k in desc):
        for k, v in src_idx.items():
            desc[k.replace('layers.6.', f'layers.{layer}.')] = copy.deepcopy(v)
with open('quant_model_description.json', 'w') as f:
    json.dump(desc, f)
print('Fixed!')
"
```

> 原因：官方发布的 `quant_model_description.json` 中 78 层只有 22 层包含 Indexer 量化描述，其余 56 个 MoE 层缺失。

### 验证

```bash
# 运行测试脚本
bash examples/glm5_2_w8a8/vllm/curl_test.sh

# 手动验证
curl http://localhost:8007/v1/models
curl http://localhost:8007/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"glm-5.2","messages":[{"role":"user","content":"Hello"}],"max_tokens":50}'
```

## 并行策略

| 场景 | TP | PP | DP | NPU | 上下文 | 状态 |
|------|-----|-----|-----|-----|--------|------|
| 单节点 A3 (128G) | 8 | 1 | 2 | 16 | 32K | 官方推荐配置 |
| 单节点 A2 (64G) | 8 | 1 | 1 | 8 | 4K+ | ❌ OOM（权重 ~60.4GB/卡） |
| **2 节点 A2 (64G)** | **8** | **2** | 1 | 16 | 32K | ✅ **已验证 PASS（本集群）** |
| 2 节点 A2 (64G) | 16 | 1 | 1 | 16 | - | ❌ TP=16 维度不兼容 |
| 4 节点 A2 (64G) | 16 | 2 | 1 | 32 | - | ❌ 同上（PP 不影响 TP 维度分割） |

> **关键结论**：
> - GLM-5.2 W8A8 在 64GB A2 NPU 上 TP=8 单节点 OOM（权重固定消耗 ~60.4GB/卡）
> - **PP=2 已在 v0.23.0 上验证可用**（早前"不支持 PP"的结论作废）；**PP>1 与 MTP 互斥**，
>   脚本 `ENABLE_MTP` 默认关
> - TP=16 不可用：MLA 注意力层维度 (`num_kv_heads=3 × head_dim=192 = 576`) 无法被 16 整除

### 内存分析（W8A8 @ 64GB A2）

| 组件 | 消耗 (TP=8) | 说明 |
|------|-------------|------|
| 模型权重 (256 专家) | ≈60.4 GiB | 256 专家 MoE 权重，每卡 32 专家 |
| KV Cache (32K ctx) | ≈2-4 GiB | 随 max_model_len 变化 |
| 编译缓存 + 临时 | ≈1-2 GiB | CUDA Graph、triton、算子缓存 |
| **总计** | **≈64 GiB** | **超出 A2 64GB 上限** |

降低 `max_model_len`、`max_num_seqs`、禁用 MTP/CUDA Graph 均无法改变权重的固定消耗。

## 环境变量

> 完整环境变量说明见 [prompts/vllm_env_vars.md](../../../prompts/vllm_env_vars.md)。
> Claude Code 集成方式见 [prompts/vllm-prompt.md](../../../prompts/vllm-prompt.md)。

## 功能验证清单

### 基础功能

| 功能 | 状态 | 脚本 |
|------|------|------|
| 基础 Chat Completion (32K) | ✅ | `run_vllm.sh` |
| 1M 超长上下文 (DSA CP) | ✅ | `run_vllm_1m.sh` |
| Tool Calling (glm47) | ✅ | `curl_test.sh` |
| Anthropic Messages API | ✅ | `curl_test.sh` |
| MTP 投机解码 | ✅ | `run_vllm.sh` (内置) |
| 多节点 PP=2 | ✅ | `run_vllm.sh` (PP=2) |

## 常见问题

### Q: GLM-5.2 和 GLM-5/5.1 的部署配置有什么不同？

A: 架构相同 (GlmMoeDsaForCausalLM)，主要区别：GLM-5.2 原生上下文扩展至 **1M**，head_dim 从 64 增至 192，新增 qk_head_dim=256、v_head_dim=256。NPU 环境变量、并行配置、量化参数通用。W8A8 比 W4A8 精度更高但占用更多显存。

### Q: 为什么必须设置 FLASHCOMM1=0？

A: GLM-5.2 的 `index_topk: 2048` 触发 DSA CP 路径，W8A8 下缺少 `aclnn_input_scale` 属性导致 crash。

### Q: W8A8 和 W4A8 有什么区别？

A: W8A8 使用 8-bit 权重 + 8-bit 激活，精度更高（MMLU 损失 < 0.5%）；W4A8 使用 4-bit 权重 + 8-bit 激活，显存占用更少但精度略低。W8A8 在昇腾上通过 `--quantization ascend` 启用。

### Q: MTP 投机解码对内存有什么影响？

A: MTP 加载第二份模型权重，减少 KV cache 可用空间。TP=8 单节点时 max_model_len 从 64K 降至 ~32K。

### Q: 为什么不用 PP？

A: TP=16 因 MLA 维度不可整除而不可用；**PP=2 已在 v0.23.0 上验证可用**。PP>1 时 MTP 不可用（与 MTP 互斥）。

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

A: **单节点不能**（权重固定消耗 ≈60.4 GiB/卡，仅剩 ~3.6GB 余量），但 **TP=8 PP=2 两节点已验证可用**（2026-07-22 PASS，每卡 ~30 GiB）。A3 (128GB) 可单节点 TP=8。注意 PP>1 时必须 `ENABLE_MTP=0`。

### Q: 为什么不能用 TP=16？

A: GLM-5.2 的 MLA 注意力层维度 `num_kv_heads × head_dim = 3 × 192 = 576`，不能被 16 整除。TP=16 时 weight sharding 计算错误：`start(0) + length(704) > 576`。需要 vLLM-Ascend 修复 `deepseek_v2.py` 中的 Indexer shard 计算逻辑。

### Q: `quant_model_description.json` 有什么问题？

A: 官方发布的 W8A8 量化描述文件中，78 层只有 22 层（密集层 + 每第 4 个 MoE 层）包含 Indexer 量化条目，其余 56 个 MoE 层缺失。需要在加载前手动补充。详见「量化文件修复」章节。

## 验证记录

| 时间 | 镜像 | 节点 | 配置 | 结果 | 日志 | 说明 |
|------|------|------|------|------|------|------|
| 2026-07-20 | `quay.io/ascend/vllm-ascend:v0.22.1rc1-a3` (CANN 8.5.1) | pair1: 10.42.11.196/197 | TP=8 PP=1, PORT=8007 | ❌ FAIL_SERVICE | `logs/parallel_deploy_remaining_v022/glm5.2-w8a8_*.log` | 权重加载失败：`KeyError: 'model.layers.3.self_attn.indexer.wq_b.weight'`，当前 vLLM-Ascend 量化配置未覆盖 GLM-5.2 的 `indexer` 权重结构 |

- 该错误发生在 `vllm_ascend/quantization/modelslim_config.py` 解析量化描述时，说明模型权重 key 与当前 vLLM-Ascend 实现不匹配。
| 2026-07-22 | `quay.io/ascend/vllm-ascend:v0.23.0rc1-a3` (CANN 8.5.1) | pair1: 10.42.11.196/197 | **TP=8 PP=2**, ENABLE_MTP=0, PORT=8007 | ✅ PASS | `logs/glm52_w8a8_pp2b_vllm.log` | curl 全项通过，质量探针输出连贯 |

### 2026-07-22 结论：v0.23.0 原生支持

- v0.23.0 的 `deepseek_v2.py` 原生实现 `index_skip_topk_offset`/`index_topk_freq`，
  与 checkpoint 的 indexer 层模式（0,1,2,6,10,…）完全吻合，07-20 的 KeyError 消失。
  （v0.22.1 手工 patch 存档在 `container_patch/`，仅旧版需要。）
- TP=8 单节点在 A2 64G 上 OOM（~60.4GiB/卡）→ **TP=8 PP=2**（两节点 ~30GiB/卡）。
- PP>1 与 MTP 互斥（v0.23.0 明确拒绝），脚本新增 `ENABLE_MTP` 开关（默认关）。
- `CACHE_ROOT` 改到项目共享路径（worker 节点 `/dev/shm` 下 clang 编译失败）。
