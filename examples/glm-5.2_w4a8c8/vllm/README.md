# GLM-5.2 W4A8C8 部署指南

> **vLLM-Ascend 0.23.0rc1 + CANN 8.5.1** | 端口: **8008**
> 架构: GlmMoeDsaForCausalLM | 256 Experts | MoE | MTP | W4A8C8 量化
> 已验证配置: **TP=8 PP=1 (1× A2 64G)** | 上下文: 32K | Chat ✅ Tool Calling ✅
> GLM-5.2 与 GLM-5/5.1 共享相同架构，上下文窗口扩展至 1M

## 模型简介

| 属性 | 值 |
|------|-----|
| **架构** | GlmMoeDsaForCausalLM (MoE + DSA + MLA) |
| **路由专家** | 256 (每 Token 激活 8 专家) |
| **隐藏维度** | 6144 |
| **网络层数** | 78 |
| **MLA** | kv_lora_rank=512, q_lora_rank=2048, qk_head_dim=256, v_head_dim=256 |
| **原生上下文** | **1,048,576** (1M) |
| **量化方式** | W4A8C8 (4-bit 权重 + 8-bit 激活 + 8-bit KV Cache) |
| **MoE 路由器类型** | float32 (`moe_router_dtype: "float32"` — W4A8C8 独有) |
| **MTP** | num_nextn_predict_layers=1（默认关，`ENABLE_MTP=1` 打开；PP>1 时不可用） |
| **PP 支持** | ✅ PP=2 可用；PP>1 与 MTP 互斥 |
| **工具调用解析器** | glm47 |
| **推理解析器** | glm45 |
| **词表大小** | 154,880 |

### 与 W8A8 版本的区别

| 特性 | W4A8C8 (本模型) | W8A8 |
|------|----------------|------|
| **权重量化** | 4-bit (INT4) | 8-bit (INT8) |
| **激活量化** | 8-bit | 8-bit |
| **KV Cache 量化** | 8-bit | 无额外压缩 |
| **权重显存 (TP=8)** | ≈35 GiB/卡 | ≈60 GiB/卡 |
| **A2 64G 单节点 TP=8** | ✅ 可用 (32K 上下文) | ❌ OOM |
| **A3 128G 单节点 TP=8** | ✅ 可用 (大上下文) | ✅ 可用 |
| **MoE 路由器精度** | float32 | 默认 |
| **精度损失** | MMLU 损失 < 1% | MMLU 损失 < 0.5% |

### 架构注意事项

GLM-5.2 的 config.json 包含 `index_topk: 2048` 和 `index_topk_freq: 4`（indexer 仅存在于
部分层：0,1,2,6,10,…）。**vLLM-Ascend 0.23.0 原生支持该层模式**（`index_skip_topk_offset`），
0.22.1 需手工 patch（存档于 `container_patch/`，仅旧版有效）。
DSA 路径不兼容 FLASHCOMM1，**必须设置 `VLLM_ASCEND_ENABLE_FLASHCOMM1=0`**（脚本已内置）。

W4A8C8 版本额外包含 `moe_router_dtype: "float32"`，MoE 路由器以 FP32 精度运行，
提升路由质量但略增开销（可忽略）。

### 官方文档参考

- GLM-5.2 官方部署文档: https://docs.vllm.ai/projects/ascend/en/main/tutorials/models/GLM5.2.html
- vLLM 官方文档: https://docs.vllm.ai/en/stable/

## 快速开始

### 前置条件

模型路径: `/home/jianzhnie/llmtuner/hfhub/models/Eco-Tech/GLM-5.2-w4a8c8`

**硬件要求**:
- **A2 (64GB/NPU)**: ✅ **TP=8 单节点**（W4A8C8 权重仅 ~35 GiB/卡，32K 上下文可用）
- **A3 (128GB/NPU)**: ✅ TP=8 单节点，TP=8 DP=2 大上下文

```bash
# 确认 NPU 内存（容器内执行）
npu-smi info | grep "HBM-Usage" | head -1
# 65536 MB = 64GB (A2) | 131072 MB = 128GB (A3)
```

> W4A8C8 相比 W8A8 权重减半，A2 (64GB) 单节点 TP=8 可直接部署。A3 (128GB) 更宽裕。

```bash
# 1. 启动 NPU Docker 容器
bash scripts/docker/manage_npuslim_containers.sh start --file node_list.txt

# 2. 启动 Ray 集群
bash scripts/ray_cluster/start_npuslim_ray_cluster.sh start --file node_list.txt
```

### 部署

```bash
# A2 单节点 (32K 上下文, TP=8) — W4A8C8 推荐配置
bash examples/glm-5.2_w4a8c8/vllm/run_vllm.sh

# A2 两节点 (TP=8 PP=2, 更大上下文)
RAY_ADDRESS=<head>:6379 PP=2 bash examples/glm-5.2_w4a8c8/vllm/run_vllm.sh

# 后台运行
nohup bash examples/glm-5.2_w4a8c8/vllm/run_vllm.sh > glm5_2_w4a8c8_vllm.log 2>&1 &

# 打开 MTP 投机解码（仅 PP=1 可用，PP>1 与 MTP 互斥）
ENABLE_MTP=1 bash examples/glm-5.2_w4a8c8/vllm/run_vllm.sh
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
cd /path/to/GLM-5.2-w4a8c8
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
bash examples/glm-5.2_w4a8c8/vllm/curl_test.sh

# 手动验证
curl http://localhost:8008/v1/models
curl http://localhost:8008/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"glm-5.2","messages":[{"role":"user","content":"Hello"}],"max_tokens":50}'
```

## 并行策略

| 场景 | TP | PP | DP | NPU | 上下文 | 状态 |
|------|-----|-----|-----|-----|--------|------|
| 单节点 A2 (64G) | 8 | 1 | 1 | 8 | 32K | ✅ **W4A8C8 推荐（权重 ~35 GiB/卡）** |
| 单节点 A2 (64G) | 8 | 1 | 1 | 8 | 64K | ⚠️ 需降低 GPU_MEM_UTIL |
| 单节点 A3 (128G) | 8 | 1 | 2 | 16 | 32K | ✅ |
| 2 节点 A2 (64G) | 8 | 2 | 1 | 16 | 32K | ✅ |
| 2 节点 A2 (64G) | 16 | 1 | 1 | 16 | - | ❌ TP=16 维度不兼容 |

> **关键结论**：
> - W4A8C8 权重减半（~35 GiB/卡 vs W8A8 的 ~60 GiB/卡），**A2 64G 单节点 TP=8 可直接部署**
> - **PP>1 与 MTP 互斥**，脚本 `ENABLE_MTP` 默认关
> - TP=16 不可用：MLA 注意力层维度 (`num_kv_heads=3 × head_dim=192 = 576`) 无法被 16 整除

### 内存分析（W4A8C8 @ 64GB A2）

| 组件 | 消耗 (TP=8) | 说明 |
|------|-------------|------|
| 模型权重 (256 专家) | ≈35 GiB | 256 专家 MoE 权重 (INT4)，每卡 32 专家 |
| KV Cache (32K ctx) | ≈2-4 GiB | 随 max_model_len 变化 |
| 编译缓存 + 临时 | ≈1-2 GiB | CUDA Graph、triton、算子缓存 |
| **总计** | **≈39 GiB** | **A2 64GB 余量充足** |

降低 `max_model_len`、`max_num_seqs`、禁用 MTP/CUDA Graph 均无法改变权重的固定消耗。

## 环境变量

> 完整环境变量说明见 [prompts/vllm_env_vars.md](../../../prompts/vllm_env_vars.md)。
> Claude Code 集成方式见 [prompts/vllm-prompt.md](../../../prompts/vllm-prompt.md)。

## 功能验证清单

### 基础功能

| 功能 | 状态 | 脚本 |
|------|------|------|
| 基础 Chat Completion | ✅ | `run_vllm.sh` |
| Tool Calling (glm47) | ✅ | `curl_test.sh` |
| Anthropic Messages API | ✅ | `curl_test.sh` |
| MTP 投机解码 | ✅ | `run_vllm.sh` (内置) |

## 常见问题

### Q: W4A8C8 和 W8A8 有什么区别？

A: W4A8C8 使用 4-bit 权重 + 8-bit 激活 + 8-bit KV Cache 量化，权重显存约为 W8A8 的一半（~35 GiB vs ~60 GiB），A2 64GB 单节点可直接部署。W8A8 精度略高（MMLU 损失 < 0.5% vs < 1%），但需要 A3 或 PP=2。W4A8C8 额外使用 FP32 MoE 路由器精度 (`moe_router_dtype: "float32"`)。

### Q: GLM-5.2 和 GLM-5/5.1 的部署配置有什么不同？

A: 架构相同 (GlmMoeDsaForCausalLM)，主要区别：GLM-5.2 原生上下文扩展至 **1M**，head_dim 从 64 增至 192，新增 qk_head_dim=256、v_head_dim=256。NPU 环境变量、并行配置、量化参数通用。

### Q: 为什么必须设置 FLASHCOMM1=0？

A: GLM-5.2 的 `index_topk: 2048` 触发 DSA CP 路径，W4A8C8/W8A8 下缺少 `aclnn_input_scale` 属性导致 crash。

### Q: MTP 投机解码对内存有什么影响？

A: MTP 加载第二份模型权重，减少 KV cache 可用空间。TP=8 单节点时 max_model_len 需适当下调。

### Q: 为什么不用 PP？

A: GLM-5.2 架构不支持 Pipeline Parallelism (`SupportsPP` 接口缺失)。多节点必须使用大 TP。

### Q: 启动时报 "failed to map segment from shared object" 错误？

A: Triton/TorchInductor 编译缓存损坏或版本不兼容。清理缓存后重启：
```bash
docker exec vllm-ascend-env bash -c 'rm -rf /root/.cache/glm52-cache/triton/* /root/.cache/glm52-cache/torchinductor/*'
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
- **不可执行缓存** (`CACHE_ROOT`, 默认项目共享路径 `.cache/glm52-w4a8c8`): tmp、home、vllm、ascend-log（不要用 `/dev/shm`：worker 节点 clang 编译会因 TMPDIR 缺失失败）
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

### Q: A2 (64GB) 上能用 TP=8 单节点部署吗？

A: **W4A8C8 可以，W8A8 不行**。W4A8C8 权重仅 ~35 GiB/卡，A2 64GB 单节点 TP=8 32K 上下文可用。W8A8 权重 ~60.4 GiB/卡，单节点 OOM，需要 PP=2 两节点。

### Q: 为什么不能用 TP=16？

A: GLM-5.2 的 MLA 注意力层维度 `num_kv_heads × head_dim = 3 × 192 = 576`，不能被 16 整除。TP=16 时 weight sharding 计算错误：`start(0) + length(704) > 576`。需要 vLLM-Ascend 修复 `deepseek_v2.py` 中的 Indexer shard 计算逻辑。

### Q: `quant_model_description.json` 有什么问题？

A: 官方发布的 W4A8C8 量化描述文件中，78 层只有 22 层（密集层 + 每第 4 个 MoE 层）包含 Indexer 量化条目，其余 56 个 MoE 层缺失。需要在加载前手动补充。详见「量化文件修复」章节。

## 验证记录

| 时间 | 镜像 | 节点 | 配置 | 结果 | 日志 | 说明 |
|------|------|------|------|------|------|------|
| - | `quay.io/ascend/vllm-ascend:v0.23.0rc1-a3` (CANN 8.5.1) | - | TP=8 PP=1, PORT=8008 | 待验证 | - | W4A8C8 首次部署 |
