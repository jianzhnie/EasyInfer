# EasyInfer 示例 README 模板

本文件为 `examples/<model>/vllm/README.md` 的统一模板。新增模型时，复制本模板并替换 `<占位符>`。

> **设计原则**（面向模板使用者）：
> - 按需保留/删除章节：非 MoE 模型删专家相关内容，非 MTP 模型删投机解码章节，纯文本模型删多模态章节
> - `<占位符>` 必须全部替换为实际值；`✅/⚠️/❌` 标记需根据验证结果更新
> - 环境变量表以 `glm5_2_w8a8` 为最完整参考；简化版以 `kimi_k2_7_code_w4a8` 为参考
> - 端口分配见 `prompts/example-scripts-template.md` 中的端口替换表

---

```
# <模型名> <量化> 部署指南

> **<框架版本>** | 端口: **<PORT>**
> 架构: <Arch> | <N> Experts | <MoE/MLA/Dense/...> | <MTP/无> | <量化> 量化
> 已验证配置: **<TP/PP/DP>** (<节点描述>) | 上下文: <MAX_LEN> | <关键特性>
> <如模型有已知问题/版本依赖/IP 限制等，在此行简要提示>
> 验证状态: <✅ PASS / ⚠️ 待验证 / 见文末「验证记录」>

<一句话描述模型定位，如"DeepSeek-V4-Flash W8A8 MTP..."。>

## 模型简介

| 属性 | 值 |
|------|-----|
| **架构** | <Arch> (<备注，如 MoE + DSA + MLA>) |
| **参数量** | <总参数量> (<激活参数量>) — 如无法确定则省略本行 |
| **路由专家** | <N> (每 Token 激活 <N> 专家) — 非 MoE 模型删除本行 |
| **隐藏维度** | <N> |
| **FFN 维度** | <N> / MoE FFN: <N> — 仅 dense 或混合结构保留 |
| **网络层数** | <N> |
| **注意力头** | <N> (GQA: <N> KV head) — 可选，有 GQA 时推荐填写 |
| **MLA** | kv_lora_rank=<N>, q_lora_rank=<N>, qk_head_dim=<N>, v_head_dim=<N> — 非 MLA 模型删除本行 |
| **Head Dim** | <N> — 非 MLA 模型使用 |
| **rope_theta** | <N> |
| **原生上下文** | **<max_position_embeddings>** |
| **量化方式** | <Quant> (<说明，如 "8-bit 权重 + 8-bit 激活">)，权重 ≈<N>G |
| **MTP** | num_nextn_predict_layers=<N>（默认<开/关>；<PP/特定条件限制>）— 无 MTP 填 ❌ 不支持 |
| **PP 支持** | ✅/❌ 支持 Pipeline Parallelism (<备注>) |
| **多模态** | ✅/❌ <Vision/Audio/...> (<N> 层) |
| **词表大小** | <N> |
| **工具调用解析器** | <parser> |
| **推理解析器** | <parser> / 不适用 |

### 架构注意事项

<关键兼容性说明，例如：
- FLASHCOMM1 必须为 0 及原因
- 必须使用特定 tool parser 及原因
- DSA CP 路径不兼容等
- 已知的量化路径缺陷（如 TP=16 乱码）
- Indexer/TopK 等特殊配置说明
>

### 官方文档参考

<如模型无特定 vLLM-Ascend 文档，可省略本节，仅保留 vLLM 官方文档。>

- <vLLM-Ascend 模型文档>: <url>
- vLLM 官方文档: https://docs.vllm.ai/en/stable/

### 硬件要求

<按需保留 A2/A3 或单节点/多节点小节。如模型支持多种硬件，推荐分小节列出。>

| 硬件 | 配置 | 推荐上下文 | 备注 |
|------|------|-----------|------|
| Atlas 800 A2 (64G × 8) | <Quant>, TP=<N> | <N>K | <备注> |
| Atlas 800 A3 (64G × 16) | <Quant>, TP=<N> | <N>K | <备注> |

**<硬件> 注意事项**:
- <关键约束，如 "A2 单节点 OOM，需 PP=2" 或 "A3 128G 单节点直接起">

<如模型有内存/权重占用分析，保留以下小节：>

### 内存分析（<Quant> @ <硬件>）

| 组件 | 消耗 (<配置>) | 说明 |
|------|-------------|------|
| 模型权重 | ≈<N> GiB | <备注> |
| KV Cache (<N>K ctx) | ≈<N> GiB | 随 max_model_len 变化 |
| 编译缓存 + 临时 | ≈<N> GiB | Triton/算子缓存 |
| **总计** | **≈<N> GiB** | **<结论>** |

## 快速开始

### 前置条件

模型路径: `<绝对路径>`

```bash
# 1. 启动 NPU Docker 容器
bash scripts/docker/manage_npuslim_containers.sh start --file node_list.txt

# 2. 启动 Ray 集群
bash scripts/ray_cluster/start_npuslim_ray_cluster.sh start --file node_list.txt
```

<快速确认 NPU 内存的提示，如：>

```bash
# 确认 NPU 内存（容器内执行）
npu-smi info | grep "HBM-Usage" | head -1
# 65536 MB = 64GB (A2) | 131072 MB = 128GB (A3)
```

### 部署

```bash
# 单节点 (<默认上下文>, TP=<N>)
bash examples/<model_dir>/vllm/run_vllm.sh

# 多节点 (<大上下文>)
TP=<TP> PP=<PP> MAX_MODEL_LEN=<LEN> bash examples/<model_dir>/vllm/run_vllm.sh

# 后台运行
nohup bash examples/<model_dir>/vllm/run_vllm.sh > <log_file>.log 2>&1 &

# <可选：传统包装器部署>
bash examples/<model_dir>/vllm/vllm_server.sh

# <可选：MTP 版本 / 无 MTP 版本 / 特殊配置>
ENABLE_MTP=1 bash examples/<model_dir>/vllm/run_vllm.sh
```

<如有多节点部署的特殊要求，添加小节：>

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

<如有量化文件修复/版本升级注意事项等，添加为可选小节。>

### 验证

```bash
# 运行测试脚本
bash examples/<model_dir>/vllm/curl_test.sh

# 手动验证
curl http://localhost:<PORT>/v1/models
curl http://localhost:<PORT>/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"<api-name>","messages":[{"role":"user","content":"Hello"}],"max_tokens":50}'
```

## 并行策略

| 场景 | TP | PP | DP | NPU | 上下文 | 量化 | 状态 |
|------|-----|-----|-----|-----|--------|------|------|
| 单节点 | <TP> | 1 | 1 | <N> | <N>K | <Quant> | ✅/⚠️/❌ |
| 多节点 | <TP> | <PP> | <DP> | <N> | <N>K | <Quant> | ✅/⚠️/❌ |

> <模型特定约束说明，例如"不支持 PP，多节点必须使用大 TP"或"PP>1 与 MTP 互斥"等。>


## 常见问题

### Q: <问题 1>?

A: <回答>

### Q: <与同类模型的部署配置有什么不同>?

A: <对比说明>

### Q: 为什么必须设置 FLASHCOMM1=<0/1>?

A: <根因说明>

### Q: <并行/量化对内存有什么影响>?

A: <说明>

### Q: 为什么 TP 默认是 <N> 而不是 <N>?

A: <硬件/量化约束说明>

### Q: 为什么不能用 PP / 为什么 PP 与 MTP 互斥?

A: <架构/版本限制说明>

<根据模型特性增减以下 FAQ 条目：>

### Q: MTP 投机解码对内存有什么影响?

A: MTP 加载第二份模型权重，减少 KV cache 可用空间。

### Q: 多节点网络如何配置?

A: 设置 `NIC_NAME` 和 `HCCL_IF_IP` 环境变量绑定高速网卡：
```bash
NIC_NAME=enp66s0f0 HCCL_IF_IP=10.42.11.130 RAY_ADDRESS=10.42.11.130:6379 PP=2 bash run_vllm.sh
```

### Q: 部署时一直卡在 "Waiting for creating a placement group"?

A: Engine Core 子进程未连接到集群 Ray。设置 `RAY_ADDRESS` 环境变量指向 Ray Head 节点。

### Q: 启动时报 "failed to map segment from shared object" 错误?

A: 编译缓存损坏。清理缓存后重启：
```bash
docker exec vllm-ascend-env bash -c 'rm -rf <CACHE_ROOT>/*'
```

### Q: <专家数>对部署有什么影响?

A: EP_SIZE 需能整除 <N> (推荐 <值>)。<N> 专家的 all-to-all 通信开销<较大/较小>。

### Q: W8A8 和 W4A8 有什么区别?

A: W8A8 精度更高但显存占用更大 (约 2× W4A8)；W4A8 省显存但精度略低。

## 验证记录

| 时间 | 镜像 | 节点 | 配置 | 结果 | 日志 | 说明 |
|------|------|------|------|------|------|------|
| <YYYY-MM-DD> | `<镜像名>` | <节点描述> | <配置> | ✅/⚠️/❌ | `<日志路径>` | <说明> |

<多轮验证时保留多行。有重要结论/根因分析时，在表格后追加小节。>

### <YYYY-MM-DD 结论：一句话总结>

<关键发现、根因分析、版本差异等。>
```