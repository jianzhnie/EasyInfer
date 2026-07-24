# LongCat-Flash-Chat-1024E-512Zero-Topk24-v2 BF16 部署指南

> **vLLM-Ascend v0.23.0rc1** | 端口: **8010**
> 架构: LongcatFlashForCausalLM | 1024 Routed + 512 Zero Experts | MoE + MLA
> 已验证配置: **TP=64 PP=1** (8 节点 × 8 NPU) | 上下文: 4096 | BF16 无量化
> 注意: 需要 EasyInfer 插件注册 EP 修复；MC2 MoE comm 与 Zero Expert 权重置零不兼容
> 验证状态: ⚠️ 待验证

超大规模 MoE 模型（1024 专家，TopK=24），最小需要 64 张 NPU 部署。

## 模型简介

| 属性 | 值 |
|------|-----|
| **架构** | LongcatFlashForCausalLM (MLA + MoE) |
| **路由专家** | 1024 (每 Token 激活 24) |
| **Zero 专家** | 512 (Identity) |
| **隐藏维度** | 6144 |
| **网络层数** | 28 |
| **KV LoRA Rank** | 512 |
| **rope_theta** | — |
| **原生上下文** | **131072** |
| **量化方式** | BF16 (无量化)，权重 ≈296G (148 个 safetensors 分片) |
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
- 模型包含 512 个 Zero (Identity) 专家，vLLM ≥ 0.23 下启用原生零号专家路径（`fix_ep_zero_expert.py`）

### 官方文档参考

- vLLM-Ascend 文档: https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/index.html

### 硬件要求

| 硬件 | 配置 | 推荐上下文 | 备注 |
|------|------|-----------|------|
| Atlas 800 A2/A3 (64G × 64) | BF16, TP=64 | 4K | 8 节点 × 8 卡最小配置 |

## 快速开始

### 前置条件

模型路径: `/home/jianzhnie/llmtuner/hfhub/models/meituan-longcat/LongCat-Flash-Chat`

```bash
# 1. 启动 NPU Docker 容器
bash scripts/docker/manage_npuslim_containers.sh start --file node_list.txt

# 2. 启动 Ray 集群
bash scripts/ray_cluster/start_npuslim_ray_cluster.sh start --file node_list.txt
```

### 部署

```bash
# 标准模式 (TP=64, 8 节点)
bash examples/longcat/vllm/run_vllm.sh

# EP 模式 (专家并行)
EP=1 bash examples/longcat/vllm/run_vllm.sh

# 自定义上下文
TP=64 MAX_MODEL_LEN=8192 MAX_NUM_SEQS=64 bash examples/longcat/vllm/run_vllm.sh
```

> 注意: 不要从 EasyInfer 根目录运行，避免插件冲突。在容器内切换到一个非 EasyInfer 目录后执行。

### 验证

```bash
bash examples/longcat/vllm/curl_test.sh

# 手动验证
curl -s http://localhost:8010/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"longcat-flash","messages":[{"role":"user","content":"你好"}],"max_tokens":50}' \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'])"
```

## 并行策略

| 场景 | TP | PP | DP | NPU | 上下文 | 量化 | 状态 |
|------|-----|-----|-----|-----|--------|------|------|
| 标准 | 64 | 1 | 1 | 64 | 4K | BF16 | ⚠️ |
| EP | 64 | 1 | 1 | 64 | 4K | BF16 | ⚠️ |

> EP=1 模式下使用 ALLGATHER comm 避免 MC2 冲突。模型加载约需 16-20 分钟。

## 关键配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| TP | 64 | 张量并行度 (8 节点 × 8 卡) |
| PP | 1 | 流水线并行度 |
| PORT | 8010 | 服务端口 |
| MAX_MODEL_LEN | 4096 | 最大序列长度 |
| MAX_NUM_SEQS | 128 | 最大并发序列数 |
| GPU_MEM_UTIL | 0.90 | 显存利用率 |
| BLOCK_SIZE | 128 | MLA 注意力 block size |
| HCCL_BUFFSIZE | 800 | HCCL 缓冲区大小 |

## 常见问题

### Q: 为什么不能使用 Chunked Prefill?

A: Chunked Prefill 与 EP token dispatch 存在冲突，默认禁用 (`CHUNKED_PREFILL=0`)。

### Q: EP 模式为什么需要覆盖 MoE Comm?

A: MC2 MoE comm 在处理 Zero Expert 权重置零时触发 MoeDistributeCombineV2 shape check 失败，导致 collective hang。EasyInfer 插件将 comm 覆盖为 ALLGATHER 规避此问题。

### Q: 为什么模型加载需要 16-20 分钟?

A: 模型包含 148 个 safetensors 分片 + 64 卡 HCCL 初始化，加载时间较长。

### Q: 部署时为什么提示 "failed to map segment from shared object"?

A: 编译缓存损坏。清理缓存后重启：
```bash
docker exec vllm-ascend-env bash -c 'rm -rf /root/.cache/vllm/*'
```

## 验证记录

| 时间 | 镜像 | 节点 | 配置 | 结果 | 日志 | 说明 |
|------|------|------|------|------|------|------|
| — | — | — | — | ⚠️ | — | 待验证 |
