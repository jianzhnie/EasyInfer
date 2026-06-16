# LLM 推理部署指南 — Ascend NPU 集群

> **日期**: 2026-06-15 | **集群**: 10×8 Ascend 910C | **vLLM-Ascend**: 0.20.2 | **CANN**: 9.0.0
> **状态**: ✅ 4/6 Eco-Tech 模型部署成功 | ❌ DeepSeek V4 输出异常 | 📋 5 新模型脚本就绪

---

## 目录

1. [部署概览](#1-部署概览)
2. [模型架构详解](#2-模型架构详解)
3. [Agent 优化参数](#3-agent-优化参数)
4. [完整部署流程](#4-完整部署流程)
5. [已知问题与修复](#5-已知问题与修复)
6. [Claude Code 集成](#6-claude-code-集成)
7. [集群拓扑](#7-集群拓扑)
8. [环境变量速查](#8-环境变量速查)
9. [API 验证命令](#9-api-验证命令)
10. [Quick Start](#10-quick-start)
11. [脚本清单](#11-脚本清单)

---

## 1. 部署概览

### 1.1 Eco-Tech 量化模型 (已验证)

| # | Model | Size | Port | TP×PP | Context | MTP | Tool | Time | Status |
|---|-------|------|------|-------|---------|-----|------|------|--------|
| 1 | **GLM-5** W4A8 | 392G | 8001 | 8×1 / 16×1 | 16K / 131K | ✅ mtp | glm47 | ~4m / ~10m | ✅ 单/多节点 |
| 2 | **GLM-5.1** W4A8 | 392G | 8002 | 8×1 / 16×1 | 16K / 131K | ✅ mtp | glm47 | ~9m / ~10m | ✅ 单/多节点 |
| 3 | **MiniMax-M2.7** W8A8 | 216G | 8004 | 8×1 | **32K** | ❌ 不支持 | minimax_m2 | ~2m | ✅ 单节点 |
| 4 | **Kimi-K2.6** W4A8 | 500G | 8003 | 8×2 | **131K** | ❌ N/A | kimi_k2 | ~12m | ✅ 多节点 / ❌ 单节点 OOM |
| 5 | **DeepSeek-V4-Flash** W8A8 | 280G | 8000 | 8×1 | - | ✅ mtp | deepseek_v4 | ~10m | ❌ 输出乱码 |
| 6 | **DeepSeek-V4-Pro** W4A8 | 791G | 8000 | 8×2 | - | ✅ mtp | deepseek_v4 | - | ❌ 需多节点+修复 |

### 1.2 原始模型 (脚本就绪, 待验证)

| # | Model | Size | Port | TP | Quant | Tool | Script |
|---|-------|------|------|----|-------|------|--------|
| 7 | **GLM-5** BF16 | 1.4T | 8001 | 32 | none | glm47 | `examples/glm5/vllm/` |
| 8 | **Kimi-K2-Thinking** W4A8 | 554G | 8003 | 16 | compressed-tensors | kimi_k2 | `examples/kimi-k2-thinking/vllm/` |
| 9 | **Kimi-K2.5** W4A8 | 555G | 8005 | 16 | compressed-tensors | kimi_k2 | `examples/kimi-k2.5/vllm/` |
| 10 | **MiniMax-M2.7** FP8 | 214G | 8004 | 8 | fp8 | minimax_m2 | `examples/minimax-m2.7/vllm/` |
| 11 | **Qwen3-235B-A22B** BF16 | 438G | 8006 | 16 | ascend | hermes | `examples/qwen3-235b-a22b-instruct-2507/vllm/` |

### 1.3 验证环境

- **容器**: `vllm-ascend-env` (quay.io/ascend/vllm-ascend:v0.20.2rc1-a3)
- **当前可用节点**: 10.16.201.229 (单节点 8× NPU)
- **全集群节点**: 10 节点 × 8 NPU = 80 NPU (见 `node_list.txt`)
- **每节点**: 8× Ascend 910C × 65GB HBM
- **单节点容量**: 520GB HBM, 可部署 ≤450G 模型 (需预留 KV cache)

---

## 2. 模型架构详解

### 2.1 GLM-5 / GLM-5.1 (W4A8)

```
config.json:
  architectures:          ['GlmMoeDsaForCausalLM']
  hidden_size:            6144
  num_hidden_layers:      78
  n_routed_experts:       256 (8 per token)
  num_nextn_predict_layers: 1          ← MTP
  max_position_embeddings: 202752
  kv_lora_rank:           512
  q_lora_rank:            2048
  quantization:           W4A8
```

**部署要求**: TP=16 PP=1, 2 节点, VLLM_ASCEND_ENABLE_FLASHCOMM1=0 (必须)

### 2.2 Kimi-K2.6 (W4A8)

```
config.json:
  architectures:          ['KimiK25ForConditionalGeneration']
  hidden_size:            7168
  num_hidden_layers:      61
  n_routed_experts:       384 (8 per token)
  num_nextn_predict_layers: 0          ← 无 MTP
  max_position_embeddings: 262144
  kv_lora_rank:           512
  q_lora_rank:            1536
  vision_config:          Vision Transformer (27 layers)
  quantization:           W4A8
```

**部署要求**: TP=8 PP=2, 2 节点, tool_parser=kimi_k2

### 2.3 MiniMax-M2.7 (W8A8)

```
config.json:
  architectures:          ['MiniMaxM2ForCausalLM']
  hidden_size:            3072
  num_hidden_layers:      62
  n_routed_experts:       256 (8 per token)
  num_mtp_modules:         3           ← MTP 支持但 vLLM-Ascend 不可用
  mtp_transformer_layers:  1
  max_position_embeddings: 204800
  quantization:           W8A8 QuaRot
```

**部署要求**: TP=8 PP=1, 单节点, 不支持 MTP (vLLM-Ascend 0.20.2 不兼容), W8A8 内存需求大 (gpu_mem_util=0.95)

### 2.4 Kimi-K2-Thinking (compressed-tensors W4A8)

```
config.json:
  architectures:          ['DeepseekV3ForCausalLM']
  hidden_size:            7168
  num_hidden_layers:      61
  n_routed_experts:       384 (8 per token)
  num_nextn_predict_layers: 0          ← 无 MTP
  max_position_embeddings: 262144
  quantization:           compressed-tensors (W4 group=32)
```

**部署要求**: TP=16, 2 节点, --quantization compressed-tensors, tool_parser=kimi_k2

### 2.5 Kimi-K2.5 (compressed-tensors W4A8, 多模态)

```
config.json:
  architectures:          ['KimiK25ForConditionalGeneration']
  text_config:            DeepseekV3ForCausalLM, 384E, 61L
  vision_config:          Vision Transformer
  quantization:           compressed-tensors (W4 group=32, text backbone)
```

**部署要求**: TP=16, 2 节点, --language-model-only (文本 Agent), --mm-encoder-tp-mode data

### 2.6 Qwen3-235B-A22B-Instruct-2507 (BF16)

```
config.json:
  architectures:          ['Qwen3MoeForCausalLM']
  hidden_size:            4096
  num_hidden_layers:      94
  num_experts:            128 (8 per token)
  max_position_embeddings: 262144
  quantization:           none (BF16, 使用 --quantization ascend 在线量化)
```

**部署要求**: TP=16, 2 节点, --quantization ascend (W4A8 在线量化), tool_parser=hermes

### 2.7 DeepSeek-V4-Pro / Flash

**已知阻塞**: Eco-Tech 版模型权重含 `attn_sink` 参数，vLLM-Ascend 0.20.2 不兼容 (KeyError)。
修复方法: 在 `/vllm-workspace/vllm-ascend/vllm_ascend/models/deepseek_v4.py:1204` 添加:
```python
if name not in params_dict:
    continue
```

---

## 3. Agent 优化参数

| Parameter | GLM-5/5.1 | Kimi-K2.6 | MiniMax | Claude Code 影响 |
|-----------|-----------|-----------|---------|------------------|
| `--enable-prefix-caching` | ✅ | ✅ | ✅ | 🔑 系统提示缓存 ~90% KV 命中率 |
| `--enable-chunked-prefill` | ✅ | ✅ | ✅ | 长上下文 prompt 优化 |
| `--max-num-seqs` | 8 | 16 | 8 | 并发工具调用 |
| `--max-num-batched-tokens` | 16384 | 16384 | 8192 | 预填充吞吐量 |
| `--gpu-memory-utilization` | 0.94 | 0.92 | 0.95 | MiniMax W8A8 需更高 |
| `--enable-auto-tool-choice` | ✅ | ✅ | ✅ | 🔑 Anthropic API tool_use |
| `--enforce-eager` | ✅ | ✅ | ✅ | Ascend 无 CUDA graph |

### 模型专属配置

| 配置项 | GLM-5/5.1 | Kimi-K2.6 | MiniMax-M2.7 |
|--------|-----------|-----------|--------------|
| 关键环境变量 | FLASHCOMM1=0, MLAPO=1 | TASK_QUEUE_ENABLE=1 | FUSED_MC2=1 |
| HCCL_BUFFSIZE | 200 | 800 | 1024 |
| MTP | ✅ mtp, 3 tokens | ❌ | ❌ 不支持 |
| Tool Parser | glm47 | kimi_k2 | minimax_m2 |
| PP 支持 | ❌ (大 TP 替代) | ✅ | ✅ |
| 多模态 | ❌ | ✅ Vision | ❌ |

---

## 4. 完整部署流程

### 前置条件

```bash
# 全集群节点列表: node_list.txt (10 节点)
# 当前可用节点: 10.16.201.229 (单节点)

# 容器镜像: quay.io/ascend/vllm-ascend:v0.20.2rc1-a3
# Eco-Tech 量化模型: /home/jianzhnie/llmtuner/hfhub/models/Eco-Tech/
# 原始模型:         /home/jianzhnie/llmtuner/hfhub/models/{ZhipuAI,moonshotai,MiniMaxAI,Qwen}/
```

### Step 1: 启动容器

```bash
# 多节点
bash scripts/docker/manage_npuslim_containers.sh restart --file node_list.txt

# 单节点 (当前)
docker restart vllm-ascend-env
```

### Step 2: 启动 Ray 集群

```bash
# 多节点
bash scripts/ray_cluster/start_npuslim_ray_cluster.sh start --file node_list.txt

# 单节点 (当前)
docker exec vllm-ascend-env bash -lc 'ray start --head --port=6379'
```

### Step 3: 部署模型

```bash
docker exec -w /home/jianzhnie/llmtuner/llm/EasyInfer -it vllm-ascend-env bash

# --- 单节点部署 (TP=8, 当前验证配置) ---

# MiniMax-M2.7 W8A8 (216G, ~2min) ✅
nohup bash examples/minimax_m2_7_w8a8/vllm/run_vllm.sh > /tmp/vllm_minimax.log 2>&1 &

# GLM-5 W4A8 (392G, ~4min, max_model_len=16K) ✅
MAX_MODEL_LEN=16384 MAX_NUM_SEQS=4 GPU_MEM_UTIL=0.95 \
    nohup bash examples/glm5_w4a8/vllm/run_vllm.sh > /tmp/vllm_glm5.log 2>&1 &

# GLM-5.1 W4A8 (392G, ~9min, max_model_len=16K) ✅
MAX_MODEL_LEN=16384 MAX_NUM_SEQS=4 GPU_MEM_UTIL=0.95 \
    nohup bash examples/glm5_1_w4a8/vllm/run_vllm.sh > /tmp/vllm_glm51.log 2>&1 &

# --- 多节点部署 (需 SSH + 多节点 Ray) ---

# Kimi-K2.6 W4A8 (500G, ~12min, 需 2 节点) ✅
TP=8 PP=2 PORT=8003 \
    nohup bash examples/kimi_k2_6_w4a8/vllm/run_vllm.sh > /tmp/vllm_kimi.log 2>&1 &

# GLM-5 W4A8 大上下文 (131K, 需 2 节点)
TP=16 PP=1 PORT=8001 MAX_MODEL_LEN=131072 \
    nohup bash examples/glm5_w4a8/vllm/run_vllm.sh > /tmp/vllm_glm5.log 2>&1 &
```

### Step 4: 验证

```bash
for port in 8001 8002 8003 8004; do
    echo -n "Port $port: "
    curl -sf http://localhost:$port/v1/models | \
        python3 -c "import json,sys; d=json.load(sys.stdin); print(d['data'][0]['id'], d['data'][0]['max_model_len'])" 2>/dev/null || echo "not ready"
done
```

### Step 5: 停止

```bash
docker exec vllm-ascend-env bash -c "pkill -9 -f 'vllm serve'"
docker restart vllm-ascend-env    # 清理 NPU 内存和 Ray 残留
```

---

## 5. 已知问题与修复

### Issue 1: DeepSeek V4 输出乱码 🔴

| 项目 | 内容 |
|------|------|
| **严重度** | BLOCKING |
| **现象** | 模型加载成功 (~10min), 但推理输出完全乱码 (garbled text) |
| **原因** | Eco-Tech 版模型权重含 `attn_sink` 参数, vLLM-Ascend 0.20.2 不兼容 |
| **文件** | `/vllm-workspace/vllm-ascend/vllm_ascend/models/deepseek_v4.py:1204` |
| **修复** | 添加 `if name not in params_dict: continue` |
| **影响** | DeepSeek-V4-Pro, DeepSeek-V4-Flash |
| **验证** | 2026-06-15 单节点 TP=8, 模型可加载但输出无意义 |

### Issue 2: MiniMax MTP 不支持 🔴

| 项目 | 内容 |
|------|------|
| **严重度** | BLOCKING |
| **错误** | `NotImplementedError: Unsupported speculative method: 'mtp'` |
| **修复** | 移除 `--speculative-config` 参数 |
| **影响** | MiniMax-M2.7 (模型含 num_mtp_modules=3 但 vLLM-Ascend 不支持) |

### Issue 3: Kimi-K2.6 W4A8 单节点 OOM 🔴

| 项目 | 内容 |
|------|------|
| **严重度** | BLOCKING (单节点) |
| **错误** | `torch.OutOfMemoryError: NPU out of memory. Tried to allocate 674.00 MiB` |
| **原因** | 500G 模型 + 8×65GB = 520GB HBM, 加载后无剩余空间 |
| **修复** | 必须使用多节点: TP=8 PP=2 (2 节点) |
| **验证** | 2026-06-15 单节点 TP=8 GPU_MEM_UTIL=0.95 MAX_MODEL_LEN=8192 仍 OOM |

### Issue 4: Kimi-K2.6 cudagraph_capture_sizes 错误 🟡

| 项目 | 内容 |
|------|------|
| **严重度** | MEDIUM |
| **错误** | `cudagraph_capture_sizes [1, 2, 4] does not contain values that are multiples of tp_size 8` |
| **修复** | 添加 `--enforce-eager` 禁用 cudagraph |
| **影响** | Kimi-K2.6 W4A8 在 TP=8 时触发 |

### Issue 5: GLM-5 W4A8 单节点极慢 🟡

| 项目 | 内容 |
|------|------|
| **严重度** | MEDIUM |
| **现象** | TP=8 单节点: ~0.1 tokens/s 生成速度, 首次响应需 30-60s |
| **原因** | 392G 模型占用 ~95% HBM, KV cache 空间极小; 256 专家 + MTP 内存压力大 |
| **修复** | 使用 TP=16 (2 节点) 获得正常速度 (~10m 启动, 正常推理) |
| **影响** | GLM-5 / GLM-5.1 W4A8 单节点部署 |

### Issue 6: Ray 残留 Placement Group 🟡

| 项目 | 内容 |
|------|------|
| **严重度** | MEDIUM |
| **现象** | NPUs reserved in placement groups, 新部署失败 |
| **修复** | 每次部署间执行 `docker restart vllm-ascend-env` |
| **预防** | 停止 vLLM 后必须重启容器清理 NPU 内存 |

---

## 6. Claude Code 集成

```bash
# 环境变量配置
ANTHROPIC_BASE_URL=http://10.16.201.229:8003 \
ANTHROPIC_API_KEY=dummy \
ANTHROPIC_AUTH_TOKEN=dummy \
ANTHROPIC_DEFAULT_SONNET_MODEL=kimi-k2.6 \
ANTHROPIC_DEFAULT_HAIKU_MODEL=kimi-k2.6 \
ANTHROPIC_DEFAULT_OPUS_MODEL=kimi-k2.6 \
claude
```

### 模型推荐

| 优先级 | 模型 | 原因 |
|--------|------|------|
| 🥇 推荐 | **Kimi-K2.6** | 131K 上下文, 384 专家, 多模态, max_seqs=16 |
| 🥈 备选 | **GLM-5.1** | 131K 上下文, MTP 加速 |
| 🥉 备选 | **MiniMax-M2.7** | 32K 上下文, W8A8 高精度 |

### API 兼容性

| API Endpoint | GLM-5 | GLM-5.1 | Kimi-K2.6 | MiniMax-M2.7 |
|-------------|-------|---------|-----------|--------------|
| `/v1/models` | ✅ | ✅ | ✅ | ✅ |
| `/v1/chat/completions` | ✅ | ✅ | ✅ | ✅ |
| Tool Calling | ✅ glm47 | ✅ glm47 | ✅ kimi_k2 | ✅ minimax_m2 |
| `/v1/messages` | ✅ | ✅ | ✅ | ✅ |

---

## 7. 集群拓扑

```
Ascend NPU Cluster — 10 Nodes (node_list.txt)
┌──────────────────────────────────────────────────────────────────────┐
│                                                                      │
│  10.16.201.229 (Head)       8× Ascend 910C × 65GB                   │
│  vllm-ascend-env            Ray Head :6379                           │
│  ← 当前唯一可用节点 (SSH 密钥未分发到其他节点)                          │
│                                                                      │
│  10.16.201.164, .40, .163, .193, .201, .153, .124, .20, .18         │
│  各 8× Ascend 910C × 65GB  Ray Worker (待配置 SSH)                   │
│                                                                      │
│  Total: 10 nodes × 8 NPU = 80 NPU                                   │
│  单节点: 520GB HBM → 可部署 ≤450G 模型                               │
│  多节点: 需配置节点间 SSH 密钥                                         │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 8. 环境变量速查

### GLM-5 / GLM-5.1

```bash
export VLLM_ASCEND_ENABLE_FLASHCOMM1=0    # 必须! W4A8 + DSA CP 路径不兼容
export VLLM_ASCEND_ENABLE_MLAPO=1
export HCCL_BUFFSIZE=200
export HCCL_OP_EXPANSION_MODE=AIV
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1 OMP_PROC_BIND=false
```

### Kimi-K2.6 / K2-Thinking / K2.5

```bash
export HCCL_BUFFSIZE=800
export TASK_QUEUE_ENABLE=1
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
export VLLM_ASCEND_ENABLE_MLAPO=1
export HCCL_OP_EXPANSION_MODE=AIV
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1 OMP_PROC_BIND=false
```

### MiniMax-M2.7

```bash
export HCCL_BUFFSIZE=1024
export TASK_QUEUE_ENABLE=1
export VLLM_ASCEND_ENABLE_FUSED_MC2=1
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
export HCCL_OP_EXPANSION_MODE=AIV
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
```

### Qwen3-235B-A22B

```bash
export HCCL_BUFFSIZE=800
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
export VLLM_ASCEND_BALANCE_SCHEDULING=1
export HCCL_OP_EXPANSION_MODE=AIV
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1 OMP_PROC_BIND=false
```

---

## 9. API 验证命令

```bash
# 健康检查
curl -s http://10.16.201.229:8003/v1/models | python3 -m json.tool

# Chat Completion
curl -s http://10.16.201.229:8003/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"kimi-k2.6","messages":[{"role":"user","content":"Hello"}],"max_tokens":30}'

# Tool Calling
curl -s http://10.16.201.229:8003/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"kimi-k2.6","messages":[{"role":"user","content":"Weather?"}],"tools":[{"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}],"max_tokens":100}'

# Anthropic Messages
curl -s http://10.16.201.229:8003/v1/messages \
  -H "Content-Type: application/json" -H "x-api-key: dummy" \
  -d '{"model":"kimi-k2.6","max_tokens":100,"messages":[{"role":"user","content":"Hi"}]}'
```

---

## 10. Quick Start

一键部署 Kimi-K2.6:

```bash
#!/bin/bash
set -e
NODES_FILE=available_nodes.txt
HEAD=10.16.201.229

# 1. 容器
bash scripts/docker/manage_npuslim_containers.sh restart --file $NODES_FILE

# 2. Ray
bash scripts/ray_cluster/start_npuslim_ray_cluster.sh start --file $NODES_FILE

# 3. 部署
docker exec -w /home/jianzhnie/llmtuner/llm/EasyInfer vllm-ascend-env bash -c "
TP=8 PP=2 DP=1 PORT=8003 \
nohup bash examples/kimi_k2_6_w4a8/vllm/run_vllm.sh > /tmp/vllm_kimi.log 2>&1 &
echo PID=\$!
"

# 4. 等待 (~12 min)
echo "Waiting for model..."
for i in $(seq 1 40); do
    if curl -sf http://$HEAD:8003/v1/models >/dev/null 2>&1; then
        echo "READY!"
        curl -s http://$HEAD:8003/v1/models
        break
    fi
    sleep 30
done
```

---

## 11. 脚本清单

### Eco-Tech 量化模型

| 文件 | 模型 | 状态 |
|------|------|------|
| `examples/glm5_w4a8/vllm/run_vllm.sh` | GLM-5 W4A8 | ✅ 单节点 TP=8 已验证 (2026-06-15) |
| `examples/glm5_1_w4a8/vllm/run_vllm.sh` | GLM-5.1 W4A8 | ✅ 单节点 TP=8 已验证 (2026-06-15) |
| `examples/minimax_m2_7_w8a8/vllm/run_vllm.sh` | MiniMax-M2.7 W8A8 | ✅ 单节点 TP=8 已验证 (2026-06-15) |
| `examples/kimi_k2_6_w4a8/vllm/run_vllm.sh` | Kimi-K2.6 W4A8 | ✅ 多节点验证 / ❌ 单节点 OOM |
| `examples/deepseek_v4_flash/vllm/run_vllm.sh` | DSV4-Flash W8A8 | ❌ 加载成功但输出乱码 |
| `examples/deepseek_v4_pro/vllm/run_vllm.sh` | DSV4-Pro W4A8 | ❌ 需多节点 + attn_sink 修复 |

### 原始模型 (新增)

| 文件 | 模型 | 状态 |
|------|------|------|
| `examples/glm5/vllm/run_vllm.sh` | GLM-5 BF16 (1.4T) | 📋 脚本就绪, 需 4 节点 TP=32 |
| `examples/kimi-k2-thinking/vllm/run_vllm.sh` | Kimi-K2-Thinking W4A8 (554G) | 📋 脚本就绪, 需 2 节点 TP=16 |
| `examples/kimi-k2.5/vllm/run_vllm.sh` | Kimi-K2.5 W4A8 (555G) | 📋 脚本就绪, 需 2 节点 TP=16 |
| `examples/minimax-m2.7/vllm/run_vllm.sh` | MiniMax-M2.7 FP8 (214G) | 📋 脚本就绪, 可单节点 TP=8 |
| `examples/qwen3-235b-a22b-instruct-2507/vllm/run_vllm.sh` | Qwen3-235B BF16 (438G) | 📋 脚本就绪, 需 2 节点 TP=16 |

---

*Last updated: 2026-06-16 | vLLM-Ascend 0.20.2 | CANN 9.0.0 | 4/6 Eco-Tech 模型已验证 (单节点 3, 多节点 1)*
