# LLM 推理部署指南 — Ascend NPU 集群

> **日期**: 2026-06-11 | **集群**: 4×8 Ascend 910C (229,164,40,193) | **vLLM-Ascend**: 0.20.2 | **CANN**: 9.0.0
> **状态**: ✅ 4/6 模型部署成功 | ❌ 2 DeepSeek V4 模型需修复 vLLM 代码

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

| # | Model | Arch | Port | TP×PP | Context | MTP | Tool | Time | Status |
|---|-------|------|------|-------|---------|-----|------|------|--------|
| 1 | **DeepSeek-V4-Pro** W4A8 | DeepseekV4 / 384E / MLA | 8000 | 8×2 | - | ✅ | deepseek_v4 | - | ❌ attn_sink KeyError |
| 2 | **DeepSeek-V4-Flash** W8A8 | DeepseekV4 / 256E / MLA | 8000 | - | - | ✅ | deepseek_v4 | - | ❌ 同 #1 |
| 3 | **GLM-5** W4A8 | GlmMoeDSA / 256E / MLA | 8001 | 16×1 | **131K** | ✅ mtp | glm47 | ~10m | ✅ 通过 |
| 4 | **GLM-5.1** W4A8 | GlmMoeDSA / 256E / MLA | 8002 | 16×1 | **131K** | ✅ mtp | glm47 | ~10m | ✅ 通过 |
| 5 | **Kimi-K2.6** W4A8 | KimiK25 / 384E / MLA | 8003 | 8×2 | **131K** | ❌ N/A | kimi_k2 | ~12m | ✅ 通过 |
| 6 | **MiniMax-M2.7** W8A8 | MiniMaxM2 / 256E | 8004 | 8×2 | **65K** | ❌ 不支持 | minimax_m2 | ~5m | ✅ 通过 |

**已验证环境**:
- **容器**: `vllm-ascend-env` (quay.io/ascend/vllm-ascend:v0.20.2rc1-a3)
- **可用节点**: 10.16.201.229, 10.16.201.164, 10.16.201.40, 10.16.201.193
- **每节点**: 8× Ascend 910C × 66GB NPU
- **总容量**: 32 NPU (4 节点 × 8), 支持最多 2 模型同时部署

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

**部署要求**: TP=8 PP=2, 2 节点, 不支持 MTP (vLLM-Ascend 0.20.2 不兼容), W8A8 内存需求大

### 2.4 DeepSeek-V4-Pro / Flash

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
| 关键环境变量 | FLASHCOMM1=0, MLAPO=1 | TASK_QUEUE=1 | FUSED_MC2=1 |
| HCCL_BUFFSIZE | 200 | 800 | 1024 |
| MTP | ✅ mtp, 3 tokens | ❌ | ❌ 不支持 |
| Tool Parser | glm47 | kimi_k2 | minimax_m2 |
| PP 支持 | ❌ (大 TP 替代) | ✅ | ✅ |
| 多模态 | ❌ | ✅ Vision | ❌ |

---

## 4. 完整部署流程

### 前置条件

```bash
# 可用节点 (2026-06-11)
# 10.16.201.229, 10.16.201.164, 10.16.201.40, 10.16.201.193

# 容器镜像: quay.io/ascend/vllm-ascend:v0.20.2rc1-a3
# 模型路径: /home/jianzhnie/llmtuner/hfhub/models/Eco-Tech/
```

### Step 1: 启动容器

```bash
bash scripts/docker/manage_npuslim_containers.sh restart \
    --file available_nodes.txt
```

### Step 2: 启动 Ray 集群

```bash
bash scripts/ray_cluster/start_npuslim_ray_cluster.sh start \
    --file available_nodes.txt
```

### Step 3: 部署模型

```bash
# 进入容器, 部署目标模型
docker exec -w /home/jianzhnie/llmtuner/llm/EasyInfer -it vllm-ascend-env bash

# GLM-5 (Port 8001, 2 节点, 131K)
MODEL_PATH=/path/GLM-5-w4a8 TP=16 PP=1 PORT=8001 \
    nohup bash examples/glm5_w4a8/vllm/run_vllm.sh > /tmp/vllm_glm5.log 2>&1 &

# GLM-5.1 (Port 8002, 2 节点, 131K)
TP=16 PP=1 PORT=8002 \
    nohup bash examples/glm5_1_w4a8/vllm/run_vllm.sh > /tmp/vllm_glm51.log 2>&1 &

# Kimi-K2.6 (Port 8003, 2 节点, 131K)
TP=8 PP=2 DP=1 PORT=8003 \
    nohup bash examples/kimi_k2_6_w4a8/vllm/run_vllm.sh > /tmp/vllm_kimi.log 2>&1 &

# MiniMax-M2.7 (Port 8004, 2 节点, 65K)
TP=8 PP=2 MAX_MODEL_LEN=65536 GPU_MEM_UTIL=0.95 PORT=8004 \
    nohup bash examples/minimax_m2_7/vllm/run_vllm.sh > /tmp/vllm_minimax.log 2>&1 &
```

### Step 4: 验证

```bash
# 检查模型可用性
for port in 8001 8002 8003 8004; do
    echo -n "Port $port: "
    curl -sf http://10.16.201.229:$port/v1/models | \
        python3 -c "import json,sys; d=json.load(sys.stdin); print(d['data'][0]['id'], d['data'][0]['max_model_len'])"
done
```

### Step 5: 停止

```bash
# 停止 vLLM
docker exec vllm-ascend-env bash -c "pkill -9 -f 'vllm serve'"

# 重启容器 (清理 NPU 内存和 Ray 残留)
bash scripts/docker/manage_npuslim_containers.sh restart \
    --file available_nodes.txt
```

---

## 5. 已知问题与修复

### Issue 1: DeepSeek V4 attn_sink KeyError 🔴

| 项目 | 内容 |
|------|------|
| **严重度** | BLOCKING |
| **错误** | `KeyError: 'model.layers.0.self_attn.attn_sink'` |
| **文件** | `/vllm-workspace/vllm-ascend/vllm_ascend/models/deepseek_v4.py:1204` |
| **修复** | 添加 `if name not in params_dict: continue` |
| **影响** | DeepSeek-V4-Pro, DeepSeek-V4-Flash |

### Issue 2: MiniMax MTP 不支持 🔴

| 项目 | 内容 |
|------|------|
| **严重度** | BLOCKING |
| **错误** | `NotImplementedError: Unsupported speculative method: 'mtp'` |
| **修复** | 移除 `--speculative-config` 参数 |
| **影响** | MiniMax-M2.7 (模型含 MTP 但 vLLM-Ascend 不支持) |

### Issue 3: MiniMax W8A8 OOM 🟡

| 项目 | 内容 |
|------|------|
| **严重度** | MEDIUM |
| **错误** | `ValueError: No available memory for the cache blocks` |
| **修复** | 使用 TP=8 PP=2 + GPU_MEM_UTIL=0.95 + MAX_MODEL_LEN≤65536 |
| **影响** | MiniMax-M2.7 (W8A8 内存需求大) |

### Issue 4: Ray 残留 Placement Group 🟡

| 项目 | 内容 |
|------|------|
| **严重度** | MEDIUM |
| **现象** | 16/16 NPUs reserved in placement groups |
| **修复** | 部署前重启容器 |
| **预防** | 每次部署间执行 `docker restart` |

### Issue 5: Ray Head Node NPU 调度 🟡

| 项目 | 内容 |
|------|------|
| **严重度** | MEDIUM |
| **错误** | `Current node has no NPU available` |
| **原因** | 残留 placement group bundle 占用 |
| **修复** | 重启容器 (释放所有 Ray 资源) |

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
| 🥉 备选 | **MiniMax-M2.7** | 65K 上下文, W8A8 高精度 |

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
Ascend NPU Cluster — 4 Nodes
┌──────────────────────────────────────────────────────────────────────┐
│                                                                       │
│  10.16.201.229 (Head)        10.16.201.164        10.16.201.40        │
│  8× Ascend 910C × 66GB      8× Ascend 910C × 66GB  8× Ascend 910C    │
│  Container: vllm-ascend-env  Container: vllm-ascend-env              │
│  Ray Head :6379              Ray Worker            Ray Worker         │
│                                                                       │
│  10.16.201.193                                                     │
│  8× Ascend 910C × 66GB                                           │
│  Container: vllm-ascend-env                                        │
│  Ray Worker                                                        │
│                                                                       │
│  Total: 4 nodes × 8 NPU = 32 NPU                                    │
│  可同时部署 2 模型 (每个用 2 节点)                                      │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 8. 环境变量速查

### GLM-5 / GLM-5.1

```bash
# 必须
export VLLM_ASCEND_ENABLE_FLASHCOMM1=0
export VLLM_ASCEND_ENABLE_MLAPO=1

# 性能
export HCCL_BUFFSIZE=200
export HCCL_OP_EXPANSION_MODE=AIV
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1 OMP_PROC_BIND=false
```

### Kimi-K2.6

```bash
# 性能
export HCCL_BUFFSIZE=800
export TASK_QUEUE_ENABLE=1
export HCCL_OP_EXPANSION_MODE=AIV
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1 OMP_PROC_BIND=false
```

### MiniMax-M2.7

```bash
# 性能
export HCCL_BUFFSIZE=1024
export TASK_QUEUE_ENABLE=1
export VLLM_ASCEND_ENABLE_FUSED_MC2=1
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
export HCCL_OP_EXPANSION_MODE=AIV
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
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

| 文件 | 模型 | 说明 | 状态 |
|------|------|------|------|
| `examples/deepseek_v4_pro/vllm/run_vllm.sh` | DSV4-Pro | 直接部署 | ❌ attn_sink |
| `examples/deepseek_v4_pro/vllm/README.md` | DSV4-Pro | 部署文档 | ✅ 含修复方法 |
| `examples/deepseek_v4_flash/vllm/run_vllm.sh` | DSV4-Flash | 直接部署 | ❌ 同 DSV4-Pro |
| `examples/deepseek_v4_flash/vllm/README.md` | DSV4-Flash | 部署文档 | ✅ 含修复方法 |
| `examples/glm5_w4a8/vllm/run_vllm.sh` | GLM-5 | 直接部署 | ✅ 已验证 |
| `examples/glm5_w4a8/vllm/README.md` | GLM-5 | 部署文档 | ✅ |
| `examples/glm5_1_w4a8/vllm/run_vllm.sh` | GLM-5.1 | 直接部署 | ✅ 已验证 |
| `examples/glm5_1_w4a8/vllm/README.md` | GLM-5.1 | 部署文档 | ✅ |
| `examples/kimi_k2_6_w4a8/vllm/run_vllm.sh` | Kimi-K2.6 | 直接部署 | ✅ 已验证 |
| `examples/kimi_k2_6_w4a8/vllm/README.md` | Kimi-K2.6 | 部署文档 | ✅ |
| `examples/minimax_m2_7/vllm/run_vllm.sh` | MiniMax-M2.7 | 直接部署 | ✅ 已验证 (无MTP) |
| `examples/minimax_m2_7/vllm/README.md` | MiniMax-M2.7 | 部署文档 | ✅ |
| `docs/AGENT_DEPLOYMENT.md` | - | 总体部署指南 | ✅ 已更新 |

---

*Last updated: 2026-06-11 | vLLM-Ascend 0.20.2 | CANN 9.0.0 | 4/6 models deployed*
