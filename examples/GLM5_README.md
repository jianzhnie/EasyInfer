# GLM-5/GLM-5.1 Deployment Guide

本文档提供 GLM-5/GLM-5.1 模型在华为 NPU 环境下的部署指南。

## 官方文档参考

完整官方文档: https://docs.vllm.ai/projects/ascend/en/v0.18.0/tutorials/models/GLM5.html

## 模型权重下载

### GLM-5
- **BF16**: [ModelScope](https://www.modelscope.cn/models/ZhipuAI/GLM-5)
- **W4A8**: [ModelScope](https://modelscope.cn/models/Eco-Tech/GLM-5-w4a8)
- **W8A8**: [ModelScope](https://www.modelscope.cn/models/Eco-Tech/GLM-5-w8a8)

### GLM-5.1
- **BF16**: [HuggingFace](https://huggingface.co/zai-org/GLM-5.1)
- **W4A8**: [Modelers](https://modelers.cn/models/Eco-Tech/GLM-5.1-w4a8)
- **W8A8**: [Modelers](https://modelers.cn/models/Eco-Tech/GLM-5.1-w8a8)

推荐下载到共享目录: `/root/.cache/`

## 量化类型

| 量化类型 | 说明 | 硬件要求 | 推荐场景 |
|---------|------|---------|---------|
| **w4a8** | 4-bit 权重 + 8-bit 激活 | A2 (8卡) / A3 (16卡) | 推荐，显存占用小 |
| **w8a8** | 8-bit 权重 + 8-bit 激活 | A3 (16卡) 仅支持 | 精度更高，需启用 MLAPO |
| **bf16** | 原生 BF16 无量化 | 多节点 (至少 2×16卡) | 最高精度，需跨节点 |

## 硬件配置矩阵

### 单节点部署

#### Atlas 800 A2 (64G × 8 NPU)
```bash
# W4A8 量化 (默认)
./glm5_server.sh

# 手动配置
QUANT_TYPE=w4a8 \
TENSOR_PARALLEL_SIZE=8 \
MAX_MODEL_LEN=32768 \
MAX_NUM_SEQS=2 \
./glm5_server.sh
```

#### Atlas 800 A3 (64G × 16 NPU)
```bash
# W4A8 量化 (支持 200k 上下文)
QUANT_TYPE=w4a8 \
TENSOR_PARALLEL_SIZE=16 \
MAX_MODEL_LEN=200000 \
MAX_NUM_SEQS=8 \
./glm5_server.sh

# W8A8 量化 (需启用 MLAPO)
QUANT_TYPE=w8a8 \
TENSOR_PARALLEL_SIZE=16 \
MAX_MODEL_LEN=40960 \
MAX_NUM_SEQS=8 \
./glm5_server.sh
```

### 多节点部署

#### 2节点 × A3 (BF16)
```bash
# Node 0 (Master)
QUANT_TYPE=bf16 \
TENSOR_PARALLEL_SIZE=16 \
PIPELINE_PARALLEL_SIZE=2 \
DATA_PARALLEL_SIZE=2 \
DATA_PARALLEL_SIZE_LOCAL=1 \
./glm5_server.sh

# Node 1 (Worker) - 需要手动配置网络参数
# 参考官方文档配置 HCCL_IF_IP, GLOO_SOCKET_IFNAME 等
```

#### 2节点 × A2 (W4A8)
```bash
# Node 0
QUANT_TYPE=w4a8 \
TENSOR_PARALLEL_SIZE=8 \
DATA_PARALLEL_SIZE=2 \
DATA_PARALLEL_SIZE_LOCAL=1 \
MAX_MODEL_LEN=131072 \
./glm5_server.sh

# Node 1 - 需要添加 --headless 和 --data-parallel-start-rank 1
# 参考官方文档
```

## 关键特性配置

### 专家并行 (Expert Parallel)
GLM-5 采用 MoE 架构，必须启用专家并行:
```bash
ENABLE_EXPERT_PARALLEL=1 ./glm5_server.sh
```

### 投机解码 (Speculative Decoding)
使用 DeepSeek MTP 加速解码阶段:
```bash
SPECULATIVE_METHOD=deepseek_mtp \
SPECULATIVE_NUM_TOKENS=3 \
./glm5_server.sh
```

推荐值: 3-5 tokens，越大加速效果越好但风险越高。

### 异步调度 (Async Scheduling)
优化大规模模型推理效率:
```bash
ENABLE_ASYNC_SCHEDULING=1 ./glm5_server.sh
```

仅支持量化模型 (w4a8/w8a8)。

### NPU Graph 优化
```bash
CUDAGRAPH_MODE=FULL_DECODE_ONLY \
ENABLE_NPUGRAPH_EX=true \
FUSE_MULS_ADD=true \
MULTISTREAM_OVERLAP_SHARED_EXPERT=true \
./glm5_server.sh
```

### MLAPO (仅 W8A8)
W8A8 量化必须启用 MLAPO:
```bash
QUANT_TYPE=w8a8 \
VLLM_ASCEND_ENABLE_MLAPO=1 \
./glm5_server.sh
```

## 环境变量详解

### HCCL 相关 (华为集合通信库)

| 变量 | 默认值 | 说明 |
|------|-------|------|
| `HCCL_OP_EXPANSION_MODE` | `AIV` | 操作扩展模式，优化性能 |
| `HCCL_BUFFSIZE` | `200` | 缓冲区大小 (MB) |
| `HCCL_IF_IP` | - | 多节点必需，节点 IP 地址 |
| `HCCL_SOCKET_IFNAME` | - | 多节点必需，网络接口名 |

### OpenMP 相关

| 变量 | 默认值 | 说明 |
|------|-------|------|
| `OMP_PROC_BIND` | `false` | 禁用线程绑定，避免干扰 NPU 调度 |
| `OMP_NUM_THREADS` | `1` | 减少线程数，降低调度开销 |

### PyTorch NPU

| 变量 | 默认值 | 说明 |
|------|-------|------|
| `PYTORCH_NPU_ALLOC_CONF` | `expandable_segments:True` | 内存分配配置 |
| `VLLM_ASCEND_BALANCE_SCHEDULING` | `1` | 负载均衡调度 |
| `VLLM_ASCEND_ENABLE_MLAPO` | `1` (仅 W8A8) | MLAPO 优化 |

## 性能调优建议

### 低延迟场景
- 单节点使用 `dp1tp16` 并关闭专家并行
- 减小 `MAX_NUM_SEQS` (如 2-4)
- 启用投机解码 (3 tokens)

### 高吞吐场景
- 增大 `MAX_NUM_SEQS` (如 8-16)
- 启用 Chunked Prefill 和 Prefix Caching
- 启用异步调度

### 长上下文场景
- 增大 `MAX_MODEL_LEN` (如 131072 或 200000)
- 增大 `GPU_MEMORY_UTILIZATION` (如 0.95)
- 多节点部署扩展显存容量

## 常见问题

### Q: W8A8 和 W4A8 有什么区别？
A: W8A8 精度更高但显存占用更大，仅支持 A3 (16卡)。W4A8 推荐用于大多数场景，支持 A2/A3。

### Q: 为什么 W8A8 需要启用 MLAPO？
A: MLAPO (Model Layer Parallel Optimization) 针对 W8A8 量化优化并行策略，必须启用否则性能会显著下降。

### Q: 如何选择 max_model_len？
A: 
- A2 (8卡) W4A8: 推荐 32k
- A3 (16卡) W4A8: 推荐 200k
- A3 (16卡) W8A8: 推荐 40k
- 多节点 BF16: 推荐 8k

根据显存和实际需求调整。

### Q: 多节点部署需要哪些额外配置？
A: 多节点部署需要:
1. 配置网络接口: `HCCL_IF_IP`, `GLOO_SOCKET_IFNAME`, `TP_SOCKET_IFNAME`, `HCCL_SOCKET_IFNAME`
2. 配置数据并行: `DATA_PARALLEL_ADDRESS`, `DATA_PARALLEL_RPC_PORT`
3. Worker 节点添加: `--headless`, `--data-parallel-start-rank`
4. BF16 权重需运行 `adjust_weight.py` 脚本启用 MTP

详细配置参考官方文档。

## 功能验证

部署完成后，可通过以下方式验证:

### 基础测试
```bash
curl http://localhost:8077/v1/models
```

### 推理测试
```bash
curl http://localhost:8077/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "glm-5",
    "prompt": "Hello, GLM-5!",
    "max_tokens": 100
  }'
```

### 性能基准测试
参考官方文档使用 AISBench 或 vLLM Benchmark 工具。

## 更多信息

- 官方文档: https://docs.vllm.ai/projects/ascend/en/v0.18.0/tutorials/models/GLM5.html
- Prefill-Decode Disaggregation: 高级部署模式，参考文档章节
- 多 Token Prediction (MTP): 投机解码技术细节
- Accuracy Evaluation: 使用 AISBench 或 lm-eval 评估精度