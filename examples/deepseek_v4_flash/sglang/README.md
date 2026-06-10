# DeepSeek-V4-Flash W8A8 MTP — SGLang 部署文档

> **部署验证状态**: ⚠️ 待验证 (SGLang Ascend 首次部署)

## 模型架构摘要

| 属性 | 值 |
|------|-----|
| **架构** | DeepseekV4ForCausalLM |
| **模型类型** | deepseek_v4 |
| **参数量** | DeepSeek-V4 Flash (MoE) |
| **隐藏维度** | 4096 (MLA) |
| **层数** | 43 |
| **专家数** | 256 (每 token 激活 6 个) |
| **注意力** | MLA (Q-LoRA rank=1024) |
| **MTP** | 1 (支持推测解码) |
| **最大位置** | 1,048,576 |
| **量化** | W8A8 (INT8 权重, FP8 激活) |
| **多模态** | 否 |

## 硬件要求

### 单节点

| 资源 | 需求 |
|------|------|
| **NPU** | 8 × 64G (Atlas 800 A2/A3) |
| **显存** | ~45G / NPU (W8A8, 64K context) |
| **并行策略** | TP=8 PP=1 EP=8 |

### 多节点 (4 节点)

| 资源 | 需求 |
|------|------|
| **NPU** | 32 × 64G (4 节点 × 8 NPU) |
| **显存** | ~45G / NPU |
| **并行策略** | TP=32 PP=1 EP=32 |
| **网络** | HCCL 多节点通信 (torch.distributed) |

## 快速开始

### 前置准备

```bash
# 1. 拉取 SGLang Ascend 镜像（所有节点）
docker pull quay.io/ascend/sglang:main-cann9.0.0-a3

# 2. 启动容器（所有节点）
# 参考: scripts/docker/manage_npuslim_containers.sh
docker run -d --name sglang-ascend-env \
    --network host \
    --privileged \
    --device=/dev/davinci0 --device=/dev/davinci1 \
    --device=/dev/davinci2 --device=/dev/davinci3 \
    --device=/dev/davinci4 --device=/dev/davinci5 \
    --device=/dev/davinci6 --device=/dev/davinci7 \
    --device=/dev/davinci_manager --device=/dev/devmm_svm \
    --device=/dev/hisi_hdc \
    -v /home/jianzhnie/llmtuner:/home/jianzhnie/llmtuner \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    quay.io/ascend/sglang:main-cann9.0.0-a3 \
    tail -f /dev/null
```

### 单节点部署

```bash
# 在容器内执行
cd /home/jianzhnie/llmtuner/llm/EasyInfer/examples/deepseek_v4_flash/sglang
TP=8 PP=1 EP=8 NNODES=1 NODE_RANK=0 bash run_sglang.sh
```

### 多节点部署 (4 节点)

```bash
# === Head 节点 (10.16.201.193) ===
docker exec sglang-ascend-env bash -c \
  'cd /home/jianzhnie/llmtuner/llm/EasyInfer/examples/deepseek_v4_flash/sglang && \
   nohup bash run_sglang.sh --nnodes 4 --node-rank 0 \
   >> /tmp/sglang_dsv4.log 2>&1 &'

# === Worker 节点 (10.16.201.201, 10.16.201.153, 10.16.201.124) ===
for rank in 1 2 3; do
  case $rank in
    1) node="10.16.201.201" ;;
    2) node="10.16.201.153" ;;
    3) node="10.16.201.124" ;;
  esac
  ssh "$node" "docker exec sglang-ascend-env bash -c \
    'cd /home/jianzhnie/llmtuner/llm/EasyInfer/examples/deepseek_v4_flash/sglang && \
     nohup bash run_sglang.sh --nnodes 4 --node-rank $rank \
     >> /tmp/sglang_dsv4.log 2>&1 &'"
done
```

### 健康检查

```bash
# 等待服务就绪 (通常 5-15 分钟)
curl -sf http://10.16.201.193:8000/health

# 查看模型列表
curl -sf http://10.16.201.193:8000/v1/models | python3 -m json.tool

# 发测试请求
curl http://10.16.201.193:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"deepseek-v4-flash","messages":[{"role":"user","content":"Hello"}],"max_tokens":100}'
```

## 环境变量参考

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `MODEL_PATH` | `.../DeepSeek-V4-Flash-w8a8-mtp` | 模型权重路径 |
| `HOST` | `0.0.0.0` | 监听地址 |
| `PORT` | `8000` | 服务端口 |
| `TP` | `32` | 张量并行度 |
| `PP` | `1` | 流水线并行度 |
| `EP` | `32` | 专家并行度 |
| `NNODES` | `4` | 节点数 |
| `NODE_RANK` | `0` | 节点序号 (0 为 Head) |
| `DIST_INIT_ADDR` | `10.16.201.193` | torch.distributed 初始化地址 |
| `CONTEXT_LEN` | `65536` | 最大上下文长度 |
| `MAX_RUNNING_REQS` | `16` | 最大并发请求 |
| `MEM_FRACTION` | `0.90` | NPU 显存分配比例 |
| `QUANTIZATION` | `modelopt_fp8` | 量化方法 |

## 并行策略推荐

| 节点数 | TP | PP | EP | 总 NPU | 状态 |
|--------|----|----|------|--------|------|
| 1 | 8 | 1 | 8 | 8 | ⚠️ 待验证 |
| 2 | 16 | 1 | 16 | 16 | ⚠️ 待验证 |
| 4 | 32 | 1 | 32 | 32 | ⚠️ 待验证 |

> 注: DeepSeek-V4-Flash 可能支持 PP，但默认使用大 TP 方案以确保兼容性。
> 若验证支持 PP，可尝试 `TP=8 PP=4 EP=32` 获得更好的通信效率。

## 性能调优建议

1. **RadixAttention 自动前缀缓存**: SGLang 默认开启，无需配置。对多轮对话场景有显著加速。
2. **推测解码**: `--speculative-algorithm EAGLE --speculative-num-draft-tokens 3` 利用 MTP 加速解码。
3. **内存分配**: W8A8 量化模型使用 `--mem-fraction-static 0.90`，可尝试提高到 0.93。
4. **上下文长度**: 默认 64K，可根据业务需求调整到 128K 或更高（需更多显存）。
5. **调度策略**: SGLang 默认 `lpm`（最长前缀匹配）适合多轮对话，无需修改。

## 功能验证

```bash
# 运行完整测试
BASE_URL=http://10.16.201.193:8000 bash curl_test.sh

# 手动验证 API
# 1. Health
curl http://10.16.201.193:8000/health

# 2. 非流式
curl http://10.16.201.193:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"deepseek-v4-flash","messages":[{"role":"user","content":"Hello"}],"max_tokens":100}'

# 3. 流式
curl http://10.16.201.193:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"deepseek-v4-flash","messages":[{"role":"user","content":"Hello"}],"max_tokens":100,"stream":true}'

# 4. Tool Calling
curl http://10.16.201.193:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"deepseek-v4-flash","messages":[{"role":"user","content":"What is the weather?"}],"tools":[{"type":"function","function":{"name":"get_weather","description":"Get weather","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}],"max_tokens":200}'
```

## 常见问题 FAQ

### Q: SGLang 启动失败 `No module named 'sglang'`
A: 确认使用的是 SGLang Ascend 镜像: `quay.io/ascend/sglang:main-cann9.0.0-a3`，不要用 vLLM 镜像。

### Q: `device type npu is not supported`
A: 需要 source CANN 环境并添加 `--device npu` 参数。

### Q: `torch.distributed initialization failed`
A: 检查节点间网络连通性，确认 `DIST_INIT_ADDR` 指向 Head 节点，所有节点使用相同端口。

### Q: 启动后 `curl /health` 长时间无响应
A: 模型加载通常需要 5-15 分钟。查看日志: `tail -f /tmp/sglang_dsv4.log`。

### Q: 如何查看 RadixAttention 缓存命中率？
A: SGLang 在日志中输出 cache hit rate，也可通过 metrics 端点查看（需 `--enable-metrics`）。

## 相关文件

| 文件 | 说明 |
|------|------|
| `run_sglang.sh` | 直接部署脚本（推荐） |
| `sglang_server.sh` | 包装器部署脚本 |
| `curl_test.sh` | API 功能测试 |
| `README.md` | 本文档 |
