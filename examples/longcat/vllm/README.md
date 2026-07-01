# LongCat-Flash-Chat-1024E-512Zero-Topk24-v2 部署指南

## 模型概况

| 属性 | 值 |
|------|-----|
| 架构 | LongcatFlashForCausalLM (MLA + MoE) |
| 专家数 | 1024 Routed + 512 Zero (Identity) |
| TopK | 24 |
| Hidden Size | 6144 |
| Layers | 28 |
| KV LoRA Rank | 512 |
| Max Position | 131072 |
| Vocab Size | 131072 |
| 精度 | bfloat16 (无量化) |

## 硬件需求

- **最小配置**: 64 × 昇腾 NPU (8 节点 × 8 卡, 每卡 64GB)
- **推荐配置**: TP=64, PP=1
- **框架**: vLLM-Ascend 0.18.0 + Ray 分布式
- **容器**: ascend910c-cann8.5.1-torch2.9.0-vllm0.18.0

## 文件说明

```
examples/longcat/vllm/
├── run_vllm.sh       ← 直接 vllm serve 部署
├── curl_test.sh      ← API 功能测试
└── README.md         ← 本文档
```

## 前置条件

## 部署步骤

### 1. 准备节点列表

使用 (8 节点 × 8 卡 = 64 NPU)

### 2. 重启容器并启动 Ray 集群

```bash
# 重启容器（确保干净状态）
bash scripts/docker/manage_npuslim_containers.sh restart --file node_list_8.txt

# 等待容器就绪
sleep 2

# 启动 Ray 集群
bash scripts/ray_cluster/start_npuslim_ray_cluster.sh start --file node_list_8.txt

# 验证 64 NPU 可用
"docker exec vllm-ascend-env ray status | grep NPU"
# 预期输出: 0.0/64.0 NPU
```

### 4. 启动模型服务

```bash
# 注意: 不要从 EasyInfer 目录运行，避免插件冲突
bash /home/jianzhnie/llmtuner/llm/EasyInfer/examples/longcat/vllm/run_vllm.sh
```

支持通过环境变量覆盖默认配置：

```bash
TP=64 MAX_MODEL_LEN=8192 MAX_NUM_SEQS=64 bash /home/jianzhnie/llmtuner/llm/EasyInfer/examples/longcat/vllm/run_vllm.sh
```

> 模型加载约需 16-20 分钟 (148 个 safetensors 分片 + 64 卡 HCCL 初始化)。

### 5. 测试模型

```bash
bash examples/longcat/vllm/curl_test.sh
```

支持指定远程地址测试：

```bash
bash examples/longcat/vllm/curl_test.sh
```

## 关键配置说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| TP | 64 | 张量并行度 (8 节点 × 8 卡) |
| PP | 1 | 流水线并行度 |
| PORT | 8010 | 服务端口 |
| MAX_MODEL_LEN | 4096 | 最大序列长度 (可调大, 受显存限制) |
| MAX_NUM_SEQS | 128 | 最大并发序列数 |
| GPU_MEM_UTIL | 0.90 | 显存利用率 |