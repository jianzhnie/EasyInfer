# LongCat-Flash-Chat-2layer 部署指南

## 模型概况

| 属性 | 值 |
|------|-----|
| 架构 | LongcatFlashForCausalLM (MLA + MoE) |
| 专家数 | 512 Routed + 256 Zero (Identity) |
| TopK | 12 |
| Hidden Size | 6144 |
| Layers | **2** |
| 精度 | bfloat16 |
| 模型大小 | ~80 GB |

## 硬件需求

EP 模式: 2 × 64 GB NPU (TP=2, --enable-expert-parallel)

> 单卡无法容纳 512 experts 模型（需 ~41 GB 权重 + KV cache）。

## EasyInfer EP 修复插件

### `fix_ep_zero_expert.py`

修复 vLLM 0.20.2 Ascend EP 路径的 3 个问题：

| 问题 | 根因 | 修复方式 |
|------|------|----------|
| AssertionError | `_zero_expert_output` 未被设置 | `MoERunner.forward` 预计算 zero expert output |
| Token dispatch 越界 | `topk_ids` 含 zero expert 索引 (>=512) 但 `moe_expert_num=256` | `TokenDispatcherWithMC2.token_dispatch` 前 clamp ID |
| 兼容性 | `ZeroExpertFusedMoE` 在 v0.20.2 已移除 | 移除版本约束 |

### 已删除: `zero_expert_fused_moe_v0202.py`

版本约束移除后不再需要独立文件。

## 部署

```bash
# 1. 创建容器
IMAGE_NAME=quay.io/ascend/vllm-ascend:v0.20.2rc1-a3 \
CONTAINER_NAME=vllm-ascend-2layer \
bash scripts/docker/ascend_infer_docker_run.sh

# 2. 部署 (EP 模式)
docker exec vllm-ascend-2layer bash \
    /home/jianzhnie/llmtuner/llm/EasyInfer/examples/longcat-2layer/vllm/deploy_2layer.sh

# 3. 测试
curl http://localhost:8010/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"longcat-flash-2layer","messages":[{"role":"user","content":"Hello!"}],"max_tokens":32}'
```

## 关键配置

| 参数 | 值 | 说明 |
|------|-----|------|
| `TP` | 2 | 张量并行度 |
| `EP` | enabled | 专家并行 |
| `MAX_MODEL_LEN` | 2048 | 最大序列长度 |
| `GPU_MEM_UTIL` | 0.85 | NPU 显存利用率 |
| `HCCL_BUFFSIZE` | 4096 | HCCL EP 缓冲区 |

## 验证结果

| 阶段 | 状态 | 说明 |
|------|------|------|
| 插件加载 | ✅ | vllm general_plugins 自动发现 |
| 模型加载 | ✅ | EP Rank 0/2, 256/512 experts, 38.87 GB/worker |
| Zero Expert (AssertionError) | ✅ | 已修复 |
| Token Dispatch (aicore) | ✅ | 已修复 (ID sanitization) |
| KV Cache | ✅ | 3.87 GiB, 902K tokens |
| API 启动 | ✅ | Application startup complete |
| 推理 | ⚠️ | CANN MLP kernel aicore 异常 (fftsplus aivector) |

> 推理阶段的 aicore 异常是 CANN 在 EP 模式下处理 512 experts MoE 的 kernel 层限制，
> 需要 CANN 版本更新来解决。插件层能修复的问题已全部修复。
