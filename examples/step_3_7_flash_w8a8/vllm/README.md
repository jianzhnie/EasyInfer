# Step-3.7-Flash W8A8 MTP 部署指南

> ⚠️ **兼容性警告**: vLLM 0.22.1 (当前容器) 注册表中**没有** `Step3p7ForConditionalGeneration`，
> 但包含其文本骨干 `Step3p5ForCausalLM` 和 `Step3p5MTP`。外层 VL 包装能否加载需实测确认。
> **vLLM-Ascend 0.22.1rc1 + CANN 8.5.1** | 端口: **8015**
> 架构: Step3p7ForConditionalGeneration (text: Step3p5ForCausalLM) | MoE | MTP | W8A8 量化
> 目标配置: TP=8 PP=1 (单节点，权重 ~204G → ~26G/NPU)

## 模型简介

| 属性 | 值 |
|------|-----|
| **架构** | Step3p7ForConditionalGeneration（文本骨干 Step3p5ForCausalLM） |
| **文本骨干** | 45 层 / hidden 4096 |
| **原生上下文** | max_seq_len 262,144（rope_scaling llama3, factor 2.0） |
| **量化方式** | W8A8，权重 ~204G |
| **投机解码** | MTP（vLLM 0.22.1 已注册 Step3p5MTP） |
| **工具调用解析器** | step3p5（注册表中另有 step3） |

## 快速开始

### 前置条件

模型路径: `/home/jianzhnie/llmtuner/hfhub/models/Eco-Tech/Step-3.7-Flash-w8a8-mtp`

```bash
bash scripts/docker/manage_npuslim_containers.sh start --file node_list3.txt
bash scripts/ray_cluster/start_npuslim_ray_cluster.sh start --file node_list3.txt
```

### 部署（容器内执行）

```bash
# 单节点 TP=8 + MTP（默认）
bash examples/step_3_7_flash_w8a8/vllm/run_vllm.sh

# 关闭 MTP
ENABLE_MTP=0 bash examples/step_3_7_flash_w8a8/vllm/run_vllm.sh

# 大上下文
MAX_MODEL_LEN=131072 bash examples/step_3_7_flash_w8a8/vllm/run_vllm.sh

# 后台运行
nohup bash examples/step_3_7_flash_w8a8/vllm/run_vllm.sh > step37_vllm.log 2>&1 &
```

### 验证

```bash
bash examples/step_3_7_flash_w8a8/vllm/curl_test.sh

# 手动验证
curl http://localhost:8015/v1/models
curl http://localhost:8015/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"step-3.7-flash","messages":[{"role":"user","content":"Hello"}],"max_tokens":50}'
```

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `MODEL_PATH` | `.../Eco-Tech/Step-3.7-Flash-w8a8-mtp` | 模型权重路径 |
| `PORT` | `8015` | 监听端口 |
| `TP` / `PP` | `8` / `1` | 并行度 |
| `MAX_MODEL_LEN` | `32768` | 最大上下文 |
| `MAX_NUM_SEQS` | `32` | 最大并发序列 |
| `GPU_MEM_UTIL` | `0.92` | 显存利用率 |
| `ENABLE_MTP` | `1` | MTP 投机解码开关 |

## 验证记录

| 日期 | 环境 | 配置 | 结果 |
|------|------|------|------|
| 待填写 | vLLM-Ascend 0.22.1rc1 + CANN 8.5.1 | TP=8 单节点 | 待验证（外层架构未注册，存疑） |

## 验证记录

| 时间 | 镜像 | 节点 | 配置 | 结果 | 日志 | 说明 |
|------|------|------|------|------|------|------|
| 2026-07-20 | `quay.io/ascend/vllm-ascend:v0.22.1rc1-a3` (CANN 8.5.1) | pair4: 10.42.11.202/203 | TP=8 PP=1, PORT=8015 | ❌ FAIL_SERVICE | `logs/parallel_deploy_remaining_v022/step-3.7-flash_*.log` | `ValueError: Unrecognized configuration class Step3p7Config for this kind of AutoModel: AutoModel`，vLLM 0.22.1 未注册 Step-3.7-Flash 的 VL 配置类 |

- 虽然内部文本模型 `Step3p5ForCausalLM` 在 vLLM 0.22.1 有注册，但外层 `Step3p7Config` VL wrapper 无法被 `AutoModel` 识别。
