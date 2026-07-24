# Step-3.7-Flash W8A8 MTP 部署指南

> ⚠️ **兼容性警告**: vLLM 0.22.1 注册表中没有 `Step3p7ForConditionalGeneration`；
> **vLLM 0.23.0rc1 已注册**，但其实现与该 checkpoint 不兼容
> （worker 初始化报 `shape '[8, -1, 128]' is invalid for input of size 128`），
> 两个版本均无法部署，需等上游修复。
> **vLLM-Ascend 0.23.0rc1 + CANN 8.5.1** | 端口: **8015**
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

## 并行策略

| 场景 | TP | PP | DP | NPU | 上下文 | 量化 | 状态 |
|------|-----|-----|-----|-----|--------|------|------|
| 单节点 | 8 | 1 | 1 | 8 | 32K | W8A8 | ❌ |

> 待上游修复 Step3p7 checkpoint 兼容性后验证。

## 环境变量

> 完整环境变量说明见 [prompts/vllm_env_vars.md](../../../prompts/vllm_env_vars.md)。
> Claude Code 集成方式见 [prompts/vllm-prompt.md](../../../prompts/vllm-prompt.md)。
## 验证记录

| 时间 | 镜像 | 节点 | 配置 | 结果 | 日志 | 说明 |
|------|------|------|------|------|------|------|
| 2026-07-20 | `quay.io/ascend/vllm-ascend:v0.22.1rc1-a3` (CANN 8.5.1) | pair4: 10.42.11.202/203 | TP=8 PP=1, PORT=8015 | ❌ FAIL_SERVICE | `logs/parallel_deploy_remaining_v022/step-3.7-flash_*.log` | `ValueError: Unrecognized configuration class Step3p7Config for this kind of AutoModel: AutoModel`，vLLM 0.22.1 未注册 Step-3.7-Flash 的 VL 配置类 |

- 虽然内部文本模型 `Step3p5ForCausalLM` 在 vLLM 0.22.1 有注册，但外层 `Step3p7Config` VL wrapper 无法被 `AutoModel` 识别。
| 2026-07-21 | 同上 | pair4: 10.42.11.202/203 | TP=8 PP=1 + `--model-impl transformers`, PORT=8015 | ❌ FAIL_SERVICE | `logs/step37_tfimpl_vllm.log` | transformers fallback 在 worker 初始化时失败，原因同上 |

### 2026-07-21 复查结论（确认当前镜像不可部署）

1. **remote code 本身可用**：transformers 5.5.4 + `trust_remote_code=True` 可正常解析
   `Step3p7Config` 和 `Step3p7ForConditionalGeneration`（auto_map 含 `AutoModelForCausalLM`）。
2. **vLLM transformers 后端不兼容**：vLLM 0.22.1 的 transformers fallback 固定调用
   `AutoModel.from_config()`（`vllm/model_executor/models/transformers/base.py`），
   而模型 `auto_map` 只注册了 `AutoConfig` / `AutoModelForCausalLM` / `AutoProcessor`，
   没有 `AutoModel` 条目 → `ValueError: Unrecognized configuration class ... for this kind of AutoModel: AutoModel`。
3. **即使绕过第 2 点也无法正确推理**：checkpoint 为 msmodelslim ascend 格式 W8A8
   （expert 权重 int8 + `weight_scale`/`weight_offset`，无标准 `quantization_config`），
   transformers 后端不会反量化，数值必然错误；`--quantization ascend` 只作用于 vLLM 原生模型层。
4. **结论**：需等待 vLLM / vllm-ascend 原生注册 `Step3p7ForConditionalGeneration`（文本侧
   `Step3p5ForCausalLM` 已注册，外层 VL wrapper + MTP 待支持）。当前镜像无脚本级 workaround。
| 2026-07-22 | `quay.io/ascend/vllm-ascend:v0.23.0rc1-a3` (CANN 8.5.1) | pair4: 10.42.11.202/203 | TP=8 PP=1, PORT=8015 | ❌ FAIL_SERVICE | `logs/step37_v023_vllm.log` | 架构已在 v0.23.0 注册，但 worker 初始化报 `shape '[8, -1, 128]' is invalid for input of size 128` |

### 2026-07-22 补充（v0.23.0）

- `Step3p7ForConditionalGeneration` 已进入 v0.23.0 注册表，配置解析通过、权重可加载，
  但在 KV cache 初始化阶段报 shape 错误 → v0.23.0rc1 的 Step3p7 实现与该 checkpoint
  不兼容，需等上游修复（稳定版或后续 rc）。
