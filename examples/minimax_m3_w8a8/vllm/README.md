# MiniMax-M3 W8A8 部署指南

> ⚠️ **兼容性警告**: vLLM **0.22.1 与 0.23.0rc1** 注册表均**不支持**
> `MiniMaxM3SparseForConditionalGeneration` 架构（0.23.0rc1 已复查确认）。
> 上游 vLLM 的支持仅有 CUDA/ROCm 路径，需等待 vllm-ascend 合入。
> **vLLM-Ascend 0.23.0rc1 + CANN 8.5.1** | 端口: **8014**
> 架构: MiniMaxM3SparseForConditionalGeneration (minimax_m3_vl) | MoE | VL | W8A8 量化
> 目标配置: TP=8 PP=1 (单节点，权重 ~418G → ~52G/NPU，显存紧张)

## 模型简介

| 属性 | 值 |
|------|-----|
| **架构** | MiniMaxM3SparseForConditionalGeneration (文本骨干 + VL) |
| **文本骨干** | 60 层 / hidden 6144 / 64 头 (GQA kv_heads=4) |
| **原生上下文** | 1,048,576 (1M tokens) |
| **量化方式** | W8A8，权重 ~418G |
| **工具调用解析器** | minimax_m2（待官方确认 M3 专用解析器） |

### 兼容性说明

当前容器 vLLM 0.22.1 注册表中的 MiniMax 系列：
`MiniMaxForCausalLM`、`MiniMaxM1ForCausalLM`、`MiniMaxM2ForCausalLM`、`MiniMaxText01ForCausalLM`、`MiniMaxVL01ForConditionalGeneration`
— 均不匹配 M3 的 `MiniMaxM3SparseForConditionalGeneration`。

脚本按已验证的 MiniMax-M2.7 配方编写，待支持落地后可直接使用。

## 快速开始

### 前置条件

模型路径: `/home/jianzhnie/llmtuner/hfhub/models/Eco-Tech/MiniMax-M3-w8a8`

```bash
bash scripts/docker/manage_npuslim_containers.sh start --file node_list3.txt
bash scripts/ray_cluster/start_npuslim_ray_cluster.sh start --file node_list3.txt
```

### 部署（容器内执行）

```bash
# 单节点 TP=8（默认）
bash examples/minimax_m3_w8a8/vllm/run_vllm.sh

# 2 节点 TP=16（更大 KV 余量）
TP=16 bash examples/minimax_m3_w8a8/vllm/run_vllm.sh

# 后台运行
nohup bash examples/minimax_m3_w8a8/vllm/run_vllm.sh > minimax_m3_vllm.log 2>&1 &
```

### 验证

```bash
bash examples/minimax_m3_w8a8/vllm/curl_test.sh
```

## 环境变量

> 完整环境变量说明见 [prompts/vllm_env_vars.md](../../../prompts/vllm_env_vars.md)。
> Claude Code 集成方式见 [prompts/vllm-prompt.md](../../../prompts/vllm-prompt.md)。
## 验证记录

| 时间 | 镜像 | 节点 | 配置 | 结果 | 日志 | 说明 |
|------|------|------|------|------|------|------|
| 2026-07-20 | `quay.io/ascend/vllm-ascend:v0.22.1rc1-a3` (CANN 8.5.1) | pair3: 10.42.11.200/201 | TP=8 PP=1, PORT=8014 | ❌ FAIL_SERVICE | `logs/minimax_m3_retry_v022/*.log` | `MiniMaxM3SparseForConditionalGeneration` 不在 vLLM 0.22.1 支持架构列表中 |
| 2026-07-22 | `quay.io/ascend/vllm-ascend:v0.23.0rc1-a3` (CANN 8.5.1) | — | — | ❌ 不支持（注册表复查） | — | v0.23.0rc1 注册表仍无 `MiniMaxM3*`，上游仅 CUDA/ROCm 路径 |

- 初次部署因脚本中包含当前版本不支持的 `--swap-space 32` 参数而直接退出，已移除该参数。
- 移除后服务仍无法启动，核心原因为 vLLM 0.22.1 registry 未注册 `MiniMaxM3SparseForConditionalGeneration`，需等后续版本支持。

### 2026-07-21 复查结论（确认当前镜像不可部署）

1. **注册表确认**：容器内 `ModelRegistry.get_supported_archs()` 无任何 `MiniMaxM3*` 条目。
2. **transformers fallback 不可行**：`AutoConfig` + `trust_remote_code` 可加载（`MiniMaxM3VLConfig`），
   但 `AutoModelForCausalLM.from_config` 报 `ValueError: Unrecognized configuration class` —
   模型目录 `auto_map` 只注册了 `AutoConfig`，没有模型实现类，transformers 后端无法加载权重。
3. **上游支持状态**：vLLM 已于 2026-06-12 宣布 MiniMax-M3 day-0 支持
   （[vLLM blog](https://vllm.ai/blog/2026-06-12-minimax-m3-vllm)、
   [recipes](https://recipes.vllm.ai/MiniMaxAI/MiniMax-M3)），但：
   - 稳定版尚未发布，CUDA/ROCm 需用专用镜像 `vllm/vllm-openai:minimax-m3`；
   - 上游实现为 hardware-isolated（`nvidia/`、`amd/` 目录），**无 Ascend (NPU) 路径**；
   - 需等待 vllm-ascend 后续版本合入 `MiniMaxM3SparseForConditionalGeneration` 支持。

**结论**：当前 `vllm-ascend:v0.22.1rc1-a3` 镜像无法部署 MiniMax-M3，无脚本级 workaround。
待 vllm-ascend 新版本发布后，`run_vllm.sh`（已按 M2.7 配方编写）可直接复用验证。
