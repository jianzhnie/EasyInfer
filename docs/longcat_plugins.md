# LongCat-Flash 部署插件说明

EasyInfer 通过插件系统（`easyinfer/plugins`）为 LongCat-Flash-Chat 在 vLLM-Ascend 上
的部署打补丁。插件按目标框架分三层注册，容器内启动 `vllm serve` 时经 entry point
自动加载，无需手动启用。

```
easyinfer/plugins/
├── transformers/            # HuggingFace transformers 层
├── vllm/                    # vLLM 核心层（模型注册、模型实现、config）
└── vllm_ascend/             # vLLM-Ascend NPU 适配层（EP 关键）
```

加载流程：`register()` → `discover_modules()` 递归 import 插件目录触发
`@register_patch` 装饰器 → `apply_all_patches()` 把补丁应用到目标模块。
补丁带版本条件（`package_version_range`），不满足时自动跳过。

## 插件清单

### vllm_ascend 层（NPU 适配，EP 部署核心）

| 插件 | 目标模块 | 作用 |
|------|---------|------|
| `vllm_ascend/ops/fused_moe/fix_ep_zero_expert.py` | `fused_moe_0_23_0` / `fused_moe` / `ascend_forward_context` / `moe_runner` | **EP 零号专家修复（vllm ≥ 0.23）**，含 4 个 patch，见下文 |
| `vllm_ascend/ops/fused_moe/zero_expert_fused_moe.py` | OOT 注册 `ZeroExpertFusedMoE` | vllm < 0.23 的 EP 路由覆盖（prepare→route→过滤零号专家→apply→finalize）。当前镜像 vllm 0.23 下自动跳过，不生效 |
| `vllm_ascend/fix_dual_attention.py` | `vllm_ascend.patch.worker.patch_deepseek_v2` | LongCat 双注意力/多 MLP 产生双整数层名（`model.layers.0.self_attn.0`），原生 `extract_layer_index` 断言失败。替换为容忍多整数前缀的版本并全局清扫所有引用 |
| `vllm_ascend/fix_mla_rotary.py` | `vllm_ascend.ops.rotary_embedding` / `mla_v1` / `sfa_v1` | LongCat 的 `model_type="longcat"` 不在 `is_deepseek_mla()` 硬编码列表中，MLA 的 `_cos_mla`/`_sin_mla` 缓存未初始化导致后端崩溃。包装 `get_cos_and_sin_mla` 按需分配/扩容缓存 |
| `vllm_ascend/fix_layernorm_dtype.py` | `vllm_ascend.ops.layernorm` | NPU 上 MLA/MoE 内核可能输出 float32 而 RMSNorm 权重为 bf16，ACLNN 算子报 EZ1001 dtype 不匹配。在算子入口把输入 cast 到权重 dtype |

#### fix_ep_zero_expert.py 的 4 个 patch

LongCat-Flash 含 512 个 Zero (Identity) 专家，EP 下这是乱码/挂起的主要来源：

1. **Patch 0b — 启用原生零号专家路径**（目标 `fused_moe_0_23_0`）：
   vllm 0.23 把零号专家配置移到 `ZeroExpertRouter`，导致
   `AscendUnquantizedFusedMoEMethod.apply` 的 `zero_expert_num > 0` 门控永假，
   零号专家的 top-k id 直接进入 dispatch kernel → aicore 崩溃。
   本 patch 把 router 配置镜像回 `AscendFusedMoE` 并给 `FusedExpertsResult` 实现 `+=`。
2. **Patch 0b2 — 拦截过早的零号专家加法**（目标 `fused_moe`）：
   包装 `zero_experts_compute`，把真实的 identity 贡献暂存（stash）并返回零，
   使 `apply` 中过早的 `final_hidden_states += zero_expert_result` 变为空操作。
   不拦截的话，EP 下每个 rank 都算了全量 identity 贡献，下游 all-reduce 会把它
   累加 world_size 次（TP=EP=64 时 ×64）→ 乱码。
3. **Patch 0c — 强制 ALLGATHER comm**（目标 `ascend_forward_context`）：
   设 `EASYINFER_MOE_COMM=allgather` 时把 MoE comm 从 MC2 覆盖为 ALLGATHER。
   MC2 的 `npu_moe_distribute_dispatch_v2` 丢弃零权重槽位，导致
   `MoeDistributeCombineV2` shape check 失败、集合通信挂起。
4. **Patch 3 — 在正确位置一次性加回**（目标 `moe_runner`）：
   在 `_maybe_add_zero_expert_output` 中取出 stash，于最终 all-reduce **之后**
   只加一次（与上游 GPU 语义一致）。无 stash 时退化为标量零空操作。

### vllm 核心层

| 插件 | 目标模块 | 作用 |
|------|---------|------|
| `vllm/model_executor/architectures.py` | `vllm.model_executor.models.registry` | 注册架构别名 `LongcatCausalLM` → `LongcatFlashForCausalLM`，避免 fallback 到 `trust_remote_code` 加载 checkpoint 自带 `modeling_longcat.py`（依赖 transformers ≥ 4.52 的 `LossKwargs`，会失败） |
| `vllm/model_executor/models/longcat_flash.py` | `vllm.model_executor.models.longcat_flash` | 两个 patch：① 分组路由（仅 `use_group_routing` 且 `expert_expansion_factor > 1` 时启用，当前 checkpoint 未启用，且仅 GPU 路径接线）；② MTP 权重过滤——加载权重时跳过 `mtp.` 前缀的 Multi-Token Prediction 键（内置 `".mtp." in name` 检查漏掉前缀形式） |
| `vllm/transformers_utils/config.py` | `vllm.transformers_utils.config` | 全局 patch `PretrainedConfig.__init__`：LongCat checkpoint 的 config 用 `num_layers` 而非 HF 标准的 `num_hidden_layers`，而 vllm_ascend 的 MLA 算子直接读后者，缺失时自动补齐 |
| `vllm/transformers_utils/model_arch_config_convertor.py` | `vllm.transformers_utils.model_arch_config_convertor` | 架构→config 转换器注册（kimi-k2 等自定义架构用，LongCat 不依赖） |

### transformers 层

| 插件 | 目标模块 | 作用 |
|------|---------|------|
| `transformers/longcat_flash.py` | `transformers.models.auto.configuration_auto` | 把 `LongcatFlashConfig` / `LongcatFlashGroupForCausalLM` 注册进 `AutoConfig` / `AutoModelForCausalLM`（含 `LongcatCausalLM` 别名），使 `from_pretrained()` 无需 `trust_remote_code=True` |

## 与部署的关系

- **TP 与 EP 都需要**：`fix_dual_attention`、`fix_mla_rotary`、`fix_layernorm_dtype`、
  `architectures`、`config`、`longcat_flash`（MTP 过滤）、`transformers/longcat_flash`。
  少了任何一个，模型要么加载失败要么输出乱码。
- **仅 EP 需要**：`fix_ep_zero_expert`（4 个 patch）。EP 部署必须配合
  `EP=1 EASYINFER_MOE_COMM=allgather`（Patch 0c 生效的前提）。
- **当前不生效**：`zero_expert_fused_moe.py`（vllm ≥ 0.23 自动跳过）、
  分组路由（checkpoint 未启用）。

## 验证插件是否生效

服务启动日志中查找：

```
Applied patch: patch_enable_native_zero_expert -> vllm_ascend.ops.fused_moe.fused_moe_0_23_0
Applied patch: patch_force_allgather_comm -> vllm_ascend.ascend_forward_context
Applied patch: fix_deepseek_v2_init -> vllm_ascend.patch.worker.patch_deepseek_v2
...
```

若目标模块被跳过（`Target module not found, skipping`），通常是 vllm_ascend 升级后
内部模块改名（如 `fused_moe_0_24_0`），需同步更新 patch 的 target。
