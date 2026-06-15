# Context

## 背景

本项目需要对 LongcatFlash 模型做 monkey patch，以支持自定义的 **Grouped Routing (分组路由)** 功能。

## 关键文件

| 文件                                                                             | 说明                              |
| ------------------------------------------------------------------------------ | ------------------------------- |
| `modeling_longcat_flash.py`                                                    | 原始 HF 模型代码                      |
| `modeling_longcat_flash_group.py`                                              | 自定义 HF 模型代码（新增 grouped routing） |
| `/Users/robin/work_dir/vllm/vllm/model_executor/models/longcat_flash.py`       | vLLM 参考实现                       |
| `/Users/robin/work_dir/EasyInfer/easyinfer/plugins/vllm/model_executor/models` | 目标目录，monkey patch 放这里           |

## 原始 vs 自定义 HF 模型的差异

自定义版本 (`modeling_longcat_flash_group.py`) 在 `LongcatFlashTopkRouter` 中新增了 **Grouped Routing**:

### 新增 config 属性

- `use_group_routing` (bool): 是否启用分组路由
- `expert_expansion_factor` (int): 每个 expert 的副本数 F

### `get_topk_indices()` 新增逻辑 (方案一)

Router 经过 expansion 后 layout 为: `[real_0..real_{N-1}, zero_0..zero_{Z-1}] × F copies`

- 将 (N+Z) 个原始 expert 分成 (N+Z) 个组，每组 F 个副本
- 组内选最优，再从 (N+Z) 个 winner 中选 top-k

核心代码:

```python
if self.use_group_routing:
    expansion_factor = self.expert_expansion_factor
    total_groups = self.n_routed_experts // expansion_factor  # N + Z
    grouped_score = scores.view(-1, expansion_factor, total_groups).transpose(-1, -2)
    group_score_best, group_best_idx = grouped_score.max(dim=-1)
    _, topk_group_ids = torch.topk(group_score_best, k=self.top_k, dim=-1, sorted=False)
    topk_indices = topk_group_ids + group_best_idx.gather(1, topk_group_ids) * total_groups
```

其余组件 (LongcatFlashMLA, LongcatFlashMoE, LongcatFlashDecoderLayer, LongcatFlashModel, LongcatFlashForCausalLM) 与原版完全一致。

## vLLM 参考实现架构

vLLM 的 `longcat_flash.py` 关键结构:

- `FlashConfig` → HF config → vLLM config 转换
- `LongcatMoe` → 包含 `LongcatRouter` + `FusedMoE`，router 只产生 logits，实际 routing 在 `FusedMoE` 内部完成
- `FlashDecoderLayer` → dual-attention (DeepseekV2MLAAttention) + dual-MLP (FlashMLP) + MoE (LongcatMoe)
- vLLM 中 **没有** `LongcatFlashTopkRouter` 的等价物，routing 逻辑在 `FusedMoE` 层内部

## 任务

在 `/Users/robin/work_dir/EasyInfer/easyinfer/plugins/vllm/model_executor/models `下创建 `longcat_flash.py`，用 `@register_patch` 对 vLLM 的 `longcat_flash` 模块做 monkey patch，使其支持 custom HF 模型的 **Grouped Routing**。

### 具体要点

1. **Patch** **`LongcatMoe`** **或** **`LongcatRouter`**：让 vLLM 的 MoE 模块支持 `use_group_routing` 和 `expert_expansion_factor` 配置
2. **遵循项目规范**：
   - 使用 `@register_patch(target="vllm.model_executor.models.longcat_flash")` 装饰器
   - 参考 `qwen3_moe.py` 的 patch 模式
   - 使用 `easyinfer.plugins.logging.patch_logger` 记录日志
3. **不影响默认行为**：未启用 `use_group_routing` 时走原 vLLM 逻辑
4. **处理 expert expansion**：expansion 后的 expert 权重在 checkpoint 中可能有不同的 key mapping，需要在 load\_weights 中正确处理

---

## 实现状态

`longcat_flash.py` 已在目标目录实现并通过代码检查。

### 实现文件

```
/Users/robin/work_dir/EasyInfer/easyinfer/plugins/vllm/model_executor/models/longcat_flash.py
```

### 已完成的要点

1. **FlashConfig patch**：新增 `use_group_routing` 和 `expert_expansion_factor` 字段，兼容 HF config。
2. **LongcatMoe patch**：在 `LongcatMoe.__init__` 后注入分组路由逻辑，支持三条运行路径：
   - **Path A (GPU + zero expert)**：直接 patch `ZeroExpertRouter._compute_routing`
   - **Path B (无 zero expert)**：设置 `FusedMoE.custom_routing_function`
   - **Path C (Ascend NPU)**：替换 `AscendZeroExpertFusedMoE.select_experts`
3. **默认行为保护**：仅当 `use_group_routing=True` 且 `expert_expansion_factor > 1` 时才启用分组路由。
4. **日志**：使用 `easyinfer.plugins.logging.patch_logger` 记录各路径的启用信息。

### 验证结果

```bash
ruff check easyinfer/plugins/vllm/model_executor/models/longcat_flash.py  # 通过
mypy easyinfer/plugins/vllm/model_executor/models/longcat_flash.py        # 通过
```

### 待处理项

- **expert expansion 权重 key mapping**：当前实现只覆盖了 routing 逻辑。如果 checkpoint 中 expansion 后的 expert 权重存在不同的参数命名（例如 `experts.w2_weight_expanded_0` 等），仍需要在 `load_weights` 或相应的模型 patch 中做额外处理。目前 `longcat_flash.py` 尚未实现这部分。
