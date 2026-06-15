# longcat_flash.py — 分组路由 Monkey Patch 详细解析

> 对应实现文件：
> `/Users/robin/work_dir/EasyInfer/easyinfer/plugins/vllm/model_executor/models/longcat_flash.py`

## 1. 背景

vLLM 原生的 `longcat_flash` 模块使用标准 top-k 路由：router 输出 (N+Z) 维 logits → softmax → top-k 选专家。但自定义 HF 模型 `modeling_longcat_flash_group.py` 引入了 **Grouped Routing（分组路由）**：将 (N+Z) 个专家分成 F 个组，先在组内选最优，再从组冠军中选 top-k。

本文件通过 `@register_patch` 将分组路由逻辑注入 vLLM 推理引擎，支持三种硬件/配置组合：纯 GPU 无零专家、GPU 有零专家、Ascend NPU。

---

## 2. 文件结构总览

```
├── _grouped_routing()                   # 核心：分组 top-k 路由算法
├── _patch_zero_expert_router()          # Path A：替换 ZeroExpertRouter._compute_routing
├── _patch_ascend_select_experts()       # Path C：替换 Ascend select_experts
└── patch_longcat_flash_grouped_routing  # 主入口 monkey patch（@register_patch 装饰）
    ├── Patch FlashConfig.__init__       #   让配置接受 use_group_routing / expert_expansion_factor
    └── Patch LongcatMoe.__init__        #   根据 zero_expert_type 选择路由注入路径
```

---

## 3. 核心算法：`_grouped_routing()`

### 函数签名

```python
def _grouped_routing(
    hidden_states,      # (tokens, hidden_dim) — 未使用，仅为 API 兼容
    gating_output,      # (tokens, N+Z) — router 原始 logits
    topk,               # 最终选取的专家数
    renormalize,        # 是否归一化 top-k 权重
    expansion_factor,   # F，每组专家数
    n_routed_experts,   # N+Z，总 router 输出维度
    e_score_correction_bias,  # softmax 后加上的 bias（HF 行为）
) -> tuple[topk_weights, topk_ids]
```

### 算法步骤

```
输入: gating_output shape = (tokens, N+Z), expansion_factor = F

Step 1 ──────────────────────────────────────────────────────
  scores = softmax(gating_output, dim=-1)           # (tokens, N+Z)

Step 2 ──────────────────────────────────────────────────────
  if e_score_correction_bias is not None:
      scores_for_choice = scores + bias             # HF: bias 加在 softmax 之后
  else:
      scores_for_choice = scores

Step 3 ── Reshape 成 F 组 ───────────────────────────────────
  total_groups = (N+Z) // F
  (tokens, N+Z) → view(-1, F, total_groups) → transpose(-1, -2)
  → grouped_scores: (tokens, total_groups, F)

Step 4 ── 每组内选最优 ──────────────────────────────────────
  group_score_best, group_best_idx = grouped_scores.max(dim=-1)
  # group_score_best: (tokens, total_groups), 每组最佳得分
  # group_best_idx:   (tokens, total_groups), 组内偏移 [0, F)

Step 5 ── 从 total_groups 个组冠军中选 top-k ──────────────
  _, topk_group_ids = torch.topk(group_score_best, k=topk, dim=-1)

Step 6 ── 映射回原始专家 index ─────────────────────────────
  best_offsets = group_best_idx.gather(1, topk_group_ids)
  topk_ids = topk_group_ids + best_offsets * total_groups
  # 映射公式: expert_idx = group_id + offset_in_group * total_groups

Step 7 ── 从原始 softmax scores 取权重（不是 bias 后的）───
  topk_weights = scores.gather(1, topk_ids)

Step 8 ── 可选归一化 ───────────────────────────────────────
  if renormalize:
      topk_weights /= topk_weights.sum(dim=-1, keepdim=True) + 1e-20
```

### 关键设计细节

| 细节 | 说明 |
|------|------|
| `hidden_states` 不使用 | 仅保留参数以匹配 `FusedMoE.custom_routing_function` 的调用签名 |
| bias 加在 softmax 之后 | 匹配 HF `LongcatFlashTopkRouter` 的行为 |
| 权重从原始 scores gather | bias 只影响**选择哪个专家**，不影响最终**权重值** |
| `sorted=False` | 与 HF 保持一致，不排序 top-k 结果 |

---

## 4. 三层补丁架构与调用栈

vLLM 的路由器工厂根据配置创建不同类型的路由器对象，且 Ascend NPU 的 `forward()` 有自己的分发路径，因此需要**三种不同的注入点**。

### 4.1 补丁入口：`LongcatMoe.__init__`

```
patch_longcat_flash_grouped_routing(module)
│
├── Patch FlashConfig.__init__
│     └── 新增字段: use_group_routing (bool), expert_expansion_factor (int)
│
└── Patch LongcatMoe.__init__
      │
      ├── 读取 config.use_group_routing, config.expert_expansion_factor
      ├── 若不启用或 expansion_factor <= 1 → 直接 return（不影响默认行为）
      │
      ├── n_routed = self.router.n_routed_experts   # N+Z，由 LongcatRouter 计算
      ├── zero_expert_type = config.zero_expert_type
      ├── is_ascend = hasattr(experts, "_temporarily_set_attrs")
      │
      ├── [Path A] zero_expert_type is not None AND not is_ascend
      │     └── _patch_zero_expert_router(experts, expansion_factor, n_routed)
      │         (仅在 GPU 上执行；Ascend 跳过，因为其 forward() 不调用 _compute_routing)
      │
      ├── [Path B] zero_expert_type is None (任何设备)
      │     └── 设置 experts.custom_routing_function
      │
      └── [Path C] is_ascend (无论 zero_expert_type)
            └── _patch_ascend_select_experts(experts, expansion_factor, n_routed)
```

> **注意：** Path A 和 Path C 互斥——GPU 有零专家走 Path A，Ascend 走 Path C。Path B 可以与 Path C 叠加（Ascend 无零专家时）。

---

### 4.2 Path B：无零专家，纯 GPU（custom_routing_function 路径）

**条件：** `zero_expert_type is None`

**注入点：** 设置 `experts.custom_routing_function`

```
LongcatMoe.__init__()
  │
  └─ self.experts.custom_routing_function = lambda(
         hidden_states, gating_output, topk, renormalize, **kw
     ):
         return _grouped_routing(
             hidden_states, gating_output, topk, renormalize,
             expansion_factor, n_routed,
             e_score_correction_bias=self.router.e_score_correction_bias,
         )
       │
       ▼
  vLLM RouterFactory 检测到 custom_routing_function != None
       │
       └─ 创建 CustomRoutingRouter 实例
            │
            ▼ 每次 forward:
            CustomRoutingRouter.route()
              └─ custom_routing_function()
                   └─ _grouped_routing()
```

**调用栈（推理时）：**
```
LongcatFlashModel.forward()
  └─ LongcatFlashDecoderLayer.forward()
       └─ LongcatMoe.forward()
            ├─ router(router_logits)                  # LongcatRouter 只产生 logits
            └─ experts(hidden_states, router_logits)  # FusedMoE.forward()
                 ├─ CustomRoutingRouter.route()
                 │    └─ custom_routing_function()
                 │         └─ _grouped_routing()      # ← 分组路由在此执行
                 └─ fused_moe()                       # 标准 MoE 计算
```

---

### 4.3 Path A：有零专家，GPU（ZeroExpertRouter 路径）

**条件：** `zero_expert_type is not None`，GPU 环境

**问题：** vLLM 路由器工厂在 `zero_expert_type` 不为 None 时，优先创建 `ZeroExpertRouter` 而非 `CustomRoutingRouter`，导致 `custom_routing_function` 被**忽略**。

**注入点：** 直接 monkey-patch `router._compute_routing` 方法

```python
def _patch_zero_expert_router(experts, expansion_factor, n_routed):
    router = experts.router  # ZeroExpertRouter 实例

    def grouped_compute_routing(hidden_states, router_logits, indices_type):
        # 1. 分组 top-k（替代原 fused_topk_bias 调用）
        topk_weights, topk_ids = _grouped_routing(...)

        # 2. routed_scaling_factor 缩放
        if router.routed_scaling_factor != 1.0:
            topk_weights *= router.routed_scaling_factor

        # 3. 零专家贡献计算
        router._zero_expert_output = zero_experts_compute_triton(
            expert_indices=topk_ids.clone(),
            expert_scales=topk_weights.clone(),
            num_experts=router.num_logical_experts,
            zero_expert_type=router.zero_expert_type,
            hidden_states=hidden_states,
        )

        # 4. Mask 零专家 ID，让下游 MoE 忽略
        zero_mask = topk_ids >= router.num_logical_experts
        topk_ids[zero_mask] = 0
        topk_weights[zero_mask] = 0.0

        return topk_weights, topk_ids

    router._compute_routing = grouped_compute_routing  # 替换
```

**调用栈（推理时）：**
```
LongcatFlashModel.forward()
  └─ LongcatFlashDecoderLayer.forward()
       └─ LongcatMoe.forward()
            └─ experts(hidden_states, router_logits)     # ZeroExpertFusedMoE
                 └─ ZeroExpertFusedMoE.forward()
                      ├─ router._compute_routing()       # ← 被替换的方法
                      │    ├─ _grouped_routing()         # 分组 top-k
                      │    ├─ routed_scaling_factor 缩放
                      │    ├─ zero_experts_compute_triton()  # 零专家贡献
                      │    └─ mask 零专家 ID
                      │
                      ├─ fused_moe()                     # 真专家计算
                      └─ + zero_expert_result            # 合并零专家贡献
```

---

### 4.4 Path C：Ascend NPU（select_experts 路径）

**条件：** Ascend NPU 环境（通过 `hasattr(experts, "_temporarily_set_attrs")` 检测）

**问题：** `AscendZeroExpertFusedMoE.forward()` 在调用 `select_experts` 之前**临时将 `custom_routing_function` 置为 None**：

```python
# AscendZeroExpertFusedMoE.forward() 关键代码（第174-186行）
temp_attrs = {"custom_routing_function": None}   # ← Path B 失效!
if self._router is not None:
    temp_attrs["e_score_correction_bias"] = self._router.e_score_correction_bias

with self._temporarily_set_attrs(**temp_attrs):
    topk_weights, topk_ids = self.select_experts(...)  # ← 这里被 Path C 拦截
```

同时 Path A 的 `_compute_routing` patch 也失效，因为 Ascend 不使用 `ZeroExpertRouter._compute_routing`，而是直接调用实例方法 `select_experts()`。

**注入点：** 直接替换 `experts.select_experts` 方法

```python
def _patch_ascend_select_experts(experts, expansion_factor, n_routed):
    def grouped_select_experts(hidden_states, router_logits):
        topk_weights, topk_ids = _grouped_routing(
            hidden_states=hidden_states,
            gating_output=router_logits,
            topk=experts.top_k,
            renormalize=experts.renormalize,
            expansion_factor=expansion_factor,
            n_routed_experts=n_routed,
            e_score_correction_bias=experts.e_score_correction_bias,
        )
        if experts.routed_scaling_factor != 1.0:
            topk_weights *= experts.routed_scaling_factor
        return topk_weights.to(torch.float32), topk_ids.to(torch.int32)

    experts.select_experts = grouped_select_experts  # 替换
```

**调用栈（推理时）：**
```
LongcatFlashModel.forward()
  └─ LongcatFlashDecoderLayer.forward()
       └─ LongcatMoe.forward()
            └─ experts(hidden_states, router_logits)
                 └─ AscendZeroExpertFusedMoE.forward()
                      │
                      ├─ 1. 临时置空 custom_routing_function
                      │    with _temporarily_set_attrs(custom_routing_function=None):
                      │
                      ├─ 2. self.select_experts()          # ← 被 Path C 替换
                      │    └─ _grouped_routing()            # 分组 top-k
                      │
                      ├─ 3. _compute_zero_expert_result()  # Ascend 零专家
                      │    ├─ ascend_zero_experts_compute()
                      │    └─ 缓存 memoized_topk_weights/ids
                      │
                      ├─ 4. AscendFusedMoE.forward()       # CANN 算子执行真专家
                      │    └─ 再次临时置空 custom_routing_function
                      │         (避免 CANN dispatch 维度不匹配)
                      │
                      └─ 5. fused_out + zero_expert_result
```

---

## 5. 完整调用栈汇总

```
LongcatFlashForCausalLM.generate()  （或 forward）
  │
  └─ LongcatFlashModel.forward()
       │
       └─ for layer in layers:
            └─ LongcatFlashDecoderLayer.forward()
                 │
                 ├─ input_layernorm(hidden_states)
                 ├─ self_attn()                     # MLA attention
                 ├─ post_attention_layernorm()
                 │
                 ├─ mlp()                           # FlashMLP（dense FFN）
                 │
                 ├─ shared_experts()                # 共享专家
                 │
                 └─ moe()                           # LongcatMoe ← 补丁生效点
                      │
                      ├─ LongcatRouter(router_logits)
                      │    └─ 只做: logits + bias → 返回 router_logits
                      │
                      └─ experts(hidden_states, router_logits)
                           │
                           ┌──────────────────────────────────────────┐
                           │ 根据配置进入三条路径之一:                    │
                           │                                          │
                           │ Path B: GPU, 无零专家                      │
                           │   CustomRoutingRouter.route()             │
                           │     └─ custom_routing_function()          │
                           │          └─ _grouped_routing()            │
                           │   → FusedMoE 计算                         │
                           │                                          │
                           │ Path A: GPU, 有零专家                      │
                           │   ZeroExpertRouter._compute_routing()     │
                           │     ├─ _grouped_routing()                 │
                           │     └─ zero_experts_compute_triton()      │
                           │   → ZeroExpertFusedMoE 计算               │
                           │                                          │
                           │ Path C: Ascend NPU                        │
                           │   select_experts()                        │
                           │     └─ _grouped_routing()                 │
                           │   → _compute_zero_expert_result()         │
                           │   → AscendFusedMoE.forward()              │
                           └──────────────────────────────────────────┘
                 │
                 └─ output = residual + moe_output + shared_output + mlp_output
       │
       └─ norm()
            └─ lm_head()
```

---

## 6. 关键设计决策

### 6.1 为何需要三层补丁而非统一方案？

vLLM 的路由器工厂在 `FusedMoE.__init__` 中根据条件创建不同类型的路由器：

| 条件 | 创建的路由器 | `custom_routing_function` 是否生效 |
|------|-------------|----------------------------------|
| `zero_expert_type is None` + `custom_routing_function` 已设置 | `CustomRoutingRouter` | 是 |
| `zero_expert_type is not None` | `ZeroExpertRouter`（优先级更高） | **否，被忽略** |
| Ascend NPU | `AscendZeroExpertFusedMoE` 直接调 `select_experts()` | **否，被临时置空** |

每种路由器有不同的分发路径，无法用单一 hook 点覆盖。

### 6.2 为何 Ascend 路径检测用 `_temporarily_set_attrs`？

`_temporarily_set_attrs` 是 `AscendZeroExpertFusedMoE`（继承自 `AscendFusedMoE`）的特征方法。在 NPU 上，`FusedMoE` 会被替换为 `AscendZeroExpertFusedMoE`，因此该属性存在即表示当前运行在 Ascend 环境。这是一种 duck-typing 检测，避免硬编码 `isinstance` 检查。

> **风险：** 这是内部实现细节，如果未来 vLLM-Ascend 版本重命名或移除此方法，检测将静默失败。更健壮的方案是使用 `isinstance` 检查或专用的标记属性。

### 6.3 为何 `e_score_correction_bias` 加在 softmax 之后？

匹配 HF `LongcatFlashTopkRouter.get_topk_indices` 的行为。HF 实现中 bias 是在 softmax 之后加上的，用于调整路由偏好但不影响概率分布本身。

### 6.4 为何权重从原始 `scores` gather 而非 `scores_for_choice`？

bias 的目的仅是**影响专家选择**（让某些专家更容易/更难被选中），而最终分配给专家的权重应该反映真实的 softmax 概率。这与 HF 实现保持一致。

### 6.5 为何 Ascend forward 要两次临时置空 `custom_routing_function`？

1. **第一次（select_experts 前）：** 确保走 Path C 的 `select_experts` patch，而非 `CustomRoutingRouter` 路径
2. **第二次（AscendFusedMoE.forward 前）：** 避免 Ascend 的 CANN dispatch kernel 收到未扩展的 hidden_states（`custom_routing_function` 返回 memoized 结果会跳过 Ascend 的 native hidden_states expansion，导致 `MoeDistributeDispatchV4 shape mismatch` 错误）

### 6.6 Ascend 第二 pass 的路由差异

`AscendZeroExpertFusedMoE.forward()` 中第二遍调用 `AscendFusedMoE.forward(router_logits=router_logits_sliced)` 时，`custom_routing_function` 被置为 None，因此 Ascend 对真实专家的选择使用**标准 top-k 而非分组路由**。只有第一遍的零专家选择（`select_experts` → `_grouped_routing`）使用了分组路由。这是 CANN fused kernel 不支持 `custom_routing_function` 的已知限制。

### 6.7 `total_groups < topk` 参数校验

`_grouped_routing()` 中已添加显式守卫：当 `expansion_factor` 过大导致 `(N+Z)//F < topk` 时，`torch.topk(k=topk)` 会因 k 超过维度大小而抛出 `RuntimeError`。新添加的 `ValueError` 给出了明确的错误信息和修复建议（减小 `expansion_factor`）。

---

## 7. 数据流图

```
Config
  ├── use_group_routing: bool
  ├── expert_expansion_factor: int (F)
  ├── n_routed_experts: int (N)
  ├── zero_expert_num: int (Z)
  └── zero_expert_type: str | None

                    ┌──────────────────────┐
                    │   router_logits       │
                    │   (tokens, N+Z)       │
                    └──────┬───────────────┘
                           │
                    ┌──────▼───────────────┐
                    │  _grouped_routing()  │
                    │                      │
                    │  softmax → +bias     │
                    │  → reshape F groups  │
                    │  → group max         │
                    │  → top-k groups      │
                    │  → map to expert id  │
                    │  → gather weights    │
                    └──────┬───────────────┘
                           │
              ┌────────────┴────────────┐
              │                         │
    Path B (no zero)           Path A/C (zero)
              │                         │
    ┌─────────▼──────────┐    ┌────────▼──────────┐
    │ custom_routing_    │    │ ZeroExpert Router  │
    │ function           │    │ or Ascend          │
    │ → CustomRouting    │    │ select_experts     │
    │   Router           │    │                    │
    └─────────┬──────────┘    │ + zero_expert      │
              │               │   compute          │
              │               │ + mask zero IDs    │
              │               └────────┬───────────┘
              │                        │
              └────────┬───────────────┘
                       │
              ┌────────▼──────────┐
              │  FusedMoE /       │
              │  AscendFusedMoE   │
              │  (real experts)   │
              └────────┬──────────┘
                       │
              ┌────────▼──────────┐
              │  final MoE output │
              │  (+ zero expert   │
              │   contribution    │
              │   if applicable)  │
              └───────────────────┘
```
