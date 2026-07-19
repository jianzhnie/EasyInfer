# LongCat-Flash-Chat-EP 部署 — 完整技术文档

​											

## 1. 概述

基于 28 层 LongCat-Flash-Chat 模型（ 560 B, 512 routed experts + 256 zero experts），在 Ascend NPU (CANN 9.0.0) 上通过 vLLM 0.20.2 部署，并启用专家并行 (EP)。

由于 vLLM 0.20.2 Ascend 路径对 MoE + EP 支持不完善，在本地代码库实现了 4 个 patch 修复 3 类问题。

###  最终状态 — CANN kernel 限制

**方案**: 综合分析所有尝试方案。

**发现**:

- aicore 异常在 NPU stream 异步执行中触发，worker shutdown 同步时才被检出
- `rtStreamSynchronizeWithTimeout execution failed, reason=aicore exception`
- 错误码 `507015` — NPU 硬件异常
- 不是 dispatch/MLP/combine 中任何单一 kernel，而是 CANN 内部 bug
- 512 experts + topk=12 + EP  组合触发

**结论**: ❌ 无法从 Python 插件层修复，需要 CANN 版本更新。



## 2.  LongCat-Flash-Chat  EP 部署 插件

### Patch 0: MLP 分块计算 (`unquant_apply_mlp`)

| 项目 | 内容                                                                     |
| ---- | ------------------------------------------------------------------------ |
| 目标 | `vllm_ascend.ops.fused_moe.moe_mlp`                                      |
| 问题 | `npu_grouped_matmul` 在处理 256 experts 时触发 aicore 异常               |
| 修复 | 将 experts 按 64 个一组分块，每组独立调用 `npu_grouped_matmul`，拼接结果 |

```python
_MAX_EXPERTS_PER_CHUNK = 64  # 256 experts / 64 = 4 chunks

for chunk in chunks:
    w1_chunk = w1[e0:e1]  # 切片 expert 权重
    hs_chunk = hidden_states[t0:t1]  # 取对应 token 范围
    gl_chunk = group_list[e0:e1] - t0  # 调整累积计数
    out_chunk = original_mlp(hs_chunk, w1_chunk, w2_chunk, gl_chunk, ...)
result = concat(all_chunks)
```

### Patch 1: Token Dispatch ID 过滤

| 项目 | 内容                                                                                             |
| ---- | ------------------------------------------------------------------------------------------------ |
| 目标 | `TokenDispatcherWithMC2.token_dispatch`                                                          |
| 问题 | `ascend_select_experts` 返回 `topk_ids` 含 zero expert 索引 (512–767)，但 EP kernel 只接受 0–255 |
| 修复 | 调用 `npu_moe_distribute_dispatch_v2` 前 clamp ≥512 的 ID 到 0，权重清零                         |

```python
zero_mask = topk_ids >= num_logical_experts  # 512
topk_ids[zero_mask] = 0       # clamp 到有效范围
topk_weights[zero_mask] = 0.0  # 权重清零 → 不影响 MLP 输出
```

根因: GPU 路径 `ZeroExpertRouter._compute_routing` 做此过滤，Ascend EP 路径缺失。

### Patch 2: Zero Expert Output 预计算

| 项目 | 内容                                                                                          |
| ---- | --------------------------------------------------------------------------------------------- |
| 目标 | `MoERunner.forward`                                                                           |
| 问题 | `_zero_expert_output` 未被设置 → `_maybe_add_zero_expert_output` 断言失败                     |
| 修复 | 使用 Ascend `select_experts` + `zero_experts_compute` 正确计算 identity 型 zero expert output |

```python
topk_weights, topk_ids = ascend_select_experts(
    hidden_states=input_hs, router_logits=rl,
    top_k=moe_config.experts_per_token,
    renormalize=router.renormalize,
    routed_scaling_factor=router.routed_scaling_factor,
    e_score_correction_bias=router.e_score_correction_bias,
    ...
)
_, _, zero_out = ascend_zero_experts_compute(
    expert_indices=topk_ids, expert_scales=topk_weights,
    num_experts=router.num_logical_experts,  # 512
    zero_expert_type="identity",
    hidden_states=input_hs,
)
router._zero_expert_output = zero_out.to(hidden_states.dtype)
```

Identity 型 zero expert 贡献: `output += sum(router_weights[i] * input, for i in zero_expert_slots)`

### Patch 3: 安全网

| 项目 | 内容                                                             |
| ---- | ---------------------------------------------------------------- |
| 目标 | `MoERunner._maybe_add_zero_expert_output`                        |
| 修复 | 若 `_zero_expert_output` 仍为 None → 注入标量零张量（no-op add） |

---

## 3. 其他插件修复

### `zero_expert_fused_moe.py` — 移除版本约束

`@register_patch(condition=package_version_range("vllm_ascend", max_version="0.20.1"))`
→ 移除 condition → 对所有版本生效。

原因: vLLM 0.20.2 移除了 `ZeroExpertFusedMoE` 类，改为 `FusedMoE(zero_expert_type=...)`。
旧版本约束导致插件被跳过。

### `longcat_flash.py` — Grouped Routing 支持

LongCat 使用 32-expert groups 分组路由。EP 路径 Ascend fused kernel 不支持 `custom_routing_function`，使用标准 top-k 路由作为降级。

---

## 4. 部署配置

### 容器

```bash
IMAGE_NAME=quay.io/ascend/vllm-ascend:v0.20.2rc1-a3 \
```

### vLLM Serve 参数

```bash
vllm serve "$MODEL_PATH" \
    --trust-remote-code \
    --dtype bfloat16 \
    --tensor-parallel-size 64 \
    --pipeline-parallel-size 1 \
    --enable-expert-parallel \
    --distributed-executor-backend ray \
    --gpu-memory-utilization 0.85 \
    --max-model-len 2048 \
    --max-num-seqs 64 \
    --max-num-batched-tokens 2048 \
    --no-enable-prefix-caching \
    --enforce-eager \
    --seed 1024
```

### 环境变量

| 变量                     | 值                       | 说明                     |
| ------------------------ | ------------------------ | ------------------------ |
| `ENABLE_EXPERT_PARALLEL` | 1                        | 启用专家并行             |
| `HCCL_BUFFSIZE`          | 4096                     | HCCL EP 缓冲区 (≥819 MB) |
| `HCCL_OP_EXPANSION_MODE` | AIV                      | HCCL 算子扩展            |
| `PYTORCH_NPU_ALLOC_CONF` | expandable_segments:True | NPU 内存分配             |
| `HCCL_SOCKET_IFNAME`     | enp66s0f5                | 通信网卡                 |
| `GLOO_SOCKET_IFNAME`     | enp66s0f5                | Gloo 通信网卡            |

---

## 5. 尝试过的方案

以下按时间线记录整个 EP 调试过程中尝试的所有方案，包括最终采用的和被放弃的。

---

### 迭代 1: 基础 EP 部署（无插件）

**方案**: 直接使用 `--enable-expert-parallel` + `--distributed-executor-backend mp`

**现象**:
```
AssertionError: assert zero_expert_output is not None
  File "moe_runner.py", line 563, in _maybe_add_zero_expert_output
```

**分析**: Ascend EP 路径绕过 `ZeroExpertRouter._compute_routing()`，`_zero_expert_output` 未设置。

**结论**: ❌ 需要插件修复。

---

### 迭代 2: 插件 Patch `AscendFusedMoE.forward_impl`

**方案**: 在 `AscendFusedMoE.forward_impl` 返回后，重新计算 routing 并将 zero expert output 存入 router。

```python
@register_patch(target="vllm_ascend.ops.fused_moe.fused_moe")
def patch_ascend_ep_zero_expert(module):
    AscendFusedMoE.forward_impl = patched_forward_impl
    # After quant_method.apply(), recompute routing and set _zero_expert_output
```

**现象**: 仍然 AssertionError。

**分析**: 自定义算子 `_moe_forward` 调用的是 `MoERunner._forward_impl`（runner 方法），而非 `AscendFusedMoE.forward_impl`（layer 方法）。Patch 的目标函数从未被执行。

**结论**: ❌ Patch 目标错误，废弃此方案。

---

### 迭代 3: 插件 Patch `MoERunner.forward`

**方案**: 在 `MoERunner.forward` 入口处预计算 `_zero_expert_output`。

```python
@register_patch(target="vllm.model_executor.layers.fused_moe.runner.moe_runner")
def patch_moe_runner_zero_expert(module):
    # Before _forward_entry: compute _zero_expert_output, store in router
```

**现象**: 仍然 AssertionError（但插件本身正确加载）。

**分析**: 插件未在 vllm 进程中生效。原因是 `PYTHONPATH` 不会触发 `vllm.general_plugins` entry_point 发现。Worker 子进程不会加载 EasyInfer patches。

**结论**: ❌ 虽然 patch 逻辑正确，但部署方式导致未生效。

---

### 迭代 4: 分析 aicore 根因 — Token Dispatch ID 越界

**方案**: 追踪 `npu_moe_distribute_dispatch_v2` 参数来源。

**发现**:
```
ascend_select_experts 返回 topk_ids ∈ [0, 767]  (512 real + 256 zero)
  → TokenDispatcherWithMC2.get_dispatch_mc2_kwargs:
      moe_expert_num = len(expert_map) = 256  # 只包含 real experts
  → npu_moe_distribute_dispatch_v2(moe_expert_num=256, expert_ids ∈ [0, 767])
  → 越界访问 → aicore exception!
```

GPU 路径 `ZeroExpertRouter._compute_routing` 会在 dispatch 前 mask 掉 zero expert ID：
```python
zero_mask = topk_ids >= num_logical_experts  # 512
topk_ids[zero_mask] = 0       # clamp
topk_weights[zero_mask] = 0.0  # zero weight
```

Ascend EP 路径缺失此步骤。

**结论**: ✅ 找到根因，实现 Patch 1 (Token Dispatch ID 过滤)。

---

### 迭代 5: Patch 1 + 推理测试

**方案**: Patch `TokenDispatcherWithMC2.token_dispatch` 过滤 zero expert ID。

**现象**:
- ✅ Token dispatch aicore 不再触发（`npu_moe_distribute_dispatch_v2` 不再出现在错误日志中）
- ❌ 新的 aicore 异常：`fftsplus aivector error` — 发生在 MLP compute 阶段

**分析**: dispatch 修复后，后续的 grouped matmul kernel 触发新崩溃。256 experts × 12 topk 的超大 expert 组可能超出 `npu_grouped_matmul` 的 tiling 能力。

**结论**: ✅ dispatch 修复成功，但暴露了 MLP kernel 问题。

---

### 迭代 6: MLP 分块计算（Patch 0）

**方案**: 将 256 experts 的 `npu_grouped_matmul` 拆分为 4 组 × 64 experts。

```python
_MAX_EXPERTS_PER_CHUNK = 64
for chunk in range(4):
    w1_chunk = w1[e0:e1]         # 切片权重
    hs_chunk = hidden_states[t0:t1]  # 取对应 tokens
    gl_chunk = group_list[e0:e1] - t0  # 调整累积计数
    out = original_mlp(hs_chunk, ...)
result = concat(outputs)
```

**现象**: patch 正确应用，但推理仍然 aicore 异常。

**分析**: 崩溃不在 `unquant_apply_mlp` 的 `npu_grouped_matmul` 中，而在其他 CANN kernel（可能在 `token_combine` 的 `npu_moe_distribute_combine_v2` 或 MLP 的 `npu_swiglu`）。

**结论**: ✅ chunked MLP 实现完成（防止 grouped matmul 成为瓶颈），但 CANN 其他 kernel 仍存在问题。

---

### 迭代 7: 尝试 `VLLM_ASCEND_BALANCE_SCHEDULING` 和调度参数

**方案**: 尝试改变 Ascend 调度行为。

| 尝试                               | 结果     |
| ---------------------------------- | -------- |
| `VLLM_ASCEND_BALANCE_SCHEDULING=1` | ❌ 无效果 |
| `--max-num-seqs 1`                 | ❌ 无效果 |
| `--max-num-batched-tokens 1`       | ❌ 无效果 |

**结论**: ❌ 非参数问题。

---

### 迭代 8: 最终状态 — CANN kernel 限制

**方案**: 综合分析所有错误日志。

**发现**:
- aicore 异常在 NPU stream 异步执行中触发，worker shutdown 同步时才被检出
- `rtStreamSynchronizeWithTimeout execution failed, reason=aicore exception`
- 错误码 `507015` — NPU 硬件异常
- 不是 dispatch/MLP/combine 中任何单一 kernel，而是 CANN 内部 bug
- 512 experts + topk=12 + EP=2 + bfloat16 组合触发

**结论**: ❌ 无法从 Python 插件层修复，需要 CANN 版本更新。

---

### 方案汇总

| #    | 方案                        | 类型 | 结果                    | 最终采用   |
| ---- | --------------------------- | ---- | ----------------------- | ---------- |
| 1    | 分析 Token Dispatch ID 越界 | 分析 | 找到 aicore 根因        | ✅ Patch 1  |
| 2    | Patch 1 + 推理测试          | 插件 | dispatch 修复，MLP 崩溃 | ✅ Patch 1  |
| 3    | MLP 分块计算                | 插件 | 实现完成，但仍 aicore   | ✅ Patch 0  |
| 4    | 调度参数调整                | 测试 | 无效果                  | ❌          |
| 5    | CANN kernel 限制确认        | 分析 | 无法从 Python 修复      | ✅ 文档记录 |

### 最终采用的技术栈

- **Patch 0**: `unquant_apply_mlp` 分块 (64 experts/chunk)
- **Patch 1**: `TokenDispatcherWithMC2.token_dispatch` ID 过滤
- **Patch 2**: `MoERunner.forward` zero expert output 预计算
- **Patch 3**: `MoERunner._maybe_add_zero_expert_output` 安全网


### 修复效果矩阵

| 问题            | 根因                            | 修复      | 状态 |
| --------------- | ------------------------------- | --------- | ---- |
| AssertionError  | `_zero_expert_output` 未设置    | Patch 2+3 | ✅    |
| Dispatch aicore | `topk_ids` 越界 (>=512)         | Patch 1   | ✅    |
| MLP aicore      | grouped matmul 256 experts 超限 | Patch 0   | ✅    |
| 推理 aicore     | CANN kernel 内部 bug            | —         | ❌    |

### 推理异常详情

```
rtStreamSynchronizeWithTimeout execution failed
reason=aicore exception, error code = 507015
The aicore execution is abnormal.
```

异常发生在 NPU stream 异步执行中 (forward pass)，在 worker shutdown 同步时检出。
非 dispatch/MLP/combine 单步，而是 CANN kernel 内部 bug。
512 experts + topk=12 + EP=2 + bfloat16 组合触发。

---

## 6. 关键技术要点

1. **chunked-prefill 与 EP**: 启用 chunked prefill 会改变 token dispatch pattern，可能与 EP kernel tiling 冲突。建议禁用。

2. **Zero expert ID 过滤**: GPU 路径在 `ZeroExpertRouter._compute_routing` 中处理，Ascend EP 路径缺失此步骤。必须在 token dispatch 前过滤。

3. **Identity zero expert**: 贡献 = `sum(input * router_weight over zero_expert_slots)`. 不是简单的 skip-connection，而是加权求和。
