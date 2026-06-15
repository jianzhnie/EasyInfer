# Kimi-K2 MCore vLLM 适配详解

本文档详细解释 `pcl_model.py` 的适配过程、每个修改的原因及其与 HF 实现（`modeling_deepseek.py`）的关系。

---

## 1. 背景：为什么需要这个文件？

Kimi-K2 的 HF 实现基于 DeepSeek V3 架构（`modeling_deepseek.py`），但有两个关键差异：

1. **注意力机制不同**：Kimi-K2 用标准 GQA（`q_proj/k_proj/v_proj` 分离），而 vLLM 内置的 `DeepseekV2Attention` 支持 MLA 和 GQA 两种路径
2. **Config 字段不同**：Kimi-K2 的 HF config 使用 `num_query_groups` 而非标准的 `num_key_value_heads`，还有 `kv_channels` 指定 head_dim

这个文件的作用是：**让 vLLM 能正确加载和运行 Kimi-K2 模型**，通过在 vLLM 的 DeepSeek V3 栈上做针对性适配。

---

## 2. 文件结构总览

```
pcl_model.py
├── _OPTIONAL_MISSING_BIAS_SUFFIXES         # 可选缺失 bias 后缀列表
├── _QKLayerNormNoBias                      # 无 bias 的 LayerNorm
├── _build_rope_parameters_from_hf_config() # HF config → vLLM rope_parameters 转换
├── _build_qk_norm_layer()                  # 构建 q/k norm 层
├── _prepare_kimi_k2_mcore_hf_config()      # Config 预处理
├── PCLAttention                    # 注意力模块
├── PCLDecoderLayer                 # 解码器层
├── PCLModel                        # 模型主体
└── PCLForCausalLM                  # 运行时入口
```

配合 `registry.py` 注册到 vLLM：

```python
# registry.py
@register_patch("vllm.model_executor.models.registry")
def patch_model_registry(module):
    module.ModelRegistry.register_model(
        "PCLForCausalLM",
        "easyinfer.plugins.vllm.model_executor.models.kimi_k2_mcore:PCLForCausalLM",
    )
```

---

## 3. 逐层适配分析

### 3.1 Config 预处理 — `_prepare_kimi_k2_mcore_hf_config`（85-112 行）

**做了什么**：在模型初始化前修改 HF config 对象。

**为什么需要**：

| 操作 | 原因 |
|------|------|
| `num_key_value_heads = num_query_groups` | Kimi config 用 `num_query_groups=2` 定义 KV 头数，但 vLLM 的 DeepSeek 栈读的是 `num_key_value_heads`。不映射的话 vLLM 会用错误的 KV 头数 |
| `q_lora_rank=None` | 这 5 个字段是 MLA（Multi-head Latent Attention）专用的。Kimi 用 GQA 不用 MLA，必须将它们清零/置 None，**强制 vLLM 走 GQA 路径**而非 MLA 路径 |
| `kv_lora_rank=0` | 同上 |
| `qk_nope_head_dim=0` | 同上，MLA 中表示非 RoPE 部分的维度 |
| `qk_rope_head_dim=0` | 同上，MLA 中表示 RoPE 部分的维度 |
| `v_head_dim=0` | 同上 |
| 构建 `rope_parameters` | vLLM 的 `get_rope()` 需要 `rope_parameters` 字典，而 HF config 把这些信息散落在 `rope_theta`、`rope_scaling` 等独立字段中。`_build_rope_parameters_from_hf_config` 把它们收集到一个字典里 |

**源码对照**：

```python
def _prepare_kimi_k2_mcore_hf_config(hf_config: Any) -> None:
    """Normalize HF config to Kimi-K2-MCore GQA behavior."""
    # 1. 映射 KV 头数
    num_query_groups = getattr(hf_config, "num_query_groups", None)
    if num_query_groups is not None:
        hf_config.num_key_value_heads = int(num_query_groups)

    # 2. 清零 MLA 字段，强制走 GQA
    for attr, value in (
        ("q_lora_rank", None),
        ("kv_lora_rank", 0),
        ("qk_nope_head_dim", 0),
        ("qk_rope_head_dim", 0),
        ("v_head_dim", 0),
    ):
        setattr(hf_config, attr, value)

    # 3. 构建 vLLM 所需的 rope_parameters
    if not hasattr(hf_config, "rope_parameters") or not getattr(hf_config, "rope_parameters"):
        rope_params = _build_rope_parameters_from_hf_config(hf_config)
        if rope_params:
            hf_config.rope_parameters = rope_params
```

**对应的 HF config（`config.json`）**：

```json
{
    "num_attention_heads": 64,
    "num_query_groups": 2,
    "kv_channels": 128,
    "qk_layernorm": true,
    "rope_theta": 50000.0,
    "rope_scaling": {
        "type": "yarn",
        "factor": 32.0,
        "mscale": 1.0,
        "mscale_all_dim": 1.0,
        "original_max_position_embeddings": 4096,
        "beta_fast": 1.0,
        "beta_slow": 1.0
    }
}
```

---

### 3.2 Attention — `PCLAttention`（115-234 行）

**为什么不能直接用 vLLM 的 `DeepseekV2Attention`**：

vLLM 的 `DeepseekV2Attention` 内部会根据 `q_lora_rank` 是否为 0 来决定走 MLA 还是 GQA。虽然上面已经把 MLA 字段清零了，但 `DeepseekV2Attention` 的 GQA 路径有几个问题：

1. **head_dim 来源不同**：vLLM DeepSeek 栈用 `hidden_size // num_heads` 算 head_dim，而 Kimi config 显式定义了 `kv_channels=128`
2. **KV 头数来源不同**：vLLM 读 `num_key_value_heads`，Kimi 用 `num_query_groups`
3. **q/k layernorm 类型不同**：Kimi 需要 LayerNorm（带 bias），vLLM 的 DeepSeek 栈用 RMSNorm
4. **softmax scale 缺少 YaRN mscale**：Kimi 用 YaRN RoPE，需要额外的 mscale 调整

#### 3.2.1 头维度计算（134-143 行）

```python
self.total_num_kv_heads = int(
    getattr(config, "num_query_groups",          # Kimi 特有字段，优先读
        getattr(config, "num_key_value_heads", num_heads))  # 兜底
)
self.head_dim = int(
    getattr(config, "kv_channels",               # Kimi 特有字段，=128
        hidden_size // self.total_num_heads)      # 兜底
)
```

Kimi config 值：`num_attention_heads=64, num_query_groups=2, kv_channels=128`

- `total_num_heads = 64`
- `total_num_kv_heads = 2`（从 `num_query_groups` 映射）
- `head_dim = 128`（由 config 显式指定）

**HF 对照**（`modeling_deepseek.py:750-761`）：

```python
self.num_heads = config.num_attention_heads          # 64
self.num_query_groups = config.num_query_groups      # 2
self.head_dim = config.kv_channels                   # 128
self.num_key_value_groups = self.num_heads // self.num_query_groups  # 32
```

两者数学一致。

#### 3.2.2 张量并行分片（145-152 行）

```python
tp_size = get_tensor_model_parallel_world_size()
self.num_heads = self.total_num_heads // tp_size           # 每个 TP rank 的 Q 头数
self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)  # 每个 TP rank 的 KV 头数
```

64 个 Q 头和 2 个 KV 头按 TP size 分片。这是 vLLM 特有的，HF 实现不涉及 TP。

#### 3.2.3 Softmax Scale + YaRN mscale（156-167 行）

```python
self.scaling = self.head_dim**-0.5  # 基础 scale

# YaRN mscale
rope_scaling = getattr(config, "rope_scaling", None)
if rope_scaling is not None:
    mscale_all_dim = rope_scaling.get("mscale_all_dim", 0)
    scaling_factor = rope_scaling.get("factor", 1.0)
    if mscale_all_dim and scaling_factor > 1.0:
        mscale = 0.1 * mscale_all_dim * math.log(scaling_factor) + 1.0
        self.scaling = self.scaling * mscale * mscale
```

**HF 对照**（`modeling_deepseek.py:807-813`）：

```python
self.softmax_scale = self.head_dim**(-0.5)
if self.config.rope_scaling is not None:
    mscale_all_dim = self.config.rope_scaling.get('mscale_all_dim', 0)
    scaling_factor = self.config.rope_scaling['factor']
    if mscale_all_dim:
        mscale = yarn_get_mscale(scaling_factor, mscale_all_dim)
        self.softmax_scale = self.softmax_scale * mscale * mscale
```

Kimi config `factor=32.0, mscale_all_dim=1.0`：

```
yarn_get_mscale(32.0, 1) = 0.1 * 1 * ln(32) + 1.0 ≈ 1.347
scaling = 128^(-0.5) × 1.347² ≈ 0.1603
```

#### 3.2.4 QKV 投影融合（169-177 行）

```python
self.qkv_proj = QKVParallelLinear(
    hidden_size,       # 7168
    self.head_dim,     # 128
    self.total_num_heads,      # 64
    self.total_num_kv_heads,   # 2
    bias=False,
    quant_config=quant_config,
    prefix=f"{prefix}.qkv_proj",
)
```

**HF 对照**（`modeling_deepseek.py:770-784`）：

```python
self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)     # [7168, 8192]
self.k_proj = nn.Linear(hidden_size, num_query_groups * head_dim, bias=False)  # [7168, 256]
self.v_proj = nn.Linear(hidden_size, num_query_groups * head_dim, bias=False)  # [7168, 256]
```

vLLM 把 3 个独立 `nn.Linear` 融合成 1 个 `QKVParallelLinear`。输出尺寸 `[batch, seq, 8192+256+256=8704]`，然后 split 为 `[q_size, kv_size, kv_size]`。数学等价，但支持 TP 自动分片。

#### 3.2.5 Q/K Layernorm（206-211 行）

```python
if getattr(config, "qk_layernorm", False):    # Kimi config 中 = True
    self.q_layernorm = _build_qk_norm_layer(self.head_dim, eps, config)
    self.k_layernorm = _build_qk_norm_layer(self.head_dim, eps, config)
```

**HF 对照**（`modeling_deepseek.py:795-802`）：

```python
if getattr(config, 'qk_layernorm', False):
    self.q_layernorm = DeepseekV3LayerNorm(self.head_dim, eps=config.rms_norm_eps)
    self.k_layernorm = DeepseekV3LayerNorm(self.head_dim, eps=config.rms_norm_eps)
```

差异：HF 用 `DeepseekV3LayerNorm`（有 weight + bias），vLLM 用 `_QKLayerNormNoBias`（只有 weight）。checkpoint 中的 bias 权重会被丢弃。

#### 3.2.6 Forward 流程（213-234 行）

```
hidden_states
    → QKV 投影 (fused)
    → split 为 [q, k, v]
    → q/k layernorm (per-head normalization)
    → RoPE (YaRN)
    → Attention (GQA)
    → O 投影
```

**HF 对照**（`modeling_deepseek.py:860-956`）：

```
hidden_states
    → q_proj / k_proj / v_proj (separate)
    → reshape 为 [batch, seq, heads, head_dim]
    → q/k layernorm
    → transpose → RoPE → KV cache update
    → repeat_kv (GQA expand)
    → matmul attention → O 投影
```

数学等价，差异仅在实现方式（fused vs separate、vLLM PagedAttention vs 标准 matmul）。

---

### 3.3 DecoderLayer — `PCLDecoderLayer`（237-330 行）

**为什么继承 `DeepseekV2DecoderLayer` 又全部重写**：

继承是为了复用 vLLM 的类型注册机制，但 `__init__` 和 `forward` 都重写，原因是：

1. **Attention 必须替换为 `PCLAttention`**
2. **MoE/Dense 选择逻辑**需要与 HF 一致
3. **`routed_scaling_factor` 的 FP16 数值稳定处理**需要适配

#### 3.3.1 MoE/Dense 层选择（273-291 行）

```python
if (config.n_routed_experts is not None
    and layer_idx >= config.first_k_dense_replace   # = 2
    and layer_idx % moe_layer_freq == 0):            # = 1
    self.mlp = DeepseekV2MoE(...)      # MoE 层：128 routed + 1 shared experts
else:
    self.mlp = DeepseekV2MLP(...)      # Dense 层：标准 MLP
```

**HF 对照**（`modeling_deepseek.py:1230-1234`）：

```python
self.mlp = (DeepseekV3MoE(config) if
            (config.n_routed_experts is not None
             and layer_idx >= config.first_k_dense_replace
             and layer_idx % config.moe_layer_freq == 0) else
            DeepseekV3MLP(config))
```

逻辑完全一致。Kimi config `first_k_dense_replace=2, moe_layer_freq=1, n_routed_experts=128`，所以：
- 层 0~1：Dense MLP
- 层 2~31：MoE

#### 3.3.2 Forward 流程（300-330 行）

```python
def forward(self, positions, hidden_states, residual, llama_4_scaling=None):
    # 1. Input LayerNorm (fused with residual)
    if residual is None:
        residual = hidden_states.clone()
        hidden_states = self.input_layernorm(hidden_states)
    else:
        hidden_states, residual = self.input_layernorm(hidden_states, residual)

    # 2. Self Attention
    hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)

    # 3. FP16 数值稳定缩放（仅 FP16 时生效）
    if hidden_states.dtype == torch.float16:
        hidden_states *= 1.0 / self.routed_scaling_factor
        if self.layer_idx == 0:
            residual *= 1.0 / self.routed_scaling_factor

    # 4. Post-attention LayerNorm (fused with residual)
    hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)

    # 5. MLP (MoE or Dense)
    hidden_states = self.mlp(hidden_states)

    # 6. Dense MLP 的 FP16 缩放回乘
    if isinstance(self.mlp, DeepseekV2MLP) and hidden_states.dtype == torch.float16:
        hidden_states *= self.routed_scaling_factor

    return hidden_states, residual
```

**HF 对照**（`modeling_deepseek.py:1269-1296`）：

```python
def forward(self, hidden_states, attention_mask, position_ids, ...):
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)

    hidden_states, _, _ = self.self_attn(hidden_states, attention_mask, position_ids, ...)
    hidden_states = residual + hidden_states

    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)

    if isinstance(self.mlp, DeepseekV3MoE):
        hidden_states, aux_loss = self.mlp(hidden_states)
    else:
        hidden_states = self.mlp(hidden_states)

    hidden_states = residual + hidden_states
```

**关键差异**：

- vLLM 用 fused LayerNorm（`self.input_layernorm(x, residual)` 同时做 norm 和残差加），HF 分别操作
- vLLM 有 FP16 的 `routed_scaling_factor` 缩放，HF 没有（HF 在 MoE Gate 内部处理 scaling）
- vLLM 不返回 `aux_loss`（推理时不需要）

#### 3.3.3 注意：第 270 行的 bug

```python
self.self_attn = PCLAttention(
    config=config,
    hidden_size=self.hidden_size,
    num_heads=config.num_attention_heads,
    ...
    topk_indices_buffer=topk_indices_buffer,  # ← 多余参数
)
```

`PCLAttention.__init__` 的 `**_: Any` 会吞掉 `topk_indices_buffer`。这是一个**无害的 bug**——参数被忽略不影响功能。这个 buffer 是给 DeepSeek V3.2 的 `index_topk` 用的，Kimi-K2 不需要。

---

### 3.4 Model — `PCLModel`（333-386 行）

**为什么继承 `DeepseekV2Model` 又重写 `__init__`**：

```python
class PCLModel(DeepseekV2Model):
    def __init__(self, *, vllm_config, prefix=""):
        nn.Module.__init__(self)  # 跳过父类 __init__
```

关键改动是 `make_layers` 中用 `PCLDecoderLayer` 替代默认的 `DeepseekV2DecoderLayer`：

```python
self.layers = make_layers(
    config.num_hidden_layers,
    lambda prefix: PCLDecoderLayer(vllm_config, prefix, topk_indices_buffer=...),
)
```

**结构对照**：

| 组件 | HF `DeepseekV3Model` | vLLM `PCLModel` |
|------|---------------------|-------------------------|
| Embedding | `nn.Embedding` | `VocabParallelEmbedding` |
| Layers | `nn.ModuleList` | `make_layers` (支持 PP) |
| DecoderLayer | `DeepseekV3DecoderLayer` | `PCLDecoderLayer` |
| Final Norm | `DeepseekV3RMSNorm` | `deepseek_v2.RMSNorm` |

其余部分（`embed_tokens`、`norm`、PP 分片、`is_v32`/`topk_indices_buffer`）与父类一致。

`aux_hidden_state_layers = ()`（386 行）：显式清空，表示不需要辅助隐状态（某些 DeepSeek 变体需要）。

---

### 3.5 ForCausalLM — `PCLForCausalLM`（389-418 行）

**这是整个适配的入口**。

#### 3.5.1 初始化时预处理 config（394-396 行）

```python
def __init__(self, *, vllm_config, prefix=""):
    _prepare_kimi_k2_mcore_hf_config(vllm_config.model_config.hf_config)  # 先改 config
    super().__init__(vllm_config=vllm_config, prefix=prefix)                # 再初始化
```

**顺序很关键**：必须先改 config，再调 `super().__init__()`。因为父类初始化时会根据 config 创建模型结构。

#### 3.5.2 权重加载后处理（398-418 行）

```python
_OPTIONAL_MISSING_BIAS_SUFFIXES = (
    ".self_attn.q_layernorm.bias",   # q/k layernorm 的 bias
    ".self_attn.k_layernorm.bias",
    ".mlp.gate.bias",                # MoE gate 的 bias
)

def load_weights(self, weights):
    loaded = super().load_weights(weights)

    # 找出模型有但 checkpoint 没提供的 bias 参数
    params_dict = dict(self.named_parameters())
    optional_missing = {
        name for name in params_dict
        if name.endswith(_OPTIONAL_MISSING_BIAS_SUFFIXES) and name not in loaded
    }
    # 将缺失的 bias zero-init
    if optional_missing:
        with torch.no_grad():
            for name in optional_missing:
                params_dict[name].zero_()
        loaded.update(optional_missing)
    return loaded
```

**为什么需要**：MCore 转换后的 checkpoint 可能缺少这些 bias（原始 MCore 模型可能没有这些参数）。如果模型定义了参数但 checkpoint 没提供，vLLM 会报错。所以检测并 zero-init。

---

## 4. 适配总结

| 适配点 | 原因 | 对应代码位置 |
|--------|------|-------------|
| Config 字段映射 | Kimi 用 `num_query_groups` / `kv_channels`，vLLM 用 `num_key_value_heads` / `hidden_size//num_heads` | `_prepare_kimi_k2_mcore_hf_config`（85-112 行） |
| 强制 GQA 路径 | vLLM DeepSeek 栈同时支持 MLA 和 GQA，Kimi 只用 GQA，需清零 MLA 字段 | 置 `q_lora_rank=None` 等（97-105 行） |
| 重写 Attention | head_dim 来源、KV 头数来源、q/k norm 类型、softmax scale 都不同 | `PCLAttention`（115-234 行） |
| 重写 DecoderLayer | 要用自定义 Attention + MoE 选择逻辑 + FP16 缩放 | `PCLDecoderLayer`（237-330 行） |
| 重写 Model | 要用自定义 DecoderLayer | `PCLModel`（333-386 行） |
| 入口类 + 权重后处理 | Config 预处理时机 + 可选 bias 容错 | `PCLForCausalLM`（389-418 行） |
| 注册到 vLLM | 让 vLLM 识别 `architectures=["PCLForCausalLM"]` | `registry.py` |

---

## 5. 已知对齐问题

| 问题 | HF 实现 | vLLM 实现 | 影响 |
|------|---------|-----------|------|
| Q/K Norm bias | `DeepseekV3LayerNorm`（有 weight + bias） | `_QKLayerNormNoBias`（只有 weight） | checkpoint 中的 bias 被丢弃，影响注意力精度 |
| YaRN mscale | `softmax_scale = head_dim^(-0.5) * mscale²` | 已修复（`scaling` 中补上 mscale） | 已修复 ✓ |
| `routed_scaling_factor` 处理 | 在 MoE Gate 内部对 `topk_weight` 缩放 | 在 Decoder 层级对 hidden_states 做 FP16 缩放 | 仅 FP16 受影响，BF16 等价 |
| `topk_indices_buffer` 参数 | 无 | 传给了 Attention 但被 `**_` 吞掉 | 无害 bug，不影响功能 |
