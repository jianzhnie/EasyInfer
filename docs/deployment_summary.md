# 4 模型部署经验总结

本文档总结在华为昇腾 NPU 集群（8 节点 × 8 NPU）上部署 4 个大语言模型的完整经验和关键技术决策。

## 部署概览

| 模型 | 架构 | 量化 | 专家数 | MTP | 多模态 | 端口 | Tool Parser |
|------|------|------|--------|-----|--------|------|-------------|
| DeepSeek-V4-Flash | DeepseekV4 MoE + MLA | W8A8 | 256 | ✓ | ✗ | 8000 | deepseek_v3 |
| GLM-5 | GlmMoeDSA | W4A8 | 256 | ✓ | ✗ | 8001 | glm47 |
| GLM-5.1 | GlmMoeDSA | W4A8 | 256 | ✓ | ✗ | 8002 | glm47 |
| Kimi-K2.6 | KimiK25 (DeepSeekV3 + ViT) | W4A8 | 384 | ✗ | ✓ | 8003 | deepseek_v3 |

## 架构分析

### 1. DeepSeek-V4-Flash-w8a8-mtp

```
config.json 关键参数:
├── architectures: DeepseekV4ForCausalLM
├── model_type: deepseek_v4
├── hidden_size: 4096          ← 相比 DeepSeek V3(7168) 更精简
├── num_hidden_layers: 43      ← 比 DeepSeek V3(61) 更少
├── n_routed_experts: 256
├── num_experts_per_tok: 6     ← 比其他 MoE 模型(8)少，更高效
├── num_nextn_predict_layers: 1 ← MTP 支持
├── max_position_embeddings: 1,048,576 ← 1M 上下文！
├── num_key_value_heads: 1     ← 极限 GQA，大幅节省 KV Cache
├── head_dim: 512              ← 比其他模型(128-256)大
└── q_lora_rank: 1024          ← MLA (Multi-head Latent Attention)
```

**关键发现**:
- DeepSeek V4 Flash 通过减少 hidden_size (4096)、层数 (43) 和 KV heads (1) 实现了极致效率
- 尽管如此，仍保持 256 专家和 1M 上下文窗口
- W8A8 量化比 W4A8 精度更高，但显存占用约 2x
- head_dim=512 是独特的，大多数模型使用 64-256
- ⚠️ **vLLM-Ascend 0.18.0rc1 不支持此架构**（见下方部署结果）

### 2. GLM-5-w4a8 / GLM-5.1-w4a8

```
config.json 关键参数 (两个模型几乎相同):
├── architectures: GlmMoeDsaForCausalLM
├── model_type: glm_moe_dsa
├── hidden_size: 6144
├── num_hidden_layers: 78       ← 最深的模型
├── n_routed_experts: 256
├── num_experts_per_tok: 8
├── num_nextn_predict_layers: 1  ← MTP 支持
├── max_position_embeddings: 202,752
├── q_lora_rank: 2048           ← MLA
├── kv_lora_rank: 512
└── vocab_size: 154,880
```

**关键发现**:
- GLM-5 和 GLM-5.1 配置完全相同（架构一致，仅训练数据不同）
- 78 层是 4 个模型中最深的，但 hidden_size 中等 (6144)
- W4A8 量化下显存占用相对较小，8 卡 A2 即可部署
- DSA (DeepSeek-style Attention) 使 GLM 也具有 MLA 的优势

### 3. Kimi-K2.6-w4a8

```
config.json 关键参数:
├── architectures: KimiK25ForConditionalGeneration (外层包装)
├── text_config:
│   ├── architectures: DeepseekV3ForCausalLM
│   ├── model_type: kimi_k2
│   ├── hidden_size: 7168       ← 最大的 hidden_size
│   ├── num_hidden_layers: 61
│   ├── n_routed_experts: 384   ← 最多的专家
│   ├── num_experts_per_tok: 8
│   ├── num_nextn_predict_layers: 0  ← 无 MTP
│   ├── max_position_embeddings: 262,144
│   ├── q_lora_rank: 1536
│   └── kv_lora_rank: 512
├── vision_config:               ← 多模态！
│   ├── mm_hidden_size: 1152
│   ├── vt_num_hidden_layers: 27
│   └── patch_size: 14
└── use_unified_vision_chunk: true
```

**关键发现**:
- 唯一的多模态模型，Vision Transformer 有 27 层
- 384 专家是最多的（Kimi-K2 系列特征），需要 EP 能整除 384
- 无 MTP 支持、无投机解码
- 基于 DeepSeek V3 架构，但带有自定义包装 (KimiK25)
- `auto_map` 需要 `--trust-remote-code`
- hidden_size=7168 是最大的，意味着单层参数量最大

## 并行策略设计

### 通用原则

在 8 节点 × 8 NPU = 64 NPU 环境下：

1. **TP (张量并行)**: 默认 8，填满单节点 8 张 NPU
2. **PP (流水线并行)**: 默认 1 (单节点)，多节点时按节点数增加
3. **EP (专家并行)**: MoE 模型必须启用，默认 EP=TP×PP，需能整除专家数
4. **DP (数据并行)**: 可选，用于提升吞吐

### 各模型推荐配置

```
模型                   节点数   TP   PP   EP   DP   适用场景       状态
──────────────────────────────────────────────────────────────────────
DeepSeek-V4-Flash W8A8   1      8    1    8    1    低延迟         ❌ 架构不支持
                         2     16    1    8    1    均衡           ❌ (待升级 vLLM)
                         8     64    1    8    1    长上下文       ❌

GLM-5/5.1 W4A8           1      8    1    8    1    低延迟         ✅ 已验证
                         2     16    1    8    1    多节点大TP     ✅ 已验证 (TP=16)
                         4     32    1    8    1    大TP           ⚠️ 未测试

Kimi-K2.6 W4A8           1      8    1    8    1    低延迟         ⚠️ 未测试
                         2      8    2    8    1    均衡           ✅ 已验证 (PP=2)
                         8      8    8    8    1    长上下文       ⚠️ 未测试
```

> **重要**: GLM-5/5.1 **不支持 Pipeline Parallelism (PP>1)**。多节点部署时必须使用更大的 TP 值
> (如 2 节点 TP=16, 4 节点 TP=32)，而非 PP。
> Kimi-K2.6 是 4 个模型中唯一支持 PP 的。

### EP 与专家数整除性

| 模型 | 专家数 | 推荐 EP 值 |
|------|--------|-----------|
| DeepSeek-V4-Flash | 256 | 8, 16, 32, 64 |
| GLM-5/5.1 | 256 | 8, 16, 32, 64 |
| Kimi-K2.6 | 384 | 8, 12, 16, 24, 32, 48, 64 |

## 量化策略对比

| 量化 | 权重精度 | 激活精度 | 显存节省 | 精度损失 | 适用模型 |
|------|---------|---------|---------|---------|---------|
| W4A8 | 4-bit | 8-bit | ~75% | 极小 | GLM-5, GLM-5.1, Kimi-K2.6 |
| W8A8 | 8-bit | 8-bit | ~50% | 极微 | DeepSeek-V4-Flash |
| BF16 | 16-bit | 16-bit | 0% | 无 | 所有模型 (需更大显存) |

- **W4A8**: 适合大多数场景，显存节省显著，精度接近 BF16
- **W8A8**: DeepSeek-V4-Flash 使用此量化，精度更高但需要更多显存
- 所有量化模型都必须设置 `QUANTIZATION=ascend` (vLLM-Ascend 专用)

## 投机解码 (MTP) 配置

| 模型 | MTP 支持 | SPECULATIVE_METHOD | 推荐 tokens |
|------|---------|-------------------|-------------|
| DeepSeek-V4-Flash | ✓ | deepseek_mtp | 3 |
| GLM-5 | ✓ | deepseek_mtp | 3 |
| GLM-5.1 | ✓ | deepseek_mtp | 3 |
| Kimi-K2.6 | ✗ | (不配置) | N/A |

判断依据: config.json 中 `num_nextn_predict_layers > 0` 则支持 MTP。

## Tool Parser 选择

Tool parser 的选择取决于模型系列：

| 模型系列 | Tool Parser | 原因 |
|---------|------------|------|
| DeepSeek V3/V4 | `deepseek_v3` | DeepSeek 系列使用自己的 tool call 格式 |
| GLM-4/5 | `glm47` | GLM 系列使用 glm47 格式 |
| Kimi-K2.6 | `deepseek_v3` | 基于 DeepSeek V3 骨干，使用 DeepSeek 格式 |
| Qwen | `hermes` | Qwen 系列使用 Hermes 格式 |

## 华为 NPU 特有配置

所有模型在华为昇腾 NPU 上都需要以下核心环境变量：

```bash
# 必须配置
export HCCL_OP_EXPANSION_MODE="AIV"        # HCCL 操作优化
export OMP_PROC_BIND="false"               # 禁用线程绑定
export OMP_NUM_THREADS="1"                  # 减少 CPU 线程
export HCCL_BUFFSIZE="200"                  # HCCL 缓冲区 (MB)
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"  # 内存分配
export VLLM_ASCEND_BALANCE_SCHEDULING="1"   # 负载均衡

# 量化模型推荐 (W4A8/W8A8)
export QUANTIZATION="ascend"               # vLLM-Ascend 专用量化标识
export ENABLE_ASYNC_SCHEDULING="1"         # 异步调度
export ENFORCE_EAGER="1"                   # 禁用 CUDA Graph

# NPU 编译优化
export CUDAGRAPH_MODE="FULL_DECODE_ONLY"
export ENABLE_NPUGRAPH_EX="true"
export FUSE_MULS_ADD="true"
export MULTISTREAM_OVERLAP_SHARED_EXPERT="true"
```

## 部署脚本设计模式

所有模型部署脚本遵循统一模式：

```bash
#!/bin/bash
set -eo pipefail  # 注意: 不用 set -u，CANN set_env.sh 与 nounset 不兼容

# 1. CANN 环境加载 (必须在其他操作之前)
set +u  # CANN 脚本引用未定义变量
if [[ -f "/usr/local/Ascend/cann/set_env.sh" ]]; then
    source /usr/local/Ascend/cann/set_env.sh
fi
if [[ -f "/usr/local/Ascend/nnal/atb/set_env.sh" ]]; then
    source /usr/local/Ascend/nnal/atb/set_env.sh
fi
set -u

# 2. 路径推导
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLLM_SCRIPT="${SCRIPT_DIR}/../../scripts/vllm/vllm_model_server.sh"

# 2. 模型路径 (环境变量覆盖)
export MODEL_PATH="${MODEL_PATH:-默认路径}"

# 3. 华为 NPU 环境变量
export HCCL_OP_EXPANSION_MODE="${HCCL_OP_EXPANSION_MODE:-AIV}"
# ... 其他 NPU 变量

# 4. 并行配置
export TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-8}"
export PIPELINE_PARALLEL_SIZE="${PIPELINE_PARALLEL_SIZE:-1}"
export ENABLE_EXPERT_PARALLEL="${ENABLE_EXPERT_PARALLEL:-1}"

# 5. 量化与内存
export DTYPE="${DTYPE:-bfloat16}"
export QUANTIZATION="${QUANTIZATION:-ascend}"

# 6. 序列调度 (根据硬件自动调整)
if [[ -z "${MAX_MODEL_LEN:-}" ]]; then
    if [[ "${TENSOR_PARALLEL_SIZE:-8}" -ge 16 ]]; then
        export MAX_MODEL_LEN=大值
    else
        export MAX_MODEL_LEN=小值
    fi
fi

# 7. 投机解码 (MTP 模型)
export SPECULATIVE_METHOD="deepseek_mtp"

# 8. NPU 编译优化
export CUDAGRAPH_MODE="FULL_DECODE_ONLY"

# 9. 工具调用
export ENABLE_TOOL_CALLING="1"
export TOOL_CALL_PARSER="模型对应的parser"

# 10. 构建 EXTRA_ARGS 并启动
EXTRA_ARGS=(--seed 1024 --trust-remote-code)
exec bash "$VLLM_SCRIPT" "${EXTRA_ARGS[@]}" "$@"
```

## 常见陷阱与解决方案

### 1. `--trust-remote-code` 必须启用

所有 4 个模型都包含自定义模型代码，不启用此参数会导致加载失败。
尤其是 Kimi-K2.6 的 `auto_map` 配置了自定义 `configuration_kimi_k25.py`。

### 2. QUANTIZATION 必须设为 "ascend"

这是 vLLM-Ascend 后端的专用量化标识。如果设为 "fp8" (vllm_model_server.sh 的默认值)，
会导致量化方式不匹配。所有 W4A8/W8A8 模型都必须覆盖为 "ascend"。

### 3. MTP 配置条件化

只有 `num_nextn_predict_layers > 0` 的模型才配置 MTP 投机解码。
Kimi-K2.6 的此值为 0，配置 MTP 会导致错误。

### 4. 量化与 MLAPO 的关系

- W4A8: 不需要 MLAPO
- W8A8: 可能需要 MLAPO (如 GLM-5 W8A8)，但 DeepSeek V4 Flash W8A8 暂不需要
- 判断标准: 查阅对应模型的 vLLM-Ascend 官方文档

### 5. EP_SIZE 必须整除专家数

MoE 模型的专家并行大小必须能整除专家总数，否则会报错：
- 256 专家: EP 可选 8, 16, 32, 64
- 384 专家: EP 可选 8, 12, 16, 24, 32, 48, 64

### 6. 端口冲突

同时运行多个模型时，确保各模型使用不同端口。默认分配：
- 8000: DeepSeek-V4-Flash
- 8001: GLM-5
- 8002: GLM-5.1
- 8003: Kimi-K2.6

## 验证清单

每个模型部署后，按以下清单逐项验证：

- [ ] 容器正常运行 (`docker ps`)
- [ ] Ray 集群健康 (`ray status`)
- [ ] vLLM 服务进程存在 (`ps aux | grep vllm`)
- [ ] `/v1/models` 端点可达
- [ ] 非流式 Chat Completion 正常返回
- [ ] 流式 Chat Completion 正常输出
- [ ] Tool Calling 功能正常 (或正确降级)
- [ ] 日志无异常错误

## 文件清单

```
examples/
├── deepseek_v4_flash/
│   ├── vllm_server.sh      ← 包装器部署 (❌ 架构不兼容)
│   ├── run_vllm.sh          ← 直接 vllm serve (❌)
│   ├── curl_test.sh         ← API 测试
│   └── README.md            ← 部署文档
├── glm5_w4a8/
│   ├── vllm_server.sh      ← 包装器部署
│   ├── run_vllm.sh          ← 直接 vllm serve (✅ TP=8 PP=1)
│   ├── curl_test.sh         ← API 测试
│   └── README.md            ← 部署文档
├── glm5_1_w4a8/
│   ├── vllm_server.sh      ← 包装器部署
│   ├── run_vllm.sh          ← 直接 vllm serve (✅ TP=8 PP=1)
│   ├── curl_test.sh         ← API 测试
│   └── README.md            ← 部署文档
└── kimi_k2_6_w4a8/
    ├── vllm_server.sh      ← 包装器部署
    ├── run_vllm.sh          ← 直接 vllm serve (✅ TP=8 PP=2)
    ├── curl_test.sh         ← API 测试
    └── README.md            ← 部署文档
```

## 待验证事项

以下项目已确认或仍需验证：

| # | 事项 | 状态 | 说明 |
|---|------|------|------|
| 1 | DeepSeek-V4-Flash 架构兼容性 | ❌ 确认不兼容 | vLLM-Ascend 0.18.0rc1 不支持 |
| 2 | GLM-5.1 与 GLM-5 配置兼容 | ✅ 确认 | 完全相同架构，部署参数通用 |
| 3 | Kimi-K2.6 多模态加载 | ⚠️ 待验证 | 文本推理正常，视觉功能未测试 |
| 4 | GLM 不支持 PP | ✅ 确认 | 使用大 TP 替代 PP 可行 |
| 5 | MTP 加速效果 (GLM-5/5.1) | ⚠️ 待测量 | deepseek_mtp 正常工作 |
| 6 | 多节点 TP=16 跨节点通信 | ✅ 确认 | 2 节点 TP=16 通过 Ray 正常 |
| 7 | tokenizer 兼容性问题 | ✅ 确认 | GLM 需修复 tokenizer_config.json |

---

## 部署实操记录 (2026-06-09)

### 环境配置

| 项目 | 详情 |
|------|------|
| 集群 | 8 节点 × 8 NPU (Atlas 800 A2/A3) |
| 容器 | `npuslim-env` (image: ascend910c-cann8.5.1-torch2.9.0-vllm0.18.0) |
| CANN | 8.5.1, 路径: `/usr/local/Ascend/cann/` |
| Ray | 每个模型独立 2 节点 Ray 集群 |

### 已修复的 Bug

#### Bug 1: `common.sh` readonly 变量重复 source 错误

**现象**: 当 `common.sh` 被多次 source 时，报错 `RED: readonly variable`

**根因**: `common.sh` 第 9 行和第 13 行直接用 `readonly VAR=value` 声明，第二次 source 时变量已为只读导致失败。

**修复**: 改为条件声明：
```bash
if ! declare -p RED &>/dev/null 2>&1; then
    readonly RED='\033[0;31m' ...
fi
```

**影响文件**: `scripts/common.sh`

#### Bug 2: CANN 环境路径错误

**现象**: `libascend_hal.so: cannot open shared object file`

**根因**: `scripts/vllm/set_env.sh` 中引用 `/usr/local/Ascend/ascend-toolkit/set_env.sh`，但实际 CANN 安装在 `/usr/local/Ascend/cann/set_env.sh`。

**修复**: 
- 更新 `scripts/vllm/set_env.sh` 优先尝试 `/usr/local/Ascend/cann/set_env.sh`
- 创建 `run_vllm.sh` 简化部署脚本，通过 `set +u`/`set -u` 包裹 CANN source

**影响文件**: `scripts/vllm/set_env.sh`, 所有 `examples/*/run_vllm.sh`

#### Bug 3: CANN set_env.sh 与 bash `set -u` 不兼容

**现象**: `/usr/local/Ascend/cann/set_env.sh: line 31: CMAKE_PREFIX_PATH: unbound variable`

**根因**: CANN 的 `set_env.sh` 脚本引用了未设置的变量（如 `CMAKE_PREFIX_PATH`, `ZSH_VERSION`），与 `set -u` (nounset) 冲突。

**修复**: 在 source CANN 脚本前临时 `set +u`，source 后恢复 `set -u`

### 简化部署方案

为了解决包装器链 (`vllm_server.sh` → `vllm_model_server.sh`) 过于复杂的问题，为每个模型创建了 `run_vllm.sh` 脚本，直接调用 `vllm serve`：

```
examples/
├── deepseek_v4_flash/run_vllm.sh   # TP=8 PP=1, W8A8 (❌ 架构不兼容)
├── glm5_w4a8/run_vllm.sh           # TP=8 PP=1, MTP, W4A8 (✅ 已验证)
├── glm5_1_w4a8/run_vllm.sh         # TP=8 PP=1, MTP, W4A8 (✅ 已验证)
└── kimi_k2_6_w4a8/run_vllm.sh      # TP=8 PP=2, no MTP, W4A8 (✅ 已验证)
```

### 节点分区

| 模型 | Head 节点 | Worker 节点 | 端口 |
|------|----------|------------|------|
| DeepSeek-V4-Flash | 10.16.201.229 | 10.16.201.164 | 8000 |
| GLM-5-w4a8 | 10.16.201.40 | 10.16.201.163 | 8001 |
| GLM-5.1-w4a8 | 10.16.201.193 | 10.16.201.201 | 8002 |
| Kimi-K2.6-w4a8 | 10.16.201.153 | 10.16.201.124 | 8003 |

#### Bug 4: 新模型 type 不被 transformers 识别

**现象**: 所有模型报 `Value error, The checkpoint you are trying to load has model type 'deepseek_v4'/'glm_moe_dsa' but Transformers does not recognize this architecture`

**根因**: transformers 4.57.6 内置注册表不包含这些新架构的 model_type，模型目录中又缺少 `auto_map` 配置和自定义 Python 配置文件。

**修复**:
1. 为每个模型创建 minimal `configuration_<type>.py` 文件，定义 `PretrainedConfig` 子类
2. 在 `config.json` 中添加 `auto_map` 指向自定义配置类

```python
# configuration_glm_moe_dsa.py
from transformers import PretrainedConfig
class GlmMoeDsaConfig(PretrainedConfig):
    model_type = "glm_moe_dsa"
```

```json
// config.json 添加
"auto_map": {"AutoConfig": "configuration_glm_moe_dsa.GlmMoeDsaConfig"}
```

**影响文件**: 模型目录中的 `config.json` 和新建的 `configuration_*.py`

#### Bug 5: `--num-scheduler-steps` 参数不被支持

**现象**: `vllm: error: unrecognized arguments: --num-scheduler-steps`

**根因**: vLLM-Ascend 0.18.0rc1 版本不支持此参数。

**修复**: 从 `run_vllm.sh` 中移除 `--num-scheduler-steps`

#### Bug 6: Tool parser 命名错误

**现象**: `KeyError: 'invalid tool call parser: deepseekv3' (chose from { deepseek_v3, deepseek_v31, ... })`

**根因**: 正确的 parser 名称应为 `deepseek_v3`（下划线分隔），而非 `deepseekv3`

**修复**: `sed -i 's/deepseekv3/deepseek_v3/g'`

#### Bug 7: GLM tokenizer extra_special_tokens 类型错误

**现象**: `AttributeError: 'list' object has no attribute 'keys'`

**根因**: GLM-5/5.1 的 `tokenizer_config.json` 中 `extra_special_tokens` 是 list 类型，但 `PreTrainedTokenizerFast.__init__` 期望 dict 类型。

**修复**: 从 `tokenizer_config.json` 中移除 `extra_special_tokens`，并显式设置 `tokenizer_class = "PreTrainedTokenizerFast"`

#### Bug 8: GLM 不支持 Pipeline Parallelism

**现象**: `NotImplementedError: Pipeline parallelism is not supported for this model. Supported models implement the SupportsPP interface.`

**根因**: GLM-5/5.1 的 GlmMoeDsaForCausalLM 未实现 `SupportsPP` 接口。

**修复**: 使用大 TP 跨节点部署替代 PP（如 TP=16 替代 TP=8,PP=2），或单节点 TP=8。Kimi-K2.6 支持 PP，不受影响。

#### Bug 9: DeepSeek V4 Flash 架构不兼容 (未解决)

**现象**: `RuntimeError: Engine core initialization failed. Failed core proc(s): {}`，根因为 `AttributeError: 'DeepseekV4Config' object has no attribute 'kv_lora_rank'`

**根因**: vLLM-Ascend 0.18.0rc1 模型注册表中无 `DeepseekV4ForCausalLM`，且将 `architectures` 改为 `DeepseekV32ForCausalLM` 后，模型特定属性 (head_dim=512, q_lora_rank=1024) 与 DeepSeek V3/V32 Config 不兼容。

**状态**: ❌ 未解决。需要升级 vLLM-Ascend 到支持 DeepSeek V4 的版本。

#### Bug 10: `--speculative-config` with `deepseek_mtp` 不支持 (DeepSeek V4)

**现象**: `NotImplementedError: Unsupported speculative method: 'mtp'`

**根因**: vLLM-Ascend 0.18.0rc1 中 DeepSeek V4 架构不支持 `deepseek_mtp` 投机解码方法。

**修复**: 从 DeepSeek V4 的 `run_vllm.sh` 中移除 `--speculative-config`。GLM 模型的 `deepseek_mtp` 正常工作。

---

## 部署最终结果 (2026-06-09)

| 模型 | 状态 | 配置 | 耗时 | 备注 |
|------|------|------|------|------|
| **Kimi-K2.6** | ✅ 成功 | TP=8 PP=2 Ray 2节点 | ~20min | 唯一支持 PP 的模型 |
| **GLM-5** | ✅ 成功 | TP=16 PP=1 Ray 2节点 | ~15min | 修复 tokenizer/config 后 |
| **GLM-5.1** | ✅ 成功 | TP=16 PP=1 Ray 2节点 | ~15min | 修复 tokenizer/config 后 |
| **DeepSeek V4 Flash** | ❌ 失败 | TP=8 PP=1 | N/A | 架构不支持 (Bug 9) |

### API 验证结果

```
Kimi-K2.6: ✅ /v1/models  ✅ Chat  ✅ Streaming  ✅ Tool Calling
GLM-5:     ✅ /v1/models  ✅ Chat  ✅ Streaming  (tool call 待测)
GLM-5.1:   ✅ /v1/models  ✅ Chat  ✅ Streaming  (tool call 待测)
```
