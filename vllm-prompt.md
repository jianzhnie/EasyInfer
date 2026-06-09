# VLLM 模型部署和测试

## 环境概况

- **集群**: 8 节点 × 8 昇腾 NPU (Atlas 800 A2/A3, 每卡 64G)
- **框架**: vLLM-Ascend 0.18.0rc1 + Ray 分布式
- **容器**: `npuslim-env` (ascend910c-cann8.5.1-torch2.9.0-vllm0.18.0)
- **CANN**: 8.5.1, 路径 `/usr/local/Ascend/cann/`
- **挂载**: `/home/jianzhnie/llmtuner` → 容器内同路径

## 任务目标

为以下模型逐一完成部署脚本、测试脚本和文档。**严格按模型逐一处理**，完成一个再开始下一个。

| # | 模型路径 | 量化 | 架构 | 专家数 | MTP | 多模态 |
|---|---------|------|------|--------|-----|--------|
| 1 | `.../DeepSeek-V4-Flash-w8a8-mtp` | W8A8 | DeepseekV4 | 256 | ✓ | ✗ |
| 2 | `.../GLM-5-w4a8` | W4A8 | GlmMoeDSA | 256 | ✓ | ✗ |
| 3 | `.../GLM-5.1-w4a8` | W4A8 | GlmMoeDSA | 256 | ✓ | ✗ |
| 4 | `.../Kimi-K2.6-w4a8` | W4A8 | KimiK25 | 384 | ✗ | ✓ |

> 模型基路径: `/home/jianzhnie/llmtuner/hfhub/models/Eco-Tech/`

## 输出要求

对每个模型，在 `examples/<模型简称>/` 下生成 **4 个文件**:

```
examples/<model_dir>/
├── run_vllm.sh       ← 直接 vllm serve（推荐，绕过包装器链问题）
├── vllm_server.sh    ← 通过 vllm_model_server.sh 包装器部署
├── curl_test.sh      ← API 功能测试
└── README.md         ← 部署与测试文档
```

### 1. `run_vllm.sh` — 直接部署脚本（推荐）

这是首选部署方式，直接调用 `vllm serve`，避免包装器链的 `common.sh` readonly 冲突。

**必须遵循的模板:**

```bash
#!/bin/bash
# <ModelName> — 直接 vllm serve 部署
# 默认 TP=8 PP=1 (单节点); 多节点: TP=16 PP=1 (2节点, 当不支持 PP 时)
set -eo pipefail  # 注意: 不用 set -u, CANN 脚本与 nounset 不兼容

# CANN 环境加载 (必须在最前面)
set +u
if [[ -f "/usr/local/Ascend/cann/set_env.sh" ]]; then
    source /usr/local/Ascend/cann/set_env.sh
fi
if [[ -f "/usr/local/Ascend/nnal/atb/set_env.sh" ]]; then
    source /usr/local/Ascend/nnal/atb/set_env.sh
fi
set -u

MODEL_PATH="${MODEL_PATH:-<默认路径>}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-<端口>}"
TP="${TP:-8}"
PP="${PP:-<1 或 2>}"  # GLM 不支持 PP>1, Kimi 支持 PP=2

export HCCL_OP_EXPANSION_MODE="${HCCL_OP_EXPANSION_MODE:-AIV}"
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export HCCL_BUFFSIZE=200
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_ASCEND_BALANCE_SCHEDULING=1

echo "[INFO] Starting <ModelName>"
echo "[INFO] TP=$TP PP=$PP PORT=$PORT"

vllm serve "$MODEL_PATH" \
    --host "$HOST" \
    --port "$PORT" \
    --served-model-name <served_name> \
    --trust-remote-code \
    --dtype bfloat16 \
    --tensor-parallel-size "$TP" \
    --pipeline-parallel-size "$PP" \
    --distributed-executor-backend ray \
    --enable-expert-parallel \              # MoE 模型必须
    --quantization ascend \
    --gpu-memory-utilization <0.90-0.95> \
    --max-model-len <根据硬件调整> \
    --max-num-seqs <根据硬件调整> \
    --max-num-batched-tokens 8192 \
    --enable-chunked-prefill \
    --enable-prefix-caching \
    --enforce-eager \
    --speculative-config "{\"num_speculative_tokens\": 3, \"method\": \"deepseek_mtp\"}" \  # 仅 MTP 模型
    --enable-auto-tool-choice \
    --tool-call-parser <parser_name> \
    --seed 1024 \
    "$@"
```

**⚠️ 禁止使用的参数 (vLLM-Ascend 0.18.0rc1 不支持):**
- `--num-scheduler-steps` — 当前版本不支持
- `--tool-call-parser deepseekv3` — 正确写法是 `deepseek_v3` (下划线)

**关键参数速查:**

| 参数 | MoE 模型 | 非 MoE | 说明 |
|------|---------|--------|------|
| `--enable-expert-parallel` | 必须 | 不需要 | MoE 必需 |
| `--speculative-config` | MTP 模型 | 不需要 | `num_nextn_predict_layers > 0` 时使用 |
| `--tool-call-parser` | 见下表 | 见下表 | 关键: 使用下划线分隔 |
| `--quantization` | `ascend` | 不设置 | W4A8/W8A8 必须设为 `ascend` |

**Tool Parser 对照表 (关键: 下划线分隔):**

| 模型系列 | Parser |
|---------|--------|
| DeepSeek V3/V32/V4 | `deepseek_v3` |
| GLM-4/5 | `glm47` |
| Kimi-K2.6 | `deepseek_v3` |
| Qwen | `hermes` |

### 2. `vllm_server.sh` — 包装器部署脚本

通过 `scripts/vllm/vllm_model_server.sh` 包装器启动。该包装器自动检测支持的参数。

注意事项:
- 参考 `examples/glm5_server.sh` 的写法 (`exec bash "$VLLM_SCRIPT"`)
- 需要通过环境变量覆盖 `QUANTIZATION` (包装器默认 `fp8`, 需改为 `ascend`)
- `scripts/vllm/set_env.sh` 中的 CANN 路径可能已过期，需验证

### 3. `curl_test.sh` — API 功能测试

参考 `examples/curl_test.sh` 模板，修改:
- `BASE_URL` 默认端口 (8000/8001/8002/8003 避免冲突)
- `MODEL_NAME` 默认值
- source `common.sh` 的相对路径 (子目录多一层: `../../scripts/common.sh`)

测试覆盖: `/v1/models` → 非流式 Chat → 流式 Chat → Tool Calling

### 4. `README.md` — 部署文档

参考 `examples/GLM5_README.md` 格式，必须包含:
- 顶部部署验证状态 banner (✅/❌/⚠️)
- 模型架构摘要表
- 硬件要求 (单节点 + 多节点方案)
- 快速开始 (含前置: tokenizer/config 修复步骤)
- 环境变量参考表 (含默认值)
- 并行策略推荐 (标注 ✅/⚠️ 验证状态)
- 性能调优建议
- 功能验证命令
- 常见问题 FAQ

## 工作流程 (每个模型 5 步)

### 步骤 1: 分析模型

读取 `config.json` 确定:

```bash
cat <model_path>/config.json | python3 -c "
import json, sys
c = json.load(sys.stdin)
print('arch:', c.get('architectures'))
print('type:', c.get('model_type'))
print('hidden:', c.get('hidden_size'))
print('layers:', c.get('num_hidden_layers'))
print('experts:', c.get('n_routed_experts', 0))
print('nextn:', c.get('num_nextn_predict_layers', 0))
print('max_pos:', c.get('max_position_embeddings'))
print('q_lora:', c.get('q_lora_rank', 'N/A'))
print('kv_lora:', c.get('kv_lora_rank', 'N/A'))
print('head_dim:', c.get('head_dim', 'N/A'))
print('has_vision:', 'vision_config' in c)
"
```

**从分析中确定:**
- MoE? → `n_routed_experts > 0` → 必须 `--enable-expert-parallel`
- MTP? → `num_nextn_predict_layers > 0` → 可使用 `deepseek_mtp`
- PP 支持? → GLM-5/5.1 **不支持**; Kimi-K2 支持
- 量化类型? → 目录名: `w4a8`/`w8a8` → `QUANTIZATION=ascend`
- EP 整除? → `n_routed_experts % EP == 0`
- **架构兼容性** → 检查 vLLM 注册表 (见下)

### 步骤 2: 检查架构兼容性

验证模型的 `architectures` 是否在 vLLM-Ascend 0.18.0rc1 支持列表中:

```bash
docker exec npuslim-env bash -c "
source /usr/local/Ascend/cann/set_env.sh 2>/dev/null
grep '<ArchitectureName>' /opt/conda/env/lib/python3.11/site-packages/vllm/model_executor/models/registry.py
"
```

**已知兼容性:**
- ✅ `DeepseekV3ForCausalLM`, `DeepseekV32ForCausalLM`
- ✅ `GlmMoeDsaForCausalLM` (需修复 config, 见步骤 3)
- ✅ `KimiK25ForConditionalGeneration` (自带完整 .py 文件)
- ❌ `DeepseekV4ForCausalLM` (不支持, 需升级 vLLM)

### 步骤 3: 修复模型目录 (模型兼容性准备)

新模型通常缺少 transformers 加载所需的文件。检查并修复:

```bash
# 检查是否有自定义 Python 文件
ls <model_path>/*.py

# 如果没有 configuration_*.py, 创建之:
cat > <model_path>/configuration_<type>.py << 'EOF'
from transformers import PretrainedConfig
class <ConfigClass>(PretrainedConfig):
    model_type = "<type>"
    tokenizer_class = "PreTrainedTokenizerFast"
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
EOF

# 在 config.json 中添加 auto_map
python3 -c "
import json
with open('<model_path>/config.json') as f:
    cfg = json.load(f)
cfg['auto_map'] = {'AutoConfig': 'configuration_<type>.<ConfigClass>'}
with open('<model_path>/config.json', 'w') as f:
    json.dump(cfg, f, indent=2)
"

# 修复 tokenizer_config.json (GLM 模型常见问题)
python3 -c "
import json
with open('<model_path>/tokenizer_config.json') as f:
    cfg = json.load(f)
cfg.pop('extra_special_tokens', None)         # 移除 list 类型的错误字段
cfg['tokenizer_class'] = 'PreTrainedTokenizerFast'
with open('<model_path>/tokenizer_config.json', 'w') as f:
    json.dump(cfg, f, indent=2)
"
```

> ⚠️ **Kimi-K2.6 无需这些修复** — 模型目录已包含完整 Python 文件。

### 步骤 4: 生成脚本和文档

- 基于上述模板生成 `run_vllm.sh` (推荐) 和 `vllm_server.sh`
- 基于 `examples/curl_test.sh` 模板生成 `curl_test.sh`
- 参考 `examples/GLM5_README.md` 编写 `README.md`
- `bash -n` 验证所有 `.sh` 文件

### 步骤 5: 验证

- [ ] 3 个 `.sh` 文件存在且有可执行权限
- [ ] `bash -n` 语法检查通过
- [ ] `MODEL_PATH` 默认值指向正确的模型目录
- [ ] `TOOL_CALL_PARSER` 使用正确的下划线格式
- [ ] `QUANTIZATION=ascend` 已设置 (量化模型)
- [ ] PP 默认值正确 (GLM=1, Kimi=2)
- [ ] MoE 模型包含 `--enable-expert-parallel`
- [ ] MTP 模型包含 `--speculative-config`
- [ ] README 包含部署验证状态

## 部署执行

生成脚本后，需要在实际集群上部署测试:

```bash
# 1. 启动容器
bash scripts/docker/manage_npuslim_containers.sh start --file node_list.txt

# 2. 启动 Ray (按模型节点组)
bash scripts/ray_cluster/start_npuslim_ray_cluster.sh start -f nodes_<model>.txt

# 3. 部署模型 (通过 SSH 到 head 节点)
ssh_run "<head_ip>" 'docker exec npuslim-env bash -c \
  "> /tmp/vllm_<model>.log; TP=<tp> PP=<pp> nohup bash \
  /home/jianzhnie/llmtuner/llm/EasyInfer/examples/<dir>/run_vllm.sh \
  >> /tmp/vllm_<model>.log 2>&1 &"'

# 4. 监控启动 (通常 10-20 分钟)
ssh_run "<head_ip>" "curl -sf http://localhost:<port>/v1/models"

# 5. 运行测试
ssh_run "<head_ip>" 'docker exec npuslim-env bash -c \
  "BASE_URL=http://localhost:<port> bash \
  /home/jianzhnie/llmtuner/llm/EasyInfer/examples/<dir>/curl_test.sh"'
```

### 多节点策略

当模型**不支持 Pipeline Parallelism (PP>1)** 时，使用大 TP 跨节点:

| 需求 | 不支持 PP | 支持 PP |
|------|----------|--------|
| 1 节点 | TP=8 PP=1 | TP=8 PP=1 |
| 2 节点 | **TP=16 PP=1** | TP=8 PP=2 |
| 4 节点 | **TP=32 PP=1** | TP=8 PP=4 |

> GLM-5/5.1 不支持 PP, 使用左列方案。Kimi-K2.6 支持 PP, 使用右列方案。

## 常见错误速查

| 错误信息 | 原因 | 修复 |
|---------|------|------|
| `libascend_hal.so: cannot open` | CANN 环境未加载 | source `/usr/local/Ascend/cann/set_env.sh` |
| `readonly variable` | common.sh 重复 source | 使用 `run_vllm.sh` 直接部署 |
| `CMAKE_PREFIX_PATH: unbound` | CANN 与 `set -u` 冲突 | `set +u` 包裹 CANN source |
| `unrecognized arguments: --num-scheduler-steps` | vLLM 版本不支持 | 移除该参数 |
| `invalid tool call parser: deepseekv3` | 命名无下划线 | 改为 `deepseek_v3` |
| `Transformers does not recognize` | config 缺少 auto_map | 创建 configuration_*.py + 添加 auto_map |
| `'list' object has no attribute 'keys'` | tokenizer 配置类型错误 | 移除 extra_special_tokens |
| `Pipeline parallelism is not supported` | 模型无 SupportsPP 接口 | 使用大 TP 替代 PP |
| `architectures ... are not supported` | 架构不在注册表 | 检查兼容性或升级 vLLM |
| `Engine core initialization failed` | 多因, 见上级日志 | grep -B30 查找根因 |

## 参考资源

### 项目内

| 文件 | 用途 |
|------|------|
| `examples/glm5_server.sh` | 包装器部署脚本参考 |
| `examples/GLM5_README.md` | 文档格式参考 |
| `examples/curl_test.sh` | API 测试模板 |
| `examples/longcat_flash-chat.sh` | 直接 vllm serve 写法 |
| `scripts/vllm/vllm_server_env_template.sh` | 环境变量完整模板 |
| `scripts/vllm/vllm_model_server.sh` | 包装器主脚本 (含自动检测) |
| `scripts/common.sh` | 共享库 (log_info/log_err/ssh_run) |
| `.claude/skills/deploy-npu-model.md` | 完整部署流程 Skill |
| `.claude/skills/diagnose-npu-errors.md` | 错误诊断 Skill |
| `DEPLOYMENT_SUMMARY.md` | 4 模型部署经验总结 |

### 外部

- vLLM 官方: https://docs.vllm.ai/en/stable/
- vLLM 配置示例: https://recipes.vllm.ai/
- vLLM-Ascend 模型: https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/index.html

## Shell 规范要点

- `run_vllm.sh`: 用 `set -eo pipefail` (**不用 `set -u`**, CANN 不兼容)
- `vllm_server.sh`: 用 `set -euo pipefail`
- CANN source 前必须 `set +u`, source 后 `set -u`
- CANN 路径: `/usr/local/Ascend/cann/set_env.sh` (不是 `ascend-toolkit`)
- `QUANTIZATION=ascend` 而非 `fp8` (vLLM-Ascend 专用)
- `TOOL_CALL_PARSER` 使用下划线: `deepseek_v3`, `glm47`
- 所有变量 `${VAR:-default}` 模式, 支持环境变量覆盖
- 模型路径使用绝对路径
- 通过 `bash -n` 语法检查
