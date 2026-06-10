# SGLang 模型部署和测试

## 环境概况

- **集群**: 8 节点 × 8 昇腾 NPU (Atlas 800 A2/A3, 每卡 64G)
- **框架**: SGLang (Ascend 版) + torch.distributed 分布式
- **容器**: `sglang-ascend-env` (镜像 `quay.io/ascend/sglang:main-cann9.0.0-a3`)
- **CANN**: 9.0.0, 路径 `/usr/local/Ascend/cann/`
- **挂载**: `/home/jianzhnie/llmtuner` → 容器内同路径
- **API**: OpenAI 兼容 `/v1/chat/completions`, `/v1/models`

> SGLang 与 vLLM 的核心区别: SGLang 使用 RadixAttention 自动前缀缓存（无需 `--enable-prefix-caching`），
> 原生支持 torch.distributed 多节点部署（无需 Ray），调度器效率更高。

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

对每个模型，在 `examples/<模型简称>/sglang/` 下生成 **4 个文件**:

```
examples/<model_dir>/
├── run_sglang.sh     ← 直接 sglang launch_server（推荐）
├── sglang_server.sh  ← 通过包装器部署
├── curl_test.sh      ← API 功能测试
└── README.md         ← 部署与测试文档
```

### 1. `run_sglang.sh` — 直接部署脚本（推荐）

直接调用 `python -m sglang.launch_server`，避免包装器链的 `common.sh` readonly 冲突。

**必须遵循的模板:**

```bash
#!/bin/bash
# <ModelName> — 直接 sglang launch_server 部署
# 默认 TP=8 (单节点); 多节点: TP=16 (2节点, 当不支持 PP 时)
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
PP="${PP:-<1 或 2>}"          # GLM 不支持 PP>1, Kimi 支持 PP=2
EP="${EP:-<值>}"               # 专家并行度, 默认 = TP×PP
CONTEXT_LEN="${CONTEXT_LEN:-<根据硬件调整>}"
MAX_RUNNING_REQS="${MAX_RUNNING_REQS:-<根据硬件调整>}"

# HCCL/NPU 环境变量
export HCCL_OP_EXPANSION_MODE="${HCCL_OP_EXPANSION_MODE:-AIV}"
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export HCCL_BUFFSIZE=200
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export SGLANG_ASCEND_BALANCE_SCHEDULING=1

echo "[INFO] Starting <ModelName> with SGLang"
echo "[INFO] TP=$TP PP=$PP EP=$EP PORT=$PORT"
echo "[INFO] CONTEXT_LEN=$CONTEXT_LEN MAX_RUNNING_REQS=$MAX_RUNNING_REQS"

python -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --host "$HOST" \
    --port "$PORT" \
    --served-model-name <served_name> \
    --trust-remote-code \
    --dtype bfloat16 \
    --tp "$TP" \
    --pp "$PP" \
    --ep "$EP" \                              # MoE 模型必须
    --device npu \
    --quantization fp8 \                       # W4A8/W8A8 使用 ascend 量化
    --mem-fraction-static <0.88-0.93> \
    --context-length "$CONTEXT_LEN" \
    --max-running-requests "$MAX_RUNNING_REQS" \
    --max-total-tokens 8192 \
    --chunked-prefill-size 8192 \
    --enable-torch-compile \                   # 等价于 vLLM 的 CUDA Graph
    --speculative-algorithm EAGLE \            # 仅 MTP 模型
    --speculative-num-draft-tokens 3 \
    --enable-tool-call \
    --tool-call-parser <parser_name> \
    --log-level info \
    "$@"
```

**⚠️ sglang vs vLLM 参数对照 (关键差异):**

| 功能 | vLLM 参数 | SGLang 参数 | 说明 |
|------|----------|------------|------|
| 张量并行 | `--tensor-parallel-size 8` | `--tp 8` | |
| 流水线并行 | `--pipeline-parallel-size 2` | `--pp 2` | |
| 专家并行 | `--enable-expert-parallel` | `--ep <N>` | sglang 显式指定 EP 度数 |
| 设备类型 | (自动) | `--device npu` | Ascend 必须显式指定 |
| GPU 内存 | `--gpu-memory-utilization 0.90` | `--mem-fraction-static 0.90` | |
| 最大长度 | `--max-model-len 65536` | `--context-length 65536` | |
| 并发请求 | `--max-num-seqs 16` | `--max-running-requests 16` | |
| 批次 Tokens | `--max-num-batched-tokens 8192` | `--max-total-tokens 8192` | |
| 分块预填充 | `--enable-chunked-prefill` | `--chunked-prefill-size 8192` | sglang 需要指定大小, -1 禁用 |
| 前缀缓存 | `--enable-prefix-caching` | **自动 (RadixAttention)** | sglang 默认开启，无需参数 |
| CUDA Graph | `--enforce-eager` (禁用) | `--enable-torch-compile` | sglang 反向: 加 flag 才启用 |
| 推测解码 | `--speculative-config "{...}"` | `--speculative-algorithm EAGLE` | sglang 支持 EAGLE/Medusa |
| 工具调用 | `--enable-auto-tool-choice` | `--enable-tool-call` | |
| 量化 | `--quantization ascend` | `--quantization fp8` | sglang 可能使用不同量化名 |
| 分布式后端 | `--distributed-executor-backend ray` | 自动 (torch.distributed) | sglang 无需 Ray |
| 多节点 | Ray Head/Worker | `--nnodes --node-rank` | sglang 用 torchrun |

**关键参数速查:**

| 参数 | MoE 模型 | 非 MoE | 说明 |
|------|---------|--------|------|
| `--ep` | 必须: `TP×PP` | 不需要 | MoE 必需, 须整除专家数 |
| `--speculative-algorithm` | MTP 模型 | 不需要 | `num_nextn_predict_layers > 0` 时使用 EAGLE |
| `--tool-call-parser` | 见下表 | 见下表 | 关键: 使用下划线分隔 |
| `--quantization` | `fp8` / `ascend` | 不设置 | 量化模型需确认 sglang 支持的量化名 |

**Tool Parser 对照表 (关键: 下划线分隔):**

| 模型系列 | Parser |
|---------|--------|
| DeepSeek V3/V32/V4 | `deepseek_v3` |
| GLM-4/5 | `glm47` |
| Kimi-K2.6 | `deepseek_v3` |
| Qwen | `hermes` |

### 2. `sglang_server.sh` — 包装器部署脚本

通过自定义包装器启动 sglang。参考项目内 `scripts/vllm/vllm_model_server.sh` 的写法，但适配为 sglang 启动命令。

注意事项:
- 环境变量驱动，支持 `${VAR:-default}` 覆盖模式
- 必须 source CANN 环境 (`set +u` / `set -u` 包裹)
- sglang 启动检测: `curl -sf http://${HOST}:${PORT}/health`
- 支持 `--tp` `--pp` `--ep` 等命令行参数覆盖

### 3. `curl_test.sh` — API 功能测试

SGLang 提供 **完全 OpenAI 兼容** 的 API，curl_test.sh 可直接复用 vLLM 版本的测试逻辑。

参考 `examples/curl_test.sh` 模板，修改:
- `BASE_URL` 默认端口 (8000/8001/8002/8003 避免冲突)
- `MODEL_NAME` 默认值
- source `common.sh` 的相对路径 (子目录多一层: `../../scripts/common.sh`)

测试覆盖: `/v1/models` → `/health` → 非流式 Chat → 流式 Chat → Tool Calling

### 4. `README.md` — 部署文档

参考 `examples/GLM5_README.md` 格式，必须包含:
- 顶部部署验证状态 banner (✅/❌/⚠️)
- 模型架构摘要表
- 硬件要求 (单节点 + 多节点方案)
- 快速开始 (含前置: tokenizer/config 修复步骤)
- 环境变量参考表 (含默认值)
- 并行策略推荐 (标注 ✅/⚠️ 验证状态)
- 性能调优建议 (RadixAttention 缓存命中率等)
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
- MoE? → `n_routed_experts > 0` → 必须 `--ep <N>` (N = TP × PP, 须整除专家数)
- MTP? → `num_nextn_predict_layers > 0` → 可使用 `--speculative-algorithm EAGLE`
- PP 支持? → GLM-5/5.1 **不支持**; Kimi-K2 支持
- 量化类型? → 目录名: `w4a8`/`w8a8` → `--quantization fp8` (需验证 sglang ascend 量化命名)
- EP 整除? → `n_routed_experts % EP == 0`
- **架构兼容性** → 检查 sglang 注册表 (sglang 支持更广泛的模型架构)

### 步骤 2: 检查架构兼容性

验证模型的 `architectures` 是否在 sglang 支持列表中:

```bash
docker exec sglang-ascend-env bash -c "
source /usr/local/Ascend/cann/set_env.sh 2>/dev/null
python3 -c \"
from sglang.srt.models.registry import get_model_architectures
print('\\\\n'.join(sorted(get_model_architectures().keys())))
\" 2>/dev/null | grep -i '<keyword>'
"
```

**已知兼容性 (sglang 通常比 vLLM 支持更多架构):**
- ✅ `DeepseekV3ForCausalLM`, `DeepseekV32ForCausalLM`
- ✅ `DeepseekV4ForCausalLM` (sglang 可能已原生支持, 需验证)
- ✅ `GlmMoeDsaForCausalLM` (需修复 config, 见步骤 3)
- ✅ `KimiK25ForConditionalGeneration` (自带完整 .py 文件)

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

- 基于上述模板生成 `run_sglang.sh` (推荐) 和 `sglang_server.sh`
- 基于 `examples/curl_test.sh` 模板生成 `curl_test.sh`
- 参考 `examples/GLM5_README.md` 编写 `README.md`
- `bash -n` 验证所有 `.sh` 文件

### 步骤 5: 验证

- [ ] 3 个 `.sh` 文件存在且有可执行权限
- [ ] `bash -n` 语法检查通过
- [ ] `MODEL_PATH` 默认值指向正确的模型目录
- [ ] `TOOL_CALL_PARSER` 使用正确的下划线格式
- [ ] `--quantization fp8` 已设置 (量化模型)
- [ ] PP 默认值正确 (GLM=1, Kimi=2)
- [ ] MoE 模型包含 `--ep`
- [ ] MTP 模型包含 `--speculative-algorithm EAGLE`
- [ ] `--device npu` 已设置
- [ ] README 包含部署验证状态

## 部署执行

生成脚本后，需要在实际集群上部署测试:

```bash
# 1. 拉取 sglang 镜像
docker pull quay.io/ascend/sglang:main-cann9.0.0-a3

# 2. 在所有节点上启动 sglang 容器
# (参考 scripts/docker/manage_npuslim_containers.sh 适配 sglang 镜像)
bash scripts/docker/manage_sglang_containers.sh start --file node_list.txt

# 3. 单节点部署 (无需 Ray!)
ssh_run "<node_ip>" "docker exec sglang-ascend-env bash -c \
  '> /tmp/sglang_<model>.log 2>&1; nohup bash \
  /home/jianzhnie/llmtuner/llm/EasyInfer/examples/<dir>/sglang/run_sglang.sh \
  >> /tmp/sglang_<model>.log 2>&1 &'"

# 4. 多节点部署 (sglang 使用 torchrun, 无需 Ray 集群!)
# Head 节点:
ssh_run "<head_ip>" "docker exec sglang-ascend-env bash -c \
  '> /tmp/sglang_<model>.log 2>&1; nohup bash \
  /home/jianzhnie/llmtuner/llm/EasyInfer/examples/<dir>/sglang/run_sglang.sh \
  --nnodes <N> --node-rank 0 --dist-init-addr <head_ip>:5000 \
  >> /tmp/sglang_<model>.log 2>&1 &'"

# Worker 节点 (并行启动):
for i in $(seq 1 $((N-1))); do
  ssh_run "<worker${i}_ip>" "docker exec sglang-ascend-env bash -c \
    '> /tmp/sglang_<model>.log 2>&1; nohup bash \
    /home/jianzhnie/llmtuner/llm/EasyInfer/examples/<dir>/sglang/run_sglang.sh \
    --nnodes <N> --node-rank ${i} --dist-init-addr <head_ip>:5000 \
    >> /tmp/sglang_<model>.log 2>&1 &'" &
done
wait

# 5. 监控启动 (通常 5-15 分钟, 比 vLLM 快)
ssh_run "<head_ip>" "curl -sf http://localhost:<port>/health"

# 6. 运行测试
ssh_run "<head_ip>" "docker exec sglang-ascend-env bash -c \
  'BASE_URL=http://localhost:<port> bash \
  /home/jianzhnie/llmtuner/llm/EasyInfer/examples/<dir>/sglang/curl_test.sh'"

# 7. 性能压测 (可选)
ssh_run "<head_ip>" "docker exec sglang-ascend-env bash -c \
  'python -m sglang.bench_serving \
  --backend sglang \
  --dataset-name random \
  --num-prompts 1000 \
  --request-rate 1.0 \
  --host localhost \
  --port <port>'"
```

### 多节点策略

sglang 通过 `torch.distributed` 实现多节点通信，**无需 Ray 集群**。

| 需求 | 不支持 PP | 支持 PP |
|------|----------|--------|
| 1 节点 | `--tp 8 --pp 1` | `--tp 8 --pp 1` |
| 2 节点 | **`--tp 16 --pp 1`** | `--tp 8 --pp 2` |
| 4 节点 | **`--tp 32 --pp 1`** | `--tp 8 --pp 4` |

> GLM-5/5.1 不支持 PP, 使用左列方案。Kimi-K2.6 支持 PP, 使用右列方案。
>
> sglang 多节点命令示例 (2 节点, TP=16):
> ```bash
> # Head
> python -m sglang.launch_server ... --tp 16 --nnodes 2 --node-rank 0 --dist-init-addr <head_ip>:5000
> # Worker
> python -m sglang.launch_server ... --tp 16 --nnodes 2 --node-rank 1 --dist-init-addr <head_ip>:5000
> ```

### SGLang vs vLLM 部署流程对比

| 步骤 | vLLM | SGLang |
|------|------|--------|
| 容器启动 | `manage_npuslim_containers.sh` | 适配 sglang 镜像的容器管理 |
| 分布式集群 | Ray Head/Worker | **无需 Ray**, torchrun 原生 |
| 启动命令 | `vllm serve <path> ...` | `python -m sglang.launch_server --model-path <path> ...` |
| 健康检查 | `curl /health` | `curl /health` (兼容) |
| 启动时间 | 10-20 分钟 | **5-15 分钟** (通常更快) |
| API 兼容性 | OpenAI 兼容 | **完全 OpenAI 兼容** |

## 常见错误速查

| 错误信息 | 原因 | 修复 |
|---------|------|------|
| `libascend_hal.so: cannot open` | CANN 环境未加载 | source `/usr/local/Ascend/cann/set_env.sh` |
| `readonly variable` | common.sh 重复 source | 使用 `run_sglang.sh` 直接部署 |
| `CMAKE_PREFIX_PATH: unbound` | CANN 与 `set -u` 冲突 | `set +u` 包裹 CANN source |
| `No module named 'sglang'` | sglang 未安装 | 确认使用 sglang 镜像: `quay.io/ascend/sglang:main-cann9.0.0-a3` |
| `device type npu is not supported` | sglang 未检测到 NPU | 确认 `--device npu` + CANN 环境已加载 |
| `Transformers does not recognize` | config 缺少 auto_map | 创建 configuration_*.py + 添加 auto_map |
| `'list' object has no attribute 'keys'` | tokenizer 配置类型错误 | 移除 extra_special_tokens |
| `Pipeline parallelism is not supported` | 模型无 SupportsPP 接口 | 使用大 TP 替代 PP |
| `torch.distributed initialization failed` | 多节点网络不通 | 检查节点间防火墙/NCCL 通信 |
| `RadixAttention cache OOM` | KV Cache 内存不足 | 降低 `--mem-fraction-static` 或 `--context-length` |
| `EAGLE speculative decode not supported` | 模型不适合 EAGLE | 检查 MTP 层数, 或移除 `--speculative-algorithm` |
| `quantization fp8 not supported for this model` | 量化格式不匹配 | 尝试 `--quantization ascend` 或移除量化参数 |
| `SGLang server failed to start` | 多因, 见上级日志 | `grep -B30 "Error\|Traceback"` 查找根因 |

## 参考资源

### 项目内

| 文件 | 用途 |
|------|------|
| `examples/glm5_w4a8/run_vllm.sh` | vLLM 部署脚本参考 (对比迁移) |
| `examples/GLM5_README.md` | 文档格式参考 |
| `examples/curl_test.sh` | API 测试模板 (sglang 兼容) |
| `scripts/docker/manage_npuslim_containers.sh` | Docker 容器管理参考 (需适配 sglang) |
| `scripts/common.sh` | 共享库 (log_info/log_err/ssh_run) |
| `.claude/skills/deploy-npu-model.md` | vLLM 部署流程 Skill (参考工作流) |
| `.claude/skills/diagnose-npu-errors.md` | 错误诊断 Skill |

### 外部

- SGLang 官方文档: https://sgl-project.github.io/
- SGLang Cookbook (AutoRegressive): https://lmsysorg.mintlify.app/cookbook/autoregressive/intro
- SGLang Ascend NPU 支持: https://sgl-project.github.io/platforms/ascend/ascend_npu_support.html
- SGLang 模型支持列表: https://sgl-project.github.io/references/supported_models.html
- SGLang Server 参数: https://sgl-project.github.io/references/supported_models.html
- SGLang GitHub: https://github.com/sgl-project/sglang

## Shell 规范要点

- `run_sglang.sh`: 用 `set -eo pipefail` (**不用 `set -u`**, CANN 不兼容)
- `sglang_server.sh`: 用 `set -euo pipefail`
- CANN source 前必须 `set +u`, source 后 `set -u`
- CANN 路径: `/usr/local/Ascend/cann/set_env.sh` (不是 `ascend-toolkit`)
- sglang 启动: `python -m sglang.launch_server` (不是 `vllm serve`)
- `--device npu` 必须显式指定
- `--tool-call-parser` 使用下划线: `deepseek_v3`, `glm47`
- 所有变量 `${VAR:-default}` 模式, 支持环境变量覆盖
- 模型路径使用绝对路径
- 通过 `bash -n` 语法检查

## SGLang 特有优化提示

1. **RadixAttention 自动前缀缓存**: 无需任何配置，sglang 自动复用相同前缀的 KV Cache。
   对 Claude Code 等多轮对话场景极其有利，system prompt 只计算一次。

2. **调度策略**: sglang 默认使用 `lpm` (longest-prefix-match) 调度，可通过 `--schedule-policy` 调整:
   - `lpm`: 前缀匹配优先 (默认，适合多轮对话)
   - `random`: 随机调度
   - `fcfs`: 先来先服务
   - `priority`: 优先级调度

3. **约束解码**: sglang 原生支持 JSON/Regex 约束解码:
   ```bash
   # JSON 约束
   curl ... -d '{"messages":[...], "response_format": {"type": "json_object"}}'
   # Regex 约束
   curl ... -d '{"messages":[...], "regex": "\\d{3}-\\d{4}"}'
   ```

4. **性能压测**: 使用 sglang 内置 benchmark 工具:
   ```bash
   python -m sglang.bench_serving \
       --backend sglang \
       --dataset-name sharegpt \  # 或 random, generated
       --num-prompts 1000 \
       --request-rate 1.0 \
       --host localhost \
       --port 8000
   ```
