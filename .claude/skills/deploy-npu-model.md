---
name: deploy-npu-model
description: |
  Deploy LLM models on Huawei Ascend NPU cluster with vLLM-Ascend.
  Covers model analysis, script generation, deployment, error diagnosis, and testing.
  Use when the user asks to deploy, test, or benchmark models on the NPU cluster.
metadata:
  trigger_keywords: [deploy, 部署, vllm, ascend, NPU, 昇腾, model server, 模型部署]
  tools_needed: [Bash, Read, Write, Edit, WebFetch, Monitor]
---

# Deploy LLM Models on Ascend NPU Cluster

Complete workflow for deploying large language models on a Huawei Ascend NPU cluster
using vLLM-Ascend + Ray.

## Environment

- **Cluster**: 8 nodes × 8 NPU (Atlas 800 A2/A3, 64G each)
- **Container**: `npuslim-env` (ascend910c-cann8.5.1-torch2.9.0-vllm0.18.0)
- **CANN**: 8.5.1 at `/usr/local/Ascend/cann/`
- **Orchestration**: Ray for distributed, vLLM-Ascend for inference
- **Scripts repo**: `/home/jianzhnie/llmtuner/llm/EasyInfer`

---

## Phase 1: Model Analysis

### 1.1 Read config.json

For each model, read `config.json` and extract key architecture info:

```bash
cat <model_path>/config.json | python3 -c "
import json, sys
c = json.load(sys.stdin)
print(f'architectures: {c.get(\"architectures\")}')
print(f'model_type: {c.get(\"model_type\")}')
print(f'hidden_size: {c.get(\"hidden_size\")}')
print(f'num_hidden_layers: {c.get(\"num_hidden_layers\")}')
print(f'n_routed_experts: {c.get(\"n_routed_experts\")}')
print(f'num_experts_per_tok: {c.get(\"num_experts_per_tok\")}')
print(f'num_nextn_predict_layers: {c.get(\"num_nextn_predict_layers\", 0)}')
print(f'max_position_embeddings: {c.get(\"max_position_embeddings\")}')
print(f'q_lora_rank: {c.get(\"q_lora_rank\", \"N/A\")}')
print(f'kv_lora_rank: {c.get(\"kv_lora_rank\", \"N/A\")}')
print(f'head_dim: {c.get(\"head_dim\", \"N/A\")}')
print(f'vocab_size: {c.get(\"vocab_size\")}')
"
```

### 1.2 Determine Key Characteristics

From config.json, derive:

| Property | How to determine | Impact |
|----------|-----------------|--------|
| **MoE** | `n_routed_experts > 0` | Requires `--enable-expert-parallel` |
| **MTP** | `num_nextn_predict_layers > 0` | Can enable `--speculative-config` with `deepseek_mtp` |
| **Multimodal** | Has `vision_config` | May need special tokenizer handling |
| **Quantization** | From directory name: `w4a8`/`w8a8`/`fp8`/`bf16` | Sets `--quantization ascend` |
| **MLA** | Has `q_lora_rank` and `kv_lora_rank` | DeepSeek-style attention, handled by vLLM |
| **Pipeline Parallel support** | Model implements `SupportsPP` interface | Only Kimi-K2 series confirmed. GLM-5 does NOT. |
| **EP divisibility** | `n_routed_experts % EP == 0` | EP must divide expert count |

### 1.3 Check vLLM Architecture Support

Verify the model architecture is in vLLM's registry:

```bash
docker exec npuslim-env bash -c "
source /usr/local/Ascend/cann/set_env.sh 2>/dev/null
grep '<ArchitectureName>' /opt/conda/env/lib/python3.11/site-packages/vllm/model_executor/models/registry.py
"
```

If the architecture is NOT in the registry, the model **cannot be deployed** with the current vLLM version.

---

## Phase 2: Model Preparation

### 2.1 Check for Required Python Files

The model directory must have custom code files if the model type is not in transformers' built-in registry:

```bash
ls <model_path>/*.py
```

If missing `configuration_<type>.py`, create a minimal one:

```python
# configuration_<model_type>.py
from transformers import PretrainedConfig

class <ConfigClassName>(PretrainedConfig):
    model_type = "<model_type>"
    tokenizer_class = "PreTrainedTokenizerFast"
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
```

### 2.2 Fix config.json

Add `auto_map` to config.json if missing:

```bash
python3 -c "
import json
with open('<model_path>/config.json') as f:
    cfg = json.load(f)
cfg['auto_map'] = {'AutoConfig': 'configuration_<type>.<ConfigClass>'}
with open('<model_path>/config.json', 'w') as f:
    json.dump(cfg, f, indent=2)
"
```

### 2.3 Fix tokenizer_config.json

Check and fix common tokenizer issues:

```bash
python3 -c "
import json
with open('<model_path>/tokenizer_config.json') as f:
    cfg = json.load(f)
# Remove problematic keys
cfg.pop('extra_special_tokens', None)
# Set explicit tokenizer class
cfg['tokenizer_class'] = 'PreTrainedTokenizerFast'
with open('<model_path>/tokenizer_config.json', 'w') as f:
    json.dump(cfg, f, indent=2)
"
```

---

## Phase 3: Generate Deployment Scripts

### 3.1 Create `run_vllm.sh` (Recommended: Direct vllm serve)

This is the preferred approach — bypasses the wrapper chain and avoids `common.sh` readonly issues.

```bash
#!/bin/bash
# <ModelName> — 直接 vllm serve 部署
# Default: TP=8 PP=1 (single node); Multi-node: TP=16 PP=1 (2 nodes)
# PP 支持情况: <check - GLM does NOT support PP> | <model> supports PP
set -eo pipefail

# Load Ascend CANN environment (required for libascend_hal.so)
set +u
if [[ -f "/usr/local/Ascend/cann/set_env.sh" ]]; then
    source /usr/local/Ascend/cann/set_env.sh
fi
if [[ -f "/usr/local/Ascend/nnal/atb/set_env.sh" ]]; then
    source /usr/local/Ascend/nnal/atb/set_env.sh
fi
set -u

MODEL_PATH="${MODEL_PATH:-<default_model_path>}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-<port>}"
TP="${TP:-8}"
PP="${PP:-<1_or_2>}"

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
    --enable-expert-parallel \
    --quantization ascend \
    --gpu-memory-utilization <0.90-0.95> \
    --max-model-len <32768-65536> \
    --max-num-seqs <2-16> \
    --max-num-batched-tokens <4096-8192> \
    --enable-chunked-prefill \
    --enable-prefix-caching \
    --enforce-eager \
    <--speculative-config if MTP> \
    --enable-auto-tool-choice \
    --tool-call-parser <parser_name> \
    --seed 1024 \
    "$@"
```

**Key parameters by model type:**

| Parameter | Dense Model | MoE (256 experts) | MoE (384 experts) |
|-----------|------------|-------------------|-------------------|
| `--enable-expert-parallel` | No | **Yes** | **Yes** |
| `--gpu-memory-utilization` | 0.92 | 0.95 (W4A8) / 0.90 (W8A8) | 0.92 |
| `--max-num-seqs` | 32 | 2-8 (W4A8) | 8-16 |
| `--speculative-config` | No | If MTP: `deepseek_mtp` | No (Kimi has 0 nextn) |
| `--tool-call-parser` | hermes/qwen | glm47 (GLM) / deepseek_v3 (DeepSeek) | deepseek_v3 |

### 3.2 Create `curl_test.sh`

Copy from `examples/curl_test.sh` template, updating:
- `BASE_URL` default port to match the model
- `MODEL_NAME` default to match served model name
- Source path for `common.sh` (adjust `../` depth)

### 3.3 Create `README.md`

Must include:
- Model architecture summary table
- Hardware requirements (single-node + multi-node)
- Quick start commands
- Environment variable reference table
- Parallel strategy recommendations (with ✅/⚠️ status per row)
- Feature verification commands
- Common issues FAQ
- Deployment verification status banner at top

---

## Phase 4: Node Assignment & Ray Setup

### 4.1 Create Node Partition Files

For N models, partition the 8 nodes into groups:

```bash
# Example: 4 models × 2 nodes each
echo -e "10.16.201.229\n10.16.201.164" > nodes_model_a.txt
echo -e "10.16.201.40\n10.16.201.163"  > nodes_model_b.txt
# ... etc
```

### 4.2 Start Docker Containers

```bash
bash scripts/docker/manage_npuslim_containers.sh start --file node_list.txt
```

### 4.3 Start Ray Clusters

Start one Ray cluster per model group:

```bash
bash scripts/ray_cluster/start_npuslim_ray_cluster.sh start -f nodes_model_a.txt
bash scripts/ray_cluster/start_npuslim_ray_cluster.sh start -f nodes_model_b.txt
# ... etc
```

---

## Phase 5: Deploy Models

### 5.1 Deploy Command Pattern

```bash
source scripts/common.sh
ssh_run "<head_ip>" 'docker exec npuslim-env bash -c \
  "> /tmp/vllm_<model>.log; TP=<tp> PP=<pp> nohup bash \
  /home/jianzhnie/llmtuner/llm/EasyInfer/examples/<dir>/run_vllm.sh \
  >> /tmp/vllm_<model>.log 2>&1 &"'
```

### 5.2 Multi-Node Strategy

When a model does NOT support Pipeline Parallelism (PP>1):

| Desired nodes | Solution | Example |
|--------------|----------|---------|
| 1 node | TP=8 PP=1 | Default |
| 2 nodes | **TP=16 PP=1** | `TP=16 PP=1` |
| 4 nodes | **TP=32 PP=1** | `TP=32 PP=1` |

When a model DOES support PP (e.g., Kimi-K2.6):

| Desired nodes | Solution | Example |
|--------------|----------|---------|
| 1 node | TP=8 PP=1 | Default |
| 2 nodes | TP=8 PP=2 | `TP=8 PP=2` |

### 5.3 Monitor Startup

```bash
Monitor --timeout_ms 900000 --description "Watch all models loading" \
  'source scripts/common.sh
   while true; do
     for model in ...; do
       curl -sf http://<ip>:<port>/v1/models && echo "READY: $model"
     done
     sleep 60
   done'
```

---

## Phase 6: Error Diagnosis

### Common Error → Fix Map

| Error Pattern | Root Cause | Fix |
|--------------|------------|-----|
| `libascend_hal.so: cannot open` | CANN env not loaded | Source `/usr/local/Ascend/cann/set_env.sh` before vllm |
| `readonly variable` | `common.sh` sourced twice | Use `run_vllm.sh` (bypasses wrapper) or update common.sh |
| `CMAKE_PREFIX_PATH: unbound variable` | CANN script + `set -u` | Wrap CANN source with `set +u`/`set -u` |
| `unrecognized arguments: --num-scheduler-steps` | vLLM version doesn't support | Remove `--num-scheduler-steps` |
| `invalid tool call parser: deepseekv3` | Wrong parser name | Use `deepseek_v3` (with underscore) |
| `Transformers does not recognize this architecture` | Missing auto_map | Add `auto_map` to config.json + create config .py |
| `'list' object has no attribute 'keys'` | tokenizer extra_special_tokens is list | Remove from tokenizer_config.json |
| `Pipeline parallelism is not supported` | Model doesn't implement SupportsPP | Use larger TP across nodes instead |
| `DeepseekV4ForCausalLM not supported` | vLLM version too old | Upgrade vLLM-Ascend or use compatible arch |
| `Engine core initialization failed` | Multiple causes | Check root cause in log above this error |

### Diagnosis Workflow

```
vllm process exists?
├── NO  → Check log for immediate crash error
│         ├── libascend_hal.so → Fix CANN env
│         ├── unrecognized arguments → Remove unsupported flag
│         ├── ModelConfig ValidationError → Fix auto_map/config
│         ├── Tokenizer error → Fix tokenizer_config.json
│         └── Engine core failed → Check root cause traceback
└── YES → Check if responding on port
          ├── YES → SUCCESS
          └── NO  → Model still loading (check for % progress in log)
```

---

## Phase 7: Testing

### 7.1 Run curl tests

```bash
ssh_run "<head_ip>" 'docker exec npuslim-env bash -c \
  "BASE_URL=http://localhost:<port> bash \
  /home/jianzhnie/llmtuner/llm/EasyInfer/examples/<dir>/curl_test.sh"'
```

### 7.2 Manual API Verification

```bash
# Health check
curl http://<ip>:<port>/v1/models

# Non-streaming chat
curl http://<ip>:<port>/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"<name>","messages":[{"role":"user","content":"Hello"}],"max_tokens":100}'

# Streaming chat
curl http://<ip>:<port>/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"<name>","messages":[{"role":"user","content":"Hello"}],"max_tokens":100,"stream":true}'
```

---

## Quick Reference: Tool Parser Names

| Model Family | Parser Name | Note |
|-------------|-------------|------|
| DeepSeek V3/V3.1/V3.2 | `deepseek_v3` / `deepseek_v31` / `deepseek_v32` | With underscore |
| DeepSeek V4 | `deepseek_v3` | Fallback (V4 not natively supported) |
| GLM-4/5 | `glm47` | GLM series |
| Kimi-K2 | `kimi_k2` | Kimi series |
| Kimi-K2.5/2.6 | `deepseek_v3` | Based on DeepSeek V3 backbone |
| Qwen | `hermes` | Qwen series |

## Quick Reference: Supported vLLM-Ascend 0.18.0rc1 Architectures

- ✅ `DeepseekV3ForCausalLM`, `DeepseekV32ForCausalLM`
- ✅ `GlmMoeDsaForCausalLM` (requires auto_map + config fix)
- ✅ `KimiK25ForConditionalGeneration` (has built-in .py files)
- ❌ `DeepseekV4ForCausalLM` (NOT supported — need upgrade)

## Script Files Produced Per Model

```
examples/<model_dir>/
├── vllm_server.sh    # Wrapper-based (uses vllm_model_server.sh)
├── run_vllm.sh       # Direct vllm serve (RECOMMENDED)
├── curl_test.sh      # API test suite
└── README.md         # Deployment documentation
```

## Critical Environment Variables

```bash
# MUST be set before vllm serve
export HCCL_OP_EXPANSION_MODE="AIV"
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export HCCL_BUFFSIZE=200
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_ASCEND_BALANCE_SCHEDULING=1

# Quantized models
export QUANTIZATION="ascend"  # NOT "fp8"!

# NPU optimizations
export CUDAGRAPH_MODE="FULL_DECODE_ONLY"
export ENABLE_NPUGRAPH_EX="true"
export FUSE_MULS_ADD="true"
export MULTISTREAM_OVERLAP_SHARED_EXPERT="true"
```
