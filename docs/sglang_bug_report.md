# SGLang Ascend NPU Bug Report — Quantized MoE Models

> **sglang version**: 0.5.12.post2.dev1045+g32bedbf88
> **Docker image**: `quay.io/ascend/sglang:main-cann9.0.0-a3`
> **Hardware**: 4× Atlas 800 A3 (Ascend 910C), 8 NPU × 64GB each
> **CANN**: 9.0.0
> **Python**: 3.11.15 | **torch**: 2.10.0+cpu | **torch_npu**: 2.10.0
> **Date**: 2026-06-10

---

## Bug 1: modelslim MoE EP — `groupList` mismatch in `aclnnGroupedMatmulWeightNz`

### Severity
**Critical** — prevents any multi-NPU deployment of quantized MoE models with EP>1.

### Affected Models
- GLM-5 W4A8 (256 experts, `GlmMoeDsaForCausalLM`)
- GLM-5.1 W4A8 (256 experts)
- Potentially all modelslim-quantized MoE models with EP>1

### Error

```
RuntimeError: npu_grouped_matmul:
  ../third_party/op-plugin/op_plugin/ops/opapi/GroupedMatmulKernelNpuOpApi.cpp:322
  NPU function error: call aclnnGroupedMatmulWeightNz failed, error code is 161002

[ERROR] ERR00100 PTA call acl api failed.
AclNN_Parameter_Error(EZ1001): When groupList is not null,
  size of groupList(tensor) 256 should be equal to weight dim 0 32 with groupType 0.
```

The `groupList` is **always 256** (total expert count), but the weight tensor is correctly EP-sharded (`weight dim 0 = 256/EP`).

### EP Size → Mismatch Pattern

| EP Size | groupList | weight dim 0 | Expected groupList |
|---------|-----------|-------------|-------------------|
| 8 | 256 | 32 | 32 |
| 2 | 256 | 128 | 128 |
| 4 | 256 | 64 | 64 |

### Stack Trace

```
File "sglang/srt/models/deepseek_v2.py", line 1005, in forward_normal
    final_hidden_states = self.experts(...)
File "sglang/srt/layers/moe/fused_moe_triton/layer.py", line 1120, in run_moe_core
    return self.quant_method.apply(...)
File "sglang/srt/layers/quantization/modelslim/modelslim.py", line 367, in apply
    return scheme.apply_weights(layer, dispatch_output)
File "sglang/srt/layers/quantization/modelslim/schemes/modelslim_w4a8_int8_moe.py", line 199, in apply_weights
    return self.kernel.apply(layer, dispatch_output)
File "sglang/srt/hardware_backend/npu/quantization/fused_moe_method_npu.py", line 926, in apply
    hidden_states = torch.ops.npu.npu_grouped_matmul(...)
```

### Root Cause Analysis

In `fused_moe_method_npu.py:926`, the `groupList` tensor passed to `npu_grouped_matmul` contains **all** expert indices (0..255) instead of only the **locally sharded** expert indices. The modelslim quantization shards expert weights by EP rank, so each rank owns `256/EP` experts' weights. The `groupList` should be filtered to match the local EP shard.

### Reproduction

```bash
# Start container
docker run -d --name sglang-ascend --net=host --privileged \
  --device=/dev/davinci0 ... --device=/dev/davinci7 \
  --device=/dev/davinci_manager --device=/dev/devmm_svm --device=/dev/hisi_hdc \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v /path/to/models:/path/to/models \
  quay.io/ascend/sglang:main-cann9.0.0-a3 tail -f /dev/null

# Trigger bug (EP=8 on 8 NPU single-node)
docker exec sglang-ascend sglang serve \
  --model-path /path/to/GLM-5-w4a8 \
  --tensor-parallel-size 8 --expert-parallel-size 8 \
  --device npu --quantization modelslim \
  --trust-remote-code --disable-cuda-graph \
  --nnodes 1 --node-rank 0
```

### Workaround
**None.** EP=1 avoids the crash but causes Bug 2 (gibberish output). Reducing EP to 1 combined with TP=8 loads all 256 experts onto each NPU, causing OOM on 64GB cards.

---

## Bug 2: modelslim EP=1 — incorrect inference output (gibberish)

### Severity
**Critical** — model loads and serves API, but all generated text is garbage.

### Affected Models
- GLM-5 W4A8 with `--expert-parallel-size 1`

### Observed Behavior

- Model config parsing: ✅ (architectures, model_type recognized)
- Weight loading: ✅ (100/100 shards loaded, ~2 min)
- Quantization detection: ✅ (`Using ModelSlimW4A8Int8MoE`)
- Server startup: ✅ (`/health`, `/v1/models` respond correctly)
- Tokenizer: ✅ (encode/decode round-trips correctly)
- **Inference**: ❌ **gibberish output**

### Example Output

```
Input:  "你好，1+1等于几？"
Output: "币 tanfol't goodness supanisAnursLibAnals.state mode statebos,
         TRLib ChenalaAnAnAn state state stateAn State region..."
```

### Root Cause Hypothesis
The modelslim NPU kernel produces incorrect numerical results when EP=1 (all experts handled by one rank). Possible causes:
- W4A8 weight layout mismatch for grouped matmul when EP is not used
- Per-channel quantization scales applied incorrectly in EP=1 path
- MoE routing indices not properly aligned with non-sharded expert weights

### Reproduction
```bash
# Same setup as Bug 1, but with --expert-parallel-size 1
docker exec sglang-ascend sglang serve \
  --model-path /path/to/GLM-5-w4a8 \
  --tensor-parallel-size 8 --expert-parallel-size 1 \
  --device npu --quantization modelslim \
  --trust-remote-code --disable-cuda-graph \
  --nnodes 1 --node-rank 0 \
  --port 8001

# Server starts, but inference broken
curl http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"glm-5","messages":[{"role":"user","content":"Hello"}],"max_tokens":50}'
```

### Workaround
**None found.** All quantization schemes tested:
- `--quantization modelslim` → gibberish (Bug 2)
- `--quantization w4afp8` → model load fails (wrong format)
- `--quantization fp8` → model load fails (wrong format)
- Without `--quantization` → model load fails (unquantized path doesn't understand modelslim weights)

---

## Bug 3: `_DeepseekV4ConfigAlias` not recognized by `AutoModel`

### Severity
**Critical** — DeepSeek V4 models cannot load at all.

### Affected Models
- DeepSeek V4 Flash W8A8 (`DeepseekV4ForCausalLM`)

### Error

```
ValueError: Unrecognized configuration class
  <class 'sglang.srt.utils.hf_transformers.common._DeepseekV4ConfigAlias'>
  for this kind of AutoModel: AutoModel.
Model type should be one of ..., DeepseekV4Config, ...
```

Note: `DeepseekV4Config` **is** in the supported types list, but the sglang alias `_DeepseekV4ConfigAlias` subclass is not.

### Stack Trace

```
File "sglang/srt/models/transformers.py", line 616, in __init__
    self.model: PreTrainedModel = AutoModel.from_config(config, ...)
File "transformers/models/auto/auto_factory.py", line 253, in from_config
    raise ValueError(f"Unrecognized configuration class ...")
```

### Root Cause Analysis

In `sglang/srt/utils/hf_transformers/common.py`, a `_DeepseekV4ConfigAlias` class is created as a wrapper around HuggingFace's `DeepseekV4Config`. This alias is registered in `AutoModel._model_mapping` (confirmed via `python3 -c` check), but `AutoModel.from_config()` fails because the alias class is not in transformers' `_model_mapping` — sglang may use a different `AutoModel` path than HuggingFace's.

### Reproduction

```bash
docker exec sglang-ascend sglang serve \
  --model-path /path/to/DeepSeek-V4-Flash-w8a8-mtp \
  --device npu --quantization modelslim --trust-remote-code \
  --tensor-parallel-size 8 --nnodes 1
```

### Confirmed

- `DeepseekV4Config` exists in transformers: ✅
- `_DeepseekV4ConfigAlias` registered in sglang's mapping: ✅
- `AutoModel.from_config(DeepseekV4Config())` works: ✅ (with native config)
- `AutoModel.from_config(_DeepseekV4ConfigAlias())` fails: ❌
- Both with and without custom `configuration_deepseek_v4.py` in model dir: ❌

### Workaround
**None found.** Removing/re-adding `auto_map` and `configuration_deepseek_v4.py` has no effect — the sglang internal alias mechanism always triggers.

---

## Environment Details

```yaml
Image: quay.io/ascend/sglang:main-cann9.0.0-a3
sglang version: 0.5.12.post2.dev1045+g32bedbf88
CANN version: 9.0.0 (cann-9.0.0 at /usr/local/Ascend/)
torch: 2.10.0+cpu
torch_npu: 2.10.0
transformers: 4.57.6
Hardware: Huawei Atlas 800 A3 (Ascend 910C)
  - 8 NPU per node × 64GB HBM
  - 4 nodes, 32 NPU total
  - Driver: 25.2.1
```

### Model Details

| Model | Architecture | Experts | Quant | Size on Disk |
|-------|-------------|---------|-------|-------------|
| GLM-5-w4a8 | GlmMoeDsaForCausalLM | 256 | W4A8 (modelslim) | 99 shards |
| DeepSeek-V4-Flash-w8a8-mtp | DeepseekV4ForCausalLM | 256 | W8A8 (modelslim) | ~100 shards |
| GLM-5.1-w4a8 | GlmMoeDsaForCausalLM | 256 | W4A8 (modelslim) | untested |
| Kimi-K2.6-w4a8 | KimiK25ForConditionalGeneration | 384 | W4A8 (modelslim) | untested |

---

## Summary

| Bug | Component | Symptom | Impact |
|-----|-----------|---------|--------|
| #1 | `fused_moe_method_npu.py` | `groupList` not EP-filtered | EP>1 crashes |
| #2 | modelslim NPU kernel (EP=1) | Incorrect numerical output | Inference gibberish |
| #3 | `_DeepseekV4ConfigAlias` | AutoModel registration | Model won't load |

All three bugs appear to be NPU-backend-specific. The CUDA backend may not have these issues (modelslim EP and DeepSeek V4 are actively used on NVIDIA GPUs).

### Suggested Fixes

1. **Bug 1**: In `fused_moe_method_npu.py:926`, filter `groupList` to only include expert indices assigned to the current EP rank before calling `npu_grouped_matmul`.

2. **Bug 2**: Review the modelslim W4A8 INT8 MoE apply logic in the EP=1 path — the grouped matmul with all 256 experts on a single rank may trigger a different code path with incorrect weight/scales indexing.

3. **Bug 3**: Register `_DeepseekV4ConfigAlias` with HuggingFace's `AutoModel._model_mapping` (not just sglang's internal mapping), or use `AutoModelForCausalLM.from_config()` instead of `AutoModel.from_config()`.
