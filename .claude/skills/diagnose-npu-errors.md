---
name: diagnose-npu-errors
description: |
  Quick diagnosis and fix for common vLLM-Ascend NPU deployment errors.
  Use when a model deployment fails with Python tracebacks or vLLM errors.
metadata:
  trigger_keywords: [error, 报错, 失败, crash, diagnose, 诊断, fix, 修复]
  tools_needed: [Bash, Read, Edit]
---

# Diagnose and Fix NPU Deployment Errors

Quick reference for diagnosing and fixing common errors when deploying LLM models
on Ascend NPU with vLLM-Ascend.

## Decision Tree

```
Deployment failed?
├── Process never started
│   ├── "libascend_hal.so" → CANN env not loaded [FIX-A]
│   ├── "readonly variable" → common.sh double-source [FIX-B]
│   └── "CMAKE_PREFIX_PATH: unbound" → set -u conflict [FIX-C]
├── Process started but crashed immediately
│   ├── "unrecognized arguments: --num-scheduler-steps" → [FIX-D]
│   ├── "invalid tool call parser: deepseekv3" → [FIX-E]
│   ├── "Transformers does not recognize" → [FIX-F]
│   ├── "'list' object has no attribute 'keys'" → [FIX-G]
│   ├── "Pipeline parallelism is not supported" → [FIX-H]
│   └── "architectures ... are not supported" → [FIX-I]
└── Process running but no API response
    ├── Model loading (< 100%) → Wait (10-20 min)
    └── "Engine core initialization failed" → [FIX-J]
```

## Error Fixes

### FIX-A: CANN Environment Not Loaded

**Symptom:** `ImportError: libascend_hal.so: cannot open shared object file`

**Fix:** Source CANN before any vllm/python call:
```bash
set +u
source /usr/local/Ascend/cann/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh 2>/dev/null
set -u
```

> Path is `/usr/local/Ascend/cann/` NOT `/usr/local/Ascend/ascend-toolkit/`.

### FIX-B: common.sh readonly Double-Source

**Symptom:** `RED: readonly variable` (repeated for GREEN, YELLOW, etc.)

**Fix:** In `scripts/common.sh`, guard readonly declarations:
```bash
if ! declare -p RED &>/dev/null 2>&1; then
    readonly RED='\033[0;31m' GREEN='\033[0;32m' ...
fi
```

**Workaround:** Use `run_vllm.sh` (direct vllm serve, bypasses wrapper chain).

### FIX-C: CANN Scripts vs set -u

**Symptom:** `/usr/local/Ascend/cann/set_env.sh: line N: VAR: unbound variable`

**Fix:** Use `set -eo pipefail` instead of `set -euo pipefail`. Always wrap CANN source with `set +u`/`set -u`.

### FIX-D: Unsupported --num-scheduler-steps

**Symptom:** `vllm: error: unrecognized arguments: --num-scheduler-steps N`

**Fix:** Remove `--num-scheduler-steps N` from the command. Not supported in v0.18.0rc1.

### FIX-E: Wrong Tool Parser Name

**Symptom:** `KeyError: 'invalid tool call parser: deepseekv3'`

**Fix:** Use underscore: `--tool-call-parser deepseek_v3` (not `deepseekv3`).

### FIX-F: Model Type Not in Transformers

**Symptom:** `Transformers does not recognize this architecture` with `model_type`

**Fix (3 steps):**
1. Create `configuration_<type>.py` in model dir with `PretrainedConfig` subclass
2. Add `auto_map` to `config.json`: `{"AutoConfig": "configuration_<type>.<Class>"}`
3. Set `tokenizer_class = "PreTrainedTokenizerFast"` in config or tokenizer_config.json

### FIX-G: Tokenizer extra_special_tokens

**Symptom:** `AttributeError: 'list' object has no attribute 'keys'`

**Fix:** Remove `extra_special_tokens` from `tokenizer_config.json`:
```bash
python3 -c "
import json
with open('tokenizer_config.json') as f: cfg = json.load(f)
cfg.pop('extra_special_tokens', None)
cfg['tokenizer_class'] = 'PreTrainedTokenizerFast'
with open('tokenizer_config.json', 'w') as f: json.dump(cfg, f, indent=2)
"
```

### FIX-H: Pipeline Parallelism Not Supported

**Symptom:** `NotImplementedError: Pipeline parallelism is not supported`

**Fix:** Use larger TP across nodes instead of PP:
- 2 nodes: `TP=16 PP=1` instead of `TP=8 PP=2`
- 4 nodes: `TP=32 PP=1` instead of `TP=8 PP=4`

> GLM-5/5.1 do NOT support PP. Kimi-K2.6 DOES support PP.

### FIX-I: Architecture Not Supported (DeepSeek V4)

**Symptom:** `architectures ['DeepseekV4ForCausalLM'] are not supported`

**Status:** ❌ Cannot be fixed in vLLM-Ascend 0.18.0rc1. Architecture not in registry. Changing to `DeepseekV32ForCausalLM` causes `Engine core initialization failed`. Requires vLLM upgrade.

### FIX-J: Engine Core Initialization Failed

**Symptom:** `RuntimeError: Engine core initialization failed. Failed core proc(s): {}`

**Diagnosis:** Check root cause ABOVE this line in the log:
```bash
grep -B30 "Engine core initialization failed" <log> | grep -E "Error|AttributeError" | tail -5
```

Common root causes: missing config attributes, Ray placement group failure, worker crash.

## Quick Check Commands

```bash
# Process alive?
ssh_run "<ip>" "docker exec npuslim-env ps aux | grep '[v]llm serve'"

# Latest log (skip harmless)
ssh_run "<ip>" "docker exec npuslim-env tail -20 /tmp/vllm_<m>.log | grep -v 'Triton\|Deprecation\|swigvarlink'"

# Error summary
ssh_run "<ip>" "docker exec npuslim-env grep -E 'Error|Traceback|unrecognized|NotImplemented|ValidationError' /tmp/vllm_<m>.log | grep -v Triton | tail -10"

# Service ready?
ssh_run "<ip>" "curl -sf --max-time 5 http://localhost:<port>/v1/models"
```
