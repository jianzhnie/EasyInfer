# Agent-Optimized LLM Deployment on Ascend NPU Cluster

> **Date**: 2026-06-09 | **Cluster**: 2×8 Ascend 910C (40 + 153) | **Status**: ✅ All 3 Models Deployed & Verified

---

## 1. Deployment Overview

| # | Model | Architecture | Port | TP×PP | Context | MTP | Tool Parser | Status |
|---|-------|-------------|------|-------|---------|-----|-------------|--------|
| 1 | GLM-5 W4A8 | GlmMoeDSA / 256E / MLA | 8001 | TP=16 PP=1 | **202,752** | ✅ deepseek_mtp | glm47 | ✅ Verified |
| 2 | GLM-5.1 W4A8 | GlmMoeDSA / 256E / MLA | 8002 | TP=16 PP=1 | **202,752** | ✅ deepseek_mtp | glm47 | ✅ Verified |
| 3 | Kimi-K2.6 W4A8 | KimiK25 / 384E / MLA / Vision | 8003 | TP=8 PP=2 | **262,144** | ❌ N/A | kimi_k2 | ✅ Verified |

All models deployed at their **max_position_embeddings** (full context length) with agent-optimized parameters.

---

## 2. Model Architecture Details

### GLM-5 / GLM-5.1 (W4A8)
- **Architecture**: `GlmMoeDsaForCausalLM` (Decoupled Sparse Attention + MoE)
- **VLLM-Ascend**: Identified as DeepSeek V3.2 (`index_topk: 2048` in config)
- **Experts**: 256 routed | **MLA**: kv_lora_rank=512, q_lora_rank=2048
- **MTP**: num_nextn_predict_layers=1 → `deepseek_mtp` speculative decoding
- **PP Support**: ❌ GLM-5/5.1 do NOT support Pipeline Parallelism
- **Strategy**: Use large TP across nodes (TP=16 on 2 nodes)

### Kimi-K2.6 (W4A8)
- **Architecture**: `KimiK25ForConditionalGeneration` wrapping `DeepseekV3ForCausalLM`
- **Experts**: 384 routed | **MLA**: kv_lora_rank=512, q_lora_rank=1536
- **Vision**: `mm_encoder_tp_mode=data` (text-only agent use)
- **MTP**: None
- **PP Support**: ✅ Supports Pipeline Parallelism
- **Strategy**: TP=8 PP=2 (balanced across 2 nodes)

---

## 3. Agent-Optimized Parameters

| Parameter | GLM-5/5.1 | Kimi-K2.6 | Rationale |
|-----------|-----------|-----------|-----------|
| `--enable-prefix-caching` | ✅ | ✅ | Critical for Claude Code system prompt reuse (~90% KV cache hit) |
| `--enable-chunked-prefill` | ✅ | ✅ | Better scheduling for long-context agent prompts |
| `--max-num-seqs` | 8 | 16 | GLM constrained by MTP memory; Kimi has no MTP overhead |
| `--max-num-batched-tokens` | 16384 | 16384 | Balanced prefill throughput |
| `--gpu-memory-utilization` | 0.94 | 0.92 | Kimi needs extra headroom for vision components |
| `--enable-auto-tool-choice` | ✅ | ✅ | Enables automatic tool selection in Anthropic API |
| `--enforce-eager` | ✅ | ✅ | Required on Ascend (no CUDA graph support) |
| `VLLM_ASCEND_ENABLE_MLAPO` | 1 | (auto) | MLA optimization for GLM DSA; Kimi uses DeepSeek attention path |
| `VLLM_ASCEND_ENABLE_FLASHCOMM1` | 0 | (auto) | **Must disable for GLM** (prevents DSA CP crash) |
| `HCCL_BUFFSIZE` | 200 | 800 | Kimi needs larger HCCL buffers for 384 experts |
| `TASK_QUEUE_ENABLE` | (default) | 1 | Kimi performance optimization |

---

## 4. Deployment Commands

### Prerequisites
```bash
# Nodes: 10.16.201.40 (head), 10.16.201.153 (worker)
# Container: npuslim-env
# Image: ascend910c-cann8.5.1-torch2.9.0-vllm0.18.0
```

### 4.1 Start Ray Cluster
```bash
# Head node (10.16.201.40)
ssh 10.16.201.40 "docker exec npuslim-env bash -c '
  source /usr/local/Ascend/cann/set_env.sh
  ray start --head --port=6379 --resources='\''{\"NPU\": 8}'\'' --num-gpus=8
'"

# Worker node (10.16.201.153)
ssh 10.16.201.153 "docker exec npuslim-env bash -c '
  source /usr/local/Ascend/cann/set_env.sh
  ray start --address=10.16.201.40:6379 --resources='\''{\"NPU\": 8}'\'' --num-gpus=8
'"

# Verify: ray status shows 2 nodes, 16 NPU
ssh 10.16.201.40 "docker exec npuslim-env ray status | grep NPU"
```

### 4.2 Deploy GLM-5 (Port 8001)
```bash
ssh 10.16.201.40 'docker exec npuslim-env bash -c "
> /tmp/vllm_glm5.log 2>&1
cd /home/jianzhnie/llmtuner/llm/EasyInfer/examples/glm5_w4a8
MAX_MODEL_LEN=202752 TP=16 PP=1 PORT=8001 nohup bash run_vllm.sh >> /tmp/vllm_glm5.log 2>&1 &
"'

# Verify: curl http://10.16.201.40:8001/v1/models
# Expected: model=glm-5, max_model_len=202752
```

### 4.3 Deploy GLM-5.1 (Port 8002)
```bash
ssh 10.16.201.40 'docker exec npuslim-env bash -c "
> /tmp/vllm_glm51.log 2>&1
cd /home/jianzhnie/llmtuner/llm/EasyInfer/examples/glm5_1_w4a8
MAX_MODEL_LEN=202752 TP=16 PP=1 PORT=8002 nohup bash run_vllm.sh >> /tmp/vllm_glm51.log 2>&1 &
"'

# Verify: curl http://10.16.201.40:8002/v1/models
# Expected: model=glm-5.1, max_model_len=202752
```

### 4.4 Deploy Kimi-K2.6 (Port 8003)
```bash
ssh 10.16.201.40 'docker exec npuslim-env bash -c "
> /tmp/vllm_kimi.log 2>&1
cd /home/jianzhnie/llmtuner/llm/EasyInfer/examples/kimi_k2_6_w4a8
MAX_MODEL_LEN=262144 TP=8 PP=2 DP=1 PORT=8003 nohup bash run_vllm.sh >> /tmp/vllm_kimi.log 2>&1 &
"'

# Verify: curl http://10.16.201.40:8003/v1/models
# Expected: model=kimi-k2.6, max_model_len=262144
```

### 4.5 Stop Deployments
```bash
# Stop vLLM on head node
ssh 10.16.201.40 "docker exec npuslim-env bash -c 'kill \$(pgrep -f \"vllm serve\")'"

# Stop Ray cluster
for ip in 10.16.201.40 10.16.201.153; do
    ssh "$ip" "docker exec npuslim-env bash -c 'ray stop --force'"
done
```

---

## 5. Bugs Discovered & Root Causes

### Bug 1: `g++: internal compiler error: Segmentation fault` on Node 163 (BLOCKING for node 163)
- **Root Cause Chain**:
  1. First inference triggers MLA attention forward pass
  2. MLA calls `rope_forward_triton_siso()` in `vllm_ascend/ops/triton/rope.py:368`
  3. Triton Ascend backend JIT compiles kernel launcher (CXX code)
  4. GCC 10.3.1 crashes with **internal compiler error: Segmentation fault signal terminated program cc1plus**
  5. EngineCore dies → EngineDeadError
- **Why Node 40 / 153 Works**: Same GCC version but compilation succeeds; likely a hardware/OS-specific issue on node 163
- **Why Simple C++ Works on 163**: The crash is specific to the complex template code generated by Triton
- **Fix**: Switched worker node from 163 → 153; node 163 is not usable for MLA-based models
- **Impact**: Node 163 excluded from this deployment

### Bug 2: Kimi-K2.6 `deepseek_v3` Tool Parser Incompatible
- **Root Cause**: Kimi-K2.6 tokenizer uses custom tool tokens (`<|tool_call_begin|>`, `<|tool_call_end|>`, etc.) but vLLM was configured with `--tool-call-parser deepseek_v3` which looks for DeepSeek-specific delimiter tokens (e.g., `"éri"`)
- **Error**: `DeepSeek-V3 Tool parser could not locate tool call start/end tokens in the tokenizer!`
- **Fix**: Changed to `--tool-call-parser kimi_k2` which uses `KimiK2ToolParser` (designed for Kimi's token format)
- **Impact**: Both OpenAI `/v1/chat/completions` (tool calling) and Anthropic `/v1/messages` API now work

### Bug 3: VLLM_ASCEND_ENABLE_FLASHCOMM1 DSA CP Crash (Pre-existing)
- **Root Cause Chain** (from previous session):
  1. `VLLM_ASCEND_ENABLE_FLASHCOMM1=1` → `enable_sp()` returns True
  2. GLM-5 config has `index_topk: 2048` → vLLM identifies as DeepSeek V3.2
  3. `enable_dsa_cp()` = True → calls `_init_o_proj_tp_full_params()` in `sfa_v1.py:682`
  4. W4A8 quantized `AscendRowParallelLinear` doesn't have `aclnn_input_scale` → crash
- **Fix**: `VLLM_ASCEND_ENABLE_FLASHCOMM1=0` in GLM-5/5.1 scripts
- **Why Kimi Works**: Uses DeepseekV3ForCausalLM attention path, not GlmMoeDSA's SFA path

### Bug 4: Zombie Process Accumulation
- **Symptom**: After vLLM crash, container accumulates 300+ zombie Ray worker processes
- **Root Cause**: Ray worker processes become defunct when EngineCore crashes; init (PID 1) in container doesn't reap them
- **Fix**: `docker restart npuslim-env` to clean zombies between deployments

---

## 6. Claude Code Integration

### 6.1 Configuration
```json
{
  "env": {
    "ANTHROPIC_BASE_URL": "http://10.16.201.40:8003",
    "ANTHROPIC_API_KEY": "dummy",
    "ANTHROPIC_AUTH_TOKEN": "dummy",
    "ANTHROPIC_DEFAULT_SONNET_MODEL": "kimi-k2.6",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL": "kimi-k2.6",
    "ANTHROPIC_DEFAULT_OPUS_MODEL": "kimi-k2.6"
  }
}
```

### 6.2 API Compatibility Matrix

| API | GLM-5 | GLM-5.1 | Kimi-K2.6 |
|-----|-------|---------|-----------|
| `/v1/chat/completions` (chat) | ✅ | ✅ | ✅ |
| `/v1/chat/completions` (tool calls) | ✅ glm47 | ✅ glm47 | ✅ kimi_k2 |
| `/v1/messages` (Anthropic Messages) | ✅ | ✅ | ✅ |
| `/v1/messages` (Anthropic tool_use) | ✅ | ✅ | ✅ |
| `/v1/models` | ✅ | ✅ | ✅ |

All 3 models fully support Claude Code's Anthropic Messages API with tool calling.

### 6.3 Model Switching
Only ONE model can run at a time (all require full 16 NPU).

```bash
# Switch to GLM-5
# 1. Stop current: kill vllm on 40 + ray stop on both
# 2. Deploy GLM-5 with PORT=8001
# 3. Update ANTHROPIC_BASE_URL=http://10.16.201.40:8001
# 4. Update ANTHROPIC_DEFAULT_*_MODEL=glm-5

# Switch to Kimi-K2.6 (currently recommended)
# ANTHROPIC_BASE_URL=http://10.16.201.40:8003
# Model name: kimi-k2.6
```

### 6.4 Recommended Model for Claude Code
**Kimi-K2.6** is recommended for Claude Code usage:
- Largest context (262K vs 202K)
- No MTP memory overhead (higher max-num-seqs=16)
- Vision capability (multimodal files supported)
- Better tool calling with kimi_k2 parser (cleaner tool token format)
- More experts (384 vs 256) = better reasoning

---

## 7. Cluster Topology

```
┌─────────────────────────────────────────────────────────────────┐
│                    2-Node Ascend NPU Cluster                      │
│                                                                   │
│  ┌─────────────────────────────┐  ┌─────────────────────────────┐ │
│  │  Node: 10.16.201.40 (HEAD)  │  │  Node: 10.16.201.153 (WORK) │ │
│  │  Host: bms-luyao-0003       │  │  Host: bms-004              │ │
│  │  NPU: 8× Ascend 910C (64G)  │  │  NPU: 8× Ascend 910C (64G)  │ │
│  │  Container: npuslim-env     │  │  Container: npuslim-env     │ │
│  │                             │  │                             │ │
│  │  Ray Head :6379             │──│  Ray Worker → :6379         │ │
│  │  Resources: {"NPU": 8}      │  │  Resources: {"NPU": 8}      │ │
│  │                             │  │                             │ │
│  │  vLLM Port: 8001/8002/8003  │  │  vLLM Worker (TP shard)     │ │
│  └─────────────────────────────┘  └─────────────────────────────┘ │
│                                                                   │
│  Node 163 EXCLUDED: GCC 10.3.1 Triton kernel compilation bug      │
│  Total: 16 NPU × 64GB = 1TB NPU memory                           │
└─────────────────────────────────────────────────────────────────┘
```

---

## 8. Key Environment Variables

### GLM-5 / GLM-5.1
```bash
VLLM_ASCEND_ENABLE_FLASHCOMM1=0  # MUST: prevents DSA CP crash
VLLM_ASCEND_ENABLE_MLAPO=1       # MLA optimization
HCCL_BUFFSIZE=200                # HCCL buffer for 256 experts
HCCL_OP_EXPANSION_MODE=AIV       # HCCL optimization mode
```

### Kimi-K2.6
```bash
VLLM_ASCEND_BALANCE_SCHEDULING=1 # Scheduling optimization
HCCL_BUFFSIZE=800                # Larger for 384 experts
TASK_QUEUE_ENABLE=1              # Performance optimization
```

### Common
```bash
PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
OMP_NUM_THREADS=1
OMP_PROC_BIND=false
```

---

## 9. Verification Commands

### API Health Check
```bash
# Check model availability
curl http://10.16.201.40:8003/v1/models

# Simple chat completion
curl http://10.16.201.40:8003/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"kimi-k2.6","messages":[{"role":"user","content":"Hi"}],"max_tokens":10}'

# Tool calling
curl http://10.16.201.40:8003/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"kimi-k2.6","messages":[{"role":"user","content":"Weather in Paris?"}],"tools":[{"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}],"max_tokens":100}'

# Anthropic Messages API
curl http://10.16.201.40:8003/v1/messages \
  -H "Content-Type: application/json" \
  -H "x-api-key: dummy" \
  -d '{"model":"kimi-k2.6","messages":[{"role":"user","content":"Hi"}],"max_tokens":30}'
```

### Monitor Logs
```bash
# vLLM server log
ssh 10.16.201.40 "docker exec npuslim-env tail -f /tmp/vllm_kimi.log"

# Check for errors
ssh 10.16.201.40 "docker exec npuslim-env grep -i 'error\|RuntimeError' /tmp/vllm_kimi.log"
```

---

## 10. Quick Start (Single Command Sequence)

```bash
# 1. Clean and restart containers
for ip in 10.16.201.40 10.16.201.153; do
    ssh "$ip" "docker restart npuslim-env"
done
sleep 15

# 2. Start Ray cluster
ssh 10.16.201.40 "docker exec npuslim-env bash -c 'source /usr/local/Ascend/cann/set_env.sh; ray start --head --port=6379 --resources='\''{\"NPU\": 8}'\'' --num-gpus=8'"
sleep 5
ssh 10.16.201.153 "docker exec npuslim-env bash -c 'source /usr/local/Ascend/cann/set_env.sh; ray start --address=10.16.201.40:6379 --resources='\''{\"NPU\": 8}'\'' --num-gpus=8'"
sleep 5

# 3. Deploy model (example: Kimi-K2.6)
ssh 10.16.201.40 'docker exec npuslim-env bash -c "
> /tmp/vllm_kimi.log 2>&1
cd /home/jianzhnie/llmtuner/llm/EasyInfer/examples/kimi_k2_6_w4a8
MAX_MODEL_LEN=262144 TP=8 PP=2 DP=1 PORT=8003 nohup bash run_vllm.sh >> /tmp/vllm_kimi.log 2>&1 &
"'

# 4. Wait 12-15 min for model loading, then verify
curl http://10.16.201.40:8003/v1/models
```

---

## 11. Lessons Learned

| Dimension | Key Finding |
|-----------|-------------|
| **Node Selection** | GCC 10.3.1 has a Triton kernel compilation bug on some nodes (e.g., 163). Always verify single-node Triton compilation before multi-node deployment. |
| **Tool Parser** | Kimi-K2.6 requires `kimi_k2` parser, NOT `deepseek_v3`. The parser name must match the tokenizer's tool tokens. |
| **GLM DSA CP** | `VLLM_ASCEND_ENABLE_FLASHCOMM1=0` is mandatory for GLM-5/5.1 W4A8 to prevent the SFA attention crash. |
| **Zombie Cleanup** | After vLLM crashes, `docker restart` is the only reliable way to clean zombie Ray processes. |
| **Triton Cache** | Pre-warming Triton kernels on each node (single-node inference) before multi-node deployment may prevent compilation race conditions. |
| **PP Strategy** | GLM-5/5.1 don't support PP → use large TP (TP=16). Kimi-K2.6 supports PP → use TP=8 PP=2 for balanced memory. |
| **MTP Tradeoff** | MTP speculative decoding improves throughput but doubles weight memory. GLM with MTP can only handle max-num-seqs=8 vs 16 for Kimi. |

---

*Generated by Claude Code | 2026-06-09*
