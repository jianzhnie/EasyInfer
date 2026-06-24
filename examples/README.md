# Examples

Per-model deployment examples for EasyInfer.

## Available Examples

| Script | Model | Notes |
|--------|-------|-------|
| `glm5_server.sh` | GLM-5 | Standard deployment |
| `glm5_full_server.sh` | GLM-5 | Full precision |
| `glm5-1_quant_server.sh` | GLM-5.1 | Quantized |
| `qwen3_server.sh` | Qwen3 | |
| `kimi2_pcl.sh` | Kimi-K2 | |
| `longcat_flash-chat.sh` | LongCAT | |
| `curl_test.sh` | — | Generic API test |
| `lm_eval.sh` | — | lm-evaluation-harness |
| `check_glm5_env.sh` | — | Environment checker |

## Structured Examples

| Directory | Model | Quantization | Port |
|-----------|-------|-------------|------|
| `glm5/vllm/` | GLM-5 | BF16 | 8001 |
| `glm5_w4a8/vllm/` | GLM-5 W4A8 | W4A8 | 8001 |
| `glm5_1_w4a8/vllm/` | GLM-5.1 W4A8 | W4A8 | 8002 |
| `glm5_2_w8a8/vllm/` | GLM-5.2 W8A8 | W8A8 | 8007 |
| `minimax_m2_7_w8a8/vllm/` | MiniMax-M2.7 W8A8 | W8A8 | 8004 |

Each structured example includes `run_vllm.sh`, `vllm_server.sh`, `curl_test.sh`, and advanced deployment scripts.

## Usage

All examples source `../scripts/common.sh` and `../scripts/vllm/set_env.sh`.

```bash
bash examples/glm5_server.sh
```
