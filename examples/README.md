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

## Usage

All examples source `../scripts/common.sh` and `../scripts/vllm/set_env.sh`.

```bash
bash examples/glm5_server.sh
```
