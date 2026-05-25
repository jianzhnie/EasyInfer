# vLLM Module

Scripts for serving LLMs with vLLM-Ascend on Ascend NPU clusters.

## Deployment Modes

| Mode | Script | Use Case |
|------|--------|----------|
| Single-node | `vllm_model_server.sh` | One node, TP=8 |
| Multi-node (Ray) | `mp/deploy_vllm_multinode.sh` | Ray backend, TP/PP across nodes |
| Multi-node (MP) | `mp/deploy_vllm_multinode_mp.sh` | Multiprocessing backend |

## Supporting Scripts

| Script | Purpose |
|--------|---------|
| `set_env.sh` | vLLM environment configuration |
| `vllm_server_env_template.sh` | Complete parameter template |
| `test/curl_test.sh` | API health check |
| `test/vllm_test.sh` | vLLM functionality test |
| `mp/_common.sh` | Shared multi-node deployment utilities |
| `mp/_node_env.sh` | Per-node environment template |

## Usage

```bash
# Single node
bash scripts/vllm/vllm_model_server.sh

# Multi-node Ray
bash scripts/vllm/mp/deploy_vllm_multinode.sh --file nodes.txt

# Multi-node MP
bash scripts/vllm/mp/deploy_vllm_multinode_mp.sh --file nodes.txt
```

## Key Environment Variables

- `MODEL_PATH` — model directory
- `TENSOR_PARALLEL_SIZE` — TP (default: 8)
- `PIPELINE_PARALLEL_SIZE` — PP (default: 1)
- `ENABLE_EXPERT_PARALLEL` — MoE EP (default: 1)
- `QUANTIZATION` — quantization method (default: fp8)
