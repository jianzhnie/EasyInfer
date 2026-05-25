# EasyInfer Script Overview

## Quick Reference

| Directory | Purpose | Key Scripts |
|-----------|---------|-------------|
| `scripts/docker/` | Docker container lifecycle | `manage_docker_containers.sh`, `ascend_infer_docker_run.sh` |
| `scripts/ray_cluster/` | Ray cluster orchestration | `start_ray_cluster.sh`, `stop_ray_cluster.sh`, `kill_multi_nodes.sh` |
| `scripts/vllm/` | vLLM model serving | `vllm_model_server.sh`, `mp/deploy_vllm_multinode.sh` |
| `tools/` | Auxiliary utilities | `hf_download.sh`, `docker_proxy.sh` |
| `examples/` | Per-model deployment examples | `glm5_server.sh`, `qwen3_server.sh` |

## Common CLI Patterns

All multi-node scripts accept:
- `--file <path>` / `-f <path>` — node list file (default: `scripts/node_list.txt`)

## Environment Variables

See individual module READMEs for complete variable lists. Key globals:
- `NODES_FILE` — cluster node list path
- `CONTAINER_NAME` — Docker container name
- `MODEL_PATH` — model weights directory
- `SSH_OPTS` — SSH options (word-split intentionally)
