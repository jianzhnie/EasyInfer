# Ray Cluster Module

Scripts for orchestrating Ray clusters across Docker containers on Ascend NPU nodes.

## Scripts

| Script | Purpose |
|--------|---------|
| `start_ray_cluster.sh` | Start Ray head + workers |
| `stop_ray_cluster.sh` | Stop Ray on all nodes |
| `kill_multi_nodes.sh` | Kill processes by keyword across nodes |
| `native_ray_start_cluster.sh` | Native (non-Docker) Ray startup |
| `start_npuslim_ray_cluster.sh` | Ray startup for npuslim containers |
| `set_ray_env.sh` | Ray/Ascend environment configuration |
| `_kill_lib.sh` | Shared kill-script utilities (sourced) |

## Usage

```bash
bash scripts/ray_cluster/start_ray_cluster.sh start --file nodes.txt
bash scripts/ray_cluster/start_ray_cluster.sh stop
bash scripts/ray_cluster/kill_multi_nodes.sh -y -k "ray,vllm"
```

## Environment Variables

See `set_ray_env.sh`. Key ones:
- `RAY_PORT` — Ray GCS port (default: 6379)
- `CONTAINER_NAME` — Docker container to exec into
- `NPUS_PER_NODE` — NPU count per node (default: 8)
