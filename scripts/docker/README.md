# Docker Module

Scripts for managing Docker containers across the Ascend NPU cluster.

## Scripts

| Script | Purpose |
|--------|---------|
| `manage_docker_containers.sh` | Start/stop/restart containers on all nodes |
| `manage_npuslim_containers.sh` | Manage npuslim-specific containers |
| `ascend_infer_docker_run.sh` | Run inference Docker container (device mounts) |
| `ascend_train_docker_run.sh` | Run training Docker container (device mounts) |
| `copy_file_to_containers.sh` | Copy files into running containers |
| `run_npuslim_container.sh` | Run a single npuslim container |
| `docker_env.sh` | Environment variables for Docker module |

## Usage

```bash
bash scripts/docker/manage_docker_containers.sh start
bash scripts/docker/manage_docker_containers.sh stop
bash scripts/docker/manage_docker_containers.sh restart --file /path/to/nodes.txt
```

## Environment Variables

See `docker_env.sh` for all variables. Key ones:
- `NODES_FILE` — node list path
- `IMAGE_NAME`, `IMAGE_TAR` — Docker image
- `CONTAINER_NAME` — target container name
- `RUN_CONTAINER_SCRIPT` — script to start container

---

## manage_npuslim_containers.sh 使用说明

  1. 默认模式（读取 scripts/node_list.txt）

  bash manage_npuslim_containers.sh start
  bash manage_npuslim_containers.sh status
  bash manage_npuslim_containers.sh stop

  2. 通过 -f/--file 指定节点列表文件

  bash manage_npuslim_containers.sh start -f /path/to/my_nodes.txt
  bash manage_npuslim_containers.sh status --file /tmp/cluster_nodes.txt

  3. 通过 --hosts 直接指定 IP

  bash manage_npuslim_containers.sh start --hosts 10.42.0.74 10.42.0.75
  bash manage_npuslim_containers.sh stop --hosts 10.42.0.76 10.42.0.77

  4. 通过环境变量指定节点文件

  NODES_FILE=/tmp/my_cluster.txt bash manage_npuslim_containers.sh start

  其他选项

  # 不挂载 npuslim
  bash manage_npuslim_containers.sh start --no-npuslim

  # 重启（先 stop 再 start）
  bash manage_npuslim_containers.sh restart

  # 组合使用
  bash manage_npuslim_containers.sh start -f /path/to/nodes.txt --no-npuslim
