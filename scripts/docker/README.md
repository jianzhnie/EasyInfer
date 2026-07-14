# Docker 容器管理模块

在昇腾 NPU 集群各节点上管理 Docker 容器的脚本集合。

## 脚本一览

| 脚本 | 用途 |
|---|---|
| `docker_env.sh` | 环境变量配置（被其他脚本 source，不直接执行） |
| `save_docker_image.sh` | 将 Docker 镜像导出为 tar 文件，可选分发到集群节点 |
| `manage_docker_containers.sh` | 集群节点批量容器管理（start / stop / restart） |
| `manage_npuslim_containers.sh` | 集群节点 NPUSlim 容器管理（start / stop / status / restart） |
| `ascend_infer_docker_run.sh` | 单节点启动昇腾推理容器（挂载 NPU 设备） |
| `ascend_train_docker_run.sh` | 单节点启动昇腾训练容器（挂载 NPU 设备） |
| `run_npuslim_container.sh` | 单节点启动 NPUSlim 容器 |
| `copy_file_to_containers.sh` | 将文件批量复制到集群各节点的容器内 |

## 典型工作流

```
1. 保存/分发镜像        2. 启动容器           3. 分发文件           4. 启动 Ray → vLLM
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────────┐
│ save_docker_     │  │ manage_docker_    │  │ copy_file_to_    │  │ start_ray_cluster.sh │
│ image.sh         │→ │ containers.sh    │→ │ containers.sh    │→ │ deploy_vllm_         │
│                  │  │ start            │  │                  │  │ multinode.sh         │
└──────────────────┘  └──────────────────┘  └──────────────────┘  └──────────────────────┘
```

---

## save_docker_image.sh

将 Docker 镜像导出为 tar 文件，可选压缩和分发到集群节点。

### 用法

```bash
# 指定镜像导出（使用默认输出路径）
bash save_docker_image.sh -i quay.io/ascend/vllm-ascend:v0.22.1rc1-a3

# 导出并压缩（大镜像推荐，自动优先使用 pigz 并行压缩）
bash save_docker_image.sh -i quay.io/ascend/vllm-ascend:v0.22.1rc1-a3 -z

# 指定输出路径
bash save_docker_image.sh -i myimage:latest -o /data/images/myimage.tar

# 导出并分发到集群所有节点（分发后默认删除本地 tar）
bash save_docker_image.sh -i myimage:latest -z -f node_list.txt

# 分发后保留本地 tar
bash save_docker_image.sh -i myimage:latest -z -f node_list.txt --no-cleanup
```

### 镜像名指定方式（优先级从高到低）

1. 命令行：`-i / --image`
2. 环境变量：`IMAGE_NAME=xxx bash save_docker_image.sh`
3. 配置文件：自动读取 `docker_env.sh` 中的 `IMAGE_NAME`

### 选项

| 选项 | 说明 |
|---|---|
| `-i, --image <NAME>` | Docker 镜像名称 |
| `-o, --output <FILE>` | 输出 tar 文件路径（默认：`${IMAGE_DIR}/<镜像名>.tar`） |
| `-z, --gzip` | 启用 gzip 压缩，自动优先使用 pigz 并行压缩 |
| `-f, --file <FILE>` | 节点列表文件，提供后将 tar 通过 SCP 分发到各节点 |
| `-h, --help` | 显示帮助信息 |
| `--no-cleanup` | 分发后保留本地 tar 文件（默认分发后删除） |

### 压缩性能

| 工具 | 16GB 镜像耗时（估算） | 说明 |
|---|---|---|
| `pigz` | ~3–4 分钟 | 并行压缩，自动检测优先使用 |
| `gzip` | ~14 分钟 | 单线程，pigz 不可用时的回退 |

安装 pigz：`yum install pigz` / `apt install pigz`

---

## manage_docker_containers.sh

在集群所有节点上管理 Docker 容器环境。

### 用法

```bash
# 启动（默认操作：确保 Docker 运行，加载镜像，启动容器）
bash manage_docker_containers.sh start --file node_list.txt

# 停止（仅停止并清理旧容器）
bash manage_docker_containers.sh stop --file node_list.txt

# 重启（停止 → 加载镜像 → 启动）
bash manage_docker_containers.sh restart --file node_list.txt
```

### 启动流程

1. `_remote_ensure_docker_running` — 确保 Docker 服务运行
2. `_remote_cleanup_containers` — 清理旧容器（restart / stop 时）
3. `_remote_load_and_run` — 加载镜像 tar（`docker load -i`）并执行 `RUN_CONTAINER_SCRIPT`

---

## manage_npuslim_containers.sh

在集群节点上管理 NPUSlim Docker 容器，支持更灵活的节点指定方式。

### 用法

```bash
# 默认模式（读取 scripts/node_list.txt）
bash manage_npuslim_containers.sh start
bash manage_npuslim_containers.sh status
bash manage_npuslim_containers.sh stop

# 通过 -f/--file 指定节点列表文件
bash manage_npuslim_containers.sh start -f /path/to/my_nodes.txt

# 通过 --hosts 直接指定 IP
bash manage_npuslim_containers.sh start --hosts 10.42.0.74 10.42.0.75

# 通过环境变量指定
NODES_FILE=/tmp/my_cluster.txt bash manage_npuslim_containers.sh start

# 不挂载 npuslim
bash manage_npuslim_containers.sh start --no-npuslim

# 重启（先 stop 再 start）
bash manage_npuslim_containers.sh restart
```

---

## 单节点容器启动脚本

以下脚本在**单节点**上直接执行，用于启动容器并挂载 NPU 设备。

### ascend_infer_docker_run.sh

启动昇腾推理容器，挂载 NPU 设备、驱动和文件系统。

```bash
# 默认镜像和容器名
bash ascend_infer_docker_run.sh

# 覆盖镜像和容器名
IMAGE_NAME=myimage:latest CONTAINER_NAME=my-infer bash ascend_infer_docker_run.sh
```

### ascend_train_docker_run.sh

启动昇腾训练容器（MindSpeed-LLM），挂载 NPU 设备、驱动和文件系统。

```bash
bash ascend_train_docker_run.sh
```

### run_npuslim_container.sh

启动 vLLM-Ascend 容器，支持指定 NPU 卡和多节点模式。

```bash
# 使用默认卡 (0)
bash run_npuslim_container.sh

# 指定 NPU 卡和多节点模式
bash run_npuslim_container.sh 0,1 --multi-node --npuslim=/path/to/npuslim
```

---

## copy_file_to_containers.sh

将文件批量复制到集群各节点的 Docker 容器内。

```bash
bash copy_file_to_containers.sh /path/to/file /container/path/
```

---

## 环境变量

完整配置见 `docker_env.sh`，关键变量：

| 变量 | 默认值 | 说明 |
|---|---|---|
| `IMAGE_DIR` | `/home/jianzhnie/llmtuner/hfhub/docker/image` | 镜像 tar 文件存放目录 |
| `IMAGE_NAME` | `quay.io/ascend/vllm-ascend:v0.22.1rc1-a3` | Docker 镜像名 |
| `IMAGE_TAR` | `${IMAGE_DIR}/vllm-ascend.v0.22.1rc1-a3.tar` | 镜像 tar 文件路径 |
| `CONTAINER_NAME` | `vllm-ascend-env` | 容器名 |
| `RUN_CONTAINER_SCRIPT` | `ascend_infer_docker_run.sh` | 容器启动脚本路径 |
| `NODES_FILE` | `scripts/node_list.txt` | 集群节点列表 |
| `SSH_OPTS` | `-o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10` | SSH 选项 |
| `SSH_USER_HOST_PREFIX` | （空） | SSH 用户前缀，如 `root@` |
| `PARALLELISM` | `8` | 并发操作数 |

---

## 节点列表文件格式

```
# 这是一个注释行，会被忽略
10.42.0.1

10.42.0.2
# 空行也会被忽略
10.42.0.3
```

解析规则：`awk 'NF && !/^#/ {print $1}'` — 跳过空行和 `#` 注释行。
