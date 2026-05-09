# EasyInfer

Ascend NPU 集群上的大模型推理部署工具集，基于 [vLLM](https://github.com/vllm-project/vllm) 和 [vLLM-Ascend](https://github.com/vllm-project/vllm-ascend)。

## 环境要求

- **硬件**: 华为 Ascend 910C NPU（每节点 8 卡）
- **系统**: 各节点已配置 SSH 免密登录
- **Docker**: 已安装并运行
- **模型**: 权重已下载到共享目录或各节点相同路径

## 项目结构

```
EasyInfer/
├── scripts/
│   ├── common.sh                          # 共享工具函数（日志、SSH、并发控制）
│   ├── node_list.txt                      # 集群节点列表
│   ├── docker/                            # Docker 容器管理
│   │   ├── set_env.sh                     #   全局环境变量配置
│   │   ├── prepare_docker_nodes.sh        #   批量准备/启停容器
│   │   ├── copy_file_to_containers.sh     #   批量拷贝文件到容器
│   │   ├── source_env_in_containers.sh    #   批量加载容器环境
│   │   ├── ascend_infer_docker_run.sh     #   推理容器启动
│   │   └── ascend_train_docker_run.sh     #   训练容器启动
│   ├── cluster/                           # Ray 集群管理
│   │   ├── start_ray_cluster.sh           #   启动 Ray 集群
│   │   ├── stop_ray_cluster.sh            #   停止 Ray 集群
│   │   ├── kill_multi_nodes.sh            #   多节点进程清理
│   │   └── native_ray_start_cluster.sh    #   快速启动示例（硬编码 IP）
│   └── vllm/                              # vLLM 推理服务
│       ├── set_env.sh                     #   vLLM/Ray 环境配置
│       ├── vllm_model_server.sh           #   vLLM 模型服务启动脚本
│       ├── vllm_server_env_template.sh    #   环境变量配置模板
│       ├── mp/                            #   多进程部署方式
│       │   ├── deploy_vllm_multinode.sh       Ray 后端多节点部署
│       │   └── deploy_vllm_multinode_mp.sh    Multiprocessing 后端多节点部署
│       └── test/                          #   测试脚本
│           ├── curl_test.sh               #   API 测试
│           └── vllm_test.sh               #   单节点启动测试
```

## 快速开始

### 1. 配置

编辑 `scripts/docker/set_env.sh`，设置镜像路径、容器名、网卡名等。

编辑 `scripts/node_list.txt`，填入集群节点主机名（每行一个）：

```
node01
node02
...
```

### 2. 准备容器

```bash
# 启动所有节点的 Docker 容器（加载镜像、运行容器）
bash scripts/docker/prepare_docker_nodes.sh start

# 重启（先清理旧容器再启动）
bash scripts/docker/prepare_docker_nodes.sh restart

# 仅停止
bash scripts/docker/prepare_docker_nodes.sh stop
```

### 3. 拷贝文件到容器

```bash
# 本地文件 → 所有节点容器
bash scripts/docker/copy_file_to_containers.sh ./local_file.py /container/path/file.py

# 远程文件 → 容器（文件已在远程节点上）
bash scripts/docker/copy_file_to_containers.sh -r /host/path/file.py /container/path/file.py

# 指定节点
bash scripts/docker/copy_file_to_containers.sh -n node01 -n node02 ./file.txt /tmp/file.txt
```

### 4. 启动 Ray 集群

```bash
bash scripts/cluster/start_ray_cluster.sh
```

### 5. 启动 vLLM 服务

```bash
# 方式一：通过 Ray 集群启动（推荐多节点）
bash scripts/vllm/vllm_model_server.sh

# 方式二：Ray 后端多节点自动化部署
bash scripts/vllm/mp/deploy_vllm_multinode.sh

# 方式三：Multiprocessing 后端多节点部署
bash scripts/vllm/mp/deploy_vllm_multinode_mp.sh
```

### 6. 测试

```bash
bash scripts/vllm/test/curl_test.sh
```

## 常用运维

```bash
# 停止 Ray 集群
bash scripts/cluster/stop_ray_cluster.sh -y

# 清理多节点上的 ray/vllm 等进程
bash scripts/cluster/kill_multi_nodes.sh -y

# 干运行（仅查看会清理哪些进程）
bash scripts/cluster/kill_multi_nodes.sh -n
```

## 配置说明

### 环境变量覆盖

所有脚本均支持通过环境变量覆盖默认值：

```bash
# vLLM 服务配置
export MODEL_PATH=/path/to/model
export TENSOR_PARALLEL_SIZE=8
export PIPELINE_PARALLEL_SIZE=4
export VLLM_PORT=8000

# 使用外部环境变量文件
VLLM_ENV_FILE=./my_env.sh bash scripts/vllm/vllm_model_server.sh
```

### 并行策略推荐

| NPU 数量 | 节点数 | TP | PP | EP |
|----------|--------|----|----|-----|
| 8        | 1      | 8  | 1  | 8   |
| 16       | 2      | 8  | 2  | 16  |
| 32       | 4      | 8  | 4  | 32  |
| 64       | 8      | 8  | 8  | 64  |
| 128      | 16     | 8  | 16 | 128 |

## License

See [LICENSE](LICENSE).
