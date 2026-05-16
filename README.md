# EasyInfer

Ascend NPU 集群上的大模型推理部署工具集，基于 [vLLM](https://github.com/vllm-project/vllm) 和 [vLLM-Ascend](https://github.com/vllm-project/vllm-ascend)。

## 环境要求

- **硬件**: 华为 Ascend 910C NPU（每节点 8 卡，支持 A2/A3 系列）
- **系统**: 各节点已配置 SSH 免密登录
- **Docker**: 已安装并运行
- **模型**: 权重已下载到共享目录或各节点相同路径

## 项目结构

```
EasyInfer/
├── scripts/
│   ├── common.sh                              # 共享工具函数（日志、SSH、并发控制）
│   ├── node_list.txt                          # 集群节点列表
│   ├── docker/                                # Docker 容器管理
│   │   ├── docker_env.sh                      #   全局环境变量配置
│   │   ├── manage_docker_containers.sh        #   批量准备/启停容器
│   │   ├── manage_npuslim_containers.sh       #   NPUSlim 容器管理
│   │   ├── copy_file_to_containers.sh         #   批量拷贝文件到容器
│   │   ├── ascend_infer_docker_run.sh         #   推理容器启动
│   │   ├── ascend_train_docker_run.sh         #   训练容器启动
│   │   └── run_npuslim_container.sh           #   NPUSlim 容器启动
│   ├── ray_cluster/                           # Ray 集群管理
│   │   ├── set_ray_env.sh                     #   Ray/Ascend 环境配置
│   │   ├── start_ray_cluster.sh               #   启动 Ray 集群
│   │   ├── stop_ray_cluster.sh                #   停止 Ray 集群
│   │   ├── kill_multi_nodes.sh                #   多节点进程清理
│   │   ├── ray_head.sh                        #   Ray Head 节点启动
│   │   ├── ray_node.sh                        #   Ray Worker 节点启动
│   │   ├── start_npuslim_ray_cluster.sh       #   NPUSlim Ray 集群
│   │   └── native_ray_start_cluster.sh        #   快速启动示例
│   └── vllm/                                  # vLLM 推理服务
│       ├── set_env.sh                         #   vLLM/Ray 环境配置
│       ├── vllm_model_server.sh               #   vLLM 模型服务主脚本
│       ├── vllm_server_env_template.sh        #   环境变量配置模板
│       ├── mp/                                #   多节点部署
│       │   ├── deploy_vllm_multinode.sh       #     Ray 后端多节点部署
│       │   ├── deploy_vllm_multinode_mp.sh    #     Multiprocessing 后端多节点部署
│       │   ├── node_1.sh                      #     Master 节点参考配置
│       │   └── node_2.sh                      #     Worker 节点参考配置
│       └── test/                              #   测试脚本
│           ├── curl_test.sh                   #     API 端点测试
│           └── vllm_test.sh                   #     单节点启动测试
├── examples/                                  # 模型部署示例
│   ├── glm5_server.sh                         #   GLM-5 量化部署 (W4A8/W8A8/BF16)
│   ├── glm5_full_server.sh                    #   GLM-5 全参数部署 (64 TP)
│   ├── glm5-1_quant_server.sh                 #   GLM-5.1 W8A8 + Claude Code
│   ├── qwen3_server.sh                        #   Qwen3-32B 部署
│   ├── kimi2_pcl.sh                           #   Kimi-K2 部署 (64 TP, Ray)
│   ├── lm_eval.sh                             #   lm-evaluation-harness 运行
│   ├── check_glm5_env.sh                      #   GLM-5 环境预检
│   └── curl_test.sh                           #   API 测试脚本
├── tools/                                     # 工具脚本
│   ├── hf_download.sh                         #   HuggingFace 模型/数据集下载
│   ├── ms_download.sh                         #   ModelScope 模型/数据集下载
│   ├── host_proxy.sh                          #   宿主机代理管理 (pon/poff/pstatus)
│   └── docker_proxy.sh                        #   容器内代理自动检测
└── docs/                                      # 文档
    ├── claude-code-vllm-setup.md              #   Claude Code + vLLM 集成指南
    ├── reverse_proxy_setup.md                 #   SSH 反向代理配置
    └── lm_eval_local_dataset.md               #   lm-eval 本地数据集使用
```

## 快速开始

### 1. 配置

编辑 `scripts/docker/docker_env.sh`，设置镜像路径、容器名等。

编辑 `scripts/node_list.txt`，填入集群节点主机名或 IP（每行一个）：

```
node01
node02
...
```

### 2. 准备容器

```bash
# 启动所有节点的 Docker 容器（加载镜像、运行容器）
bash scripts/docker/manage_docker_containers.sh start

# 重启（先清理旧容器再启动）
bash scripts/docker/manage_docker_containers.sh restart

# 仅停止
bash scripts/docker/manage_docker_containers.sh stop
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
bash scripts/ray_cluster/start_ray_cluster.sh start
```

### 5. 启动 vLLM 服务

```bash
# 方式一：单节点/通过 Ray 集群启动
bash scripts/vllm/vllm_model_server.sh

# 方式二：Ray 后端多节点自动化部署
bash scripts/vllm/mp/deploy_vllm_multinode.sh

# 方式三：Multiprocessing 后端多节点部署
bash scripts/vllm/mp/deploy_vllm_multinode_mp.sh
```

所有变量均支持环境变量覆盖：

```bash
# 直接覆盖
MODEL_PATH=/path/to/model TENSOR_PARALLEL_SIZE=16 bash scripts/vllm/vllm_model_server.sh

# 使用外部环境文件
VLLM_ENV_FILE=./my_env.sh bash scripts/vllm/vllm_model_server.sh
```

### 6. 测试

```bash
bash scripts/vllm/test/curl_test.sh
# 或
BASE_URL=http://10.0.0.1:9000 MODEL_NAME=qwen3-32b bash scripts/vllm/test/curl_test.sh
```

## 常用运维

```bash
# 停止 Ray 集群
bash scripts/ray_cluster/stop_ray_cluster.sh -y

# 清理多节点上的 ray/vllm 等进程
bash scripts/ray_cluster/kill_multi_nodes.sh -y

# 干运行（仅查看会清理哪些进程）
bash scripts/ray_cluster/kill_multi_nodes.sh -n
```

## 模型部署示例

| 模型 | 脚本 | 架构 | 量化 | 并行策略 |
|------|------|------|------|----------|
| GLM-5 | `glm5_server.sh` | MoE | W4A8 / W8A8 / BF16 | TP=8 |
| GLM-5 全参数 | `glm5_full_server.sh` | MoE | BF16 | TP=64, Ray |
| GLM-5.1 | `glm5-1_quant_server.sh` | MoE | W8A8 (Ascend) | TP=32, Ray |
| Qwen3-32B | `qwen3_server.sh` | Dense | 无 / FP8 | TP=8 |
| Kimi-K2 | `kimi2_pcl.sh` | MoE | 无 | TP=64, Ray |

每个示例脚本均支持通过环境变量覆盖所有配置参数。

## 并行策略推荐

| NPU 数量 | 节点数 | TP | PP | EP |
|----------|--------|----|----|-----|
| 8        | 1      | 8  | 1  | 8   |
| 16       | 2      | 8  | 2  | 16  |
| 32       | 4      | 8  | 4  | 32  |
| 64       | 8      | 8  | 8  | 64  |
| 128      | 16     | 8  | 16 | 128 |

## 工具脚本

```bash
# HuggingFace 下载（使用国内镜像）
bash tools/hf_download.sh

# ModelScope 下载
bash tools/ms_download.sh

# 容器内代理（自动检测宿主机地址）
source tools/docker_proxy.sh && pon
```

## License

See [LICENSE](LICENSE).
