# CLAUDE.md

本文件为 Claude Code (claude.ai/code) 在本仓库中工作时提供项目指引。

## 项目简介

EasyInfer 是一个 Bash/Shell 工具包，用于在华为昇腾 NPU 集群上部署大语言模型（LLM）推理服务。它封装了 vLLM 和 vLLM-Ascend，提供集群级 Docker 容器管理、Ray 集群编排和 vLLM 模型服务，支持多节点并行（TP/PP/EP）。目标硬件为昇腾 910C NPU（每节点 8 卡，A2/A3 系列）。

## 验证命令

```bash
# 静态分析（主要 lint 工具）
shellcheck scripts/**/*.sh tools/*.sh examples/*.sh

# 单脚本语法检查
bash -n scripts/vllm/vllm_model_server.sh

# pre-commit 全量检查（flake8、isort、yapf、trailing-whitespace 等）
pre-commit run --all-files
```

注意：功能测试需要昇腾 NPU 集群环境，无法在本地运行。

## 项目结构

```
EasyInfer/
├── scripts/
│   ├── common.sh                     ← 共享库（日志、SSH、并发、节点解析）
│   ├── node_list.txt                 ← 集群节点 IP 列表
│   ├── docker/                       ← Docker 容器生命周期管理
│   ├── ray_cluster/                  ← Ray 集群编排（Head/Worker）
│   └── vllm/                         ← vLLM 模型服务
│       ├── mp/                       ← 多节点部署（Ray / 多进程）
│       └── test/                     ← 推理测试脚本
├── tools/                            ← 辅助工具（模型下载、代理配置）
├── examples/                         ← 各模型部署示例
└── docs/                             ← 文档
```

## 架构

### 核心依赖链

所有脚本遵循分层的 `source` 模式：

```
scripts/common.sh                          ← 共享库（日志、SSH、并发、节点解析）
  ├─ scripts/docker/docker_env.sh          ← 容器/镜像配置（被 Docker 脚本 source）
  ├─ scripts/ray_cluster/set_ray_env.sh    ← Ray/Ascend 环境配置（被 Ray 脚本 source）
  └─ scripts/vllm/set_env.sh               ← vLLM 环境配置（被 vLLM 脚本 source）
       └─ scripts/vllm/vllm_server_env_template.sh  ← 完整参数模板（复制为 vllm_server_env.sh 使用）
```

关键设计：
- `common.sh` 是 **被 source 的库**，不是直接执行的脚本。它故意不设 `set -euo pipefail` 以避免覆盖调用者的 shell 选项。
- `SCRIPTS_ROOT` 通过 `BASH_SOURCE[0]` 的位置推导。
- 所有配置均为 **环境变量驱动** — 每个脚本使用 `${VAR:-default}` 模式，支持通过环境变量、env 文件或 CLI 参数覆盖。

### 三个操作层

| 层         | 目录                   | 说明                                                                                                                                |
| ---------- | ---------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| **Docker** | `scripts/docker/`      | 集群节点批量容器管理。`manage_docker_containers.sh` 通过 `declare -f` 序列化 `_remote_prepare_node` 函数，经 SSH 发送到各节点执行。 |
| **Ray**    | `scripts/ray_cluster/` | Head/Worker 节点编排。`start_ray_cluster.sh` 通过 SSH → `docker exec` 连接容器，使用 base64 编码命令避免引号问题。                  |
| **vLLM**   | `scripts/vllm/`        | 模型服务，三种部署模式见下。                                                                                                        |

vLLM 部署模式：
- 单节点：`vllm_model_server.sh`（通过 `vllm serve --help` 自动检测支持的参数）
- 多节点（Ray）：`mp/deploy_vllm_multinode.sh`
- 多节点（多进程）：`mp/deploy_vllm_multinode_mp.sh`

### 并发模型

`common.sh` 提供 `limit_jobs <max>`（轮询 `jobs -rp` + `wait -n`）和 `ssh_run`（刻意使用 `SSH_OPTS` 的分词行为）。并行操作遍历 `node_list.txt` 中的节点（由 `read_nodes` 解析：跳过空行和 `#` 注释）。

### 辅助工具

| 脚本                    | 功能                                                            |
| ----------------------- | --------------------------------------------------------------- |
| `tools/hf_download.sh`  | HuggingFace 模型/数据集下载（默认使用 hf-mirror.com 国内镜像）  |
| `tools/ms_download.py`  | ModelScope 模型/数据集下载（含完整性校验、精准补下、自动重试）  |
| `tools/docker_proxy.sh` | 容器内代理配置（自动检测宿主机代理，支持 host/bridge 网络模式） |
| `tools/host_proxy.sh`   | 宿主机代理配置（`source && pon` 模式）                          |

### 示例脚本

`examples/` 目录包含各模型的部署示例：GLM-5、GLM-5.1 量化版、Kimi-K2、Qwen3，以及通用的 `curl_test.sh` 和 `lm_eval.sh`。

## Shell 脚本规范

- 直接执行的脚本必须 `set -euo pipefail`；被 source 的文件（如 `common.sh`）不设
- 所有变量引用必须双引号：`"$var"`
- 条件判断用 `[[ ]]`，命令替换用 `$(command)`
- 函数内变量用 `local`，常量用 `readonly`
- 4 空格缩进，最大行宽 120 字符
- 单脚本不超过 400 行，单函数不超过 50 行
- 必须通过 `shellcheck` 检查（允许 `disable=SC2086` 并附说明）

## 兼容性约束

以下接口被运维流程依赖，禁止修改：

- 所有脚本的 CLI 参数和环境变量名
- `node_list.txt` 解析格式（`awk 'NF && !/^#/ {print $1}'`）
- `common.sh` 中 `ssh_run` 的调用约定和 `SSH_OPTS` 分词行为
- `ascend_infer_docker_run.sh` 和 `ascend_train_docker_run.sh` 中的设备/驱动挂载路径
- 必须兼容 bash 4.2+
- 不引入新依赖，仅限：bash 4+、coreutils、openssh、docker、ray、vllm

## 关键环境变量

所有脚本支持环境变量覆盖。vLLM 服务端变量最多，完整列表见 `vllm_server_env_template.sh`。

| 变量                     | 默认值                  | 用途                            |
| ------------------------ | ----------------------- | ------------------------------- |
| `NODES_FILE`             | `scripts/node_list.txt` | 集群节点列表路径                |
| `CONTAINER_NAME`         | `vllm-ascend-0.18-env`  | Docker 容器名称                 |
| `MODEL_PATH`             | 视场景而定              | 模型权重目录                    |
| `TENSOR_PARALLEL_SIZE`   | `8`                     | 张量并行度（匹配每节点 NPU 数） |
| `PIPELINE_PARALLEL_SIZE` | `1`                     | 流水线并行度（匹配节点数）      |
| `ENABLE_EXPERT_PARALLEL` | `1`                     | MoE 模型的专家并行              |
| `QUANTIZATION`           | `fp8`                   | 量化方法                        |
| `VLLM_ENV_FILE`          | `vllm_server_env.sh`    | vLLM 服务自定义环境文件         |
| `AUTO_DETECT_FLAGS`      | `1`                     | 自动检测 vLLM 支持的参数        |
| `DTYPE`                  | `auto`                  | 模型数据类型                    |
| `MAX_MODEL_LEN`          | 视模型而定              | 最大模型序列长度                |
