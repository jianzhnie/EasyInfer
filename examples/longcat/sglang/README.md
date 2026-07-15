# LongCat-Flash-Chat SGLang 部署

## 前置条件

- Docker 容器 `sglang-ascend-env` 已在各节点创建好
- 容器已挂载 `/home/jianzhnie/llmtuner/llm/EasyInfer`，容器内路径与 host 一致
- 模型已下载到共享路径

## 文件说明

| 文件 | 用途 |
|---|---|
| `run_sglang.sh` | 部署/停止 SGLang 服务 |
| `test_sglang.sh` | API 功能验证 |

## 用法

### 启动服务

```bash
# 默认配置（8 节点，TP=64）
bash run_sglang.sh

# 指定节点列表和模型路径
NODES_FILE=/path/to/nodes.txt MODEL_PATH=/path/to/model bash run_sglang.sh

# 自定义参数
TP_SIZE=32 SERVER_PORT=8000 bash run_sglang.sh
```

### 停止服务

```bash
bash run_sglang.sh --stop
```

### 验证服务

```bash
# 默认验证
bash test_sglang.sh

# 指定地址
HOST=10.42.11.130 PORT=6677 bash test_sglang.sh
```

## 环境变量

| 变量 | 默认值 | 说明 |
|---|---|---|
| `NODES_FILE` | `node_list1.txt` | 节点列表文件 |
| `MODEL_PATH` | `.../LongCat-Flash-Chat` | 模型路径 |
| `TP_SIZE` | `64` | 张量并行大小 |
| `SERVER_PORT` | `6677` | 服务端口 |
| `MASTER_PORT` | `5000` | 节点通信端口 |
| `SERVED_MODEL_NAME` | `longcat-flash` | 服务模型名 |
| `MEM_FRACTION` | `0.65` | 显存占用比例 |
| `MAX_RUNNING` | `16` | 最大并发请求 |
| `CONTEXT_LENGTH` | `8192` | 上下文长度 |

## 节点列表格式

```
10.42.11.130
10.42.11.131
10.42.11.132
# 注释行和空行会被忽略
```

## 工作原理

`run_sglang.sh` 在 host 上执行：

1. 读取节点列表
2. 通过 SSH 连接各节点
3. 在各节点的容器内直接执行 SGLang 启动命令

由于容器已挂载项目目录，无需 `scp` 或 `docker cp` 分发文件。
