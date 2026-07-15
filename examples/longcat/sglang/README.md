# LongCat-Flash-Chat SGLang 部署

## 模型概况

| 属性 | 值 |
|------|-----|
| 架构 | LongCatFlashForCausalLM (MLA + MoE) |
| 参数量 | LongCat-Flash-Chat |
| 精度 | bfloat16 |
| 最大位置编码 | 131072 |

## 硬件需求

| 资源 | 需求 |
|------|------|
| NPU | 64 × 昇腾 910C (8 节点 × 8 卡) |
| 显存 | ~64GB / NPU |
| 网络 | HCCL 多节点通信 (需要 host 网络模式) |
| 容器 | sglang-ascend-env (基于 SGLang Ascend 镜像) |

## 前置条件

- Docker 容器 `sglang-ascend-env` 已在各节点创建好，必须使用 `--network host`
- 容器已挂载 `/home/jianzhnie/llmtuner/llm/EasyInfer`，容器内路径与 host 一致
- 模型已下载到共享路径

## 文件说明

| 文件 | 用途 |
|------|------|
| `run_sglang.sh` | 部署/停止 SGLang 服务 (在管理节点上执行) |
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
|------|--------|------|
| `NODES_FILE` | `node_list1.txt` (EasyInfer 根目录) | 节点列表文件路径 |
| `MODEL_PATH` | `.../LongCat-Flash-Chat` | 模型权重路径 |
| `CONTAINER_NAME` | `sglang-ascend-env` | Docker 容器名称 |
| `TP_SIZE` | `64` | 张量并行大小 (8 节点 × 8 卡) |
| `SERVER_HOST` | `0.0.0.0` | 服务监听地址 |
| `SERVER_PORT` | `6677` | 服务端口 |
| `MASTER_PORT` | `5000` | 节点间通信端口 (torch.distributed) |
| `SERVED_MODEL_NAME` | `longcat-flash` | API 中显示的模型名称 |
| `MEM_FRACTION` | `0.65` | NPU 显存占用比例 |
| `MAX_RUNNING` | `16` | 最大并发请求数 |
| `CONTEXT_LENGTH` | `8192` | 最大上下文长度 |
| `CHUNKED_PREFILL` | `8192` | Chunked Prefill 分块大小 |
| `WATCHDOG_TIMEOUT` | `9000` | 服务 Watchdog 超时 (秒) |
| `SGLANG_PYTHONPATH` | `/home/jianzhnie/llmtuner/llm/sglang/python` | SGLang Python 源码路径 |
| `HCCL_SOCKET_IFNAME` | `enp66s0f0` | HCCL 通信网卡接口名 |
| `GLOO_SOCKET_IFNAME` | `enp66s0f0` | GLOO 通信网卡接口名 |

## 节点列表格式

```
10.42.11.130
10.42.11.131
10.42.11.132
# 注释行和空行会被忽略
```

## 工作原理

`run_sglang.sh` 在管理节点上执行：

1. 读取节点列表，计算 `NNODES` 和 `TP_SIZE`
2. 为每个节点生成容器内 SGLang 启动命令（base64 编码）
3. 通过 SSH → `docker exec -i` → `base64 -d | bash` 在各节点容器内执行
4. 等待主节点 (rank 0) 服务端口就绪

关键设计点：
- **NODE_RANK 由管理节点直接传入**，无需容器内 IP 匹配检测
- **base64 编码**避免多层引号转义（与 `scripts/ray_cluster/` 模式一致）
- 容器必须使用 `--network host` 以确保 HCCL 多节点通信正常

## 常见问题 FAQ

### Q: SGLang 启动失败 `No module named 'sglang'`
A: 确认使用的是 SGLang Ascend 镜像，不要用 vLLM 镜像。同时确认 `SGLANG_PYTHONPATH` 指向正确的 `sglang/python` 目录。

### Q: `torch.distributed initialization failed` 或 `Connection refused`
A: 检查：
1. 容器是否使用了 `--network host`（HCCL 多节点通信必须）
2. 节点间网络是否互通（`ping` / `nc -z <master_ip> 5000`）
3. Master 节点 (rank 0) 是否先于其他节点启动
4. `HCCL_SOCKET_IFNAME` 是否指向正确的网卡接口

### Q: `device type npu is not supported`
A: 容器内需要 source CANN 环境。脚本已自动 source `/usr/local/Ascend/ascend-toolkit/set_env.sh`，如果路径不同请确认容器内 CANN 安装位置。

### Q: 启动后 `curl /health` 长时间无响应
A: 模型加载通常需要 10-20 分钟（148 个 safetensors 分片 + 64 卡 HCCL 初始化）。查看容器日志确认进度。

### Q: NODE_RANK 检测失败 `local IPs [...] not found in node list`
A: 此问题已在最新版修复——NODE_RANK 由管理节点直接传入循环下标，不再依赖容器内 IP 匹配。如果仍有此问题，请确认节点列表 IP 与容器内 `hostname -I` 输出一致（容器需使用 `--network host`）。

### Q: 如何修改 HCCL 网卡接口？
A: 设置环境变量 `HCCL_SOCKET_IFNAME` 和 `GLOO_SOCKET_IFNAME`。默认值为 `enp66s0f0`，如果节点使用不同网卡（如 `enp66s0f5`），启动时指定：
```bash
HCCL_SOCKET_IFNAME=enp66s0f5 GLOO_SOCKET_IFNAME=enp66s0f5 bash run_sglang.sh
```
