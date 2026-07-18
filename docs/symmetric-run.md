# Ray symmetric-run 使用指南

> 参考文档：[Streamlined multi-node serving with Ray symmetric-run](https://vllm.ai/blog/2025-11-22-ray-symmetric-run) (vLLM Blog, 2025-11-22)

## 背景

传统 Ray 多节点工作流需要**两套命令**：一套启动 Ray 集群、一套执行任务。典型流程如下：

```bash
# Step 1: 在 head 节点启动 Ray
ray start --head --port=6379

# Step 2: 在每个 worker 节点加入集群
ray start --address=<head_ip>:6379 --block

# Step 3: 在 head 节点另开终端，提交任务
vllm serve Qwen/Qwen3-32B --tensor-parallel-size 8 --pipeline-parallel-size 2

# Step 4: 任务完成后，逐节点清理
ray stop   # 每个节点都要执行
```

这个流程有几个痛点：
- **步骤繁琐**：head/worker 各自需要不同命令，容易出错
- **环境变量不透明**：忘记设置 `VLLM_HOST_IP` 等变量就得推倒重来
- **生命周期割裂**：Ray 启停与任务执行分离，清理残留是常态

`symmetric-run` 将这些步骤合并为**一条命令**，在所有节点对称运行，自动识别角色、管理生命周期。

## 概览

```
节点1 (IP 匹配 --address)  →  Head: ray start --head → 执行 entrypoint → ray stop
节点2 (IP 不匹配)           →  Worker: 等待 head 就绪 → ray start --address --block → ray stop
节点3 (IP 不匹配)           →  Worker: 等待 head 就绪 → ray start --address --block → ray stop
```

对比传统方式（`start_ray_cluster.sh` + `stop_ray_cluster.sh` → 手动启动 vLLM → 逐个 `ray stop`），`symmetric-run` 省去了中间的 SSH 编排和生命周期管理。

## 基本语法

```
ray symmetric-run --address <head_ip:port> [Ray启动选项] -- <entrypoint命令>
```

| 参数 | 必填 | 说明 |
|------|------|------|
| `--address` | 是 | Ray 集群地址，填 head 节点的**业务网 IP**（多节点时禁止 `127.0.0.1`） |
| `--min-nodes` | 否 | 等待指定数量节点就绪后再执行 entrypoint，默认 `1` |
| `--` | 是 | 分隔符，前面是 Ray 启动参数，后面是要执行的命令 |

`--` 之前可传大部分 `ray start` 参数（如 `--num-cpus`、`--num-gpus`、`--resources`）。以下参数**禁止使用**（由 symmetric-run 自动管理）：`--head`、`--node-ip-address`、`--port`、`--block`。

## 环境变量传递

entrypoint 前设置的环境变量会自动传播到 Ray 运行时：

```bash
ENV=VAR ray symmetric-run --address 127.0.0.1:6379 -- python test.py
```

这是解决 "忘记设 `VLLM_HOST_IP` 就得重来" 的关键机制 — 环境变量与 symmetric-run 在同一行设置即可。

## 快速上手

### 单节点

```bash
ray symmetric-run --address 127.0.0.1:6379 -- python -c "import ray; ray.init(); print(ray.cluster_resources())"
```

单节点时 `127.0.0.1` 可用。

### 多节点（手动 SSH）

在**每个节点**上运行相同命令：

```bash
# head 节点 (192.168.1.10) — IP 匹配，自动成为 head
ray symmetric-run --address 192.168.1.10:6379 --min-nodes 4 -- python train.py

# worker 节点 (192.168.1.11-13) — IP 不匹配，自动成为 worker
ray symmetric-run --address 192.168.1.10:6379 --min-nodes 4 -- python train.py
```

### 多节点（pdsh 一键分发）

```bash
pdsh -w 192.168.1.[10-13] \
  'ray symmetric-run --address 192.168.1.10:6379 --min-nodes 4 -- python train.py'
```

## vLLM 多节点部署示例

以下命令等价于传统四步流程，一条命令完成 DeepSeek-V3 规模模型的多节点推理：

```bash
ray symmetric-run \
  --address 10.42.1.66:6379 \
  --min-nodes 2 \
  --num-gpus 8 \
  -- vllm serve Qwen/Qwen3-32B \
    --tensor-parallel-size 8 \
    --pipeline-parallel-size 2
```

## 在 EasyInfer 集群中使用

### 前置条件

- 所有节点 SSH 免密互通
- Docker 容器已运行（镜像包含 ray + vllm-ascend）
- Ascend NPU 驱动正常

### 场景一：Ray 多节点 TP/PP（推荐）

适合模型大到单节点放不下，需要跨节点张量/流水线并行。

```bash
HEAD_IP=10.42.1.66
NODES=16

pdsh -w 10.42.1.[66-81] \
  "ray symmetric-run \
    --address ${HEAD_IP}:6379 \
    --min-nodes ${NODES} \
    --resources='{\"NPU\":8}' \
    -- \
    bash -c '
      source /llm_workspace_1P/robin/EasyInfer/scripts/vllm/set_env.sh
      source /llm_workspace_1P/robin/EasyInfer/scripts/ray_cluster/set_ray_env.sh

      export HCCL_IF_IP=\$(ip -4 addr show enp66s0f0 | awk \"/inet /{print \\\$2}\" | cut -d/ -f1 | head -1)

      vllm serve /path/to/model \
        --host 0.0.0.0 \
        --port 8077 \
        --trust-remote-code \
        --tensor-parallel-size 8 \
        --pipeline-parallel-size 16 \
        --distributed-executor-backend ray \
        --max-model-len 8192 \
        --gpu-memory-utilization 0.92
    '"
```

### 场景二：Docker 容器内执行

```bash
HEAD_IP=10.42.1.66
CONTAINER_NAME=vllm-ascend-0.18-env

pdsh -w 10.42.1.[66-81] \
  "ray symmetric-run \
    --address ${HEAD_IP}:6379 \
    --min-nodes 16 \
    --resources='{\"NPU\":8}' \
    -- \
    docker exec -i ${CONTAINER_NAME} \
      bash -c '
        source /llm_workspace_1P/robin/EasyInfer/scripts/ray_cluster/set_ray_env.sh
        vllm serve /path/to/model \
          --host 0.0.0.0 --port 8077 \
          --tensor-parallel-size 8 --pipeline-parallel-size 16 \
          --distributed-executor-backend ray
      '"
```

### 场景三：带 Ascend 环境变量

```bash
VLLM_USE_V1=1 \
RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1 \
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
ray symmetric-run \
  --address 10.42.1.66:6379 \
  --min-nodes 4 \
  --resources='{"NPU":8}' \
  -- \
  vllm serve /path/to/model --tensor-parallel-size 8 --distributed-executor-backend ray
```

环境变量设置在 `ray symmetric-run` 之前即可自动传播到 Ray 运行时。

## 角色判断机制

源码 `ray/scripts/symmetric_run.py` 中的判断逻辑：

```python
is_head = resolved_gcs_host in my_ips
```

每个节点：
1. 解析 `--address` 得到 `host:port`
2. DNS 解析 host → IP（支持 IPv4/IPv6，通过 `socket.getaddrinfo` + `AF_UNSPEC`）
3. 枚举本机所有网卡 IP（通过 `psutil.net_if_addrs()`）
4. **IP 匹配 → Head**，否则 → Worker

多节点时（`--min-nodes > 1`）会自动排除 `127.0.0.1` 和 `::1`，防止所有节点都误判为 head。

## 生命周期详解

```
Head 节点:
  1. subprocess.run(["ray", "start", "--head", ...])    # 后台启动
  2. check_cluster_ready(min_nodes)                     # 等待 worker 注册
  3. subprocess.run(entrypoint)                         # 执行用户命令
  4. subprocess.run(["ray", "stop"])                    # finally 块保证执行

Worker 节点:
  1. check_head_node_ready(address)                     # 轮询等待 head 就绪
  2. subprocess.run(["ray", "start", "--address", ..., "--block"])  # 阻塞
  3. subprocess.run(["ray", "stop"])                    # head stop 后 --block 解除
```

关键保证：
- **`finally` 块**：无论 entrypoint 成功还是失败，`ray stop` 都会执行
- **退出码传递**：entrypoint 的非零退出码会被 `sys.exit()` 传递
- **`Ctrl+C` 安全**：`KeyboardInterrupt` 被捕获，依然触发 `finally` 清理

## 超时配置

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `RAY_SYMMETRIC_RUN_CLUSTER_WAIT_TIMEOUT` | `30` | Worker 等待 head 就绪的超时（秒） |

```bash
# 集群启动较慢时增加超时
export RAY_SYMMETRIC_RUN_CLUSTER_WAIT_TIMEOUT=120
```

## 与现有脚本的对比

| | `manage_ray_cluster.sh` | `deploy_vllm_multinode.sh` | `ray symmetric-run` |
|---|---|---|---|
| Ray 管理 | 手动 start/stop | 无（各节点独立 DP） | 自动 start → run → stop |
| vLLM 进程数 | 不启动 vLLM | 每个节点一个 | 仅 head 一个 |
| 并行策略 | 需自行编排 | TP(单节点) + DP(跨节点) | TP + PP（跨节点 Ray） |
| 适用场景 | Ray 集群管理 | 多实例独立推理 | 大模型跨节点分布式推理 |
| entrypoint 执行 | 不需要 | 每个节点独立执行 | 仅 head 执行 |
| SSH 编排 | 需要（head/worker 分别处理） | 需要（逐个节点启动） | 不需要（对称命令） |
| 环境变量 | 需在容器内 source | 通过 build_env_exports 拼接 | 命令行前缀自动传播 |

## 注意事项

1. **`--address` 必须用真实 IP**，多节点场景下不能用 `127.0.0.1`，否则所有节点都会认为自己是 head
2. **所有节点必须能相互访问** `--address` 中指定的 IP:Port
3. **entrypoint 退出 → 整个集群自动停止**，worker 上的 `ray start --block` 随之解除阻塞
4. **`Ctrl+C` 会触发所有节点的 `ray stop`** 清理（由 `finally` 块保证）
5. **entrypoint 的退出码会被传递**，方便脚本判断任务成功与否
6. **symmetric-run 只在 head 节点执行 entrypoint**，适合 Ray 分布式任务；如需每个节点各自执行（如 DP 模式），请继续使用 `deploy_vllm_multinode.sh`
7. 环境变量设置在 `ray symmetric-run` 命令之前即可，会自动传播到 Ray 运行时
