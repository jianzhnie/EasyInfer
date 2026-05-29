# Ray symmetric-run 使用指南

## 概览

`symmetric-run` 是 Ray 内置的对称集群启动命令，将 **启动 Ray → 执行任务 → 停止 Ray** 三段封装为一条命令。在所有节点上运行相同命令，各节点自动识别角色（head/worker），无需手动编排。

```
节点1 (IP 匹配 address)  →  Head: ray start --head → 执行 entrypoint → ray stop
节点2 (IP 不匹配)         →  Worker: 等待 head 就绪 → ray start --address --block → ray stop
节点3 (IP 不匹配)         →  Worker: 等待 head 就绪 → ray start --address --block → ray stop
...
```

对比传统方式（`start_ray_cluster.sh` → 手动启动 vLLM → `ray stop`），`symmetric-run` 省去了中间的 SSH 编排和生命周期管理。

## 基本语法

```
ray symmetric-run --address <head_ip:port> [Ray启动选项] -- <entrypoint命令>
```

| 参数 | 必填 | 说明 |
|------|------|------|
| `--address` | 是 | Ray 集群地址，填 head 节点的**业务网 IP**（禁止 `127.0.0.1`） |
| `--min-nodes` | 否 | 等待指定数量节点就绪后再执行 entrypoint，默认 `1` |
| `--` | 是 | 分隔符，前面是 Ray 启动参数，后面是要执行的命令 |

`--` 之前可以传大部分 `ray start` 参数（如 `--num-cpus`、`--resources`）。以下参数**禁止使用**（由 symmetric-run 自动管理）：`--head`、`--node-ip-address`、`--port`、`--block`。

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


## 注意事项

1. **`--address` 必须用真实 IP**，多节点场景下不能用 `127.0.0.1`，否则所有节点都会认为自己是 head
2. **所有节点必须能相互访问** `--address` 中指定的 IP:Port
3. **entrypoint 退出 → 整个集群自动停止**，worker 上的 `ray start --block` 随之解除阻塞
4. **`Ctrl+C` 会触发所有节点的 `ray stop`** 清理
5. **entrypoint 的退出码会被传递**，方便脚本判断任务成功与否
6. **symmetric-run 只在 head 节点执行 entrypoint**，适合 Ray 分布式任务；
