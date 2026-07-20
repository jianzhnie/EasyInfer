# DeepSeek-V4-Pro 分布式推理部署脚本

> 架构: 4 PNode (Prefill, TP8×DP4) + 4 DNode (Decode, TP4×DP8)

---

## 1. 脚本套件总览

| 文件 | 作用 | 对应原始文档 |
|------|------|-------------|
| `deploy.conf` | **全局配置文件**（唯一需要编辑的文件） | — |
| `pnode.sh` | Prefill 节点启动模板（4 个 PNode 共用 1 份） | pnode0~3.sh |
| `dnode.sh` | Decode 节点启动模板（4 个 DNode 共用 1 份） | dnode0~3.sh |
| `launch_online_dp.py` | 通用 DP 启动器（所有节点共用 1 份） | launch_online_dp.py ×8 |
| `start_pnode.sh` | PNode 启动入口（按编号自动注入配置） | — |
| `start_dnode.sh` | DNode 启动入口（按编号自动注入配置） | — |
| `stop_node.sh` | 停止当前节点的 vLLM 进程 | — |
| `check_status.sh` | 检查所有节点健康状态 | — |

### 去重效果

原始文档需要维护 **16 个文件**（8 个 .sh + 8 份 launch_online_dp.py），
其中 .sh 脚本仅 `local_ip` 不同、Python 脚本仅硬编码的 .sh 文件名不同。

本套件压缩为 **8 个文件**，差异化参数全部集中到 `deploy.conf` 一个文件中。

---

## 2. 前置条件

1. **Docker 容器已启动** — 在所有 PNode 和 DNode 上执行 `start_glm5_docker.sh` 进入容器
2. **模型权重就位** — `/data/GLM-5.2-w8a8` 路径可访问（共享存储或本地拷贝）
3. **脚本目录挂载** — 本套件需在容器内可访问，建议放在 `/data/scripts/` 下（已挂载到容器）
4. **网络互通** — 8 台 NPU 节点之间网络连通

---

## 3. 快速开始

### 3.1 编辑配置

编辑 `deploy.conf`，确认以下内容与你的环境一致：

```
MODEL_PATH="/data/GLM-5.2-w8a8"    # 模型路径
NIC_NAME="enp66s0f1"               # 网卡名（所有节点需一致）
LOG_DIR="/data/scripts"            # 日志输出目录

PNODE_IPS=(10.18.1.10 10.18.1.11 10.18.1.12 10.18.1.13)
DNODE_IPS=(10.18.1.14 10.18.1.15 10.18.1.16 10.18.1.17)
```

如果你的 IP 或网卡名不同，只需修改这一个文件。

### 3.2 分发脚本

将整个目录拷贝到所有 8 台节点的相同路径下（例如 `/data/scripts/glm5.2-deploy-scripts/`）：

```bash
# 从管理节点分发到所有 NPU 节点
for ip in 10.18.1.{10..17}; do
    scp -r /data/scripts/glm5.2-deploy-scripts root@${ip}:/data/scripts/
done
```

### 3.3 赋予执行权限

在所有节点上执行：

```bash
cd /data/scripts/glm5.2-deploy-scripts
chmod +x *.sh
```

### 3.4 启动 Prefill 节点（PNode）

**按顺序**在 4 台 PNode 上分别执行（建议先启动 rank-0）：

| 节点 IP | 命令 | dp-rank-start |
|---------|------|---------------|
| 10.18.1.10 | `./start_pnode.sh 0` | 0 |
| 10.18.1.11 | `./start_pnode.sh 1` | 1 |
| 10.18.1.12 | `./start_pnode.sh 2` | 2 |
| 10.18.1.13 | `./start_pnode.sh 3` | 3 |

每个节点执行后会打印配置摘要并启动 vLLM，日志输出到 `$LOG_DIR/pnode_<ip>_rank<N>.log`。

### 3.5 启动 Decode 节点（DNode）

**在所有 PNode 就绪后**，在 4 台 DNode 上分别执行：

| 节点 IP | 命令 | dp-rank-start |
|---------|------|---------------|
| 10.18.1.14 | `./start_dnode.sh 0` | 0 |
| 10.18.1.15 | `./start_dnode.sh 1` | 2 |
| 10.18.1.16 | `./start_dnode.sh 2` | 4 |
| 10.18.1.17 | `./start_dnode.sh 3` | 6 |

注意 DNode 的 `node_index` 不等于 `dp-rank-start`（映射关系为 0/2/4/6），
脚本已自动处理，无需手动计算。

### 3.6 验证部署

在任意节点上执行：

```bash
./check_status.sh          # 检查所有 8 个节点
./check_status.sh pnode    # 只检查 PNode
./check_status.sh dnode    # 只检查 DNode
```

期望输出：所有节点显示 `OK (200)`。

也可手动验证单个节点：

```bash
curl http://10.18.1.10:9081/v1/models   # PNode 0
curl http://10.18.1.14:9900/v1/models   # DNode 0
```

---

## 4. 停止服务

在需要停止的节点上执行：

```bash
./stop_node.sh             # 停止当前节点所有 vllm 进程
./stop_node.sh pnode       # 只停止 PNode 进程
./stop_node.sh dnode       # 只停止 DNode 进程
```

停止策略：先发 SIGTERM 等待 5 秒，若进程仍在则发 SIGKILL。

---

## 5. 文件参数详解

### 5.1 deploy.conf 参数对照表

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `MODEL_PATH` | /data/GLM-5.2-w8a8 | 模型权重路径 |
| `NIC_NAME` | enp66s0f1 | NPU 通信网卡名 |
| `LOG_DIR` | /data/scripts | 日志输出目录 |
| `P_DP_SIZE` | 4 | PNode 总 DP 数 |
| `P_TP_SIZE` | 8 | PNode 每实例 TP 数 |
| `P_DP_SIZE_LOCAL` | 1 | PNode 本机 DP 数 |
| `P_DP_RPC_PORT` | 16591 | PNode DP RPC 端口 |
| `P_VLLM_START_PORT` | 9081 | PNode vLLM API 起始端口 |
| `D_DP_SIZE` | 8 | DNode 总 DP 数 |
| `D_TP_SIZE` | 4 | DNode 每实例 TP 数 |
| `D_DP_SIZE_LOCAL` | 2 | DNode 本机 DP 数 |
| `D_DP_RPC_PORT` | 16600 | DNode DP RPC 端口 |
| `D_VLLM_START_PORT` | 9900 | DNode vLLM API 起始端口 |
| `PNODE_IPS` | 10.18.1.10~13 | PNode IP 列表 |
| `DNODE_IPS` | 10.18.1.14~17 | DNode IP 列表 |
| `DNODE_RANK_STARTS` | 0 2 4 6 | DNode dp-rank-start 映射 |

### 5.2 脚本调用链

```
start_pnode.sh 0                     start_dnode.sh 0
      |                                    |
      | source deploy.conf                 | source deploy.conf
      | 查 PNODE_IPS[0] -> local_ip        | 查 DNODE_IPS[0] -> local_ip
      | export LOCAL_IP / NIC_NAME / ...   | export LOCAL_IP / NIC_NAME / ...
      v                                    v
launch_online_dp.py                  launch_online_dp.py
  --script ./pnode.sh                  --script ./dnode.sh
  --dp-size 4 --tp-size 8 ...          --dp-size 8 --tp-size 4 ...
      |                                    |
      | 为每个 local DP rank fork 进程       | 为每个 local DP rank fork 进程
      v                                    v
pnode.sh <args>                      dnode.sh <args>
  设置 ASCEND_RT_VISIBLE_DEVICES        设置 ASCEND_RT_VISIBLE_DEVICES
  导出 HCCL/GLOO/... 环境变量            导出 HCCL/GLOO/... 环境变量
  nohup vllm serve ... &                nohup vllm serve ... &
```

---

## 6. PNode vs DNode 关键差异

| 维度 | PNode (Prefill) | DNode (Decode) |
|------|-----------------|----------------|
| KV 角色 | `kv_producer` | `kv_consumer` |
| KV 端口 | 30000 | 30100 |
| engine_id | 0 | 1 |
| TP size | 8 | 4 |
| DP size | 4 | 8 |
| DP size local | 1 | 2 |
| max-model-len | 135000 | 135168 |
| max-num-batched-tokens | 4096 | 164 |
| max-num-seqs | 64 | 48 |
| gpu-memory-utilization | 0.95 | 0.92 |
| enforce-eager | 是 | 否（用 cudagraph） |
| DNode 专有环境变量 | — | TASK_QUEUE_ENABLE, DYNAMIC_EPLB, VLLM_ASCEND_ENABLE_MLAPO |
| additional-config | 含 enable_dsa_cp | 不含 enable_dsa_cp |
| HCCL_BUFFSIZE | 400 | 500 |
| compilation-config | — | FULL_DECODE_ONLY |

这些差异已分别固化在 `pnode.sh` 和 `dnode.sh` 中，无需手动管理。

---

## 7. 常见问题

**Q: 启动顺序有要求吗？**
A: 有。先启动 PNode rank-0（`start_pnode.sh 0`），它是 DP master；
   再启动其余 PNode；所有 PNode 就绪后启动 DNode。

**Q: 如何查看日志？**
A: 日志在 `$LOG_DIR` 下，文件名格式为 `pnode_<ip>_rank<N>.log` 或 `dnode_<ip>_rank<N>.log`。

**Q: 网卡名怎么查？**
A: 在节点上执行 `ip addr` 或 `ifconfig`，找到与 DP 通信网络对应的网卡名。
   所有节点需使用相同名称（默认 `enp66s0f1`）。

**Q: 修改了 deploy.conf 后需要重新分发吗？**
A: 是。`deploy.conf` 是各节点本地读取的，修改后需同步到所有节点。

**Q: 如何只重启示例而不重启容器？**
A: 先 `./stop_node.sh` 停止进程，再 `./start_pnode.sh <index>` 或 `./start_dnode.sh <index>` 重启。

**Q: DNode 的第二个实例监听哪个端口？**
A: DNode 的 `dp-size-local=2`，第二个实例监听 `D_VLLM_START_PORT + 1 = 9901`。
   `check_status.sh` 默认只检查 rank-0 的端口（9900），如需检查 9901 可手动 curl。
