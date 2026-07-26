# DeepSeek-V4-Flash PD 分离部署

> **官方 §5.2 A2**: 4×PNode + 4×DNode | DP=8/32 TP=1 | MooncakeHybridConnector | 135K

## 文件

| 文件 | 用途 |
|------|------|
| `deploy.conf` | **唯一配置文件** (IP, SSH, DP/TP 全部在此) |
| `remote_deploy.sh` | 一键远程部署入口 |
| `pnode.sh` | Prefill 节点启动模板 |
| `dnode.sh` | Decode 节点启动模板 |
| `launch_online_dp.py` | DP 多进程启动器 |

## 使用

```bash
# 1. 编辑配置
vim deploy.conf   # 修改 PNODE_IPS, DNODE_IPS

# 2. 一键操作
bash remote_deploy.sh deploy     # 部署 (清理 → P×4 → D×4 → 健康检查)
bash remote_deploy.sh status     # 状态
bash remote_deploy.sh stop       # 停止
bash remote_deploy.sh restart    # 重启
```

## 架构

```
PNode 0-3 (kv_producer)           DNode 0-3 (kv_consumer)
DP=8 TP=1, engine_id=0..3         DP=32 TP=1, engine_id=4..7
enable-prefix-caching             no-enable-prefix-caching
enforce-eager                     FULL_DECODE_ONLY
FLASHCOMM1=1                      recompute_scheduler + npugraph_ex
         │                                │
         └──── MooncakeHybridConnector ────┘
