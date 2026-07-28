# GLM-5.2 W8A8 PD 分离部署（1M 上下文）

> A2 PD: 4×PNode + 4×DNode | P/D 均 DP=4 TP=8 PCP1 **DCP=8** | MooncakeConnectorV1 | **1M ctx**
> 节点: `nodes/node_list4.txt`（10.42.11.202-209）

## 架构

```
PNode 0-3 (kv_producer, 202-205)         DNode 0-3 (kv_consumer, 206-209)
DP=4 TP=8 DCP=8  每节点1实例×8卡          DP=4 TP=8 DCP=8  每节点1实例×8卡
enforce-eager, MTP 1 token               FULL_DECODE_ONLY, MTP 3 tokens
FLASHCOMM1=1 (DCP 强制要求)               FLASHCOMM1=1 (DCP 强制要求)
max-num-seqs 8,  batched 16384           max-num-seqs 16, batched 128
gpu-mem-util 0.85                        gpu-mem-util 0.92
         │                                      │
         └────── MooncakeConnectorV1 ───────────┘
          use_ascend_direct, prefill dp4tp8 / decode dp4tp8
```

> **设计约束**（实测结论）：
> - A2 (8卡/节点) 每个 DP rank 的 TP×PP 必须 ≤ 8：vLLM v1 按 `local_dp_rank × world_size`
>   枚举本机设备，TP=16 的 DP rank 在 8 卡节点上直接 IndexError，故 **TP=8 是 PD 部署上限**
> - **DCP>1 强制 FLASHCOMM1=1**（`DSA CP requires SP`，本镜像实测）；1M KV 由 DCP=8
>   分片（MLA 压缩后 ~11.8 GiB/条/卡），单 rank 约 3 条全 1M 并发，短序列更多
> - TP=16 维度本身可行（官方 1M 单节点即 TP16 DCP16），但仅限 16 卡节点（A3 64G×16）

## 节点分配

| IP | 角色 | 实例 | NPU | dp_rank | vllm 端口 |
|-----|------|------|-----|---------|-----------|
| 10.42.11.202 | PNode 0 (DP master) | 1 | 8 | 0 | 9081 |
| 10.42.11.203 | PNode 1 | 1 | 8 | 1 | 9081 |
| 10.42.11.204 | PNode 2 | 1 | 8 | 2 | 9081 |
| 10.42.11.205 | PNode 3 | 1 | 8 | 3 | 9081 |
| 10.42.11.206 | DNode 0 (DP master) | 1 | 8 | 0 | 9900 |
| 10.42.11.207 | DNode 1 | 1 | 8 | 1 | 9900 |
| 10.42.11.208 | DNode 2 | 1 | 8 | 2 | 9900 |
| 10.42.11.209 | DNode 3 | 1 | 8 | 3 | 9900 |

## 与官方 1M PD 配置的对齐/偏差

| 项 | 官方 1M PD (A3+W4A8C8) | 本配置 (A2+W8A8) | 说明 |
|----|------------------------|------------------|------|
| P/D 并行 | DP4 TP8 PCP1 DCP8 | 同左 | A2 每 rank=1 节点 |
| FLASHCOMM1 | P: true / D: **false** | P/D 均 **true** | 本镜像 DCP 强制 SP（实测报错） |
| Connector | MooncakeConnectorV1 | 同左 | use_ascend_direct |
| MTP | P: 1 token / D: 3 tokens | 同左 | enforce_eager=true |
| recompute_scheduler | P: true / D: true | P: **false** / D: true | vllm-ascend 在 P 节点仅告警忽略，直接关闭 |
| max-num-seqs | P: 8 / D: 32 | P: 8 / D: **16** | W8A8 KV 池更小，保守起步 |
| gpu-mem-util | P: 0.75 / D: 0.93 | P: **0.85** / D: 0.92 | W8A8 权重 ~19G/卡（EP=32） |
| enable_balance_scheduling | — | ❌ 不用 | 仅 kv_both 模式可用（PD 分离下报错） |

## 文件

| 文件 | 用途 |
|------|------|
| `deploy.conf` | 全局配置（节点 IP、DP/TP 拓扑、端口） |
| `remote_deploy.sh` | 一键部署/停止/状态（SSH 自动化） |
| `pnode.sh` | PNode 启动模板（kv_producer, 1M 参数） |
| `dnode.sh` | DNode 启动模板（kv_consumer, 1M 参数） |
| `launch_online_dp.py` | DP 启动器（`--script` 外部化，P/D 共用） |

## 使用

```bash
# 一键部署（P+D 并行启动，自动健康检查）
bash remote_deploy.sh deploy

# 查看状态 / 停止 / 彻底清理（重启容器）
bash remote_deploy.sh status
bash remote_deploy.sh stop
bash remote_deploy.sh clean
```

## 验证

```bash
# 各节点 /v1/models
bash remote_deploy.sh status

# 请求转发代理（仓库示例）
python3 examples/prefill_decode_separation_deploy/load_balance_proxy_server_example.py \
  --host 0.0.0.0 --port 8000 \
  --prefiller-hosts 10.42.11.202 --prefiller-ports 9081 \
  --decoder-hosts 10.42.11.206 --decoder-ports 9900

# 功能测试
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"glm-52","messages":[{"role":"user","content":"Hello"}],"max_tokens":50}'
```

## 参考

- [GLM-5.2 官方教程 — PD 分离与 1M 配置](https://docs.vllm.ai/projects/ascend/zh-cn/latest/tutorials/models/GLM5.2.html)
- 已验证参考实现（132K, v0.22.1 时代）: `batch_remote_deploy_pd_seg/glm52-deploy-scripts/`
