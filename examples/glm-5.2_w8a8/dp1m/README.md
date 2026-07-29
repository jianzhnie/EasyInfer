# GLM-5.2 W8A8 共部署（1M 上下文，8 节点原生 DP）

> ✅ **2026-07-28 已验证 PASS** | 8×A2 (64G×8) | DP=8 TP=8 PCP1 DCP=8 EP=64 | MTP 3 tokens
> 入口: `http://10.42.11.194:8007` | 节点: `nodes/node_list3.txt`

## 拓扑

```
Node 0 (194, DP master, --api-server-count 1)   ← 唯一 API 入口 :8007
Node 1-7 (195-201, --headless)                  ← 内部 DP 负载均衡
每节点 1 个 DP rank, TP=8 节点内, EP=64 跨节点, DCP=8 KV 分片
```

- 权重：EP=64 → 4 专家/卡（~12 GiB）；KV：DCP=8 后 ~11.8 GiB/条 1M，util 0.90 下每 rank 约 3 条全 1M 并发
- 客户端只需访问 master（194:8007），DP 组内部自动负载均衡（`--data-parallel-start-rank` 内部 LB 模式）

## 部署

```bash
bash remote_deploy.sh deploy    # 一键部署（8 节点并行启动 + master 健康检查，~22 分钟）
bash remote_deploy.sh status    # master API + 各节点进程状态
bash remote_deploy.sh stop      # 停止
bash remote_deploy.sh restart   # 重启
```

## 验证（2026-07-28 PASS）

| 项目 | 结果 |
|------|------|
| `/v1/models` (max_model_len=1024000) | ✅ |
| 中文/英文/数学/代码/流式 | ✅ 全 PASS |
| Tool Calling (glm47) | ✅ |
| Anthropic Messages API (`/v1/messages`) | ✅（thinking 块正常） |
| MTP (deepseek_mtp 3 tokens) | ✅ 启用 |

> ⚠️ 从集群外机器 curl 时请绕过本机代理（`no_proxy` 的 `10.` 前缀写法 curl 不识别）：
> `curl --noproxy '*' ...` 或 `env -u http_proxy -u https_proxy ...`

## 关键约束（实测结论）

| 约束 | 结论 |
|------|------|
| 多节点 DP 部署方式 | **必须原生逐节点启动**（`--data-parallel-start-rank` + rank0 API / 其余 `--headless`）。单次 `vllm serve` 的 DP>1 会把所有 engine core 放在启动节点（设备按 `local_dp_rank × TP×PP` 本机枚举），Ray backend 也无效——Ray 只管单个 engine 内部的 worker 放置 |
| `--data-parallel-size-local` | **必须显式传 1**。缺省时 headless 节点按全局 rank 切设备（`local range: [8,16)` IndexError） |
| TP 上限 | TP×PP ≤ 单节点卡数（8）→ A2 上 TP=8；TP=16 共部署两轮实证内存不可行（EP=16 权重 ~35G/卡，util 0.95 仍无 KV） |
| FLASHCOMM1 | DCP>1 强制开启（`DSA CP requires SP`），本配置 =1 |
| FUSED_MC2 | W8A8 下 `aclnnDispatchFFNCombine` 崩溃，**必须关**（=0） |
| MTP | PP=1 故可用（3 tokens, enforce_eager） |

## 文件

| 文件 | 用途 |
|------|------|
| `deploy.conf` | 节点/拓扑/端口（唯一需按集群修改） |
| `remote_deploy.sh` | 一键部署/停止/状态（含 conf 校验） |
| `conode.sh` | 节点启动模板（rank0 API / 其余 headless，1M 参数） |
| `launch_online_dp.py` | DP 实例启动器（复用 pd-separation 版） |

## 参考

- [GLM-5.2 官方教程 — 1M 双节点共部署](https://docs.vllm.ai/projects/ascend/zh-cn/latest/tutorials/models/GLM5.2.html)
- 备选：PD 分离部署见 `../../pd-separation/`（P/D 各 4 节点，MooncakeConnectorV1）
