# GLM-5.2 W8A8 共部署（1M 上下文，8/16 节点原生 DP）

> ✅ **2026-07-29 已验证 PASS（16 节点）** | 16×A2 (64G×8) = 128 NPU | **DP=16 TP=8 PCP1 DCP=8 EP=128** | MTP 3 tokens
> 入口: `http://10.42.11.194:8007` | 节点: `nodes/node_list0.txt`（10.42.11.194-209）
> 8 节点（node_list3.txt, EP=64）2026-07-28 已验证 PASS；节点数由 `deploy.conf` 一处切换

## 拓扑

```
Node 0 (194, DP master, --api-server-count 1)   ← 唯一 API 入口 :8007
Node 1-15 (195-209, --headless)                 ← 内部 DP 负载均衡
每节点 1 个 DP rank, TP=8 节点内, EP=128 跨 16 节点, DCP=8 KV 分片
```

- EP=128 → 2 专家/卡（~6 GiB，较 8 节点 EP=64 的 4 专家/卡省 ~6 GiB/卡给 KV 池）
- KV：DCP=8 后 ~11.8 GiB/条 1M；util 0.85 下每 rank 约 3 条全 1M 并发，短序列更多
- 客户端只访问 master（194:8007），DP 组内部所有 engine 统一 LB

## 切换 8 / 16 节点

只改 `deploy.conf`：`NODES`（8 节点用 `nodes/node_list3.txt` = 194-201）与
`DP_SIZE`（= 节点数），`DP_ADDRESS` 保持 `NODES[0]`。conode.sh 参数对两种规模通用。

## 关键约束（实测结论）

| 约束 | 结论 |
|------|------|
| 多节点 DP 部署方式 | **必须原生逐节点启动**（`--data-parallel-start-rank` + rank0 API / 其余 `--headless`）。单次 `vllm serve` 的 DP>1 会把所有 engine core 放在启动节点（设备按 `local_dp_rank × TP×PP` 本机枚举），Ray backend 也无效 |
| `--data-parallel-size-local` | **必须显式传 1**。缺省时 headless 节点按全局 rank 切设备（`local range: [8,16)` IndexError） |
| TP 上限 | TP×PP ≤ 单节点卡数（8）→ A2 上 TP=8；TP=16 共部署两轮实证内存不可行 |
| FLASHCOMM1 | DCP>1 强制开启（`DSA CP requires SP`），本配置 =1 |
| FUSED_MC2 | W8A8 下 `aclnnDispatchFFNCombine` 崩溃，**必须关**（=0） |
| MTP | PP=1 故可用（3 tokens, enforce_eager） |
| gpu-mem-util | **0.85 上限**。0.90 时长文 chunk prefill 的 MoE dispatch 瞬时 buffer(~2.8G) OOM（16 节点实测） |
| max-num-batched-tokens | **8192 上限**。EP=128 的 all2all 元数据随组宽翻倍，16384 chunk 的 dispatch buffer(~5.6G) OOM（16 节点 rank15 实测）；8192 与官方 A2 全部配置一致（16384 是官方 A3 128G 卡的值；8 节点 EP=64 时 16384 可用） |
| HCCL 超时 | CONNECT/EXEC=600：跨节点 EP all2all；单 rank 故障时全组挂起，watchdog 600s 收尾 |
| PD 分离 1M | 不可用：MooncakeConnectorV1 对 DSA 模型有 indexer KV 传输 bug（上游 #12863，v0.23.0rc1 未修复），1M 只能共部署 |

## 部署

```bash
bash remote_deploy.sh deploy    # 全部节点并行启动 + master 健康检查（~29 分钟）
bash remote_deploy.sh status    # master API + 各节点进程状态
bash remote_deploy.sh stop      # 停止
bash remote_deploy.sh restart   # 重启
bash remote_deploy.sh clean     # 彻底清理（重启容器）
```

## 验证

### 16 节点（2026-07-29 PASS）

三轮调参：轮 1 util 0.90 长文 OOM；轮 2 util 0.85 + batched 16384 长文 rank15 OOM（全组
all2all 挂起 600s 后 watchdog 关闭）；轮 3 util 0.85 + batched 8192 全部通过：

| 项目 | 结果 |
|------|------|
| 16 rank 启动 / master 就绪 | ✅ ~29 分钟 |
| `/v1/models` (max_model_len=1024000) | ✅ |
| 短请求 e2e | ✅ |
| 59,252 token 长文 | ✅ 41.5s（prefill ~1430 tok/s/rank） |
| **500,790 token 超长文** | ✅ 358s，无 OOM |
| 16 并发 × 128 tok | ✅ 16/16，38s |

### 8 节点（2026-07-28 PASS，EP=64, batched 16384, max-num-seqs 8）

| 项目 | 结果 |
|------|------|
| `/v1/models` (max_model_len=1024000) | ✅ |
| 中文/英文/数学/代码/流式 | ✅ 全 PASS |
| Tool Calling (glm47) | ✅ |
| Anthropic Messages API (`/v1/messages`) | ✅（thinking 块正常） |
| MTP (deepseek_mtp 3 tokens) | ✅ 启用 |

> ⚠️ 从集群外机器 curl 时请绕过本机代理：`curl --noproxy '*' ...` 或 `env -u http_proxy -u https_proxy ...`

## 已知限制

- batched 8192 的 chunk prefill 峰值吞吐低于 16384（A2 64G 显存约束，非配置失误）
- 单 DP 组故障域为全部节点；任一 rank 崩溃全组不可用（HCCL watchdog 600s 后全组退出）

## 文件

| 文件 | 用途 |
|------|------|
| `deploy.conf` | 节点/拓扑/端口（8↔16 节点切换只改这里） |
| `remote_deploy.sh` | 一键部署/停止/状态（含 conf 校验；pkill 自匹配已修） |
| `conode.sh` | 节点启动模板（rank0 API / 其余 headless，1M 参数） |
| `launch_online_dp.py` | DP 实例启动器（复用 pd-separation 版） |

## 参考

- [GLM-5.2 官方教程 — 1M 共部署配置](https://docs.vllm.ai/projects/ascend/zh-cn/latest/tutorials/models/GLM5.2.html)
- 135K PD 分离（已验证）: `../pd-separation-135k/`
