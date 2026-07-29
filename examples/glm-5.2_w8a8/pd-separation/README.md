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
| FUSED_MC2 | 未开启 | P/D 均 **false** | W8A8 跨节点 EP 下 aclnnDispatchFFNCombine 崩溃（实测） |
| Connector | MooncakeConnectorV1 | 同左 | use_ascend_direct；DSA 模型有上游 bug，见「已知问题」 |
| MTP | P: 1 token / D: 3 tokens | 同左 | enforce_eager=true |
| recompute_scheduler | P: true / D: true | P/D 均 **false** | P 节点仅告警忽略；D 侧排查期间关闭（非根因，未恢复） |
| D 图模式 | FULL_DECODE_ONLY | **enforce-eager** | kv_consumer + FULL_DECODE capture 触发量化 op bug（见「已知问题」） |
| D batched tokens | 128 | 2048 | 官方值偏小；非根因，未回改 |
| sparse c8 | sfa/li: true | 均 **false** | W4A8C8 的 KV C8 量化，W8A8 无对应 scale（非根因，保守关闭） |
| max-num-seqs | P: 8 / D: 32 | P: 8 / D: **16** | W8A8 KV 池更小，保守起步 |
| gpu-mem-util | P: 0.75 / D: 0.93 | P: 0.93 / D: 0.93 | W8A8 权重大；P 实测 util 0.85 时 KV 不足 1M（需 ≥8.91 GiB/卡） |
| enable_balance_scheduling | — | ❌ 不用 | 仅 kv_both 模式可用（PD 分离下报错） |
| DYNAMIC_EPLB / MLAPO | D 端优化 | EPLB=0 / MLAPO=1 | EPLB 曾为嫌疑关闭（非根因），MLAPO 保留 |

## 验证记录

| 时间 | 配置 | 结果 | 说明 |
|------|------|------|------|
| 2026-07-28 | P: DP4 TP8 DCP8, FUSED_MC2=0, util 0.93 | ✅ 4/4 就绪 | curl /v1/models 全通 |
| 2026-07-28 | D: DP4 TP8 DCP8, **enforce-eager** | ✅ 4/4 就绪 | 可独立服务；PD 端到端被 Mooncake bug 阻断（见下） |

### 关键调试记录（D 侧，10 轮）

1. ~~FUSED_MC2=1~~ → `aclnnDispatchFFNCombine`（跨节点 EP 崩溃）→ 关
2. ~~DP 握手竞态~~ → rendezvous 2/4 → rank0 错峰先行 60s
3. ~~`aclnnMoeDistributeDispatchV4` 561000~~ → plog 显示 HCCL transport init timeout → `HCCL_CONNECT/EXEC_TIMEOUT=600`
4. ~~`aclnnAscendQuantV3` 161002（scale 2048 vs x 16384）~~ → 排除 EPLB/batched/c8/recompute，定位于 **kv_consumer + FULL_DECODE_ONLY capture** → 改 enforce-eager 通过
5. PD 端到端 → Mooncake `_transfer_kv_cache_all_groups` IndexError（见「已知问题」#1）

## 已知问题（均为上游 vllm-ascend 问题，v0.23.0rc1）

### 1. MooncakeConnectorV1 不支持 DSA 模型分离的 indexer KV cache（阻断 PD 端到端）

- **现象**：D 接收 KV 时 `IndexError`（`_transfer_kv_cache_all_groups` → `remote_kv_caches_base_addrs[layer_idx][cache_idx]`），请求 500
- **根因**：GLM-5.2（DSA）的 indexer KV cache 与 MLA cache 分组，connector 的 `use_hybrid` 判定未适配
- **上游修复**：[#12863](https://github.com/vllm-project/vllm-ascend/pull/12863)（`a0cc4ce1d`，2026-07-28，main）——**不在 v0.23.0rc1**；且依赖 v0.23.0rc1 中不存在的 `AscendSFAIndexerCacheSpec`，无法直接 cherry-pick
- **对策**：等包含该修复的新镜像；或先在 kv_both（共部署，见 `../dp1m/`）下承载 1M

### 2. kv_consumer + FULL_DECODE_ONLY capture 量化 op 形状错误

- **现象**：`aclnnAscendQuantV3` 161002，`scale dim(0)=2048 vs x dim(1)=16384`（q_lora/q_proj 维度错绑）
- **规避**：D 侧 `--enforce-eager`（当前配置）；代价是 decode 无 CUDA Graph、吞吐降低
- P 侧（producer、enforce-eager）与 dp1m（kv_both、FULL_DECODE_ONLY）均不触发

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

# 请求转发代理（运行于 P0 容器内 :8123，注册全部 4P+4D 端点）
bash remote_deploy.sh proxy
bash remote_deploy.sh proxy-stop
```

## 当前运行状态（2026-07-28）

- **P 侧 4/4 就绪**（9081），**D 侧 4/4 存活**（9900，enforce-eager 形态）
- 代理已部署于 `10.42.11.202:8123`（`/v1/models` 正常，`max_model_len=1024000`）
- **端到端请求仍被 Mooncake bug 阻断**（见「已知问题」#1）：代理链路 P→D 转发正常，
  D 接收 KV 时 IndexError → 500。1M 生产流量请走共部署 `../dp1m/`（194:8007）

## 验证

```bash
# 各节点 /v1/models
bash remote_deploy.sh status

# 代理出口（P0 容器内执行，或任何可达 202 的机器）
curl http://10.42.11.202:8123/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"glm-52","messages":[{"role":"user","content":"Hello"}],"max_tokens":50}'
```

## 参考

- [GLM-5.2 官方教程 — PD 分离与 1M 配置](https://docs.vllm.ai/projects/ascend/zh-cn/latest/tutorials/models/GLM5.2.html)
- 已验证参考实现（132K, v0.22.1 时代）: `batch_remote_deploy_pd_seg/glm52-deploy-scripts/`
