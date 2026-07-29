# GLM-5.2 W8A8 PD 分离部署（135K 上下文）

> ✅ **2026-07-29 已验证 PASS（端到端）** | 8×A2 (64G×8) | P: DP4 TP8 / D: DP8 TP4 | MooncakeConnector (kv_p2p)
> 入口: `http://10.42.11.202:8123`（代理） | 节点: `nodes/node_list4.txt`（10.42.11.202-209）

## 架构

```
PNode 0-3 (kv_producer, 202-205)         DNode 0-3 (kv_consumer, 206-209)
DP=4 TP=8  每节点 1 实例×8卡 :9081         DP=8 TP=4  每节点 2 实例×4卡 :9900/:9901
enforce-eager, MTP 1 token               enforce-eager, MTP 3 tokens
FLASHCOMM1=1, batched 4096, seqs 64      MLAPO=1, EPLB=0, batched 164, seqs 48
gpu-mem-util 0.95                        gpu-mem-util 0.92
         │                                      │
         └──── MooncakeConnector (kv_p2p, use_ascend_direct) ────┘
              prefill dp4tp8 / decode dp8tp4
         ┌──── 代理 (202:8123, 4×P 端点 + 8×D 端点) ────┐
```

## 节点分配

| IP | 角色 | DP rank | vllm 端口 |
|-----|------|---------|-----------|
| 10.42.11.202 | PNode 0 (DP master + 代理) | 0 | 9081 |
| 10.42.11.203 | PNode 1 | 1 | 9081 |
| 10.42.11.204 | PNode 2 | 2 | 9081 |
| 10.42.11.205 | PNode 3 | 3 | 9081 |
| 10.42.11.206 | DNode 0 (DP master) | 0, 1 | 9900, 9901 |
| 10.42.11.207 | DNode 1 | 2, 3 | 9900, 9901 |
| 10.42.11.208 | DNode 2 | 4, 5 | 9900, 9901 |
| 10.42.11.209 | DNode 3 | 6, 7 | 9900, 9901 |

## 与官方 8×A2 PD 配置的对齐/偏差

拓扑（P dp4tp8 / D dp8tp4、每节点 rank 划分、端口、rank-start 0/2/4/6）与
[官方 GLM-5.2 教程](https://docs.vllm.ai/projects/ascend/zh-cn/latest/tutorials/models/GLM5.2.html)
的 8×A2 布局一致。偏差项：

| 项 | 官方 (W4A8C8) | 本配置 (W8A8, v0.23.0rc1) | 说明 |
|----|--------------|--------------------------|------|
| Connector | MooncakeConnectorV1 | **旧版 MooncakeConnector (kv_p2p)** | V1 在本镜像对 DSA 模型有 indexer KV 传输 bug（见 `../pd-separation/README.md` 已知问题 #1），kv_p2p 为已验证路径 |
| D 图模式 | FULL_DECODE_ONLY | **enforce-eager** | kv_consumer + capture 触发 `aclnnAscendQuantV3` 161002（1M 方案实测）；代价是 decode 无图、吞吐降低 |
| DYNAMIC_EPLB | D 端开启 | **=0** | v0.23.0rc1 下导致 `aclnnMoeDistributeDispatchV4` 561000（2026-07-29 8/8 D rank 崩溃实测） |
| HCCL 超时 | 未显式设置 | CONNECT/EXEC=600 | 561000 transport init timeout 对策 |
| max-model-len | 256000 | 135168 | 权重 W8A8 更大，KV 池受限，保守取值 |
| FUSED_MC2 | 开启 | **=0** | W8A8 跨节点 EP 下 `aclnnDispatchFFNCombine` 崩溃（实测） |

## 验证记录

| 时间 | 结果 | 说明 |
|------|------|------|
| 2026-07-29 上午 | ❌ D 8/8 崩溃 | `DYNAMIC_EPLB=1` + FULL_DECODE_ONLY → capture 阶段 `aclnnMoeDistributeDispatchV4` 561000，rank5 伴随 glibc 堆损坏 |
| 2026-07-29 13:35 | ✅ P 4/4 + D 8/8 就绪 | 应用下述修复后 ~25 分钟全部就绪 |
| 2026-07-29 13:38 | ✅ 端到端 PASS | 代理 8123 连续 chat 请求成功，fingerprint `tp4-dp8-ep` 确认 P→D KV 传输链路工作 |

### 2026-07-29 修复清单

- `dnode.sh`：`DYNAMIC_EPLB=1→0`；补 `HCCL_CONNECT/EXEC_TIMEOUT=600`；去掉
  `FULL_DECODE_ONLY` 改 `--enforce-eager`
- `pnode.sh`：MTP `num_speculative_tokens` 3→1（对齐官方 P 侧配置）
- `remote_deploy.sh`：
  - `wait_ready`/`cmd_status` 的 curl 在 `set -e + pipefail` 下让健康检查子 shell
    静默自杀（此前"部署完成"为误报）→ 加 `|| true`
  - D 就绪/状态检查补齐每节点第二个实例端口 9901
  - 代理 D 端点 4→8（官方 8×A2 布局要求每 D 节点注册 9900+9901）
  - `pkill`/`pgrep` 自匹配（pattern 命中外层 bash 自身导致误杀/误报）→ `[v]` 括号技巧；
    `stop` 补杀 `VLLM::EngineCore` 进程

## 使用

```bash
bash remote_deploy.sh deploy    # P 全部就绪后再起 D，健康检查（~25 分钟）
bash remote_deploy.sh status    # P×4 + D×8 端点状态
bash remote_deploy.sh stop      # 停止所有节点 vLLM 进程
bash remote_deploy.sh clean     # 彻底清理（重启容器，回收驱动级显存残留）
bash remote_deploy.sh proxy     # 启动代理 (202:8123)
bash remote_deploy.sh proxy-stop
```

## 验证

```bash
curl --noproxy '*' http://10.42.11.202:8123/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"glm-52","messages":[{"role":"user","content":"Hello"}],"max_tokens":50}'
```

## 已知限制

- D 侧 enforce-eager：decode 无 CUDA Graph，吞吐有损失；待上游修复
  `aclnnAscendQuantV3`（kv_consumer + FULL_DECODE_ONLY capture）后可恢复图模式
- 1M 上下文需求请走共部署 `../dp1m/`（194:8007）；1M PD 分离（`../pd-separation/`）
  被 MooncakeConnectorV1 的 DSA indexer KV bug（上游 #12863）阻断

## 文件

| 文件 | 用途 |
|------|------|
| `deploy.conf` | 节点 IP、DP/TP 拓扑、端口、显存利用率等全部差异化参数 |
| `remote_deploy.sh` | 一键部署/停止/状态/代理（SSH 自动化） |
| `start_pnode.sh` / `start_dnode.sh` | 单节点启动入口（读 deploy.conf 注入环境） |
| `pnode.sh` / `dnode.sh` | P/D 节点 vllm 启动模板（由 launch_online_dp.py 调用） |
| `launch_online_dp.py` | DP 启动器（`--script` 外部化，P/D 共用） |
| `check_status.sh` / `stop_node.sh` | 单节点辅助脚本 |
