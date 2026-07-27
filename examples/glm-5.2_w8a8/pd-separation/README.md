# GLM-5.2 W8A8 PD 分离部署

> A2 PD: 4×PNode + 4×DNode | DP=4/4 TP=8/8 | MooncakeHybridConnector | 4K ctx

## 架构

```
PNode (4节点, kv_producer)            DNode (4节点, kv_consumer)
DP=4 TP=8  每节点1实例×8卡             DP=4 TP=8  每节点1实例×8卡
rank 0..3  全局统一编址                rank 0..3  全局统一编址
                                       │
FLASHCOMM1=0 (DSA CP incompatible)     FLASHCOMM1=1 (DSA CP requires SP)
enforce-eager                          enforce-eager (no FULL_DECODE_ONLY)
enable-prefix-caching                  no-enable-prefix-caching
MTP: deepseek_mtp 3 tokens             MTP: disabled
enable_dsa_cp ✅                       enable_dsa_cp ✅
         │                                      │
         └────── MooncakeHybridConnector ───────┘
                TP_P(8) >= TP_D(8) ✅
```

> **约束**: A2 ~60.4 GiB/NPU，max-model-len=4096。TP=16 不支持 (MLA dim 576%16≠0)

## 节点分配

| IP | 角色 | 实例 | NPU | engine_id | dp_rank |
|-----|------|------|-----|-----------|---------|
| 10.42.11.194 | PNode 0 | 1 | 8 | 0 | 0 |
| 10.42.11.195 | PNode 1 | 1 | 8 | 1 | 1 |
| 10.42.11.196 | PNode 2 | 1 | 8 | 2 | 2 |
| 10.42.11.197 | PNode 3 | 1 | 8 | 3 | 3 |
| 10.42.11.198 | DNode 0 | 1 | 8 | 4 | 0 |
| 10.42.11.199 | DNode 1 | 1 | 8 | 5 | 1 |
| 10.42.11.200 | DNode 2 | 1 | 8 | 6 | 2 |
| 10.42.11.201 | DNode 3 | 1 | 8 | 7 | 3 |

## GLM-5.2 特有配置

| 参数 | 值 | 说明 |
|------|-----|------|
| FLASHCOMM1 | P:0 D:1 | DSA CP 与 FLASHCOMM1 冲突(P), 但 DSA CP 需要 SP(D) |
| MTP method | deepseek_mtp | 3 tokens (vs mtp 1 token for DeepSeek) |
| tool-call-parser | glm47 | GLM 系解析器 |
| reasoning-parser | glm45 | GLM 系推理解析器 |
| chat-template | string | GLM 使用 string 格式 |
| enable_balance_scheduling | ❌ PD 不支持 | 仅 kv_both 模式可用 |

## 文件

| 文件 | 用途 |
|------|------|
| `deploy.conf` | 全局配置 |
| `remote_deploy.sh` | 一键部署 |
| `pnode.sh` | PNode 模板 |
| `dnode.sh` | DNode 模板 |
| `launch_online_dp.py` | DP 启动器 |

## 使用

```bash
bash remote_deploy.sh clean && bash remote_deploy.sh deploy
```

## 验证

```bash
for ip in 194 195 196 197 198 199 200 201; do
  ssh 10.42.11.$ip "docker exec vllm-ascend-env bash -c 'curl -s --max-time 3 http://localhost:8200/v1/models'"
done
```
