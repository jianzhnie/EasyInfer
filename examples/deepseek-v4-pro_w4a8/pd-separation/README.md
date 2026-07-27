# DeepSeek-V4-Pro W4A8 PD 分离部署

> 官方 A2 PD: 4×PNode + 4×DNode | DP=4/8 TP=8/4 | MooncakeHybridConnector | 135K

## 架构

```
PNode (4节点, kv_producer)            DNode (4节点, kv_consumer)
DP=4 TP=8  每节点1实例×8卡             DP=8 TP=4  每节点2实例×4卡
rank 0..3  全局统一编址                rank 0..7  全局统一编址
                                       │
enforce-eager (无async-scheduling)     async-scheduling (无enforce-eager)
FLASHCOMM1=1 ✅                        no FLASHCOMM1
enable-prefix-caching                  no-enable-prefix-caching
enable_dsa_cp ✅                       recompute_scheduler + npugraph_ex
         │                                      │
         └────── MooncakeHybridConnector ───────┘
                TP_P(8) >= TP_D(4) ✅
```

> **与 Flash 差异**: V4-Pro 384 experts, Prefill FLASHCOMM1=1 (TP=8), enable_dsa_cp

## 节点分配

| IP | 角色 | 实例 | NPU/实例 | engine_id | dp_rank |
|-----|------|------|---------|-----------|---------|
| 10.42.11.194 | PNode 0 (主P) | 1 | 8 | 0 | 0 |
| 10.42.11.195 | PNode 1 | 1 | 8 | 1 | 1 |
| 10.42.11.196 | PNode 2 | 1 | 8 | 2 | 2 |
| 10.42.11.197 | PNode 3 | 1 | 8 | 3 | 3 |
| 10.42.11.198 | DNode 0 (主D) | 2 | 4 | 4 | 0,1 |
| 10.42.11.199 | DNode 1 | 2 | 4 | 5 | 2,3 |
| 10.42.11.200 | DNode 2 | 2 | 4 | 6 | 4,5 |
| 10.42.11.201 | DNode 3 | 2 | 4 | 7 | 6,7 |

## 文件

| 文件 | 用途 |
|------|------|
| `deploy.conf` | 全局配置 (IP, DP/TP) |
| `remote_deploy.sh` | 一键部署 (deploy/clean/stop/status/restart) |
| `pnode.sh` | PNode 模板 (kv_producer) |
| `dnode.sh` | DNode 模板 (kv_consumer) |
| `launch_online_dp.py` | DP 多进程启动器 |

## 使用

```bash
bash remote_deploy.sh clean     # 彻底清理 (重启容器)
bash remote_deploy.sh deploy    # 部署
bash remote_deploy.sh status    # 状态
bash remote_deploy.sh stop      # 停止
bash remote_deploy.sh restart   # 重启
```

## 参数速查

| 参数 | PNode | DNode |
|------|-------|-------|
| DP (全局) | 4 | 8 |
| TP | 8 | 4 |
| max-model-len | 135000 | 135000 |
| max-num-batched-tokens | 4096 | 120 |
| max-num-seqs | 16 | 60 |
| GPU_MEM_UTIL | 0.85 | 0.85 |
| compilation-config | ❌ | FULL_DECODE_ONLY |
| prefix-caching | enable | no-enable |
| enforce-eager | ✅ | ❌ |
| async-scheduling | ❌ | ✅ |
| FLASHCOMM1 | ✅ (TP=8) | ❌ |
| enable_dsa_cp | ✅ | ❌ |
| Mooncake kv_port | 30000-30003 | 30400-30403 |

## 实测

| 指标 | 值 |
|------|-----|
| 权重 | 41.6 GB (W4A8) |
| Shards | 205 |
| 加载时间 | ~7 min |
| 推理延迟 | < 1s (8 tokens) |

## 约束

- **Mooncake 硬性要求: Prefill TP(8) ≥ Decode TP(4)**
- V4-Pro 384 experts 编译慢于 256 experts 模型
- 修改 TP/DP 需同步更新 `kv_connector_extra_config`

## 验证

```bash
# 推理测试
ssh 10.42.11.194 "docker exec vllm-ascend-env bash -c 'curl -s --max-time 60 http://localhost:8100/v1/chat/completions -H \"Content-Type: application/json\" -d '\''{\"model\":\"deepseek-v4-pro\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":16}'\'''"

# 进程检查 (PNode 1/节点, DNode 2/节点)
for ip in 194 195 196 197 198 199 200 201; do
  ssh 10.42.11.$ip "docker exec vllm-ascend-env bash -c 'ps aux|grep vllm|grep -v grep|grep -v defunct|wc -l'"
done
```
