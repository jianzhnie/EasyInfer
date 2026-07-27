# DeepSeek-V4-Flash PD 分离部署

> 官方 §5.2: MooncakeHybridConnector | 135K | A2

## 资源分配

```
PNode (4×8 NPU, kv_producer)        DNode (4×8 NPU, kv_consumer)
10.42.11.194-197                     10.42.11.198-201
DP=4 TP=8  eng=0..3                 DP=4 TP=8  eng=4..7
rank=0,1,2,3                         rank=0,1,2,3
1 实例 × 8 NPU/节点                  1 实例 × 8 NPU/节点
                                       │
enforce-eager                        FULL_DECODE_ONLY
enable-prefix-caching                no-enable-prefix-caching
         │                                    │
         └────── MooncakeHybridConnector ─────┘
```

| 参数 | PNode | DNode |
|------|-------|-------|
| DP (全局) | 4 | 4 |
| TP | 8 | 8 |
| 每节点实例 | 1 (8 NPU) | 1 (8 NPU) |
| max-model-len | 135000 | 135000 |
| max-num-seqs | 16 | 30 |
| max-num-batched-tokens | 4096 | 60 |
| enforce-eager | ✅ | ❌ |
| compilation-config | ❌ | FULL_DECODE_ONLY |
| prefix-caching | enable | no-enable |
| additional-config | cpu_binding+shared_expert_dp | +npugraph_ex+recompute_scheduler |
| kv_role | producer | consumer |
| engine_id | 0-3 | 4-7 |
| Mooncake port | 30000-30003 | 30400-30403 |

## 文件

| 文件 | 用途 |
|------|------|
| `deploy.conf` | 全局配置 |
| `remote_deploy.sh` | 一键部署 (deploy/clean/stop/status/restart) |
| `pnode.sh` | Prefill 模板 (kv_producer) |
| `dnode.sh` | Decode 模板 (kv_consumer) |
| `launch_online_dp.py` | DP 多进程启动器 |

## 节点分配

| IP | 角色 | engine_id | dp_rank |
|-----|------|-----------|---------|
| 10.42.11.194 | PNode 0 (主) | 0 | 0 |
| 10.42.11.195 | PNode 1 | 1 | 1 |
| 10.42.11.196 | PNode 2 | 2 | 2 |
| 10.42.11.197 | PNode 3 | 3 | 3 |
| 10.42.11.198 | DNode 0 (主) | 4 | 0 |
| 10.42.11.199 | DNode 1 | 5 | 1 |
| 10.42.11.200 | DNode 2 | 6 | 2 |
| 10.42.11.201 | DNode 3 | 7 | 3 |

## 使用

```bash
bash remote_deploy.sh clean     # 彻底清理(重启容器, 清除僵尸)
bash remote_deploy.sh deploy    # 部署
bash remote_deploy.sh status    # 状态
bash remote_deploy.sh stop      # 停止(不重启容器)
bash remote_deploy.sh restart   # 重启

# 修改 TP/DP 等参数: 编辑 deploy.conf 后重新部署
```

## 验证

```bash
# 进程检查 (每节点应 1 实例 ≈ 8 vllm 进程)
for ip in 194 195 196 197 198 199 200 201; do
  ssh 10.42.11.$ip "docker exec vllm-ascend-env bash -c 'ps aux|grep vllm|grep -v grep|grep -v defunct|wc -l'"
done

# 推理测试
ssh 10.42.11.194 "docker exec vllm-ascend-env bash -c 'curl -s --max-time 60 http://localhost:7100/v1/chat/completions -H \"Content-Type: application/json\" -d '\''{\"model\":\"dsv4\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":16}'\'''"
```
