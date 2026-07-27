# DeepSeek-V4-Flash PD 分离部署

> 官方 §5.2: MooncakeHybridConnector | 135K | A2 | vLLM v0.23.0

## 架构

```
PNode (4节点, kv_producer)            DNode (4节点, kv_consumer)
DP=8 TP=4  每节点2实例×4卡             DP=4 TP=4  每节点1实例×4卡
rank 0..7  全局统一编址                rank 0..3  全局统一编址
                                       │
enable-prefix-caching                  no-enable-prefix-caching
enforce-eager                          FULL_DECODE_ONLY
         │                                      │
         └────── MooncakeHybridConnector ───────┘
                TP_P(4) >= TP_D(4) ✅
```

## 节点分配

| IP | 角色 | DP实例 | NPU/实例 | engine_id | dp_rank |
|-----|------|--------|---------|-----------|---------|
| 10.42.11.194 | PNode 0 (主P) | 2 | 4 | 0 | 0,1 |
| 10.42.11.195 | PNode 1 | 2 | 4 | 1 | 2,3 |
| 10.42.11.196 | PNode 2 | 2 | 4 | 2 | 4,5 |
| 10.42.11.197 | PNode 3 | 2 | 4 | 3 | 6,7 |
| 10.42.11.198 | DNode 0 (主D) | 1 | 4 | 4 | 0 |
| 10.42.11.199 | DNode 1 | 1 | 4 | 5 | 1 |
| 10.42.11.200 | DNode 2 | 1 | 4 | 6 | 2 |
| 10.42.11.201 | DNode 3 | 1 | 4 | 7 | 3 |

## 参数速查

| 参数 | PNode | DNode |
|------|-------|-------|
| DP (全局) | 8 | 4 |
| TP | 4 | 4 |
| max-model-len | 135000 | 135000 |
| max-num-batched-tokens | 4096 | 60 |
| max-num-seqs | 16 | 30 |
| compilation-config | ❌ | FULL_DECODE_ONLY |
| prefix-caching | enable | no-enable |
| enforce-eager | ✅ | ❌ |
| Mooncake kv_port | 30000-30003 | 30400-30403 |

## 文件

| 文件 | 用途 |
|------|------|
| `deploy.conf` | 全局配置 (IP, DP/TP 全部在此) |
| `remote_deploy.sh` | 一键部署 |
| `pnode.sh` | PNode 模板 (kv_producer) |
| `dnode.sh` | DNode 模板 (kv_consumer) |
| `launch_online_dp.py` | DP 多进程启动器 |

## 使用

```bash
bash remote_deploy.sh clean     # 彻底清理 (重启容器, 清除僵尸)
bash remote_deploy.sh deploy    # 部署 (P+D 并行)
bash remote_deploy.sh status    # 状态
bash remote_deploy.sh stop      # 停止
bash remote_deploy.sh restart   # 重启
```

## 约束

- **Mooncake 硬性要求: Prefill TP ≥ Decode TP**
- 修改 TP/DP 需同时更新 `deploy.conf` 和 `pnode.sh`/`dnode.sh` 中的 `kv_connector_extra_config`

## 验证

```bash
# 推理测试
curl -s http://10.42.11.194:7100/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"dsv4","messages":[{"role":"user","content":"hi"}],"max_tokens":16}'

# 进程检查 (PNode 2/节点, DNode 1/节点)
for ip in 194 195 196 197 198 199 200 201; do
  ssh 10.42.11.$ip "docker exec vllm-ascend-env bash -c 'npu-smi info' 2>/dev/null | head -4"
done
```
