# LongCat-Flash-Chat-1024E-512Zero-Topk24-v2 部署指南

## 模型概况

| 属性 | 值 |
|------|-----|
| 架构 | LongcatFlashForCausalLM (MLA + MoE) |
| 专家数 | 1024 Routed + 512 Zero (Identity) |
| TopK | 24 |
| Hidden Size | 6144 |
| Layers | 28 |
| KV LoRA Rank | 512 |
| Max Position | 131072 |
| Vocab Size | 131072 |
| 精度 | bfloat16 (无量化) |

## 硬件需求

- **最小配置**: 64 × 昇腾 NPU (8 节点 × 8 卡, 每卡 64GB)
- **推荐配置**: TP=64, PP=1, EP enabled
- **框架**: vLLM-Ascend 0.18.0 + Ray 分布式
- **容器**: ascend910c-cann8.5.1-torch2.9.0-vllm0.18.0

## 文件说明

```
examples/longcat/vllm/
├── run_vllm.sh       ← 直接 vllm serve 部署
├── curl_test.sh      ← API 功能测试
└── README.md         ← 本文档
```

## 前置条件

### 1. MC2 Patch（必须）

由于 TP=64 时每卡本地专家数 (1024/64=16) 小于 topk (24)，MC2 MoE dispatch 内核
(`aclnnMoeDistributeDispatchV4`) 会报错 561002。**部署前必须在所有节点上 patch
`ascend_forward_context.py`**，强制使用 ALLGATHER 通信：

```bash
# 在所有 8 个节点上执行
for ip in 17 18 19 20 21 22 23 24; do
    ssh 10.1.0.$ip "docker exec vllm-ascend-env sed -i \
      's/if num_experts_per_device <= 24 and ep_world_size >= 16 and num_tokens <= mc2_tokens_capacity:/if False:  # Patched: topk>local_experts/' \
      /opt/conda/env/lib/python3.11/site-packages/vllm_ascend/ascend_forward_context.py"
    ssh 10.1.0.$ip "docker exec vllm-ascend-env sed -i \
      's/elif soc_version in {AscendDeviceType.A3}:/elif False:  # Patched: force ALLGATHER/' \
      /opt/conda/env/lib/python3.11/site-packages/vllm_ascend/ascend_forward_context.py"
    # 清除 .pyc 缓存
    ssh 10.1.0.$ip "docker exec vllm-ascend-env find /opt/conda/env/lib/python3.11/site-packages/vllm_ascend/ \
      -path '*__pycache__*' -name 'ascend_forward*' -delete"
done
```

> **原因**: vLLM-Ascend 的 `select_moe_comm_method()` 在 A2/A3 设备上会选择 MC2
> 通信方式，但 MC2 要求 `local_experts >= topk`。LongCat 在 TP=64 时不满足此条件。

### 2. 避免 EasyInfer 与 NPUSlim 冲突

容器中已预装 npuslim 包（editable install），两者都注册了 `ZeroExpertFusedMoE` op。
**运行 `run_vllm.sh` 时不要从 EasyInfer 目录启动**，否则 Python 会同时加载两个
插件导致 `AssertionError: Duplicate op name: ZeroExpertFusedMoE`。

```bash
# 正确方式 (不 cd 到 EasyInfer)
bash /home/jianzhnie/llmtuner/llm/EasyInfer/examples/longcat/vllm/run_vllm.sh

# 错误方式 (会触发 easyinfer 插件自动发现)
cd /home/jianzhnie/llmtuner/llm/EasyInfer && bash examples/longcat/vllm/run_vllm.sh
```

## 部署步骤

### 1. 准备节点列表

使用 `scripts/node_list_8.txt` (8 节点 × 8 卡 = 64 NPU):

```
10.1.0.17
10.1.0.18
10.1.0.19
10.1.0.20
10.1.0.21
10.1.0.22
10.1.0.23
10.1.0.24
```

### 2. 重启容器并启动 Ray 集群

```bash
# 重启容器（确保干净状态）
bash scripts/docker/manage_npuslim_containers.sh restart --file scripts/node_list_8.txt

# 等待容器就绪
sleep 20

# 启动 Ray 集群
bash scripts/ray_cluster/start_npuslim_ray_cluster.sh start --file scripts/node_list_8.txt

# 验证 64 NPU 可用
ssh 10.1.0.17 "docker exec vllm-ascend-env ray status | grep NPU"
# 预期输出: 0.0/64.0 NPU
```

### 3. 应用 MC2 Patch

参见「前置条件：MC2 Patch」章节。

### 4. 启动模型服务

```bash
# 注意: 不要从 EasyInfer 目录运行，避免插件冲突
bash /home/jianzhnie/llmtuner/llm/EasyInfer/examples/longcat/vllm/run_vllm.sh
```

支持通过环境变量覆盖默认配置：

```bash
TP=64 MAX_MODEL_LEN=8192 MAX_NUM_SEQS=64 bash /home/jianzhnie/llmtuner/llm/EasyInfer/examples/longcat/vllm/run_vllm.sh
```

> 模型加载约需 16-20 分钟 (148 个 safetensors 分片 + 64 卡 HCCL 初始化)。

### 5. 测试模型

```bash
bash examples/longcat/vllm/curl_test.sh
```

支持指定远程地址测试：

```bash
HOST=10.1.0.17 PORT=8010 bash examples/longcat/vllm/curl_test.sh
```

## 关键配置说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| TP | 64 | 张量并行度 (8 节点 × 8 卡) |
| PP | 1 | 流水线并行度 |
| PORT | 8010 | 服务端口 |
| MAX_MODEL_LEN | 4096 | 最大序列长度 (可调大, 受显存限制) |
| MAX_NUM_SEQS | 128 | 最大并发序列数 |
| GPU_MEM_UTIL | 0.90 | 显存利用率 |
| HCCL_CONNECT_TIMEOUT | 1800 | HCCL 连接超时 (秒) |
| HCCL_SOCKET_IFNAME | enp66s0f1 | HCCL 网络接口名 |

## 已知问题与注意事项

- **MC2 内核不兼容**: TP=64 时 local_experts(16) < topk(24)，必须 patch 禁用 MC2
- **EasyInfer/NPUSlim 冲突**: 容器预装 npuslim，不要从 EasyInfer 目录启动 vllm
- **TP=32 OOM**: 每卡需加载 ~59GB 模型权重，超过 64GB 卡可用显存，不可用
- **HCCL 初始化慢**: 64 卡跨 8 节点的 HCCL 通信建立需 5-15 分钟，属正常
- **HCCL 间歇性卡死**: 集群 RDMA 网络不稳定时可能需要重试（重启容器后重新部署）
- **节点 10.1.0.28 NPU 驱动故障**: drvErr=87，已从节点列表中排除
- 模型使用自定义 `modeling_longcat_flash.py`，需 `--trust-remote-code`
- MoE 模型启用 `--enable-expert-parallel` 进行专家并行
- 使用 `--no-enable-prefix-caching` 和 `--enforce-eager` 保证稳定性
- 148 个 safetensors 分片文件，首次加载约 16 分钟（~6.7s/shard）
- 服务端口 8010，避免与其他模型冲突
- NPUSlim 插件自动注册 `AscendZeroExpertFusedMoE` 支持 zero expert

## 故障排查

### Ray placement group 资源泄露

如果 kill 模型进程后 `ray status` 显示 NPU 仍被占用：

```bash
# 方法 1: 通过 Python 清理
ssh 10.1.0.17 "docker exec vllm-ascend-env python3 -c '
import ray; ray.init()
for pg_id in ray.util.placement_group_table():
    try:
        ray.util.remove_placement_group(ray.util.get_placement_group(pg_id))
    except: pass
'"

# 方法 2: 重启 Ray 集群
bash scripts/ray_cluster/start_npuslim_ray_cluster.sh stop --file scripts/node_list_8.txt
bash scripts/ray_cluster/start_npuslim_ray_cluster.sh start --file scripts/node_list_8.txt
```

### HCCL 通信卡住

如果日志停在 `parallel_state.py` 初始化超过 15 分钟：

1. 确认 `HCCL_SOCKET_IFNAME` 与实际网卡匹配: `ip addr | grep enp`
2. 确认所有节点间网络互通: `for ip in 17..24; do ping -c1 10.1.0.$ip; done`
3. 增大超时: `HCCL_CONNECT_TIMEOUT=3600 bash run_vllm.sh`
