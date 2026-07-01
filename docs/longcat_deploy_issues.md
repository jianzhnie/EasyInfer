# LongCat-Flash-Chat 部署问题总结

> 模型: `LongCat-Flash-Chat-1024E-512Zero-E-Topk24-v2`
> 节点列表:
> - `node_list1.txt` (16 nodes: 10.1.0.17 ~ 10.1.0.32)
> - `node_list2.txt` (14 nodes: 10.1.1.173 ~ 10.1.1.205)
> 目标配置: TP=64, 8 nodes × 8 NPU

---

## 问题 1: DP=2 不支持多节点 TP

**现象**: 设置 `TP=64, DP=2` 启动时报错:

```
Exception: Error setting ASCEND_RT_VISIBLE_DEVICES: local range: [0, 64) base value: "0,1,2,3,4,5,6,7"
```

**原因**: vLLM v1 引擎的 `set_device_control_env_var()` 尝试按 DP rank 切分本地设备，但每节点仅有 8 个 NPU，无法满足 local range [0, 64) 的需求。`--data-parallel-size` 在 Ray 多节点 TP 场景下不可用。

**解决**: 回退至 `TP=64, DP=1`。若需利用全部 16 节点，需启动两个独立 vLLM 实例（各 8 节点），通过外部负载均衡分发请求。

---

## 问题 2: Ray GCS 连接超时（反复出现）

**现象**: vLLM 启动后，Worker 节点大量报错:

```
rpc_client.h:153: Failed to connect to GCS at address 10.1.0.17:6379 within 5 seconds.
ray.exceptions.RaySystemError: Failed to connect to GCS.
worker_pool.cc:590: Some workers of the worker process have not registered within the timeout. The process is dead/hanging.
```

**原因**: 每个节点有 320 个 CPU 核心，Ray 默认按 CPU 数量预启动 Worker 进程。当 vLLM 请求创建 64-NPU 的 Placement Group 时，大量 Worker 进程同时尝试连接 GCS Server，导致 GCS 过载或网络拥塞。

**尝试的缓解措施**:
- 使用 `--num-cpus=16` 启动 Ray（限制资源声明）→ GCS 压力降低但问题仍存在
- 增大 `HCCL_CONNECT_TIMEOUT=3600` → 无效
- 设置 `RAY_DEDUP_LOGS=0` 观察详细日志 → 确认问题在 Worker 连接阶段

**建议**:
1. 检查集群网络 QoS，确认 port 6379 无限速/防火墙规则
2. 在 GCS 所在节点增加 `ulimit -n 65535` 提升文件描述符上限
3. 设置 `RAY_GCS_SERVER_REQUEST_TIMEOUT_SECONDS=60`
4. 考虑使用 `--distributed-executor-backend mp`（多进程模式）绕过 Ray

---

## 问题 3: VLLM_HOST_IP 检测错误导致 Placement Group 创建失败

**现象**: vLLM 创建 Ray placement group 时请求 `node:8.x.x.x`（如 `node:8.3.224.23`、`node:8.7.130.152`）而非正确的 `node:10.1.1.x` 或 `node:10.1.0.x`，导致 "No available node types can fulfill resource request"。

**原因**: 容器使用 `--net=host` 但存在多个网络接口，vLLM 自动检测 IP 时选取了错误的网络接口（8.x.x.x 是某内部接口的 IP，非 HCCL/Ray 通信使用的 10.1.x.x 接口）。这不仅影响 head 节点，还影响所有 Ray Worker，导致 "Every node should have a unique IP address" 错误。

**修复方案**: 在**每个节点**容器的 `/root/.bashrc` 中设置 `export VLLM_HOST_IP=<节点实际IP>`：

```bash
for ip in $(grep -v '^#' node_list.txt | grep -v '^$'); do
    ssh $ip "docker exec vllm-ascend-env bash -c 'echo export VLLM_HOST_IP=$ip >> /root/.bashrc'"
done
```

**关键点**: 必须在启动 Ray 集群之前设置。Raylet 通过 `bash -lc` 启动时 source `.bashrc`，Worker 进程继承 raylet 的环境变量。

---

## 问题 3b: 同一物理节点多容器导致 Ray 节点 IP 重复

**现象**: vLLM 报错:

```
RuntimeError: Every node should have a unique IP address. Got 8 nodes with [...] and 7 unique IP addresses.
```

**原因**: 物理节点上同时运行了 `vllm-ascend-env` 和 `mindspeed-llm-env` 容器（均使用 `--net=host`），两个容器内的 Ray 进程共享同一 IP，同时注册到集群导致 IP 冲突。

**解决**:
1. 部署前在所有容器中执行 `ray stop --force`（不仅仅是 vllm-ascend-env）
2. 使用独立端口（如 `--port=6380`）避免与其他容器 Ray 实例混淆
3. 或停止/删除不需要的容器（`mindspeed-llm-env`, `npuslim-env`）

---

## 问题 4: Ray Actor RPC 连接重置

**现象**: Engine Core 初始化阶段报错:

```
ray.exceptions.ActorUnavailableError: The actor is temporarily unavailable:
RpcError: RPC error: recvmsg:Connection reset by peer rpc_code: 14
```

随后:

```
RuntimeError: Engine core initialization failed. See root cause above.
```

**原因**: Worker Actor 在远程节点上被成功创建并激活了 Ascend 插件，但在 `collective_rpc` 调用时，Actor 的 RPC 连接被对端重置。可能是 RDMA 网络不稳定或节点间带宽竞争。

**建议**:
1. 确认 RDMA 网络健康: `ibstat` / `perftest` 跨节点验证
2. 设置 `VLLM_RPC_TIMEOUT=600` 和 `VLLM_V1_FRONTEND_ENGINE_CORE_TIMEOUT=1200`
3. 如反复出现，可能需要运维排查物理网络

---

## 问题 5: 容器 NPU 设备访问失败 (drvErr=87, node_list2 集群)

**现象**: node_list2 (10.1.1.x) 集群中，容器内 `torch.npu.device_count()` 返回 0，`npu-smi` 报错 "dcmi model initialized failed, because the device is used. ret is -8020"，所有 Ray Worker 初始化时 crash (`basic_string::_S_construct null not valid`)。

**原因**: `run_npuslim_container.sh` 脚本在 multi-node 模式下使用 `--device=/dev/davinci0..7` 挂载 NPU 设备，但在 node_list2 集群的机器上，仅 `--device` 方式不足以提供完整的 NPU 访问权限（与 cgroup 设备权限或 Ascend 驱动的设备管理机制有关）。

**验证**: 使用 `--privileged` 启动容器后 `torch.npu.device_count()` 正确返回 8。

**修复方案**: 在 node_list2 集群上必须使用 `--privileged` 标志启动容器：

```bash
docker run -d --privileged --net=host --shm-size=10g \
    --name vllm-ascend-env \
    -e ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    -e HCCL_IF_IP=<node_ip> \
    -e HCCL_SOCKET_IFNAME=<nic_name> \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
    -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware:ro \
    -v /usr/local/dcmi:/usr/local/dcmi:ro \
    -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
    -v /etc/hccn.conf:/etc/hccn.conf:ro \
    -v /home/jianzhnie/llmtuner:/home/jianzhnie/llmtuner \
    ascend910c-cann8.5.1-torch2.9.0-vllm0.18.0:latest bash -lc 'sleep infinity'
```

**建议**: 在 `run_npuslim_container.sh` 中增加 `--privileged` 选项或环境变量控制。

---

## 问题 5b: 部分节点 NPU 驱动故障

**现象**:
- 节点 `10.1.0.28`: `drvErr=87`（已在 README 中记录）
- 节点 `10.1.0.30`: `Can't get ascend_hal device count`
- 节点 `10.1.0.31`: `aclrtGetDeviceCountImpl: get device count failed, runtime result = 507899`

**原因**: NPU 驱动异常或硬件故障，`libascend_hal.so` 无法正常枚举设备。

**解决**: 将这些节点从部署列表中排除，仅使用 `10.1.0.17 ~ 10.1.0.27`（排除 28）中的 8 个健康节点。

---

## 问题 6: Docker 容器重启卡死

**现象**: 执行 `docker restart vllm-ascend-env` 时，部分节点报错:

```
Error response from daemon: Cannot restart container vllm-ascend-env: tried to kill container, but did not receive an exit event
```

或长时间无响应超时。

**原因**: 容器内 vLLM/Ray 进程持有 NPU 设备句柄，Docker 的 SIGTERM 无法正常终止进程，导致容器停止超时。

**解决**: 使用 `docker kill` 代替 `docker restart`，然后再 `docker start`:

```bash
docker kill vllm-ascend-env && sleep 2 && docker start vllm-ascend-env
```

---

## 问题 7: MC2 MoE 内核不兼容

**现象**: 若不打 patch，TP=64 时 vLLM 会触发 `aclnnMoeDistributeDispatchV4` 报错 561002。

**原因**: `local_experts = 1024 / TP = 16`，小于 `topk = 24`。MC2 内核要求 `local_experts >= topk`。

**解决**: 已通过 sed patch `ascend_forward_context.py` 强制禁用 MC2，使用 ALLGATHER 通信方式。每次容器重启后需重新 patch。

---

## 问题 8: CANN 环境未在 docker exec -d 中加载

**现象**: 使用 `docker exec -d` 启动 vLLM 时报错:

```
ImportError: libascend_hal.so: cannot open shared object file: No such file or directory
RuntimeError: Failed to load the backend extension: torch_npu
```

**原因**: `docker exec -d` 默认不启动 login shell，CANN 环境变量（LD_LIBRARY_PATH 等）未通过 `.bashrc` / `set_env.sh` 加载。

**解决**: 使用 `bash -lc` 确保 login shell:

```bash
docker exec -d vllm-ascend-env bash -lc "cd /tmp && bash run_vllm.sh > /tmp/vllm_serve.log 2>&1"
```

---

## 总结

| # | 问题 | 严重程度 | 状态 |
|---|------|----------|------|
| 1 | DP=2 不支持多节点 TP | 中 | 已绕过（DP=1） |
| 2 | Ray GCS 连接超时 | **高** | 未解决（集群网络） |
| 3 | VLLM_HOST_IP 检测错误 | **高** | 已修复（设置 .bashrc） |
| 3b | 多容器 IP 冲突 | **高** | 需停止其他容器 Ray |
| 4 | Actor RPC 连接重置 | **高** | 未解决（网络不稳定） |
| 5 | 容器 NPU 设备访问失败 | **高** | 已修复（--privileged） |
| 5b | 节点 NPU 驱动故障 | 中 | 已排除问题节点 |
| 6 | Docker 重启/删除卡死 | 中 | 用 kill+start 代替 |
| 7 | MC2 内核不兼容 | 中 | 已 patch |
| 8 | CANN 环境未加载 | 低 | 用 bash -lc |

**核心阻塞项**:
1. **Ray 模式 (node_list1)**: GCS 过载（问题 2）+ 多容器 IP 冲突（问题 3b）导致持续失败
2. **MP 模式 (node_list1)**: 已取得重大进展 — 64 rank 全部通过 torch distributed 连接，Gloo 建立成功（63/64 peers），但 Rank 10（node 10.1.0.18, local_rank 2）Gloo 连接失败导致 HCCL init 超时

**MP 模式部署进展**:
- ✅ 所有 64 个 worker 进程启动成功
- ✅ torch.distributed 初始化 (tcp://10.1.0.17:29501, backend=hccl)
- ✅ Gloo 通信建立 (63/64 peers connected)
- ❌ Rank 10 (node 10.1.0.18, device 2) Gloo 连接失败，阻塞全集群

**建议**:
1. 使用 MP 模式而非 Ray（已创建 `examples/longcat/vllm/run_vllm_mp.sh`）
2. 排除 10.1.0.18（device 2 网络异常），替换为 10.1.0.25/26/27
3. 部署前确保每个物理节点只有一个容器运行 Ray/vLLM
4. 设置 `HCCL_CONNECT_TIMEOUT=3600` 容忍长 HCCL 初始化
5. 使用 `docker exec -d bash -lc` 并行启动所有节点（避免顺序启动的时间差）

---

## HCCL 初始化时间参考

| 阶段 | 耗时 |
|------|------|
| 容器启动 (8-14 nodes) | ~10-30s |
| Ray 集群启动 | ~3-4 min |
| Placement group 分配 | ~1-3 min |
| HCCL 初始化 (64 NPU, 8 nodes) | 15-25 min |
| 模型权重加载 (148 shards, 34.5GB/rank) | 10-17 min |
| **总计** | **~30-50 min** |
