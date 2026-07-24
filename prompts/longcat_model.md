# LongCat-Flash-Chat 专家并行部署

## 环境概况

- **集群**: 16 节点 × 8 昇腾 NPU (Atlas 800 A2/A3, 每卡 64G)
- **框架**: vLLM-Ascend + Ray 分布式
- **镜像**: `quay.io/ascend/vllm-ascend:v0.23.0rc1-a3`
- **容器名**: `vllm-ascend-env`
- **挂载**: `/home/jianzhnie/llmtuner` → 容器内同路径
- **插件**: `/home/jianzhnie/llmtuner/llm/EasyInfer/easyinfer/plugins` → vLLM 插件系统
- **节点文件**: `node_list3.txt`

## 任务目标

实现 `meituan-longcat/LongCat-Flash-Chat` 模型的**专家并行 (EP)** 部署，确保模型输出正常、无乱码。

### 成功标准

1. `curl_test.sh` 返回的 `content` 字段为可读中文/英文，无乱码字符
2. 多轮对话输出一致、连贯
3. 所有 EP rank 无 dispatch kernel 报错

## 部署与测试流程

### Step 1: 启动容器群

```bash
bash scripts/docker/manage_npuslim_containers.sh start \
    --file node_list3.txt
```

### Step 2: 启动 Ray 集群

```bash
bash scripts/ray_cluster/start_npuslim_ray_cluster.sh start \
    --file node_list3.txt

# 验证: 确认所有节点 NPU 可用
ssh $HEAD "docker exec vllm-ascend-env ray status | grep -E 'NPU|Active'"
```

### Step 3: 部署模型

进入容器内执行：

```bash
ssh 10.42.11.194
docker exec -it vllm-ascend-env /bin/bash
cd /home/jianzhnie/llmtuner/llm/EasyInfer
bash examples/longcat/vllm/run_vllm.sh
```

### Step 4: 测试模型

```bash
bash examples/longcat/vllm/curl_test.sh
```

### Step 5: 迭代修复

如果输出乱码，检查以下关键模块后修改代码，然后**重启容器 → 重新部署 → 重新测试**：

| 模块 | 路径 | 作用 |
|------|------|------|
| EP 零号专家 | `easyinfer/plugins/vllm_ascend/ops/fused_moe/fix_ep_zero_expert.py` | vllm ≥ 0.23 下启用原生零号专家路径 |
| EP forward_impl | `easyinfer/plugins/vllm_ascend/ops/fused_moe/zero_expert_fused_moe.py` | vllm < 0.23 的 EP 路由覆盖 |
| 双注意力 | `easyinfer/plugins/vllm_ascend/fix_dual_attention.py` | LongCat 双 attention layer_index 提取 |
| MLA 旋转编码 | `easyinfer/plugins/vllm_ascend/fix_mla_rotary.py` | 非 DeepSeek 模型的 MLA cos/sin 缓存 |

### 常用调试命令

```bash
# 查看服务日志
tail -f /home/jianzhnie/llmtuner/llm/EasyInfer/longcat_flash-chat.log

# 检查 Ray 任务状态
docker exec vllm-ascend-env ray status

# 快速 API 测试
curl -s http://localhost:8010/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"longcat","messages":[{"role":"user","content":"你好"}],"max_tokens":50}' \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'])"
```