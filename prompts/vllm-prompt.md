# VLLM 模型部署和测试

## 环境概况

- **集群**: 8 节点 × 8 昇腾 NPU (Atlas 800 A2/A3, 每卡 64G)
- **框架**: vLLM-Ascend 0.20.2 + Ray 分布式
- **容器**: `vllm-ascend-env` (quay.io/ascend/vllm-ascend:v0.20.2rc1-a3)
- **CANN**: 9.0.0
- **挂载**: `/home/jianzhnie/llmtuner` → 容器内同路径

## 任务目标

为以下模型逐一完成部署脚本、测试脚本和文档, 并验证模型功能正常。

> 模型基路径: `/home/jianzhnie/llmtuner/hfhub/models`

- /home/jianzhnie/llmtuner/hfhub/models/ZhipuAI/GLM-5
- /home/jianzhnie/llmtuner/hfhub/models/moonshotai/Kimi-K2-Thinking
- /home/jianzhnie/llmtuner/hfhub/models/moonshotai/Kimi-K2.5
- /home/jianzhnie/llmtuner/hfhub/models/MiniMaxAI/MiniMax-M2.7
- /home/jianzhnie/llmtuner/hfhub/models/Qwen/Qwen3-235B-A22B-Instruct-2507

## 输出要求

对每个模型，在 `examples/<模型简称>/vllm/ `下生成 **5 个文件**, (glm5, kimi-k2-thinking, kimi-k2.5, minimax-m2.7, qwen3-235b-a22b-instruct-2507):

```
examples/<model_dir>/
├── run_vllm.sh       ← 直接 vllm serve（首选）
├── vllm_server.sh    ← 传统包装器部署（备份）
├── curl_test.sh      ← API 功能测试
└── README.md         ← 部署与测试文档
```

## 部署与测试

参考 /home/jianzhnie/llmtuner/llm/EasyInfer/prompts/example-scripts-template.md 中的脚本模板，完成以下步骤：

### Step 1: 清理并重启容器

```bash
# 确保所有节点容器状态干净
for ip in $HEAD $WORKER; do
    ssh "$ip" "docker restart vllm-ascend-env"
done
sleep 15
```

### Step2: 启动 vLLM-Ascend 容器群

```bash
bash scripts/docker/manage_npuslim_containers.sh start \
    --file node_list.txt
```

### Step 3: 启动 Ray 集群

```bash
bash scripts/ray_cluster/start_npuslim_ray_cluster.sh start \
    --file node_list.txt

# 验证: 1 node, 8 NPU
ssh $HEAD "docker exec vllm-ascend-env ray status | grep -E 'NPU|Active'"
```

### Step 4: 部署模型

```bash
bash examples/<model_dir>/vllm/run_vllm.sh
```

#### Step 5: 测试模型

```bash
bash examples/<model_dir>/vllm/curl_test.sh
```
