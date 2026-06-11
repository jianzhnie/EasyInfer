# VLLM 模型部署和测试

## 环境概况

- **集群**: 8 节点 × 8 昇腾 NPU (Atlas 800 A2/A3, 每卡 64G)
- **框架**: vLLM-Ascend 0.20.2 + Ray 分布式
- **容器**: `vllm-ascend-env` (quay.io/ascend/vllm-ascend:v0.20.2rc1-a3)
- **CANN**: 9.0.0
- **挂载**: `/home/jianzhnie/llmtuner` → 容器内同路径

## 任务目标

为以下模型逐一完成部署脚本、测试脚本和文档。**严格按模型逐一处理**，完成一个再开始下一个。

> 模型基路径: `/home/jianzhnie/llmtuner/hfhub/models/Eco-Tech/`

| # | 模型路径                             | 量化   | 架构           | 专家数 | MTP | 多模态 |
| - | -------------------------------- | ---- | ------------ | --- | --- | --- |
| 1 | `.../DeepSeek-V4-Pro-w4a8-mtp`   | W4A8 | DeepseekV4   | 384 | ✓   | ✗   |
| 2 | `.../DeepSeek-V4-Flash-w8a8-mtp` | W8A8 | DeepseekV4   | 256 | ✓   | ✗   |
| 3 | `.../GLM-5-w4a8`                 | W4A8 | GlmMoeDSA    | 256 | ✓   | ✗   |
| 4 | `.../GLM-5.1-w4a8`               | W4A8 | GlmMoeDSA    | 256 | ✓   | ✗   |
| 5 | `.../Kimi-K2.6-w4a8`             | W4A8 | KimiK25      | 384 | ✗   | ✓   |
| 6 | `.../MiniMax-M2.7-w8a8-QuaRot`   | W8A8 | MiniMax-M2.7 | 256 | ✓   | ✗   |

## 输出要求

对每个模型，在 `examples/<模型简称>/vllm/ `下生成 **4 个文件**, (glm5\_1\_w4a8 和 glm5\_w4a8 使用相同配置):

```
examples/<model_dir>/
├── run_vllm.sh       ← 直接 vllm serve（首选）
├── vllm_server.sh    ← 传统包装器部署（备份）
├── curl_test.sh      ← API 功能测试
└── README.md         ← 部署与测试文档
```

## 部署与测试

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
bash /home/jianzhnie/llmtuner/llm/EasyInfer/scripts/docker/manage_npuslim_containers.sh start \
    --file /home/jianzhnie/llmtuner/llm/EasyInfer/node_list.txt
```

### Step 3: 启动 Ray 集群

```bash
bash /home/jianzhnie/llmtuner/llm/EasyInfer/scripts/ray_cluster/start_npuslim_ray_cluster.sh start \
    --file /home/jianzhnie/llmtuner/llm/EasyInfer/node_list.txt

# 验证: 1 node, 8 NPU
ssh $HEAD "docker exec vllm-ascend-env ray status | grep -E 'NPU|Active'"
```

### Step 4: 直接部署脚本 `run_vllm.sh` 模板

**必须遵循的模板:**

```bash
#!/bin/bash
# <ModelName> — 直接 vllm serve 部署
# 默认 TP=8 PP=1 (单节点)
set -eo pipefail

# CANN 环境加载 (必须在最前面)
set +u
if [[ -f "/usr/local/Ascend/cann/set_env.sh" ]]; then
    source /usr/local/Ascend/cann/set_env.sh
fi
if [[ -f "/usr/local/Ascend/nnal/atb/set_env.sh" ]]; then
    source /usr/local/Ascend/nnal/atb/set_env.sh
fi
set -u

# 基础路径配置
BASE_MODEL_PATH="/home/jianzhnie/llmtuner/hfhub/models/Eco-Tech"
MODEL_PATH="${MODEL_PATH:-$BASE_MODEL_PATH/<model_relative_path>}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-<端口>}"
TP="${TP:-8}"
PP="${PP:-1}"

# 环境变量优化
export HCCL_OP_EXPANSION_MODE=AIV
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export HCCL_BUFFSIZE=200
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_ASCEND_BALANCE_SCHEDULING=1
export VLLM_USE_MODELSCOPE=False

echo "[INFO] Starting <ModelName> at $MODEL_PATH"
echo "[INFO] TP=$TP PP=$PP PORT=$PORT"

vllm serve "$MODEL_PATH" \
    --host "$HOST" \
    --port "$PORT" \
    --served-model-name "<served_name>" \
    --trust-remote-code \
    --dtype bfloat16 \
    --tensor-parallel-size "$TP" \
    --pipeline-parallel-size "$PP" \
    --distributed-executor-backend ray \
    --quantization ascend \
    --gpu-memory-utilization 0.92 \
    --max-model-len 32768 \
    --max-num-seqs 64 \
    --max-num-batched-tokens 8192 \
    --enable-chunked-prefill \
    --enable-prefix-caching \
    --enforce-eager \
    --seed 1024 \
    <ARGS_PLACEHOLDER> \
    "$@"
```

**`<ARGS_PLACEHOLDER>`** **说明:**

- **MoE 模型**: 必须添加 `--enable-expert-parallel`
- **MTP 模型**: 必须添加 `--speculative-config "{\"num_speculative_tokens\": 3, \"method\": \"mtp\"}"`
- **多模态模型**: 如果是 Kimi-K2.6，确保相关多模态参数开启。

### Step 5: 测试脚本 `curl_test.sh` 模板

```bash
#!/bin/bash
PORT=<端口>
MODEL_NAME="<served_name>"

echo "Testing Chat Completions API..."
curl http://localhost:$PORT/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "'$MODEL_NAME'",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "你好，请简单介绍一下你自己。"}
    ],
    "max_tokens": 128
  }'
```

