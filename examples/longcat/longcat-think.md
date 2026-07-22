# LongCat-Flash-Thinking-2601 部署指南

## 环境信息

| 项目 | 值 |
|------|-----|
| 模型 | Meituan LongCat-Flash-Thinking-2601 |
| 架构 | 28 层 / 512 routed experts + 256 zero experts / moe_topk=12 / MLA |
| 节点 | 10.42.11.138 ~ 10.42.11.145（8 节点） |
| 头节点 | 10.42.11.138 |
| 容器 | vllm-ascend-env |
| TP | 64 |
| 端口 | 8200 |
| EasyInfer | `/home/jianzhnie/llmtuner/llm/EasyInfer`（已挂载到容器内，插件自动生效） |

## Step 1：启动 Docker 容器

在**本机**执行，检查容器状态：

```bash
bash /home/jianzhnie/llmtuner/llm/EasyInfer/scripts/docker/manage_docker_containers.sh status \
    --file /home/jianzhnie/llmtuner/llm/EasyInfer/node_list2.txt
```

如果 8 个节点都是 `running`，跳过这一步。否则启动容器：

```bash
bash /home/jianzhnie/llmtuner/llm/EasyInfer/scripts/docker/manage_npuslim_containers.sh start \
    --file /home/jianzhnie/llmtuner/llm/EasyInfer/node_list2.txt
```

## Step 2：启动 Ray 集群

在**本机**执行：

```bash
bash /home/jianzhnie/llmtuner/llm/EasyInfer/scripts/ray_cluster/manage_npuslim_ray_cluster.sh start \
    --file /home/jianzhnie/llmtuner/llm/EasyInfer/node_list2.txt
```

检查 Ray 状态（在容器内执行）：

```bash
ssh 10.42.11.138
docker exec vllm-ascend-env ray status
```

预期看到 8 个 Active 节点。

## Step 3：进入容器，部署模型

```bash
# SSH 到头节点
ssh 10.42.11.138

# 进入容器
docker exec -it vllm-ascend-env /bin/bash
```

进入容器后，**先设置 Ascend 环境变量**（关键！否则 `libascend_hal.so` 找不到）：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

然后启动模型：

```bash
cd /home/jianzhnie/llmtuner/llm/EasyInfer

MODEL_PATH=/home/jianzhnie/llmtuner/hfhub/models/meituan-longcat/LongCat-Flash-Thinking-2601 \
TENSOR_PARALLEL_SIZE=64 \
PORT=8200 \
MAX_MODEL_LEN=4096 \
MAX_NUM_SEQS=128 \
SERVED_MODEL_NAME=longcat-flash \
nohup bash examples/longcat/longcat_flash-chat.sh \
    > /home/jianzhnie/llmtuner/llm/EasyInfer/logs/longcat-think-$(date +%Y%m%d-%H%M%S).log 2>&1 &

# 记录 PID，退出容器
exit
exit
```

> **说明**：EasyInfer 插件（`easyinfer/plugins/`）通过 `/home/jianzhnie/llmtuner` 挂载在容器内已自动生效。
> - `architectures.py` 将 `LongcatCausalLM` 指向 vLLM 内置 `longcat_flash` 实现
> - `config.py` 将 `model_type="longcat"` 映射到 vLLM 的 `LongcatFlashConfig`
> - **无需修改任何模型文件**

## Step 4：监控日志

在**本机**执行，查看启动进度：

```bash
tail -f /home/jianzhnie/llmtuner/llm/EasyInfer/logs/longcat-think-*.log
```

模型加载通常需要 5-10 分钟。看到以下信息表示加载完成：

```
INFO: Application startup complete.
```

## Step 5：验证

在**本机**执行：

```bash
# 检查模型列表
curl -s http://10.42.11.138:8200/v1/models | python3 -m json.tool

# 发送测试请求
curl -s http://10.42.11.138:8200/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "longcat-flash",
        "messages": [{"role": "user", "content": "你好，请用一句话介绍自己"}],
        "max_tokens": 64
    }' | python3 -m json.tool
```

## 常见问题

### `libascend_hal.so: cannot open shared object file`

容器内未设置 Ascend 环境变量。执行：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

### `Model architectures ['LongcatCausalLM'] are not supported`

EasyInfer 插件未生效。确认 `/home/jianzhnie/llmtuner/llm/EasyInfer` 已挂载到容器内，且 `easyinfer` 包已安装（`pip install -e /home/jianzhnie/llmtuner/llm/EasyInfer`）。

### Ray 集群节点异常

在容器内检查：

```bash
ray status
# 如果某些节点未连接，重启 Ray：
bash /home/jianzhnie/llmtuner/llm/EasyInfer/scripts/ray_cluster/manage_npuslim_ray_cluster.sh restart \
    --file /home/jianzhnie/llmtuner/llm/EasyInfer/node_list2.txt
```
