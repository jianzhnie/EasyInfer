# VLLM 模型部署工作流

## 环境概况

- **集群**: <N> 节点 × 8 昇腾 NPU (Atlas 800 A2/A3, 每卡 64G)
- **框架**: vLLM-Ascend + Ray 分布式
- **镜像**: `<镜像名>` (quay.io/ascend/vllm-ascend)
- **容器**: `<容器名>`
- **挂载**: `/home/jianzhnie/llmtuner` → 容器内同路径
- **节点文件**: `<node_list.txt>`

## 任务目标

为指定模型完成部署脚本、测试脚本和文档，并验证模型功能正常。

> 模型基路径: `/home/jianzhnie/llmtuner/hfhub/models`

## 输出要求

对每个模型，在 `examples/<模型简称>/vllm/` 下生成：

```
examples/<model_dir>/vllm/
├── run_vllm.sh       ← 直接 vllm serve（首选）
├── curl_test.sh      ← API 功能测试
└── README.md         ← 部署与测试文档
```

## 部署前确认清单

- [ ] `MODEL_PATH` 默认值正确，模型文件存在
- [ ] `PORT` 不与其他模型冲突（见端口分配表）
- [ ] 架构特性已确认：MoE / MLA / MTP / 多模态 / 量化
- [ ] 并行策略已确认：TP / PP / EP 取值及约束
- [ ] 关键环境变量已确认：FLASHCOMM1 / MLAPO / BUFFSIZE

## 部署与测试流程

参考 `prompts/example-scripts-template.md` 中的脚本模板和 `prompts/vllm_env_vars.md` 中的环境变量。

### Step 1: 清理并重启容器

```bash
for ip in $HEAD $WORKER; do
    ssh "$ip" "docker restart <容器名>"
done
sleep 15
```

### Step 2: 启动容器群

```bash
bash scripts/docker/manage_npuslim_containers.sh start --file <node_list.txt>
```

### Step 3: 启动 Ray 集群

```bash
bash scripts/ray_cluster/start_npuslim_ray_cluster.sh start --file <node_list.txt>

# 验证 NPU 可用
ssh $HEAD "docker exec <容器名> ray status | grep -E 'NPU|Active'"
```

### Step 4: 部署模型

进入容器内执行：

```bash
docker exec -it <容器名> /bin/bash
cd /home/jianzhnie/llmtuner/llm/EasyInfer
bash examples/<model_dir>/vllm/run_vllm.sh
```

多节点部署时，**必须设置 `RAY_ADDRESS`**：

```bash
# 获取 Ray 集群地址
docker exec <容器名> python3 -c "
import ray; ray.init(address='auto', ignore_reinit_error=True)
print(ray.get_runtime_context().gcs_address)
"

# 部署时导出
RAY_ADDRESS=<ip>:6379 PP=2 bash run_vllm.sh
```

### Step 5: 测试模型

```bash
bash examples/<model_dir>/vllm/curl_test.sh
```

## Claude Code 集成

部署完成后，通过以下环境变量将模型接入 Claude Code：

```bash
ANTHROPIC_BASE_URL=http://localhost:<PORT> \
ANTHROPIC_API_KEY=dummy \
ANTHROPIC_AUTH_TOKEN=dummy \
ANTHROPIC_DEFAULT_SONNET_MODEL=<api-name> \
ANTHROPIC_DEFAULT_HAIKU_MODEL=<api-name> \
ANTHROPIC_DEFAULT_OPUS_MODEL=<api-name> \
claude
```

> Agent 优化参数参见 `prompts/vllm_env_vars.md` 中的 "Agent 优化参数" 节。

## 参考模板

| 模板 | 文件 | 用途 |
|------|------|------|
| 脚本模板 | `prompts/example-scripts-template.md` | `run_vllm.sh` / `curl_test.sh` 格式 |
| README 模板 | `prompts/example-readme-template.md` | 模型 README 文档结构 |
| 环境变量 | `prompts/vllm_env_vars.md` | 完整环境变量参考 |
