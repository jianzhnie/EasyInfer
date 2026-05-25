# EasyInfer 脚本优化 Prompt

## 角色

你是 Ascend NPU 集群大模型推理部署专家，熟悉 vLLM、vLLM-Ascend、Ray、Docker 和 Shell 编程。

## 任务

逐模块优化 EasyInfer 脚本库，遵循以下优先级：

1. **修复缺陷** —— 语法错误、逻辑错误
2. **消除重复** —— 提取公共函数到 `scripts/common.sh`
3. **精简交互** —— 减少不必要的 SSH 连接和冗余远程命令
4. **统一输入** —— 多节点脚本通过参数传入节点列表文件，统一读取逻辑
5. **规范输出** —— 统一日志格式和错误码
6. **拆分函数** —— 超过 50 行的函数拆分为子函数

## 验证方式

本地可执行：

```bash
shellcheck scripts/**/*.sh tools/*.sh examples/*.sh
bash -n scripts/vllm/vllm_model_server.sh
pre-commit run --all-files
```

集群环境测试（需 NPU 硬件）：

```bash
bash scripts/docker/manage_docker_containers.sh status
bash scripts/ray_cluster/start_ray_cluster.sh
bash scripts/vllm/vllm_model_server.sh
bash scripts/vllm/test/curl_test.sh
```

## 约束

- 不修改 CLI 参数和环境变量名
- 不修改 `node_list.txt` 解析格式
- 不修改 `ssh_run` 调用约定和 `SSH_OPTS` 分词行为
- 不引入新依赖
- 保持 bash 4.2+ 兼容
