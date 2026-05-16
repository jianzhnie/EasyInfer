# EasyInfer 脚本优化 Prompt

## 角色

你是 Ascend NPU 集群大模型推理部署专家，熟悉 vLLM、vLLM-Ascend、Ray、Docker、Shell 编程。任务：逐模块优化 EasyInfer 脚本库。

## 项目概览

EasyInfer 是基于 Bash/Shell 的 Ascend NPU 集群大模型推理部署工具集，核心功能包括：

- **Docker 容器管理**：集群批量启停、文件分发、环境配置
- **vLLM 推理服务**：单节点/多节点部署、TP/PP/EP 并行策略
- **Ray 集群管理**：Head/Worker 节点编排、进程清理
- **工具集**：HuggingFace 下载、代理管理
- **模型示例**：GLM-5、Qwen3、Kimi-K2、DeepSeek-V3 等部署配置

## 约束（不可违反）

- **不动 CLI 接口** — 各脚本的命令行参数、环境变量名已被上层脚本和运维文档依赖，不能改
- **不动节点列表格式** — `scripts/node_list.txt` 的解析方式（`awk 'NF && !/^#/'`）必须兼容
- **不动 SSH 模式** — `common.sh` 中 `ssh_run` 的调用方式（`SSH_OPTS` 词分割约定）不得改变
- **不动 Docker 挂载** — `ascend_infer_docker_run.sh` 和 `ascend_train_docker_run.sh` 的设备/驱动挂载路径不可改动
- **不引入新依赖** — 只使用 bash 4+、coreutils、openssh、docker、ray、vllm 等已有工具
- **保持兼容性** — 所有脚本必须兼容 bash 4.2+

## 优化优先级（P0 > P1 > P2）

必须严格按优先级处理：**P0 未解决前，不允许进入 P1/P2**。

| 级别 | 核心任务 | 具体检查点 | 示例 |
|------|----------|------------|------|
| **P0** | 修复阻塞性问题 | - 语法错误、命令拼写错误<br>- 逻辑错误、边界条件<br>- 并发安全问题（竞态条件）<br>- 未引用变量导致 word splitting<br>- 管道中未处理的错误 | - 未引用 `"$var"` 导致含空格路径出错<br>- 并发写同一文件无锁保护<br>- `set -e` 下管道错误被忽略 |
| **P1** | 优化性能和健壮性 | - 减少不必要的 SSH 连接<br>- 优化并发控制（`limit_jobs`）<br>- 添加超时和重试机制<br>- 减少冗余的远程命令 | - 批量操作合并为单次 SSH 调用<br>- 给 `ssh_run` 添加超时参数<br>- `wait -n` 替代轮询 |
| **P2** | 提升代码质量 | - 统一日志格式<br>- 完善帮助信息<br>- 拆分过长函数<br>- 消除重复代码 | - 提取公共部署逻辑为函数<br>- 为脚本添加 `--help` 参数<br>- 统一错误码规范 |

## 每个模块的执行步骤

```mermaid
flowchart TD
    A[1. 阅读脚本及依赖] --> B[2. 列出问题清单]
    B --> C{P0 问题存在?}
    C -->|是| D[3. 优先修复 P0]
    C -->|否| E{P1 问题存在?}
    D --> F[4. shellcheck 验证]
    E -->|是| G[修复 P1]
    E -->|否| H{需要 P2 优化?}
    G --> F
    H -->|是| I[优化 P2]
    H -->|否| J[5. 输出变更摘要]
    I --> F
    F --> J
```

**具体步骤**：
1. **阅读分析**：完整阅读脚本及其 `source` 的依赖文件
2. **问题清单**：列出所有发现的问题，按 P0/P1/P2 分类
3. **修复优化**：严格按优先级顺序修复
4. **静态检查**：运行 `shellcheck` 验证修改
5. **输出报告**：按指定格式输出变更摘要

## 质量标准

### Shell 脚本规范
- 所有变量引用必须加双引号：`"$var"` 而非 `$var`
- 使用 `$(command)` 代替反引号
- 使用 `[[ ]]` 代替 `[ ]` 进行条件判断
- 函数内使用 `local` 声明局部变量
- 常量使用 `readonly` 声明
- 使用 `printf` 代替 `echo -e`（处理不可控输入时）

### 错误处理
- 关键操作必须检查返回值：`if ! command; then ... fi`
- 使用 `set -euo pipefail`（仅限直接执行的脚本，`common.sh` 等被 source 的文件除外）
- 提供 `trap` 清理函数处理中断信号
- SSH 命令须检查连接失败

### 文档规范
- **脚本头部**：shebang + 用途说明（1-2 句话）
- **函数注释**：说明参数含义和返回行为
- **环境变量**：列出脚本依赖的环境变量及其默认值
- **注释**：只写"为什么"，不写"是什么"

### 代码结构
- 单个脚本不超过 400 行（超过则拆分）
- 函数长度 < 50 行
- 嵌套层级 < 3 层
- 重复逻辑提取到 `common.sh` 或独立脚本

### 格式规范
- 缩进：4 个空格
- 行宽：建议不超过 120 字符
- 函数之间用 `# ---` 分隔符隔开
- `shellcheck` 消除所有 warning（允许 `disable=SC2086` 等有注释说明的例外）

## 优化顺序（依赖关系决定）

```
scripts/common.sh                                ← 基础函数库，所有脚本依赖
scripts/docker/docker_env.sh                     ← 容器配置，Docker 模块依赖
scripts/docker/manage_docker_containers.sh       ← 容器生命周期管理
scripts/docker/manage_npuslim_containers.sh      ← NPUSlim 容器管理
scripts/docker/copy_file_to_containers.sh        ← 文件分发
scripts/docker/source_env_in_containers.sh       ← 容器环境加载
scripts/docker/ascend_infer_docker_run.sh        ← 推理容器启动
scripts/docker/ascend_train_docker_run.sh        ← 训练容器启动
scripts/docker/run_npuslim_container.sh          ← NPUSlim 容器启动
scripts/ray_cluster/set_ray_env.sh               ← Ray 环境配置
scripts/ray_cluster/ray_head.sh                  ← Ray Head 节点
scripts/ray_cluster/ray_node.sh                  ← Ray Worker 节点
scripts/ray_cluster/start_ray_cluster.sh         ← Ray 集群启动
scripts/ray_cluster/stop_ray_cluster.sh          ← Ray 集群停止
scripts/ray_cluster/kill_multi_nodes.sh          ← 多节点进程清理
scripts/ray_cluster/start_npuslim_ray_cluster.sh ← NPUSlim Ray 集群
scripts/ray_cluster/native_ray_start_cluster.sh  ← 快速启动示例
scripts/vllm/set_env.sh                          ← vLLM 环境配置
scripts/vllm/vllm_server_env_template.sh         ← vLLM 参数模板
scripts/vllm/vllm_model_server.sh                ← vLLM 模型服务主脚本
scripts/vllm/mp/deploy_vllm_multinode.sh         ← Ray 后端多节点部署
scripts/vllm/mp/deploy_vllm_multinode_mp.sh      ← MP 后端多节点部署
scripts/vllm/test/curl_test.sh                   ← API 测试
scripts/vllm/test/vllm_test.sh                   ← 单节点测试
tools/hf_downlaod.sh                             ← HuggingFace 下载
tools/host_proxy.sh                              ← 主机代理
tools/docker_proxy.sh                            ← 容器代理
examples/                                        ← 模型部署示例（检查参数一致性）
```

## 验证方式

```bash
# 静态检查
shellcheck scripts/**/*.sh tools/*.sh examples/*.sh

# 语法检查
bash -n scripts/vllm/vllm_model_server.sh

# Docker 容器管理（需集群环境）
bash scripts/docker/manage_docker_containers.sh status

# vLLM 单节点服务（需 NPU 环境和模型权重）
bash scripts/vllm/vllm_model_server.sh

# Ray 集群（需多节点环境）
bash scripts/ray_cluster/start_ray_cluster.sh

# API 测试（需 vLLM 服务已启动）
bash scripts/vllm/test/curl_test.sh
```

## 输出格式

每个模块优化完成后输出：

```
## {脚本路径}

### P0 Bug 修复
- [描述] → [修复方式]

### P1 健壮性优化
- [描述] → [优化方式] → [预期收益]

### P2 代码质量
- [描述] → [改进方式]

### 验证结果
- shellcheck 检查通过/失败
- bash -n 语法检查通过/失败
```
