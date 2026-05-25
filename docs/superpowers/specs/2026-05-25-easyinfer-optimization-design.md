# EasyInfer 脚本优化设计规约

日期: 2026-05-25

## 目标

逐模块优化 EasyInfer 脚本库:
1. 修复缺陷 (语法/逻辑错误)
2. 消除重复 (提取公共函数到 common.sh)
3. 精简交互 (减少不必要的 SSH 连接)
4. 统一输入 (统一使用 common.sh 节点读取)
5. 规范输出 (统一日志格式和错误码)
6. 拆分函数 (超 50 行拆子函数)

## 方案: 自底向上五阶段

### 阶段 1: 增强 common.sh
- 新增 is_local_ip(), resolve_nodes()
- 增强 confirm() 支持 --yes/-y
- 确保 bash 4.2 兼容

### 阶段 2: Docker 层
- manage_npuslim_containers.sh: 删除自定义 resolve_hosts/is_local, 修复 =~
- manage_docker_containers.sh: 统一使用 resolve_nodes
- copy_file_to_containers.sh: 使用公共节点解析

### 阶段 3: Ray 层
- start_npuslim_ray_cluster.sh: 修复 =~, 删除重复
- start_ray_cluster.sh: 合并重复清理循环
- stop_ray_cluster.sh: 统一 confirm() 和 parse_nodes_file_arg()
- kill_multi_nodes.sh: 拆分 _gen_kill_remote_script

### 阶段 4: vLLM 层
- mp/_common.sh: 提取公共模式
- mp/deploy_vllm_multinode.sh / mp/deploy_vllm_multinode_mp.sh: 精简

### 阶段 5: Examples & Tools
- 统一使用公共函数

## 验证
- shellcheck scripts/**/*.sh tools/*.sh examples/*.sh
- bash -n <changed_script>
- pre-commit run --all-files
