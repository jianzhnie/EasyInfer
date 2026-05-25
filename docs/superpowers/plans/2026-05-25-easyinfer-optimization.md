# EasyInfer 脚本优化实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 逐模块优化 EasyInfer 脚本库 — 修复缺陷、消除重复、精简 SSH、统一输入、规范输出、拆分长函数

**Architecture:** 自底向上五阶段: 先增强 common.sh 基础库, 再逐层优化 Docker → Ray → vLLM → Examples。每个阶段完成后运行 shellcheck + bash -n + pre-commit 验证。

**Tech Stack:** Bash 4.2+, shellcheck, pre-commit

---

### Task 1: 增强 common.sh — 新增 `is_local_ip()` 和统一节点解析

**Files:**
- Modify: `scripts/common.sh`

- [ ] **Step 1: 在 common.sh 末尾添加 `is_local_ip()` 函数**

在 `SCRIPTS_ROOT` 之后追加:

```bash
# ------------------------------------------------------------------------------
# 判断 IP 是否为本机
# ------------------------------------------------------------------------------
is_local_ip() {
    local ip="$1"
    local lip
    for lip in $(hostname -I 2>/dev/null || true); do
        [[ "$ip" == "$lip" ]] && return 0
    done
    [[ "$ip" == "$(hostname -s 2>/dev/null)" || "$ip" == "$(hostname 2>/dev/null)" ]] && return 0
    return 1
}
```

- [ ] **Step 2: 运行 shellcheck 验证**

```bash
shellcheck scripts/common.sh
```

Expected: 0 errors, 0 warnings (existing SC2034 disables for error codes are fine)

- [ ] **Step 3: 运行 bash -n 语法检查**

```bash
bash -n scripts/common.sh
```

Expected: no output (syntax OK)

- [ ] **Step 4: Commit**

```bash
git add scripts/common.sh
git commit -m "feat(scripts): add is_local_ip() to common.sh"
```

---

### Task 2: 增强 common.sh — 增强 `confirm()` 支持全局跳过标志

**Files:**
- Modify: `scripts/common.sh`

- [ ] **Step 1: 修改 `confirm()` 函数**

将 `confirm()` 替换为:

```bash
# 用法: confirm "确认操作?" [default_yes|default_no]
# 返回 0 表示用户确认, 1 表示取消
# 若 SKIP_CONFIRM=true，自动跳过交互返回 0
confirm() {
    local msg="${1:?用法: confirm <message> [default]}"
    local default="${2:-default_no}"
    [[ "${SKIP_CONFIRM:-false}" == "true" ]] && return 0
    local prompt
    if [[ "$default" == "default_yes" ]]; then
        prompt="$msg [Y/n] "
    else
        prompt="$msg [y/N] "
    fi
    read -r -p "$prompt" answer 2>/dev/null || answer=""
    case "$answer" in
        y|Y|yes|YES) return 0 ;;
        n|N|no|NO)   return 1 ;;
        "")          [[ "$default" == "default_yes" ]] && return 0 || return 1 ;;
        *)           return 1 ;;
    esac
}
```

- [ ] **Step 2: 在 `parse_nodes_file_arg()` 后面添加 `resolve_nodes()` 函数**

```bash
# ------------------------------------------------------------------------------
# 统一节点解析: CLI --hosts > --file/-f > NODES_FILE 环境变量 > 默认文件
# 用法: resolve_nodes "$@" → 将结果存入全局数组 RESOLVED_NODES
# 返回: 0 成功, 节点数通过 ${#RESOLVED_NODES[@]} 获取
# ------------------------------------------------------------------------------
resolve_nodes() {
    RESOLVED_NODES=()
    local nodes_file=""

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --hosts) shift
                while [[ $# -gt 0 && "$1" != -* ]]; do
                    RESOLVED_NODES+=("$1"); shift
                done ;;
            --file|-f)
                [[ -n "${2:-}" && "$2" != -* ]] || log_fatal "选项 $1 需要一个参数: 节点列表文件路径"
                nodes_file="$2"; shift 2 ;;
            *) shift ;;
        esac
    done

    # 如果已通过 --hosts 指定，直接返回
    [[ ${#RESOLVED_NODES[@]} -gt 0 ]] && return 0

    # 从文件读取
    local file="${nodes_file:-${NODES_FILE:-scripts/node_list.txt}}"
    [[ -f "$file" ]] || log_fatal "节点列表文件未找到: $file"

    while IFS= read -r line; do
        [[ -n "$line" ]] && RESOLVED_NODES+=("$line")
    done < <(read_nodes "$file")

    [[ ${#RESOLVED_NODES[@]} -gt 0 ]] || log_fatal "节点列表为空: $file"
}
```

- [ ] **Step 3: 运行 shellcheck + bash -n 验证**

```bash
shellcheck scripts/common.sh && bash -n scripts/common.sh
```

- [ ] **Step 4: Commit**

```bash
git add scripts/common.sh
git commit -m "feat(scripts): enhance confirm() with SKIP_CONFIRM and add resolve_nodes()"
```

---

### Task 3: Docker 层 — 优化 `manage_npuslim_containers.sh`

**Files:**
- Modify: `scripts/docker/manage_npuslim_containers.sh`

- [ ] **Step 1: 重写脚本，删除自定义节点解析和 is_local**

关键改动:
1. 删除 `resolve_hosts()` 函数 (替换为 common.sh 的 `resolve_nodes`)
2. 删除 `is_local()` 函数 (替换为 common.sh 的 `is_local_ip`)
3. 修复 bash 4.2 `=~` 兼容性问题 (改为 glob 匹配 `[[ "$1" != -* ]]`)
4. 删除 `read -ra LOCAL_IPS` (不再需要)
5. 使用 `RESOLVED_NODES` 数组替代 `HOSTS`

改动 `cmd_start()`:
```bash
cmd_start() {
    resolve_nodes "$@"
    local with_npuslim=true

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --no-npuslim) with_npuslim=false; shift ;;
            --npuslim) with_npuslim=true; shift ;;
            --hosts) shift
                while [[ $# -gt 0 && "$1" != -* ]]; do shift; done ;;
            --file|-f) shift 2 ;;
            *) shift ;;
        esac
    done

    echo "========================================"
    echo "Starting Containers"
    echo "========================================"
    echo "Hosts:   ${RESOLVED_NODES[*]}"
    echo "NPUSlim: ${with_npuslim}"
    echo ""

    for host in "${RESOLVED_NODES[@]}"; do
        echo "--- Starting on ${host} ---"
        local npuslim_arg=""
        if $with_npuslim; then
            local npath
            npath=$(npuslim_path_for "$host")
            npuslim_arg="--npuslim=${npath}"
        fi
        remote_bash "$host" "bash ${RUN_CONTAINER} --multi-node --daemon ${npuslim_arg}"
        echo ""
    done

    echo "========================================"
    echo "All containers started."
    echo "========================================"
}
```

改动 `remote_docker_cmd()`:
```bash
remote_docker_cmd() {
    local host="$1"; shift
    if is_local_ip "$host"; then
        docker "$@"
    else
        # shellcheck disable=SC2029
        ssh "${SSH_USER}@${host}" docker "$@"
    fi
}
```

改动 `remote_bash()`:
```bash
remote_bash() {
    local host="$1"; shift
    if is_local_ip "$host"; then
        bash -c "$*"
    else
        # shellcheck disable=SC2029
        ssh "${SSH_USER}@${host}" bash -c "$*"
    fi
}
```

改动 `cmd_stop()`, `cmd_status()` 同理: 用 `RESOLVED_NODES` 替代 `HOSTS`, 用 `is_local_ip` 替代 `is_local`。

- [ ] **Step 2: 验证**

```bash
shellcheck scripts/docker/manage_npuslim_containers.sh
bash -n scripts/docker/manage_npuslim_containers.sh
```

- [ ] **Step 3: Commit**

```bash
git add scripts/docker/manage_npuslim_containers.sh
git commit -m "refactor(docker): deduplicate node resolution and fix bash 4.2 compat in manage_npuslim_containers.sh"
```

---

### Task 4: Docker 层 — 优化 `manage_docker_containers.sh`

**Files:**
- Modify: `scripts/docker/manage_docker_containers.sh`

- [ ] **Step 1: 简化 prepare_node 使用 parse_nodes_file_arg**

改动说明: 脚本已经使用 `parse_nodes_file_arg` 和 `read_nodes`, 主要改动是精简重复的参数解析和确保一致。

`prepare_node()` 中的 SSH 调用保持不变（兼容性约束）, 但在主流程中增加对 `--yes/-y` 全局跳过确认的支持。

- [ ] **Step 2: 验证**

```bash
shellcheck scripts/docker/manage_docker_containers.sh
bash -n scripts/docker/manage_docker_containers.sh
```

- [ ] **Step 3: Commit**

```bash
git add scripts/docker/manage_docker_containers.sh
git commit -m "refactor(docker): streamline manage_docker_containers.sh"
```

---

### Task 5: Ray 层 — 优化 `start_npuslim_ray_cluster.sh`

**Files:**
- Modify: `scripts/ray_cluster/start_npuslim_ray_cluster.sh`

- [ ] **Step 1: 修复 bash 4.2 兼容性和删除重复实现**

关键改动:
1. 删除 `read_cluster_nodes()` 函数, 改用 `resolve_nodes`
2. 删除 `is_local()` 函数, 改用 `is_local_ip`
3. 删除 `read -ra LOCAL_IPS`, 不再需要
4. 修复 `[[ ! "$1" =~ ^-- ]]` → `[[ "$1" != -* ]]` (bash 4.2 兼容)
5. 在 `node_exec()` 中用 `is_local_ip` 替代 `is_local`

`node_exec()` 改为:
```bash
node_exec() {
    local host="$1"
    shift
    local container
    if is_local_ip "$host"; then
        container=$(get_container)
        if [[ -z "$container" ]]; then
            log_err "本地未找到运行中的容器"
            return 1
        fi
        docker exec "$container" bash -lc "$*"
    else
        # shellcheck disable=SC2029
        container=$(ssh "${SSH_USER}@${host}" "docker ps -q --filter ancestor=${IMAGE_NAME} | head -1" 2>/dev/null)
        if [[ -z "$container" ]]; then
            log_err "${host} 上未找到运行中的容器"
            return 1
        fi
        # shellcheck disable=SC2029
        ssh "${SSH_USER}@${host}" "docker exec ${container} bash -lc $(printf '%q' "$*")" 2>/dev/null
    fi
}
```

各命令 (start/stop/status) 中使用 `RESOLVED_NODES`:
```bash
# 替换原有的 read_cluster_nodes 调用
resolve_nodes "$@"
# 之后使用 ${RESOLVED_NODES[@]} 替代 ${_CLUSTER_NODES[@]}
```

- [ ] **Step 2: 验证**

```bash
shellcheck scripts/ray_cluster/start_npuslim_ray_cluster.sh
bash -n scripts/ray_cluster/start_npuslim_ray_cluster.sh
```

- [ ] **Step 3: Commit**

```bash
git add scripts/ray_cluster/start_npuslim_ray_cluster.sh
git commit -m "refactor(ray): fix bash 4.2 compat and deduplicate in start_npuslim_ray_cluster.sh"
```

---

### Task 6: Ray 层 — 优化 `start_ray_cluster.sh`

**Files:**
- Modify: `scripts/ray_cluster/start_ray_cluster.sh`

- [ ] **Step 1: 合并重复的 stop_ray_on_node 清理循环**

将两次串行清理循环合并为一次，但保留两次执行以确保清理干净:

```bash
# Step 2: 清理已有 Ray 进程（执行两次确保干净）
log_info "[2/5] 清理已有 Ray 进程..."
for _ in 1 2; do
    for node in "${NODES[@]}"; do
        limit_jobs "$MAX_SSH_PARALLELISM"
        stop_ray_on_node "$node" &
    done
    wait
    sleep 2
done
```

(此逻辑已合理，保留不变。主要改动是确保变量引用一致。)

其他改动:
- 将 `mktemp -d` + 自定义 `cleanup()` 改为使用 common.sh 的 `mktemp_dir`

- [ ] **Step 2: 验证**

```bash
shellcheck scripts/ray_cluster/start_ray_cluster.sh
bash -n scripts/ray_cluster/start_ray_cluster.sh
```

- [ ] **Step 3: Commit**

```bash
git add scripts/ray_cluster/start_ray_cluster.sh
git commit -m "refactor(ray): use mktemp_dir from common.sh in start_ray_cluster.sh"
```

---

### Task 7: Ray 层 — 优化 `stop_ray_cluster.sh`

**Files:**
- Modify: `scripts/ray_cluster/stop_ray_cluster.sh`

- [ ] **Step 1: 使用公共 confirm() 和 parse_nodes_file_arg()**

关键改动:
1. 删除自定义确认逻辑, 改用 `confirm()` + `SKIP_CONFIRM` 支持
2. 使用 `parse_nodes_file_arg` 获取节点文件路径, 而非直接用 `NODE_LIST`
3. 使用 `${RESOLVED_NODES[@]}` 读取节点

替换确认部分:
```bash
# 确认 (替换原手工 read -r -p)
if ! confirm "将停止以下节点的 Ray 集群: $nodes"; then
    log_info "已取消"
    exit 0
fi
```

替换节点读取:
```bash
NODE_FILE=$(parse_nodes_file_arg "$@")
nodes=$(read_nodes "$NODE_FILE")
```

- [ ] **Step 2: 验证**

```bash
shellcheck scripts/ray_cluster/stop_ray_cluster.sh
bash -n scripts/ray_cluster/stop_ray_cluster.sh
```

- [ ] **Step 3: Commit**

```bash
git add scripts/ray_cluster/stop_ray_cluster.sh
git commit -m "refactor(ray): use common confirm() and parse_nodes_file_arg() in stop_ray_cluster.sh"
```

---

### Task 8: Ray 层 — 拆分 `_kill_lib.sh` 长函数

**Files:**
- Modify: `scripts/ray_cluster/_kill_lib.sh`

- [ ] **Step 1: 拆分 `_gen_kill_remote_script` (75行 → 多个子函数)**

将单一 heredoc 拆分, 在远程脚本中调用子函数:

改动: 将 PS 过滤逻辑提取到远程脚本内的独立函数。原函数保持不变但通过调用更小的内部单元组成:

```bash
_gen_kill_remote_script() {
    local pattern="$1" kill_timeout="$2" dry_run="$3"
    local script
    read -r -d '' script << 'REMOTE_SCRIPT'
        set -euo pipefail
        PATTERN="__PATTERN__"
        KILL_TIMEOUT="__KILL_TIMEOUT__"
        DRY_RUN="__DRY_RUN__"

        get_matching_pids() {
            ps aux | grep -E "$PATTERN" | grep -v grep | \
                grep -v -E '(vscode-server|code-server|sshd:|/bin/sh -c|extension|/agent/|ssh.*:)' | \
                awk '{print $2}' | sort -u | tr '\n' ' ' || true
        }

        get_process_info() {
            local pids="$1"
            ps -p $pids -o pid,ppid,user,%cpu,%mem,etime,args 2>/dev/null || true
        }

        try_terminate() {
            local pids="$1" timeout="$2"
            kill -15 $pids 2>/dev/null || true
            sleep "$timeout"
            local remaining=""
            for pid in $pids; do
                kill -0 "$pid" 2>/dev/null && remaining="$remaining $pid"
            done
            printf '%s' "${remaining# }"
        }

        try_kill() {
            local pids="$1"
            kill -9 $pids 2>/dev/null || true
            sleep 1
            local still_alive=""
            for pid in $pids; do
                kill -0 "$pid" 2>/dev/null && still_alive="$still_alive $pid"
            done
            printf '%s' "${still_alive# }"
        }

        all_pids=$(get_matching_pids)
        if [ -z "$all_pids" ] || [ "$all_pids" = " " ]; then
            echo "STATUS:NO_PROCESSES"
            exit 0
        fi

        echo "STATUS:FOUND"
        echo "PIDS:$all_pids"
        echo "PROCESS_INFO:"
        get_process_info "$all_pids"

        if [ "$DRY_RUN" = "true" ]; then
            echo "ACTION:SKIP_DRY_RUN"
            exit 0
        fi

        echo "ACTION:SIGTERM"
        remaining=$(try_terminate "$all_pids" "$KILL_TIMEOUT")

        if [ -z "$remaining" ]; then
            echo "STATUS:TERMINATED"
            exit 0
        fi

        echo "ACTION:SIGKILL:$remaining"
        still_alive=$(try_kill "$remaining")

        if [ -n "$still_alive" ]; then
            echo "STATUS:FAILED:$still_alive"
            exit 1
        fi
        echo "STATUS:KILLED"
REMOTE_SCRIPT
    script="${script//__PATTERN__/$pattern}"
    script="${script//__KILL_TIMEOUT__/$kill_timeout}"
    script="${script//__DRY_RUN__/$dry_run}"
    printf '%s' "$script"
}
```

- [ ] **Step 2: 验证**

```bash
shellcheck scripts/ray_cluster/_kill_lib.sh
bash -n scripts/ray_cluster/_kill_lib.sh
```

- [ ] **Step 3: Commit**

```bash
git add scripts/ray_cluster/_kill_lib.sh
git commit -m "refactor(ray): split _gen_kill_remote_script into smaller functions"
```

---

### Task 9: vLLM 层 — 提取 mp 公共模式

**Files:**
- Modify: `scripts/vllm/mp/_common.sh`
- Modify: `scripts/vllm/mp/deploy_vllm_multinode.sh`
- Modify: `scripts/vllm/mp/deploy_vllm_multinode_mp.sh`

- [ ] **Step 1: 将 `launch_on_node()` 公共部分提取到 `_common.sh`**

两个脚本的 `launch_on_node()` 高度相似。在 `_common.sh` 中添加:

```bash
# 在两个多节点部署脚本间共享的 launch_on_node 公共逻辑
# 用法: _launch_vllm_on_node <node> <local_ip> <is_headless> <args_declare_cmd> <env_exports_cmd>
#   args_declare_cmd: 生成 declare -p args 的命令 (已字符串化)
#   env_exports_cmd: 生成环境变量导出的命令
_launch_vllm_on_node() {
    local node="$1" local_ip="$2" is_headless="$3"
    local array_decl_cmd="$4" env_exports_cmd="$5" log_label="${6:-}"

    local array_decl env_exports
    array_decl=$(eval "$array_decl_cmd")
    env_exports=$(eval "$env_exports_cmd")

    local inner_cmd ssh_cmd
    inner_cmd="export SCRIPT_DIR='${SCRIPT_DIR}' && cd '${SCRIPT_DIR}' && source ../set_env.sh"$'\n'"${env_exports}"$'\n'"${array_decl}"$'\n'"nohup vllm \"\${args[@]}\" > ${SCRIPT_DIR}/vllm_${node}${log_label}.log 2>&1 &"$'\n'"echo PID:\$!"

    ssh_cmd="export SCRIPT_DIR='${SCRIPT_DIR}' && cd '${SCRIPT_DIR}' && source ../set_env.sh && docker exec -i \"\${CONTAINER_NAME:-vllm-ascend-env-a3}\" bash -s"

    log_info "Launching on ${node} (IP: ${local_ip})..."
    if [[ "${DRY_RUN}" == "true" || "${DRY_RUN}" == "1" ]]; then
        echo "---------- Node: ${node} (host command) ----------"
        echo "${ssh_cmd}"
        echo "---------- Node: ${node} (container inner command) ----------"
        echo "${inner_cmd}"
        echo "-----------------------------------"
    else
        local pid
        # shellcheck disable=SC2086,SC2029
        pid=$(echo "${inner_cmd}" | ssh ${SSH_OPTS} "${node}" "${ssh_cmd}")
        log_info "Started vLLM on ${node}, PID=${pid}, log=${SCRIPT_DIR}/vllm_${node}${log_label}.log"
    fi
}
```

- [ ] **Step 2: 更新 `deploy_vllm_multinode.sh` 使用 `_launch_vllm_on_node`**

简化 `launch_on_node()`:
```bash
launch_on_node() {
    local node="$1" local_ip="$2" is_headless="$3" idx="$4"
    local dp_start_rank=$((idx * NPUS_PER_NODE / CARDS_PER_INSTANCE))
    _launch_vllm_on_node "$node" "$local_ip" "$is_headless" \
        "build_vllm_args_declare '${is_headless}' '${idx}' '${dp_start_rank}' '${NODE0_IP}'" \
        "build_env_exports '${local_ip}'" \
        "_${node}"
}
```

- [ ] **Step 3: 更新 `deploy_vllm_multinode_mp.sh` 同理**

简化 `launch_on_node()`:
```bash
launch_on_node() {
    local node="$1" local_ip="$2" is_headless="$3" node_rank="$4"
    local dp_start_rank="$5" dp_size_local="$6" master_addr="$7" nnodes="$8"
    local vllm_port="$9" use_internal_dp="${10}"
    _launch_vllm_on_node "$node" "$local_ip" "$is_headless" \
        "build_vllm_args_declare '${is_headless}' '${node_rank}' '${dp_start_rank}' '${dp_size_local}' '${master_addr}' '${nnodes}' '${vllm_port}' '${use_internal_dp}'" \
        "build_env_exports '${local_ip}'" \
        "_${node}_${vllm_port}"
}
```

- [ ] **Step 4: 验证全部三个文件**

```bash
shellcheck scripts/vllm/mp/_common.sh scripts/vllm/mp/deploy_vllm_multinode.sh scripts/vllm/mp/deploy_vllm_multinode_mp.sh
bash -n scripts/vllm/mp/_common.sh
bash -n scripts/vllm/mp/deploy_vllm_multinode.sh
bash -n scripts/vllm/mp/deploy_vllm_multinode_mp.sh
```

- [ ] **Step 5: Commit**

```bash
git add scripts/vllm/mp/_common.sh scripts/vllm/mp/deploy_vllm_multinode.sh scripts/vllm/mp/deploy_vllm_multinode_mp.sh
git commit -m "refactor(vllm): extract shared launch_on_node logic into mp/_common.sh"
```

---

### Task 10: 全局验证

**Files:** 全部

- [ ] **Step 1: 运行 shellcheck 全量检查**

```bash
shellcheck scripts/**/*.sh tools/*.sh examples/*.sh
```

Expected: 0 new errors/warnings (仅保留已有的 SC1091 info 级别提示)

- [ ] **Step 2: 运行 bash -n 全量语法检查**

```bash
for f in scripts/**/*.sh tools/*.sh examples/*.sh; do bash -n "$f" && echo "OK: $f" || echo "FAIL: $f"; done
```

Expected: all OK

- [ ] **Step 3: 运行 pre-commit 全量检查**

```bash
pre-commit run --all-files
```

Expected: all pass

- [ ] **Step 4: 最终 commit**

```bash
git add -A
git commit -m "chore: final verification after optimization pass"
```
