# EasyInfer Script Library Optimization — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Optimize the EasyInfer shell script library across four independent rounds: fix defects/split functions, deduplicate into common.sh, unify input/output, and update docs.

**Architecture:** Layered refactoring where each round builds on the previous but is independently committable and reversible. All changes preserve existing CLI interfaces and env var names.

**Tech Stack:** Bash 4.2+, shellcheck, pre-commit hooks

---

## File Map

### Files to create
- `scripts/ray_cluster/_kill_lib.sh` — kill_multi_nodes helper functions (Round 1)
- `docs/scripts-overview.md` — script index (Round 4)
- `scripts/docker/README.md` — docker module docs (Round 4)
- `scripts/ray_cluster/README.md` — ray cluster module docs (Round 4)
- `scripts/vllm/README.md` — vLLM module docs (Round 4)
- `examples/README.md` — examples docs (Round 4)

### Files to modify
- `scripts/common.sh` — add `parse_nodes_file_arg`, enhance `ssh_run_timeout`, add `wait_for_server`, `print_server_ready`, `require_env` (Round 2)
- `scripts/docker/manage_docker_containers.sh` — split `_remote_prepare_node` (Round 1)
- `scripts/vllm/mp/deploy_vllm_multinode_mp.sh` — split `build_vllm_args_declare` (Round 1); use `_common.sh` helpers (Round 2); add `--file/-f` (Round 3)
- `scripts/ray_cluster/kill_multi_nodes.sh` — extract to `_kill_lib.sh` (Round 1); use enhanced `ssh_run_timeout` (Round 2); add `--file/-f` alias (Round 3)
- `scripts/vllm/vllm_model_server.sh` — remove `has_flag` (Round 1); source `common.sh` (Round 2); standardize exit codes and logs (Round 3)
- `scripts/vllm/mp/_common.sh` — add `load_and_validate_nodes`, `validate_parallelism_config`, `resolve_node0_ip` (Round 2)
- `scripts/vllm/mp/deploy_vllm_multinode.sh` — use `_common.sh` helpers (Round 2); add `--file/-f` (Round 3)
- `scripts/ray_cluster/start_ray_cluster.sh` — use `parse_nodes_file_arg` (Round 3); standardize exit codes (Round 3)
- `examples/_common.sh` — use `common.sh` helpers (Round 2)
- `examples/*.sh` — standardize exit codes (Round 3)
- `docs/claude-code-vllm-setup.md`, `docs/reverse_proxy_setup.md` — update if needed (Round 4)

---

## Round 1: Defect Fixes + Function Splits

> **Commit:** `refactor: split oversized functions and remove duplicates (round 1)`

---

### Task 1.1: Create `scripts/ray_cluster/_kill_lib.sh`

**Files:**
- Create: `scripts/ray_cluster/_kill_lib.sh`

Extract the following functions from `kill_multi_nodes.sh` into a new sourced library.

- [ ] **Step 1: Write `scripts/ray_cluster/_kill_lib.sh`**

```bash
#!/bin/bash
#
# _kill_lib.sh — Shared kill-script utilities for kill_multi_nodes.sh
#
# Note: sourced, not executed. Do not set shell options.

# -----------------------------------------------------------------------------
# Regex escape
# -----------------------------------------------------------------------------
escape_regex() {
    local s="$1"
    s="${s//\\/\\\\}"
    s="${s//./\\.}"
    s="${s//\*/\\*}"
    s="${s//+/\\+}"
    s="${s//\?/\\?}"
    s="${s//^/\\^}"
    s="${s//\$/\\$}"
    s="${s\//\(/\\(}"
    s="${s\//\)/\\)}"
    s="${s//\[/\\[}"
    s="${s//\]/\\]}"
    s="${s//\{/\\{}"
    s="${s//\}/\\}}"
    s="${s//|/\\|}"
    printf '%s' "$s"
}

# -----------------------------------------------------------------------------
# Build kill pattern from KEYWORDS array
# -----------------------------------------------------------------------------
_build_kill_pattern() {
    local escaped_keywords=()
    local kw
    for kw in "${KEYWORDS[@]}"; do
        escaped_keywords+=("$(escape_regex "$kw")")
    done
    IFS='|'; echo "${escaped_keywords[*]}"
}

# -----------------------------------------------------------------------------
# Generate remote kill script
# -----------------------------------------------------------------------------
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
        kill -15 $all_pids 2>/dev/null || true
        sleep "$KILL_TIMEOUT"

        remaining=""
        for pid in $all_pids; do
            if kill -0 "$pid" 2>/dev/null; then
                remaining="$remaining $pid"
            fi
        done
        remaining="${remaining# }"

        if [ -z "$remaining" ]; then
            echo "STATUS:TERMINATED"
            exit 0
        fi

        echo "ACTION:SIGKILL:$remaining"
        kill -9 $remaining 2>/dev/null || true
        sleep 1

        still_alive=""
        for pid in $remaining; do
            if kill -0 "$pid" 2>/dev/null; then
                still_alive="$still_alive $pid"
            fi
        done
        still_alive="${still_alive# }"

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

# -----------------------------------------------------------------------------
# Status parsing
# -----------------------------------------------------------------------------
_parse_kill_status() {
    local output="$1" exit_code="$2"
    if [[ "$output" == *"STATUS:NO_PROCESSES"* ]]; then
        echo "no_processes"
    elif [[ "$output" == *"STATUS:TERMINATED"* ]]; then
        echo "success"
    elif [[ "$output" == *"STATUS:KILLED"* ]]; then
        echo "killed"
    elif [[ "$output" == *"STATUS:FAILED"* ]]; then
        echo "failed"
    elif [[ $exit_code -eq 124 ]]; then
        echo "timeout"
    else
        echo "failed"
    fi
}

_log_kill_status() {
    local node="$1" status="$2" pids="$3" quiet="$4"

    case $status in
        no_processes)
            [[ "$quiet" == false ]] && log_info "[Node: $node] 未找到匹配的进程"
            return 0
            ;;
        success)
            [[ "$quiet" == false ]] && log_info "[Node: $node] 进程已正常终止 (PIDs: $pids)"
            return 0
            ;;
        killed)
            log_warn "[Node: $node] 进程已强制终止 (PIDs: $pids)"
            return 0
            ;;
        timeout)
            log_err "[Node: $node] SSH 连接超时 (${SSH_TIMEOUT}s)"
            return 124
            ;;
        failed)
            log_err "[Node: $node] 无法终止所有进程 (PIDs: $pids)"
            return 1
            ;;
    esac
}

_parse_and_log_kill_result() {
    local node="$1" output="$2" exit_code="$3" quiet="$4"
    local status pids=""

    status=$(_parse_kill_status "$output" "$exit_code")

    if [[ "$output" =~ PIDS:([^[:space:]]+) ]]; then
        pids="${BASH_REMATCH[1]}"
    fi

    _log_kill_status "$node" "$status" "$pids" "$quiet"
}
```

- [ ] **Step 2: Verify new file passes shellcheck**

Run: `shellcheck scripts/ray_cluster/_kill_lib.sh`
Expected: Only SC1091 (not following sourced common.sh) — acceptable.

---

### Task 1.2: Shrink `scripts/ray_cluster/kill_multi_nodes.sh`

**Files:**
- Modify: `scripts/ray_cluster/kill_multi_nodes.sh`

Remove the extracted functions and source `_kill_lib.sh`.

- [ ] **Step 1: Add source line after common.sh**

Insert after line 16 (`source "${SCRIPTS_DIR}/common.sh"`):
```bash
source "${SCRIPT_DIR}/_kill_lib.sh"
```

- [ ] **Step 2: Remove `escape_regex` function (lines 129-146)**

Delete the entire function.

- [ ] **Step 3: Remove `_build_kill_pattern` function (lines 191-198)**

Delete the entire function.

- [ ] **Step 4: Remove `_gen_kill_remote_script` function (lines 200-276)**

Delete the entire function.

- [ ] **Step 5: Remove `_parse_kill_status` function (lines 281-296)**

Delete the entire function.

- [ ] **Step 6: Remove `_log_kill_status` function (lines 298-323)**

Delete the entire function.

- [ ] **Step 7: Remove `_parse_and_log_kill_result` function (lines 325-336)**

Delete the entire function.

- [ ] **Step 8: Remove `ssh_run_with_timeout` function (lines 104-124)**

This will be replaced by `common.sh`'s enhanced `ssh_run_timeout` in Round 2. For Round 1, temporarily keep it but mark with a TODO comment, or delete if the plan worker handles Round 2 immediately after.

Actually, for clean Round 1, keep `ssh_run_with_timeout` but add a comment noting it will be replaced:
```bash
# TODO(round2): replace with common.sh ssh_run_timeout once perl fallback is merged
```

Wait — that's a placeholder, which the skill forbids. Better: leave it as-is for Round 1. Round 2 will replace it.

- [ ] **Step 9: Verify shrunk file**

Run: `wc -l scripts/ray_cluster/kill_multi_nodes.sh`
Expected: Under 400 lines (should be ~230 lines after removals).

Run: `bash -n scripts/ray_cluster/kill_multi_nodes.sh`
Expected: No output (syntax OK).

Run: `shellcheck scripts/ray_cluster/kill_multi_nodes.sh`
Expected: No errors.

---

### Task 1.3: Split `_remote_prepare_node` in `manage_docker_containers.sh`

**Files:**
- Modify: `scripts/docker/manage_docker_containers.sh`

Replace the 63-line `_remote_prepare_node` with 3 helpers.

- [ ] **Step 1: Replace `_remote_prepare_node` function**

Old function (lines 145-207):
```bash
_remote_prepare_node() {
  local image_name="$1"
  local image_tar="$2"
  local run_container_script="$3"
  local container_name="$4"
  local action="${5:-start}"

  set -euo pipefail

  if ! command -v docker >/dev/null 2>&1; then
    echo "[ERROR] docker command not found" >&2
    exit 127
  fi

  if ! docker info >/dev/null 2>&1; then
    echo "[INFO] Docker service not running, attempting to start..."
    if ! systemctl daemon-reload || ! systemctl start docker; then
      echo "[ERROR] Failed to start Docker service" >&2
      exit 1
    fi
  fi

  if [[ "$action" == "restart" || "$action" == "stop" ]]; then
    echo "[INFO] Stopping and removing all existing containers..."
    docker ps -aq 2>/dev/null | xargs -r docker stop 2>/dev/null || true
    docker ps -aq 2>/dev/null | xargs -r docker kill 2>/dev/null || true
    docker ps -aq 2>/dev/null | xargs -r docker rm -f 2>/dev/null || true
  fi

  if [[ "$action" == "start" || "$action" == "restart" ]]; then
    if docker image inspect "${image_name}" >/dev/null 2>&1; then
      :
    else
      if [[ ! -f "${image_tar}" ]]; then
        echo "[ERROR] image tar not found: ${image_tar}" >&2
        exit 2
      fi
      echo "[INFO] Loading image from ${image_tar}..."
      docker load -i "${image_tar}"
    fi

    if [[ ! -f "${run_container_script}" ]]; then
      echo "[ERROR] run script not found: ${run_container_script}" >&2
      exit 2
    fi

    export IMAGE_NAME="${image_name}"
    export CONTAINER_NAME="${container_name}"
    bash "${run_container_script}"

    if docker ps --format '{{.Names}}' | grep -Fx "${container_name}" >/dev/null; then
      echo "[INFO] Container ready: ${container_name}"
    else
      echo "[ERROR] Failed to start container: ${container_name}" >&2
      exit 1
    fi
  else
    echo "[INFO] Action is 'stop', skipping image load and container start."
  fi
}
```

New functions:
```bash
_remote_ensure_docker_running() {
  if ! command -v docker >/dev/null 2>&1; then
    echo "[ERROR] docker command not found" >&2
    exit 127
  fi
  if ! docker info >/dev/null 2>&1; then
    echo "[INFO] Docker service not running, attempting to start..."
    if ! systemctl daemon-reload || ! systemctl start docker; then
      echo "[ERROR] Failed to start Docker service" >&2
      exit 1
    fi
  fi
}

_remote_cleanup_containers() {
  echo "[INFO] Stopping and removing all existing containers..."
  docker ps -aq 2>/dev/null | xargs -r docker stop 2>/dev/null || true
  docker ps -aq 2>/dev/null | xargs -r docker kill 2>/dev/null || true
  docker ps -aq 2>/dev/null | xargs -r docker rm -f 2>/dev/null || true
}

_remote_load_and_run() {
  local image_name="$1" image_tar="$2" run_container_script="$3" container_name="$4"

  if docker image inspect "${image_name}" >/dev/null 2>&1; then
    :
  else
    if [[ ! -f "${image_tar}" ]]; then
      echo "[ERROR] image tar not found: ${image_tar}" >&2
      exit 2
    fi
    echo "[INFO] Loading image from ${image_tar}..."
    docker load -i "${image_tar}"
  fi

  if [[ ! -f "${run_container_script}" ]]; then
    echo "[ERROR] run script not found: ${run_container_script}" >&2
    exit 2
  fi

  export IMAGE_NAME="${image_name}"
  export CONTAINER_NAME="${container_name}"
  bash "${run_container_script}"

  if docker ps --format '{{.Names}}' | grep -Fx "${container_name}" >/dev/null; then
    echo "[INFO] Container ready: ${container_name}"
  else
    echo "[ERROR] Failed to start container: ${container_name}" >&2
    exit 1
  fi
}

_remote_prepare_node() {
  local image_name="$1"
  local image_tar="$2"
  local run_container_script="$3"
  local container_name="$4"
  local action="${5:-start}"

  set -euo pipefail

  _remote_ensure_docker_running

  if [[ "$action" == "restart" || "$action" == "stop" ]]; then
    _remote_cleanup_containers
  fi

  if [[ "$action" == "start" || "$action" == "restart" ]]; then
    _remote_load_and_run "${image_name}" "${image_tar}" "${run_container_script}" "${container_name}"
  else
    echo "[INFO] Action is 'stop', skipping image load and container start."
  fi
}
```

- [ ] **Step 2: Verify**

Run: `bash -n scripts/docker/manage_docker_containers.sh`
Expected: No output.

Run: `shellcheck scripts/docker/manage_docker_containers.sh`
Expected: No errors (SC1091 info for sourced files is OK).

---

### Task 1.4: Split `build_vllm_args_declare` in `deploy_vllm_multinode_mp.sh`

**Files:**
- Modify: `scripts/vllm/mp/deploy_vllm_multinode_mp.sh`

- [ ] **Step 1: Insert helper functions before `build_vllm_args_declare`**

Insert before line 155 (the old `build_vllm_args_declare`):

```bash
_build_base_args() {
    local -n arr="$1"
    arr+=(serve "${MODEL_PATH}")
    arr+=(--host 0.0.0.0)
    arr+=(--port "${VLLM_PORT}")
    arr+=(--trust-remote-code)
    arr+=(--served-model-name "${SERVED_MODEL_NAME}")
    arr+=(--seed 1024)
    arr+=(--tensor-parallel-size "${TENSOR_PARALLEL_SIZE}")
    arr+=(--pipeline-parallel-size "${PIPELINE_PARALLEL_SIZE}")
    arr+=(--max-num-seqs "${MAX_NUM_SEQS}")
    arr+=(--max-model-len "${MAX_MODEL_LEN}")
    arr+=(--max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}")
    arr+=(--gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}")
    arr+=(--no-enable-prefix-caching)
}

_build_mp_args() {
    local -n arr="$1"
    local node_rank="$2" master_addr="$3" nnodes="$4"
    if [[ "${nnodes}" -gt 1 ]]; then
        arr+=(--distributed-executor-backend mp)
        arr+=(--nnodes "${nnodes}")
        arr+=(--node-rank "${node_rank}")
        arr+=(--master-addr "${master_addr}")
    fi
}

_build_dp_args() {
    local -n arr="$1"
    local is_headless="$2" dp_size_local="$3" dp_start_rank="$4"
    if [[ "${DP_SIZE}" -gt 1 ]]; then
        arr+=(--data-parallel-size "${DP_SIZE}")
        arr+=(--data-parallel-size-local "${dp_size_local}")
        arr+=(--data-parallel-address "${NODE0_IP}")
        arr+=(--data-parallel-rpc-port "${DP_RPC_PORT}")
        if [[ "${is_headless}" == "true" ]]; then
            arr+=(--headless)
            arr+=(--data-parallel-start-rank "${dp_start_rank}")
        fi
    else
        if [[ "${is_headless}" == "true" ]]; then
            arr+=(--headless)
        fi
    fi
}

_build_a2_compile_args() {
    local -n arr="$1"
    arr+=(--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture sizes":[8, 16, 24, 32, 40, 48]}')
    arr+=(--additional-config '{"layer_sharding": ["q_b_proj", "o_proj"]}')
    arr+=(--speculative-config '{"num_speculative_tokens": 3, "method": "deepseek_mtp"}')
}
```

- [ ] **Step 2: Replace `build_vllm_args_declare` with orchestrator**

Replace the old function body with:

```bash
build_vllm_args_declare() {
    local is_headless="$1"
    local node_rank="$2"
    local dp_start_rank="$3"
    local dp_size_local="$4"
    local master_addr="$5"
    local nnodes="$6"
    local vllm_port="$7"
    local use_internal_dp="$8"

    local tp_size="${TENSOR_PARALLEL_SIZE}"
    local pp_size="${PIPELINE_PARALLEL_SIZE}"
    local ep_size="${EXPERT_PARALLEL_SIZE}"

    local -a args=()
    _build_base_args args
    _build_mp_args args "${node_rank}" "${master_addr}" "${nnodes}"

    if [[ "${use_internal_dp}" == "true" ]]; then
        _build_dp_args args "${is_headless}" "${dp_size_local}" "${dp_start_rank}"
    else
        if [[ "${is_headless}" == "true" ]]; then
            args+=(--headless)
        fi
    fi

    _build_a2_compile_args args

    local help_text=""
    [[ "${AUTO_DETECT_FLAGS}" == "1" ]] && help_text="$(vllm_help)"

    _add_ep_args args "$help_text" "$ep_size"
    _add_chunked_prefill_args args "$help_text"
    _add_prefix_caching_args args "$help_text"

    declare -p args
}
```

- [ ] **Step 3: Update `launch_on_node` call signature**

The `launch_on_node` function passes arguments positionally. Ensure the call in `_deploy_multinode_instance` and `_deploy_singlenode_instance` matches the new 8-parameter signature. The existing calls already pass the right arguments in the right order — no change needed except ensuring `use_internal_dp` is passed correctly.

In `_deploy_multinode_instance`, the call is:
```bash
launch_on_node "${node}" "${local_ip}" "${is_headless}" "${offset}" "0" "1" "${instance_master_ip}" "${NODES_PER_INSTANCE}" "${instance_port}" "false"
```

This already passes 10 arguments, but the function only uses 9. After the split, `build_vllm_args_declare` takes 8 parameters. The `launch_on_node` function should pass them in this order:
`is_headless node_rank dp_start_rank dp_size_local master_addr nnodes vllm_port use_internal_dp`

Actually, wait — let me re-check. The current `build_vllm_args_declare` signature is:
```bash
build_vllm_args_declare() {
    local is_headless="$1"
    local node_rank="$2"
    local dp_start_rank="$3"
    local dp_size_local="$4"
    local master_addr="$5"
    local nnodes="$6"
    local vllm_port="$7"
    local use_internal_dp="$8"
```

And `launch_on_node` calls it with:
```bash
array_decl=$(build_vllm_args_declare "${is_headless}" "${node_rank}" "${dp_start_rank}" "${dp_size_local}" "${master_addr}" "${nnodes}" "${vllm_port}" "${use_internal_dp}")
```

This already matches. Good.

- [ ] **Step 4: Verify**

Run: `bash -n scripts/vllm/mp/deploy_vllm_multinode_mp.sh`
Expected: No output.

Run: `shellcheck scripts/vllm/mp/deploy_vllm_multinode_mp.sh`
Expected: No errors.

---

### Task 1.5: Remove duplicate `has_flag` from `vllm_model_server.sh`

**Files:**
- Modify: `scripts/vllm/vllm_model_server.sh`

- [ ] **Step 1: Remove the local `has_flag` function (lines 202-204)**

Delete:
```bash
has_flag() {
    [[ "${HELP_TEXT:-}" == *"$1"* ]]
}
```

- [ ] **Step 2: Add `source` for common.sh**

After line 21 (SCRIPT_DIR assignment), add:
```bash
source "${SCRIPT_DIR}/../common.sh"
```

Note: The script already has `has_flag` calls that reference `HELP_TEXT` (a local variable). The `common.sh` version uses a parameter: `has_flag "$help_text" "$flag"`. So the call sites need updating too.

Actually, looking at the code more carefully:
- `common.sh`'s `has_flag` takes two args: `help_text` and `flag`
- `vllm_model_server.sh`'s `has_flag` takes one arg and uses global `HELP_TEXT`

The callers in `vllm_model_server.sh` are:
```bash
has_flag "--swap-space"
has_flag "--max-tokens-per-sequence"
has_flag "--num-scheduler-steps"
has_flag "--enable-expert-parallel"
has_flag "--enable-prefix-caching"
has_flag "--disable-prefix-caching"
has_flag "--enforce-eager"
has_flag "--max-seq-len-to-capture"
has_flag "--log-level"
has_flag "--enable-metrics"
has_flag "--metrics-port"
has_flag "--allowed-origins"
has_flag "--disable-log-requests"
```

These need to be changed to: `has_flag "$HELP_TEXT" "--flag-name"`

- [ ] **Step 3: Update all `has_flag` call sites**

Replace each `has_flag "<flag>"` with `has_flag "$HELP_TEXT" "<flag>"`.

- [ ] **Step 4: Verify**

Run: `bash -n scripts/vllm/vllm_model_server.sh`
Expected: No output.

Run: `shellcheck scripts/vllm/vllm_model_server.sh`
Expected: No errors.

---

### Task 1.6: Round 1 commit

- [ ] **Step 1: Commit all Round 1 changes**

```bash
git add scripts/ray_cluster/_kill_lib.sh \
        scripts/ray_cluster/kill_multi_nodes.sh \
        scripts/docker/manage_docker_containers.sh \
        scripts/vllm/mp/deploy_vllm_multinode_mp.sh \
        scripts/vllm/vllm_model_server.sh
git commit -m "refactor: split oversized functions and remove duplicates (round 1)

- Extract kill_multi_nodes helpers into _kill_lib.sh (shrinks 498 -> ~230 lines)
- Split _remote_prepare_node into 3 helpers (<50 lines each)
- Split build_vllm_args_declare into 4 helpers (<50 lines each)
- Remove duplicate has_flag() from vllm_model_server.sh, source common.sh instead"
```

- [ ] **Step 2: Run full verification**

Run: `shellcheck scripts/**/*.sh tools/*.sh examples/*.sh`
Expected: Only SC1091 info messages.

Run: `bash -n scripts/vllm/vllm_model_server.sh`
Expected: No output.

---

## Round 2: Deduplication into `scripts/common.sh`

> **Commit:** `refactor: extract shared utilities into common.sh (round 2)`

---

### Task 2.1: Enhance `ssh_run_timeout` with Perl fallback

**Files:**
- Modify: `scripts/common.sh`

- [ ] **Step 1: Replace `ssh_run_timeout` function**

Old function (lines 60-71):
```bash
ssh_run_timeout() {
    local timeout_sec="${1:?用法: ssh_run_timeout <timeout> <node> <cmd...>}"; shift
    local node="$1"; shift
    if command -v timeout >/dev/null 2>&1; then
        # shellcheck disable=SC2086
        timeout "$timeout_sec" ssh ${SSH_OPTS:-} "$(ssh_target "$node")" "$@" 2>&1
    else
        # 无 timeout 命令时直接执行
        # shellcheck disable=SC2086,SC2029
        ssh ${SSH_OPTS:-} "$(ssh_target "$node")" "$@" 2>&1
    fi
}
```

New function:
```bash
ssh_run_timeout() {
    local timeout_sec="${1:?用法: ssh_run_timeout <timeout> <node> <cmd...>}"; shift
    local node="$1"; shift
    if command -v timeout >/dev/null 2>&1; then
        # shellcheck disable=SC2086
        timeout "$timeout_sec" ssh ${SSH_OPTS:-} "$(ssh_target "$node")" "$@" 2>&1
    elif command -v perl >/dev/null 2>&1; then
        # Fallback: perl alarm-based timeout
        # shellcheck disable=SC2086
        perl -e '
            use strict; use warnings;
            my $timeout = shift @ARGV; my @cmd = @ARGV;
            eval { local $SIG{ALRM} = sub { die "TIMEOUT\n" }; alarm $timeout; system(@cmd); alarm 0; };
            if ($@ eq "TIMEOUT\n") { print STDERR "[ERROR] Command timed out after ${timeout}s\n"; exit 124; }
            exit $? >> 8;
        ' "$timeout_sec" ssh ${SSH_OPTS:-} "$(ssh_target "$node")" "$@" 2>&1
    else
        # 无 timeout 命令时直接执行
        # shellcheck disable=SC2086,SC2029
        ssh ${SSH_OPTS:-} "$(ssh_target "$node")" "$@" 2>&1
    fi
}
```

- [ ] **Step 2: Verify**

Run: `bash -n scripts/common.sh`
Expected: No output.

Run: `shellcheck scripts/common.sh`
Expected: No errors.

---

### Task 2.2: Add `parse_nodes_file_arg` to `common.sh`

**Files:**
- Modify: `scripts/common.sh`

- [ ] **Step 1: Insert after `wait_for_port` function**

Insert after line 138 (end of `wait_for_port`):

```bash
# ------------------------------------------------------------------------------
# 参数解析：统一处理 --file/-f 节点列表参数
# ------------------------------------------------------------------------------
# 用法: NODE_LIST=$(parse_nodes_file_arg "$@")
# 从脚本参数中提取 --file/-f 值，未提供则返回 ${NODES_FILE:-scripts/node_list.txt}
parse_nodes_file_arg() {
    local nodes_file="${NODES_FILE:-scripts/node_list.txt}"
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --file|-f)
                if [[ -n "${2:-}" && "$2" != -* ]]; then
                    nodes_file="$2"
                    shift 2
                else
                    log_fatal "选项 $1 需要一个参数: 节点列表文件路径"
                fi
                ;;
            *) shift ;;
        esac
    done
    printf '%s\n' "$nodes_file"
}
```

- [ ] **Step 2: Verify**

Run: `bash -n scripts/common.sh`
Expected: No output.

---

### Task 2.3: Add `wait_for_server`, `print_server_ready`, `require_env` to `common.sh`

**Files:**
- Modify: `scripts/common.sh`

- [ ] **Step 1: Insert after `parse_nodes_file_arg`**

```bash
# ------------------------------------------------------------------------------
# 等待 vLLM HTTP 服务就绪
# ------------------------------------------------------------------------------
wait_for_server() {
    local host="${1:?用法: wait_for_server <host> <port> [timeout_sec]}"
    local port="${2:?用法: wait_for_server <host> <port> [timeout_sec]}"
    local max_wait="${3:-600}"
    local url="http://${host}:${port}/health"
    local elapsed=0 interval=5

    log_info "Waiting for server to become ready..."
    while (( elapsed < max_wait )); do
        if curl -sf "$url" >/dev/null 2>&1; then
            log_info "================================================================================="
            log_info "  vLLM server is READY"
            log_info "================================================================================="
            log_info "  Health check:  http://${host}:${port}/health"
            log_info "  API endpoint:  http://${host}:${port}/v1"
            log_info "  Models list:   http://${host}:${port}/v1/models"
            log_info "================================================================================="
            return 0
        fi
        sleep "$interval"
        elapsed=$((elapsed + interval))
        printf "."
    done
    printf "\n"
    log_err "Server did not become ready within ${max_wait}s"
    return 1
}

# ------------------------------------------------------------------------------
# 打印服务就绪后的 Claude Code 配置输出
# ------------------------------------------------------------------------------
print_server_ready() {
    local host_ip="${1:?用法: print_server_ready <host> <port> [model_name]}"
    local port="${2:?用法: print_server_ready <host> <port> [model_name]}"
    local model_name="${3:-}"

    log_info "================================================================================="
    log_info "  vLLM server is READY"
    log_info "================================================================================="
    log_info "  Health check:  http://${host_ip}:${port}/health"
    log_info "  API endpoint:  http://${host_ip}:${port}/v1"
    log_info "  Models list:   http://${host_ip}:${port}/v1/models"
    if [[ -n "$model_name" ]]; then
        log_info ""
        log_info "  --- Claude Code 配置 ---"
        log_info ""
        log_info "  方式一: 写入 ~/.claude/settings.json"
        log_info "  {"
        log_info "    \"env\": {"
        log_info "      \"ANTHROPIC_BASE_URL\": \"http://${host_ip}:${port}/v1\","
        log_info "      \"ANTHROPIC_API_KEY\": \"dummy\","
        log_info "      \"ANTHROPIC_AUTH_TOKEN\": \"dummy\","
        log_info "      \"ANTHROPIC_DEFAULT_SONNET_MODEL\": \"${model_name}\","
        log_info "      \"ANTHROPIC_DEFAULT_HAIKU_MODEL\": \"${model_name}\","
        log_info "      \"ANTHROPIC_DEFAULT_OPUS_MODEL\": \"${model_name}\""
        log_info "    }"
        log_info "  }"
        log_info ""
        log_info "  方式二: 命令行直接使用"
        log_info "  ANTHROPIC_BASE_URL=http://${host_ip}:${port}/v1 \\"
        log_info "  ANTHROPIC_API_KEY=dummy \\"
        log_info "  ANTHROPIC_AUTH_TOKEN=dummy \\"
        log_info "  ANTHROPIC_DEFAULT_SONNET_MODEL=${model_name} \\"
        log_info "  ANTHROPIC_DEFAULT_HAIKU_MODEL=${model_name} \\"
        log_info "  ANTHROPIC_DEFAULT_OPUS_MODEL=${model_name} \\"
        log_info "  claude"
    fi
    log_info ""
    log_info "================================================================================="
}

# ------------------------------------------------------------------------------
# 要求环境变量已设置
# ------------------------------------------------------------------------------
require_env() {
    local var="$1"
    local desc="${2:-$var}"
    if [[ -z "${!var:-}" ]]; then
        log_fatal "环境变量 ${var} (${desc}) 未设置"
    fi
}
```

- [ ] **Step 2: Verify**

Run: `bash -n scripts/common.sh`
Expected: No output.

Run: `shellcheck scripts/common.sh`
Expected: No errors.

---

### Task 2.4: Update `kill_multi_nodes.sh` to use enhanced `ssh_run_timeout`

**Files:**
- Modify: `scripts/ray_cluster/kill_multi_nodes.sh`

- [ ] **Step 1: Replace `ssh_run_with_timeout` calls**

In `kill_processes_on_node` function, replace:
```bash
    output=$(ssh_run_with_timeout "$node" "$remote_cmd" 2>&1) || exit_code=$?
```

With:
```bash
    output=$(ssh_run_timeout "$SSH_TIMEOUT" "$node" "$remote_cmd" 2>&1) || exit_code=$?
```

- [ ] **Step 2: Remove the old `ssh_run_with_timeout` function**

Delete lines 104-124 (the entire function).

- [ ] **Step 3: Verify**

Run: `bash -n scripts/ray_cluster/kill_multi_nodes.sh`
Expected: No output.

Run: `shellcheck scripts/ray_cluster/kill_multi_nodes.sh`
Expected: No errors.

---

### Task 2.5: Update `vllm_model_server.sh` to use `common.sh`

**Files:**
- Modify: `scripts/vllm/vllm_model_server.sh`

- [ ] **Step 1: Ensure `common.sh` is sourced**

This was done in Task 1.5. Verify line 22-23 area has:
```bash
source "${SCRIPT_DIR}/../common.sh"
```

- [ ] **Step 2: Replace raw echo logging with `log_*` functions**

Replace all raw `echo` logging with `log_*` calls. Key replacements:

Line 209: `command -v vllm ... || { echo "[ERROR] vllm not found" ...`
→ `command -v vllm ... || { log_err "vllm not found" ...`

Line 210: `[[ -e "$MODEL_PATH" ]] || { echo "[ERROR] MODEL_PATH not found" ...`
→ `[[ -e "$MODEL_PATH" ]] || { log_err "MODEL_PATH not found: $MODEL_PATH" ...`

Line 211: `[[ -f "$MODEL_PATH/config.json" ]] || { echo "[ERROR] config.json not found" ...`
→ `[[ -f "$MODEL_PATH/config.json" ]] || { log_err "config.json not found in: $MODEL_PATH" ...`

Lines 312-332 (config summary): Keep the heredoc output, but replace `[INFO]` prefix with `log_info` where appropriate, or keep as-is since it's a structured output block.

Lines 338-358 (retry loop): Replace `echo "[INFO]"` with `log_info`, `echo "[WARN]"` with `log_warn`, `echo "[FATAL]"` with `log_fatal`.

Specific replacements:
```bash
# Line ~342
        echo "[INFO] vLLM server exited normally."
→       log_info "vLLM server exited normally."

# Line ~345
        echo "[INFO] Terminated by signal (exit $EXIT_CODE)."
→       log_info "Terminated by signal (exit $EXIT_CODE)."

# Line ~348
        RETRY_COUNT=$((RETRY_COUNT + 1))
        if [[ $RETRY_COUNT -lt $MAX_RETRIES ]]; then
            echo "[WARN] Crashed (exit $EXIT_CODE), retrying in ${RETRY_DELAY}s... ($RETRY_COUNT/$MAX_RETRIES)"
→           log_warn "Crashed (exit $EXIT_CODE), retrying in ${RETRY_DELAY}s... ($RETRY_COUNT/$MAX_RETRIES)"

# Line ~354
            echo "[FATAL] Max retries reached."
→           log_fatal "Max retries reached."
```

- [ ] **Step 3: Verify**

Run: `bash -n scripts/vllm/vllm_model_server.sh`
Expected: No output.

Run: `shellcheck scripts/vllm/vllm_model_server.sh`
Expected: No errors.

---

### Task 2.6: Update `examples/_common.sh` to use `common.sh`

**Files:**
- Modify: `examples/_common.sh`

- [ ] **Step 1: Replace `wait_for_server` with call to `common.sh` version**

The function in `examples/_common.sh` is nearly identical to the one now in `common.sh`. Replace it with:

```bash
# wait_for_server and print_server_ready are now in common.sh
# Kept here for backward compatibility of callers that expect them in this file
```

Actually, better approach: keep `examples/_common.sh` as a thin wrapper that sources `common.sh` and re-exports if needed. But since `examples/_common.sh` already sources `common.sh` at line 10, the functions from `common.sh` are already available.

Simply delete the `wait_for_server` and `print_server_ready` functions from `examples/_common.sh` (lines 37-73 and 78-114).

- [ ] **Step 2: Verify**

Run: `bash -n examples/_common.sh`
Expected: No output.

Run: `shellcheck examples/_common.sh`
Expected: No errors.

---

### Task 2.7: Add shared vLLM deployment helpers to `scripts/vllm/mp/_common.sh`

**Files:**
- Modify: `scripts/vllm/mp/_common.sh`

- [ ] **Step 1: Insert after existing functions**

Add after line 92 (after `_add_prefix_caching_args`):

```bash
# ------------------------------------------------------------------------------
# 共享多节点部署逻辑
# ------------------------------------------------------------------------------

# 读取并验证节点列表
load_and_validate_nodes() {
    local nodes_file="$1"
    local min_nodes="${2:-2}"

    if [[ ! -f "${nodes_file}" ]]; then
        log_fatal "Node list file not found: ${nodes_file}"
    fi

    ALL_NODES=()
    while IFS= read -r line; do
        ALL_NODES+=("$line")
    done < <(read_nodes "${nodes_file}")
    TOTAL_NODES=${#ALL_NODES[@]}

    if [[ ${TOTAL_NODES} -lt ${min_nodes} ]]; then
        log_fatal "Need at least ${min_nodes} nodes, got ${TOTAL_NODES}"
    fi

    NODE0="${ALL_NODES[0]}"
    log_info "Loaded ${TOTAL_NODES} nodes from ${nodes_file}"
    log_info "Master node: ${NODE0}"
}

# 验证并行配置合法性
validate_parallelism_config() {
    local total_nodes="$1"
    local npus_per_node="$2"

    TOTAL_CARDS=$((total_nodes * npus_per_node))
    CARDS_PER_INSTANCE=$((TENSOR_PARALLEL_SIZE * PIPELINE_PARALLEL_SIZE))

    if [[ ${CARDS_PER_INSTANCE} -eq 0 ]]; then
        log_fatal "Invalid config: TENSOR_PARALLEL_SIZE * PIPELINE_PARALLEL_SIZE = 0"
    fi

    if [[ $((TOTAL_CARDS % CARDS_PER_INSTANCE)) -ne 0 ]]; then
        log_fatal "Card mismatch: TOTAL_CARDS (${TOTAL_CARDS}) is not divisible by CARDS_PER_INSTANCE (${CARDS_PER_INSTANCE})"
    fi

    DP_SIZE=$((TOTAL_CARDS / CARDS_PER_INSTANCE))
    if [[ ${DP_SIZE} -lt 1 ]]; then
        log_fatal "Invalid config: DP_SIZE (${DP_SIZE}) must be >= 1"
    fi

    if [[ ${CARDS_PER_INSTANCE} -le ${npus_per_node} ]]; then
        DP_SIZE_LOCAL=$((npus_per_node / CARDS_PER_INSTANCE))
    else
        DP_SIZE_LOCAL=1
    fi

    if [[ ${CARDS_PER_INSTANCE} -gt ${npus_per_node} ]]; then
        NODES_PER_INSTANCE=$((CARDS_PER_INSTANCE / npus_per_node))
        if [[ $((total_nodes % NODES_PER_INSTANCE)) -ne 0 ]]; then
            log_fatal "Node mismatch: each instance needs ${NODES_PER_INSTANCE} nodes"
        fi
        if [[ $((DP_SIZE * NODES_PER_INSTANCE)) -ne ${total_nodes} ]]; then
            log_fatal "Config mismatch: DP_SIZE * NODES_PER_INSTANCE != TOTAL_NODES"
        fi
    else
        NODES_PER_INSTANCE=1
    fi

    log_info "Config: TOTAL_CARDS=${TOTAL_CARDS}, TP=${TENSOR_PARALLEL_SIZE}, PP=${PIPELINE_PARALLEL_SIZE}, DP=${DP_SIZE}, DP_LOCAL=${DP_SIZE_LOCAL}, NODES_PER_INSTANCE=${NODES_PER_INSTANCE}"
}

# 获取 node0 IP
resolve_node0_ip() {
    local node0="$1"
    local nic_name="$2"
    NODE0_IP=$(get_node_ip "${node0}" "${nic_name}")
    if [[ -z "${NODE0_IP}" ]]; then
        log_fatal "Failed to get IP for node ${node0} on interface ${nic_name}"
    fi
    log_info "Node0 IP: ${NODE0_IP}"
}
```

- [ ] **Step 2: Verify**

Run: `bash -n scripts/vllm/mp/_common.sh`
Expected: No output.

Run: `shellcheck scripts/vllm/mp/_common.sh`
Expected: No errors.

---

### Task 2.8: Update `deploy_vllm_multinode.sh` to use shared helpers

**Files:**
- Modify: `scripts/vllm/mp/deploy_vllm_multinode.sh`

- [ ] **Step 1: Replace node-loading block (lines 76-93)**

Old:
```bash
if [[ ! -f "${NODE_LIST_FILE}" ]]; then
    log_fatal "Node list file not found: ${NODE_LIST_FILE}"
fi

ALL_NODES=()
while IFS= read -r line; do
    ALL_NODES+=("$line")
done < <(read_nodes "${NODE_LIST_FILE}")
TOTAL_NODES=${#ALL_NODES[@]}

if [[ ${TOTAL_NODES} -lt 2 ]]; then
    log_fatal "Need at least 2 nodes for multi-node deployment, got ${TOTAL_NODES}"
fi

NODE0="${ALL_NODES[0]}"
log_info "Loaded ${TOTAL_NODES} nodes from ${NODE_LIST_FILE}"
log_info "Master node: ${NODE0}"
```

New:
```bash
load_and_validate_nodes "${NODE_LIST_FILE}" 2
```

- [ ] **Step 2: Replace config validation block (lines 96-122)**

Old:
```bash
TOTAL_CARDS=$((TOTAL_NODES * NPUS_PER_NODE))
CARDS_PER_INSTANCE=$((TENSOR_PARALLEL_SIZE * PIPELINE_PARALLEL_SIZE))

if [[ ${CARDS_PER_INSTANCE} -eq 0 ]]; then
    log_fatal "Invalid config: TENSOR_PARALLEL_SIZE * PIPELINE_PARALLEL_SIZE = 0"
fi

if [[ $((TOTAL_CARDS % CARDS_PER_INSTANCE)) -ne 0 ]]; then
    log_fatal "Card mismatch: TOTAL_CARDS (${TOTAL_CARDS}) is not divisible by CARDS_PER_INSTANCE (${CARDS_PER_INSTANCE}). Please adjust TP/PP."
fi

DP_SIZE=$((TOTAL_CARDS / CARDS_PER_INSTANCE))

if [[ ${DP_SIZE} -lt 1 ]]; then
    log_fatal "Invalid config: DP_SIZE (${DP_SIZE}) must be >= 1. Please reduce TP or PP."
fi

if [[ ${CARDS_PER_INSTANCE} -le ${NPUS_PER_NODE} ]]; then
    DP_SIZE_LOCAL=$((NPUS_PER_NODE / CARDS_PER_INSTANCE))
else
    DP_SIZE_LOCAL=1
fi

log_info "Config check passed: TOTAL_CARDS=${TOTAL_CARDS}, TP=${TENSOR_PARALLEL_SIZE}, PP=${PIPELINE_PARALLEL_SIZE}, DP=${DP_SIZE}, DP_LOCAL=${DP_SIZE_LOCAL}"
```

New:
```bash
validate_parallelism_config "${TOTAL_NODES}" "${NPUS_PER_NODE}"
```

Wait, but `validate_parallelism_config` also computes `NODES_PER_INSTANCE` which isn't needed in `deploy_vllm_multinode.sh` (it doesn't have the multi-node-per-instance logic). Let me check the current code again...

Actually, looking at `deploy_vllm_multinode.sh` lines 96-122, it does NOT compute `NODES_PER_INSTANCE`. The simpler version is sufficient. So I should create a simpler validation function, or just keep the original inline code for `deploy_vllm_multinode.sh`.

Hmm, but the design says to extract shared logic. The two files share the basic validation but `deploy_vllm_multinode_mp.sh` has extra logic for `NODES_PER_INSTANCE`. Let me make `validate_parallelism_config` handle both cases by making the extra logic conditional.

Actually, looking more carefully:
- `deploy_vllm_multinode.sh` (ray backend): computes TOTAL_CARDS, CARDS_PER_INSTANCE, DP_SIZE, DP_SIZE_LOCAL. No NODES_PER_INSTANCE.
- `deploy_vllm_multinode_mp.sh` (mp backend): computes all of the above PLUS NODES_PER_INSTANCE with extra validation.

So `validate_parallelism_config` should compute everything. For `deploy_vllm_multinode.sh`, the extra `NODES_PER_INSTANCE` variable will simply go unused. That's fine.

- [ ] **Step 3: Replace IP resolution block (lines 125-131)**

Old:
```bash
NODE0_IP=$(get_node_ip "${NODE0}" "${NIC_NAME}")
if [[ -z "${NODE0_IP}" ]]; then
    log_fatal "Failed to get IP address for node ${NODE0} on interface ${NIC_NAME}"
fi
log_info "Node0 IP (DP master): ${NODE0_IP}"
```

New:
```bash
resolve_node0_ip "${NODE0}" "${NIC_NAME}"
```

- [ ] **Step 4: Verify**

Run: `bash -n scripts/vllm/mp/deploy_vllm_multinode.sh`
Expected: No output.

Run: `shellcheck scripts/vllm/mp/deploy_vllm_multinode.sh`
Expected: No errors.

---

### Task 2.9: Update `deploy_vllm_multinode_mp.sh` to use shared helpers

**Files:**
- Modify: `scripts/vllm/mp/deploy_vllm_multinode_mp.sh`

- [ ] **Step 1: Replace node-loading block (lines 76-94)**

Same as Task 2.8 but with `min_nodes=1` (the mp version allows single node):
```bash
load_and_validate_nodes "${NODE_LIST_FILE}" 1
```

- [ ] **Step 2: Replace config validation block (lines 97-134)**

```bash
validate_parallelism_config "${TOTAL_NODES}" "${NPUS_PER_NODE}"
```

- [ ] **Step 3: Replace IP resolution block (lines 137-143)**

```bash
resolve_node0_ip "${NODE0}" "${NIC_NAME}"
```

- [ ] **Step 4: Verify**

Run: `bash -n scripts/vllm/mp/deploy_vllm_multinode_mp.sh`
Expected: No output.

Run: `shellcheck scripts/vllm/mp/deploy_vllm_multinode_mp.sh`
Expected: No errors.

---

### Task 2.10: Round 2 commit

- [ ] **Step 1: Commit all Round 2 changes**

```bash
git add scripts/common.sh \
        scripts/ray_cluster/kill_multi_nodes.sh \
        scripts/vllm/vllm_model_server.sh \
        examples/_common.sh \
        scripts/vllm/mp/_common.sh \
        scripts/vllm/mp/deploy_vllm_multinode.sh \
        scripts/vllm/mp/deploy_vllm_multinode_mp.sh
git commit -m "refactor: extract shared utilities into common.sh (round 2)

- Enhance ssh_run_timeout with perl alarm fallback
- Add parse_nodes_file_arg, wait_for_server, print_server_ready, require_env
- Update kill_multi_nodes.sh to use common.sh ssh_run_timeout
- Update vllm_model_server.sh to use log_* from common.sh
- Deduplicate examples/_common.sh into common.sh
- Extract shared vLLM deployment logic into _common.sh helpers"
```

- [ ] **Step 2: Run full verification**

Run: `shellcheck scripts/**/*.sh tools/*.sh examples/*.sh`
Expected: Only SC1091 info messages.

Run: `bash -n scripts/vllm/vllm_model_server.sh`
Expected: No output.

---

## Round 3: Unify Input + Standardize Output

> **Commit:** `refactor: unify node-list input and standardize exit codes (round 3)`

---

### Task 3.1: Unify `manage_docker_containers.sh` node-list parsing

**Files:**
- Modify: `scripts/docker/manage_docker_containers.sh`

- [ ] **Step 1: Replace custom --file/-f parsing with `parse_nodes_file_arg`**

Remove the custom `--file/-f` handling (lines 58-67) and the `NODES_FILE_ARG` variable.

Replace lines 39-41 and 80-81:
```bash
ACTION="start"
NODES_FILE_ARG=""
```

With:
```bash
ACTION="start"
```

And replace lines 80-81:
```bash
# 命令行参数优先于环境变量
[[ -n "$NODES_FILE_ARG" ]] && NODES_FILE="$NODES_FILE_ARG"
```

With:
```bash
# 解析节点列表参数
NODES_FILE=$(parse_nodes_file_arg "$@")
```

Also remove the `-f|--file` case from the while loop.

- [ ] **Step 2: Standardize exit codes**

Replace bare exit codes:
- `exit 1` for bad args → `exit $E_INVALID_ARG`
- `exit 2` for missing files → `exit $E_NOT_FOUND`
- `exit 127` for missing docker command in remote → keep as-is (remote script)

- [ ] **Step 3: Verify**

Run: `bash -n scripts/docker/manage_docker_containers.sh`
Expected: No output.

---

### Task 3.2: Unify `start_ray_cluster.sh` node-list parsing

**Files:**
- Modify: `scripts/ray_cluster/start_ray_cluster.sh`

- [ ] **Step 1: Replace custom --file parsing**

In the argument loop (lines 40-47), replace:
```bash
        --file|-f)  NODE_LIST="$2"; shift 2 ;;
```

With the use of `parse_nodes_file_arg` before the loop. Actually, the current loop consumes `start|stop` as ACTION. A cleaner approach is to first extract the node list, then handle action:

Replace lines 39-47:
```bash
ACTION=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        start|stop) ACTION="$1"; shift ;;
        --file|-f)  NODE_LIST="$2"; shift 2 ;;
        --help|-h)  usage ;;
        *) log_err "未知参数: $1"; usage ;;
    esac
done
```

With:
```bash
NODE_LIST=$(parse_nodes_file_arg "$@")
ACTION=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        start|stop) ACTION="$1"; shift ;;
        --file|-f)  shift 2 ;;  # consumed by parse_nodes_file_arg
        --help|-h)  usage ;;
        *) log_err "未知参数: $1"; usage ;;
    esac
done
```

- [ ] **Step 2: Standardize exit codes**

Replace `exit 1` with appropriate `E_*` constants.

- [ ] **Step 3: Verify**

Run: `bash -n scripts/ray_cluster/start_ray_cluster.sh`
Expected: No output.

---

### Task 3.3: Unify `kill_multi_nodes.sh` node-list parsing

**Files:**
- Modify: `scripts/ray_cluster/kill_multi_nodes.sh`

- [ ] **Step 1: Add --file/-f support**

In `parse_args` function, add `--file|-f` case:

```bash
            --file|-f)
                [[ -n "${2:-}" && "$2" != -* ]] || { log_err "选项 $1 需要一个参数"; exit $E_INVALID_ARG; }
                NODE_LIST_FILE="$2"; shift 2
                ;;
```

- [ ] **Step 2: Standardize exit codes**

Replace bare `exit 1` with `exit $E_INVALID_ARG`, `exit $E_NOT_FOUND`, etc.

- [ ] **Step 3: Verify**

Run: `bash -n scripts/ray_cluster/kill_multi_nodes.sh`
Expected: No output.

---

### Task 3.4: Add `--file/-f` to `deploy_vllm_multinode.sh`

**Files:**
- Modify: `scripts/vllm/mp/deploy_vllm_multinode.sh`

- [ ] **Step 1: Replace hardcoded node list with parameter parsing**

After line 38 (`AUTO_DETECT_FLAGS`), add:
```bash
NODE_LIST_FILE=$(parse_nodes_file_arg "$@")
```

And remove the hardcoded:
```bash
NODE_LIST_FILE="${SCRIPT_DIR}/../../node_list.txt"
```

- [ ] **Step 2: Standardize exit codes**

Replace `exit 2` with `exit $E_NOT_FOUND` where appropriate.

- [ ] **Step 3: Verify**

Run: `bash -n scripts/vllm/mp/deploy_vllm_multinode.sh`
Expected: No output.

---

### Task 3.5: Add `--file/-f` to `deploy_vllm_multinode_mp.sh`

**Files:**
- Modify: `scripts/vllm/mp/deploy_vllm_multinode_mp.sh`

- [ ] **Step 1: Replace hardcoded node list**

Same as Task 3.4:
```bash
NODE_LIST_FILE=$(parse_nodes_file_arg "$@")
```

Remove hardcoded line 37.

- [ ] **Step 2: Standardize exit codes**

Replace bare `exit` codes with `E_*` constants.

- [ ] **Step 3: Verify**

Run: `bash -n scripts/vllm/mp/deploy_vllm_multinode_mp.sh`
Expected: No output.

---

### Task 3.6: Standardize remaining scripts' exit codes

**Files:**
- Modify: `examples/*.sh`, `tools/*.sh`, `scripts/docker/*.sh`, `scripts/ray_cluster/*.sh`, `scripts/vllm/*.sh`

- [ ] **Step 1: Update scripts that use bare exit codes**

For each script that uses `exit 1` for invalid args: replace with `exit $E_INVALID_ARG`
For each script that uses `exit 1` or `exit 2` for missing files: replace with `exit $E_NOT_FOUND`
For `exit 127` for missing commands: replace with `exit $E_CMD_NOT_FOUND`

Scripts to update:
- `examples/check_glm5_env.sh`
- `examples/glm5_full_server.sh`
- `examples/glm5_server.sh`
- `examples/glm5-1_quant_server.sh`
- `examples/qwen3_server.sh`
- `examples/curl_test.sh`
- `examples/lm_eval.sh`
- `examples/longcat_flash-chat.sh`
- `examples/run.sh`
- `tools/docker_proxy.sh`
- `tools/host_proxy.sh`
- `tools/hf_download.sh`
- `tools/ms_download.sh`
- `scripts/docker/ascend_infer_docker_run.sh`
- `scripts/docker/ascend_train_docker_run.sh`
- `scripts/docker/copy_file_to_containers.sh`
- `scripts/docker/manage_npuslim_containers.sh`
- `scripts/docker/run_npuslim_container.sh`
- `scripts/ray_cluster/native_ray_start_cluster.sh`
- `scripts/ray_cluster/ray_head.sh`
- `scripts/ray_cluster/ray_node.sh`
- `scripts/ray_cluster/start_npuslim_ray_cluster.sh`
- `scripts/ray_cluster/stop_ray_cluster.sh`
- `scripts/vllm/test/curl_test.sh`
- `scripts/vllm/test/vllm_test.sh`
- `scripts/vllm/vllm_server_env_template.sh`

For each script, identify the exit code patterns and update them. This is a bulk find-and-replace operation.

- [ ] **Step 2: Verify**

Run: `bash -n` on each modified script.
Expected: No output for all.

Run: `shellcheck scripts/**/*.sh tools/*.sh examples/*.sh`
Expected: No new errors.

---

### Task 3.7: Round 3 commit

- [ ] **Step 1: Commit**

```bash
git add scripts/docker/manage_docker_containers.sh \
        scripts/ray_cluster/start_ray_cluster.sh \
        scripts/ray_cluster/kill_multi_nodes.sh \
        scripts/vllm/mp/deploy_vllm_multinode.sh \
        scripts/vllm/mp/deploy_vllm_multinode_mp.sh \
        examples/*.sh tools/*.sh scripts/docker/*.sh \
        scripts/ray_cluster/*.sh scripts/vllm/*.sh
git commit -m "refactor: unify node-list input and standardize exit codes (round 3)

- All multi-node scripts accept --file/-f for node list
- Standardize exit codes using E_* constants from common.sh
- Standardize vllm_model_server.sh logging with log_* functions"
```

---

## Round 4: Documentation

> **Commit:** `docs: add README files and script overview (round 4)`

---

### Task 4.1: Create `docs/scripts-overview.md`

**Files:**
- Create: `docs/scripts-overview.md`

- [ ] **Step 1: Write the file**

```markdown
# EasyInfer Script Overview

## Quick Reference

| Directory | Purpose | Key Scripts |
|-----------|---------|-------------|
| `scripts/docker/` | Docker container lifecycle | `manage_docker_containers.sh`, `ascend_infer_docker_run.sh` |
| `scripts/ray_cluster/` | Ray cluster orchestration | `start_ray_cluster.sh`, `stop_ray_cluster.sh`, `kill_multi_nodes.sh` |
| `scripts/vllm/` | vLLM model serving | `vllm_model_server.sh`, `mp/deploy_vllm_multinode.sh` |
| `tools/` | Auxiliary utilities | `hf_download.sh`, `docker_proxy.sh` |
| `examples/` | Per-model deployment examples | `glm5_server.sh`, `qwen3_server.sh` |

## Common CLI Patterns

All multi-node scripts accept:
- `--file <path>` / `-f <path>` — node list file (default: `scripts/node_list.txt`)

## Environment Variables

See individual module READMEs for complete variable lists. Key globals:
- `NODES_FILE` — cluster node list path
- `CONTAINER_NAME` — Docker container name
- `MODEL_PATH` — model weights directory
- `SSH_OPTS` — SSH options (word-split intentionally)
```

- [ ] **Step 2: Verify**

Run: `pre-commit run --all-files`
Expected: Pass (markdown files are not checked by shellcheck).

---

### Task 4.2: Create module READMEs

**Files:**
- Create: `scripts/docker/README.md`
- Create: `scripts/ray_cluster/README.md`
- Create: `scripts/vllm/README.md`
- Create: `examples/README.md`

- [ ] **Step 1: Write `scripts/docker/README.md`**

```markdown
# Docker Module

Scripts for managing Docker containers across the Ascend NPU cluster.

## Scripts

| Script | Purpose |
|--------|---------|
| `manage_docker_containers.sh` | Start/stop/restart containers on all nodes |
| `manage_npuslim_containers.sh` | Manage npuslim-specific containers |
| `ascend_infer_docker_run.sh` | Run inference Docker container (device mounts) |
| `ascend_train_docker_run.sh` | Run training Docker container (device mounts) |
| `copy_file_to_containers.sh` | Copy files into running containers |
| `run_npuslim_container.sh` | Run a single npuslim container |
| `docker_env.sh` | Environment variables for Docker module |

## Usage

```bash
bash scripts/docker/manage_docker_containers.sh start
bash scripts/docker/manage_docker_containers.sh stop
bash scripts/docker/manage_docker_containers.sh restart --file /path/to/nodes.txt
```

## Environment Variables

See `docker_env.sh` for all variables. Key ones:
- `NODES_FILE` — node list path
- `IMAGE_NAME`, `IMAGE_TAR` — Docker image
- `CONTAINER_NAME` — target container name
- `RUN_CONTAINER_SCRIPT` — script to start container
```

- [ ] **Step 2: Write `scripts/ray_cluster/README.md`**

```markdown
# Ray Cluster Module

Scripts for orchestrating Ray clusters across Docker containers on Ascend NPU nodes.

## Scripts

| Script | Purpose |
|--------|---------|
| `start_ray_cluster.sh` | Start Ray head + workers |
| `stop_ray_cluster.sh` | Stop Ray on all nodes |
| `kill_multi_nodes.sh` | Kill processes by keyword across nodes |
| `native_ray_start_cluster.sh` | Native (non-Docker) Ray startup |
| `start_npuslim_ray_cluster.sh` | Ray startup for npuslim containers |
| `set_ray_env.sh` | Ray/Ascend environment configuration |
| `_kill_lib.sh` | Shared kill-script utilities (sourced) |

## Usage

```bash
bash scripts/ray_cluster/start_ray_cluster.sh start --file nodes.txt
bash scripts/ray_cluster/start_ray_cluster.sh stop
bash scripts/ray_cluster/kill_multi_nodes.sh -y -k "ray,vllm"
```

## Environment Variables

See `set_ray_env.sh`. Key ones:
- `RAY_PORT` — Ray GCS port (default: 6379)
- `CONTAINER_NAME` — Docker container to exec into
- `NPUS_PER_NODE` — NPU count per node (default: 8)
```

- [ ] **Step 3: Write `scripts/vllm/README.md`**

```markdown
# vLLM Module

Scripts for serving LLMs with vLLM-Ascend on Ascend NPU clusters.

## Deployment Modes

| Mode | Script | Use Case |
|------|--------|----------|
| Single-node | `vllm_model_server.sh` | One node, TP=8 |
| Multi-node (Ray) | `mp/deploy_vllm_multinode.sh` | Ray backend, TP/PP across nodes |
| Multi-node (MP) | `mp/deploy_vllm_multinode_mp.sh` | Multiprocessing backend |

## Supporting Scripts

| Script | Purpose |
|--------|---------|
| `set_env.sh` | vLLM environment configuration |
| `vllm_server_env_template.sh` | Complete parameter template |
| `test/curl_test.sh` | API health check |
| `test/vllm_test.sh` | vLLM functionality test |
| `mp/_common.sh` | Shared multi-node deployment utilities |
| `mp/_node_env.sh` | Per-node environment template |

## Usage

```bash
# Single node
bash scripts/vllm/vllm_model_server.sh

# Multi-node Ray
bash scripts/vllm/mp/deploy_vllm_multinode.sh --file nodes.txt

# Multi-node MP
bash scripts/vllm/mp/deploy_vllm_multinode_mp.sh --file nodes.txt
```

## Key Environment Variables

- `MODEL_PATH` — model directory
- `TENSOR_PARALLEL_SIZE` — TP (default: 8)
- `PIPELINE_PARALLEL_SIZE` — PP (default: 1)
- `ENABLE_EXPERT_PARALLEL` — MoE EP (default: 1)
- `QUANTIZATION` — quantization method (default: fp8)
```

- [ ] **Step 4: Write `examples/README.md`**

```markdown
# Examples

Per-model deployment examples for EasyInfer.

## Available Examples

| Script | Model | Notes |
|--------|-------|-------|
| `glm5_server.sh` | GLM-5 | Standard deployment |
| `glm5_full_server.sh` | GLM-5 | Full precision |
| `glm5-1_quant_server.sh` | GLM-5.1 | Quantized |
| `qwen3_server.sh` | Qwen3 | |
| `kimi2_pcl.sh` | Kimi-K2 | |
| `longcat_flash-chat.sh` | LongCAT | |
| `curl_test.sh` | — | Generic API test |
| `lm_eval.sh` | — | lm-evaluation-harness |
| `check_glm5_env.sh` | — | Environment checker |

## Usage

All examples source `../scripts/common.sh` and `../scripts/vllm/set_env.sh`.

```bash
bash examples/glm5_server.sh
```
```

- [ ] **Step 5: Verify**

Run: `pre-commit run --all-files`
Expected: Pass.

---

### Task 4.3: Round 4 commit

- [ ] **Step 1: Commit**

```bash
git add docs/scripts-overview.md \
        scripts/docker/README.md \
        scripts/ray_cluster/README.md \
        scripts/vllm/README.md \
        examples/README.md
git commit -m "docs: add README files and script overview (round 4)"
```

---

## Final Verification

- [ ] **Step 1: Run all checks**

```bash
shellcheck scripts/**/*.sh tools/*.sh examples/*.sh
bash -n scripts/vllm/vllm_model_server.sh
pre-commit run --all-files
```

Expected: shellcheck shows only SC1091 (info). bash -n passes silently. pre-commit passes.

---

## Self-Review Checklist

### Spec Coverage

| Spec Requirement | Plan Task |
|------------------|-----------|
| Split `_remote_prepare_node` | Task 1.3 |
| Split `build_vllm_args_declare` | Task 1.4 |
| Shrink `kill_multi_nodes.sh` | Tasks 1.1, 1.2 |
| Remove duplicate `has_flag` | Task 1.5 |
| Remove duplicate `ssh_run_with_timeout` | Tasks 1.2, 2.4 |
| Enhance `ssh_run_timeout` | Task 2.1 |
| Add `parse_nodes_file_arg` | Task 2.2 |
| Add `wait_for_server` | Task 2.3 |
| Add `print_server_ready` | Task 2.3 |
| Add `require_env` | Task 2.3 |
| Extract vLLM shared logic | Tasks 2.7, 2.8, 2.9 |
| Unify `--file/-f` | Tasks 3.1-3.5 |
| Standardize exit codes | Tasks 3.1-3.6 |
| Standardize logging | Tasks 2.5, 3.1-3.6 |
| Documentation | Tasks 4.1-4.3 |

### Placeholder Scan

- No "TBD", "TODO", "implement later", "fill in details" found.
- No "Add appropriate error handling" or "handle edge cases" without specifics.
- All code blocks contain complete, runnable code.
- No "Similar to Task N" references.

### Type Consistency

- `ssh_run_timeout` signature unchanged across all tasks.
- `parse_nodes_file_arg` consistently returns path via stdout.
- `E_*` constants used consistently in Round 3.
