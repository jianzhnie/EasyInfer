# EasyInfer Script Library Optimization Design

## Overview

Optimize the EasyInfer shell script library (39 scripts, ~6,200 lines) across four independent rounds. Each round is self-contained, verifiable via `shellcheck` and `bash -n`, and introduces no behavioral changes until explicitly noted.

## Constraints (must not violate)

- `node_list.txt` parsing format: `awk 'NF && !/^#/ {print $1}'`
- `ssh_run` calling convention and `SSH_OPTS` word-split behavior
- Device/driver mount paths in `ascend_infer_docker_run.sh` and `ascend_train_docker_run.sh`
- `source` dependency chain: `common.sh` -> `docker_env.sh` / `set_ray_env.sh` / `set_env.sh`
- CLI parameters and env var names of all scripts
- Bash 4.2+ compatibility; no bash 4.3+ features (`declare -A`, namerefs)
- No new dependencies beyond bash 4+, coreutils, openssh, docker, ray, vllm

## Round 1: Defect Fixes + Function Splits

### Goal
Fix duplicate function definitions and split functions exceeding 50 lines. Shrink `kill_multi_nodes.sh` below 400 lines. No behavioral changes.

### Changes

#### `scripts/docker/manage_docker_containers.sh`
Split `_remote_prepare_node` (63 lines -> 3 helpers):
- `_remote_ensure_docker_running` — checks `docker` command, starts service if needed
- `_remote_cleanup_containers` — stop / kill / rm all existing containers
- `_remote_load_and_run` — load image tar, execute run script, verify container started

#### `scripts/vllm/mp/deploy_vllm_multinode_mp.sh`
Split `build_vllm_args_declare` (69 lines -> 5 helpers):
- `_build_base_args` — model path, host, port, TP/PP, memory settings
- `_build_mp_args` — nnodes, node-rank, master-addr (multiprocessing backend)
- `_build_dp_args` — data-parallel sizing, headless flag, rank assignment
- `_build_a2_compile_args` — compilation-config, additional-config, speculative-config
- `build_vllm_args_declare` — orchestrates the above 4 helpers

#### `scripts/ray_cluster/kill_multi_nodes.sh` (498 lines -> ~200 lines)
Extract into `scripts/ray_cluster/_kill_lib.sh` (sourced library, not executable):
- `escape_regex`, `_build_kill_pattern`, `_gen_kill_remote_script`
- `_parse_kill_status`, `_log_kill_status`, `_parse_and_log_kill_result`

`kill_multi_nodes.sh` retains:
- `usage`, `parse_args`
- `cleanup_jobs`, `confirm_operation`, `print_summary`
- `kill_processes_on_node` (thin wrapper)
- Main loop and result aggregation

#### Minor fixes
- Remove duplicate `has_flag()` from `vllm_model_server.sh` (use `common.sh` version)
- Remove duplicate `ssh_run_with_timeout()` from `kill_multi_nodes.sh` (merge Perl fallback into `common.sh`)

## Round 2: Deduplication into `scripts/common.sh`

### Goal
Extract duplicated patterns into `common.sh`, update all callers.

### New utilities in `common.sh`

1. **`parse_nodes_file_arg "$@"`**
   - Standardizes `--file/-f` parsing across all scripts
   - Returns resolved node-list path or `${NODES_FILE:-scripts/node_list.txt}`
   - Usage: `NODE_LIST=$(parse_nodes_file_arg "$@")`

2. **`ssh_run_timeout` enhancement**
   - Merge the Perl `alarm` fallback from `kill_multi_nodes.sh`
   - Single implementation: prefers `timeout` command, falls back to `perl` alarm, then bare ssh
   - Signature unchanged: `ssh_run_timeout <timeout_sec> <node> <cmd...>`

3. **`wait_for_server <host> <port> [timeout_sec]`**
   - Moved from `examples/_common.sh` to `common.sh`
   - Uses `wait_for_port` internally, adds HTTP health-check loop

4. **`print_server_ready <host> <port> [model_name]`**
   - Moved from `examples/_common.sh` to `common.sh`
   - Prints "server is ready" banner with endpoints and optional Claude Code config

5. **`require_env <var> [description]`**
   - Wrapper around `: "${VAR:?msg}"` with `log_fatal` formatting
   - Usage: `require_env MODEL_PATH "模型权重路径"`

### Update callers

- `kill_multi_nodes.sh` — use enhanced `ssh_run_timeout()`, drop local Perl fallback
- `vllm_model_server.sh` — source `common.sh`, use `log_*` instead of raw `echo`, use `has_flag` from `common.sh`
- `examples/_common.sh` — source `common.sh`, drop duplicated `wait_for_server` and `print_server_ready`

### Extract shared vLLM deployment logic

Both `deploy_vllm_multinode.sh` and `deploy_vllm_multinode_mp.sh` share ~80 lines of identical node-loading, config-validation, IP-detection, and SSH-check logic. Extract into `scripts/vllm/mp/_common.sh`:

- `load_and_validate_nodes <file> <min_nodes>` — reads node list, validates minimum node count
- `validate_parallelism_config <total_nodes> <npus_per_node>` — computes TOTAL_CARDS, DP_SIZE, validates divisibility constraints
- `resolve_node0_ip <node0> <nic_name>` — gets NODE0_IP, fatal-on-failure

## Round 3: Unify Input + Standardize Output

### Goal
All multi-node scripts accept `--file/-f` for node lists. Standardize exit codes and logging.

### Node-list input unification

Every multi-node script accepts `--file <path>` / `-f <path>` (or positional argument), falling back to `${NODES_FILE:-scripts/node_list.txt}`.

| Script | Current state | Change |
|--------|--------------|--------|
| `manage_docker_containers.sh` | Already has `--file/-f` | Tighten to use `parse_nodes_file_arg` |
| `start_ray_cluster.sh` | Already has `--file` | Add `-f` short form, use `parse_nodes_file_arg` |
| `kill_multi_nodes.sh` | Accepts positional arg | Add `--file/-f` alias, keep positional for compat |
| `deploy_vllm_multinode.sh` | Hardcodes `../../node_list.txt` | Add `--file/-f`, use `parse_nodes_file_arg` |
| `deploy_vllm_multinode_mp.sh` | Hardcodes `../../node_list.txt` | Add `--file/-f`, use `parse_nodes_file_arg` |

### Exit code standardization

Use `E_*` constants already defined in `common.sh`:
- `E_INVALID_ARG=2` — bad CLI arguments
- `E_NOT_FOUND=3` — missing files, directories
- `E_TIMEOUT=124` — SSH or service timeouts
- `E_CMD_NOT_FOUND=127` — missing required commands
- `E_GENERAL=1` — other failures

Replace bare `exit 1` / `exit 2` across all scripts with appropriate constant.

### Logging standardization

- `vllm_model_server.sh` — convert raw `echo "[INFO]"` / `echo "[ERROR]"` to `log_info` / `log_err` / `log_warn` / `log_fatal` from `common.sh`
- All other scripts already use `log_*`; no changes needed

## Round 4: Documentation

### New files
- `docs/scripts-overview.md` — index of all scripts, purpose, CLI interface, and key env vars
- `scripts/docker/README.md` — docker container lifecycle scripts
- `scripts/ray_cluster/README.md` — Ray cluster orchestration scripts
- `scripts/vllm/README.md` — vLLM model serving scripts (single-node, multi-node Ray, multi-node MP)
- `examples/README.md` — per-model deployment examples

### Updated files
- `docs/claude-code-vllm-setup.md` — update if CLI interfaces changed
- `docs/reverse_proxy_setup.md` — update if port/env var references changed

## Verification

Each round must pass:

```bash
# Static analysis
shellcheck scripts/**/*.sh tools/*.sh examples/*.sh
bash -n scripts/vllm/vllm_model_server.sh

# Full pre-commit suite
pre-commit run --all-files
```

## Rollback Strategy

Each round is an independent git commit. If a round causes issues, revert that single commit. No round depends on changes from a later round.
