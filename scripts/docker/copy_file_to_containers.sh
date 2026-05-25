#!/bin/bash
# ==========================================
# Docker 容器文件复制脚本 (copy_file_to_containers.sh)
# 将文件复制到集群各节点的 Docker 容器内
# ==========================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/docker_env.sh"

# 加载共享工具函数
source "${SCRIPT_DIR}/../common.sh"

# ------------------------------------------
# 引入环境变量
# ------------------------------------------
if [[ -f "${ENV_FILE}" ]]; then
  # shellcheck source=/dev/null
  source "${ENV_FILE}"
else
  log_err "环境配置文件未找到: ${ENV_FILE}"
  exit 1
fi

# ------------------------------------------
# 帮助信息
# ------------------------------------------
usage() {
  cat <<'USAGE'
Usage:
  bash copy_to_docker.sh [OPTIONS] <source_file> <dest_path_in_container>

Description:
  将文件复制到集群各节点的 Docker 容器内。
  节点列表从 node_list.txt 读取，环境变量在 set_env.sh 中配置。

Arguments:
  source_file              源文件路径
  dest_path_in_container   容器内的目标路径（绝对路径）

Options:
  -h, --help               显示帮助信息
  -n, --node <node>        仅复制到指定节点（可多次使用）
  -p, --parallel <num>     并发数（默认: 8）
  -r, --remote             源文件在远程节点上（直接在远程执行 docker cp）

Examples:
  # 本地文件复制到所有节点的容器（先 scp 到远程，再 docker cp）
  bash copy_to_docker.sh ./local_file.py /container/path/file.py

  # 远程文件复制到容器（源文件已在远程节点上）
  bash copy_to_docker.sh -r /llm_workspace_1P/wf/deepseek_v2.py /vllm-workspace/vllm/vllm/model_executor/models/deepseek_v2.py

  # 仅复制到指定节点
  bash copy_to_docker.sh -n bms1905 -n bms1906 ./file.txt /container/path/file.txt

  # 批量复制多个文件（使用配置文件）
  bash copy_to_docker.sh -r -c copy_files.conf
USAGE
}

# ------------------------------------------
# 参数解析
# ------------------------------------------
PARALLELISM_ARG=""
SPECIFIC_NODES=()
REMOTE_MODE=false
CONFIG_FILE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    -n|--node)
      if [[ -z "${2:-}" || "$2" == -* ]]; then
          log_err "选项 $1 需要一个参数: <node>"
          exit 1
      fi
      SPECIFIC_NODES+=("$2")
      shift 2
      ;;
    -p|--parallel)
      if [[ -z "${2:-}" || "$2" == -* ]]; then
          log_err "选项 $1 需要一个参数: <num>"
          exit 1
      fi
      PARALLELISM_ARG="$2"
      shift 2
      ;;
    -r|--remote)
      REMOTE_MODE=true
      shift
      ;;
    -c|--config)
      if [[ -z "${2:-}" || "$2" == -* ]]; then
          log_err "选项 $1 需要一个参数: <config_file>"
          exit 1
      fi
      CONFIG_FILE="$2"
      shift 2
      ;;
    -*)
      log_err "未知选项: $1"
      usage
      exit 1
      ;;
    *)
      break
      ;;
  esac
done

# 设置并发数
PARALLELISM="${PARALLELISM_ARG:-${PARALLELISM:-8}}"

# ------------------------------------------
# 文件复制函数 - 远程模式（文件已在远程节点）
# ------------------------------------------
copy_remote_to_docker() {
    local node="$1" source="$2" dest="$3"
    local container="${CONTAINER_NAME:-vllm-ascend-env-a3}"

    if ! ssh_run "$node" "docker cp '${source}' '${container}:${dest}'"; then
        log_err "[${node}] docker cp 失败: ${source} -> ${container}:${dest}"
        return 1
    fi
    log_info "[${node}] OK"
}

# ------------------------------------------
# 文件复制函数 - 本地模式（文件在本地）
# ------------------------------------------
copy_local_to_docker() {
    local node="$1" source="$2" dest="$3"
    local container="${CONTAINER_NAME:-vllm-ascend-env-a3}"
    local target
    target=$(ssh_target "$node")
    local temp_remote_file
    temp_remote_file="/tmp/copy_to_docker_$(basename "$source").$$.$(date +%s%N)"

    # shellcheck disable=SC2086
    if ! scp ${SSH_OPTS:--o BatchMode=yes -o ConnectTimeout=10} "$source" "${target}:${temp_remote_file}" 2>/dev/null; then
        log_err "[${node}] scp 传输失败"
        return 1
    fi

    if ! ssh_run "$node" "docker cp '${temp_remote_file}' '${container}:${dest}' && rm -f '${temp_remote_file}'"; then
        ssh_run "$node" "rm -f '${temp_remote_file}'" 2>/dev/null || true
        log_err "[${node}] docker cp 失败"
        return 1
    fi
    log_info "[${node}] OK"
}

# ------------------------------------------
# 批量复制（配置文件模式）
# ------------------------------------------
run_batch_copy() {
    local config_file="$1"

    if [[ ! -f "$config_file" ]]; then
        log_err "配置文件不存在: $config_file"
        exit 2
    fi

    log_info "使用配置文件: $config_file"

    local line_num=0
    while IFS='|' read -r source dest || [[ -n "$source" ]]; do
        ((line_num++))

        # 跳过空行和注释
        [[ -z "$source" || "$source" == \#* ]] && continue

        # 去除前后空格
        source=$(echo "$source" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
        dest=$(echo "$dest" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')

        log_info "复制任务 #${line_num}: ${source} -> ${dest}"

        for node in $nodes; do
            limit_jobs "$PARALLELISM"
            if $REMOTE_MODE; then
                (copy_remote_to_docker "$node" "$source" "$dest") &
            else
                (copy_local_to_docker "$node" "$source" "$dest") &
            fi
        done
        wait
    done < "$config_file"
}

# ------------------------------------------
# 单文件复制
# ------------------------------------------
run_single_copy() {
  local source="$1"
  local dest="$2"

  log_info "源文件: $source"
  log_info "容器内目标: $dest"

  if ! $REMOTE_MODE; then
    # 本地模式：检查源文件
    if [[ ! -f "$source" ]]; then
      log_err "源文件不存在: $source"
      exit 2
    fi
  fi

  for node in $nodes; do
    limit_jobs "$PARALLELISM"
    if $REMOTE_MODE; then
      (copy_remote_to_docker "$node" "$source" "$dest") &
    else
      (copy_local_to_docker "$node" "$source" "$dest") &
    fi
  done
  wait
}

# ------------------------------------------
# 主流程入口
# ------------------------------------------

# 获取节点列表
if [[ ${#SPECIFIC_NODES[@]} -gt 0 ]]; then
  nodes="${SPECIFIC_NODES[*]}"
  log_info "指定节点: $nodes"
else
  nodes="$(read_nodes "$NODES_FILE")"
  if [[ -z "$nodes" ]]; then
    log_err "NODES_FILE 中未找到任何节点信息"
    exit 2
  fi
  log_info "目标节点: $nodes"
fi

log_info "容器名称: ${CONTAINER_NAME:-vllm-ascend-env-a3}"
log_info "并发数: $PARALLELISM"
log_info "模式: $([[ "$REMOTE_MODE" == true ]] && echo "远程文件" || echo "本地文件")"
log_info "=== 开始复制 ==="

# 判断是配置文件模式还是单文件模式
if [[ -n "$CONFIG_FILE" ]]; then
  run_batch_copy "$CONFIG_FILE"
elif [[ $# -ge 2 ]]; then
  run_single_copy "$1" "$2"
else
  log_err "缺少必需参数"
  usage
  exit 1
fi

log_info "=== 复制完成 ==="
