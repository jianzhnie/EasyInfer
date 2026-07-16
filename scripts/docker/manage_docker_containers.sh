#!/bin/bash
# ==========================================
# Docker 容器集群管理脚本（并行版）
# 用于在集群节点上管理 Docker 容器环境 (start/stop/restart)
# ==========================================

# 注意: 顶层不设 set -e，因为本脚本使用后台并行任务模型（& + wait），
# set -e 在函数内的后台任务失败时会意外退出整个脚本。
# 各 _remote_ 前缀的函数内部自行设置 set -eo pipefail。
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 加载共享库（日志、SSH、并发、节点解析）
source "${SCRIPT_DIR}/../common.sh"

# ------------------------------------------
# 帮助信息
# ------------------------------------------
usage() {
    cat <<'USAGE'
Usage:
  bash manage_docker_containers.sh [start|stop|restart] [OPTIONS]

操作:
  start     确保 Docker 可用，加载镜像并启动容器（默认）
  stop      仅停止并清理旧容器，不启动新容器
  restart   停止并清理旧容器，加载镜像并启动新容器

Options:
  -h, --help            显示帮助信息
  -f, --file <FILE>     节点列表文件路径（必传）
  -j, --jobs <N>        并行度（默认: 8，可通过 PARALLELISM 环境变量设置）
  -r, --retries <N>     失败重试次数（默认: 1）
  --image <IMAGE>       容器镜像（覆盖 docker_env.sh 中的 IMAGE_NAME）
  --name <NAME>         容器名称（覆盖 docker_env.sh 中的 CONTAINER_NAME）
  --timeout <SEC>       SSH 操作超时秒数（默认: 600）
  --keep-logs           执行完毕后保留临时日志目录

环境变量配置: scripts/docker/docker_env.sh
USAGE
}

# ------------------------------------------
# 参数解析与初始化
# ------------------------------------------
ACTION="start"
RETRIES=1
KEEP_LOGS=0
SSH_TIMEOUT="${SSH_TIMEOUT:-600}"
# ascend_infer_docker_run.sh 已内置 docker run -d，--daemon 对它无操作（无害）；
# run_npuslim_container.sh 则需要 --daemon 来以分离模式运行。
CONTAINER_RUN_EXTRA_ARGS="${CONTAINER_RUN_EXTRA_ARGS:---daemon}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        -f|--file)
            NODES_FILE="${2:?错误: $1 需要一个参数}"
            shift 2
            ;;
        -j|--jobs)
            PARALLELISM="${2:?错误: $1 需要一个参数}"
            shift 2
            ;;
        -r|--retries)
            RETRIES="${2:?错误: $1 需要一个参数}"
            shift 2
            ;;
        --keep-logs) KEEP_LOGS=1; shift ;;
        --image)
            IMAGE_NAME="${2:?错误: $1 需要一个参数}"
            shift 2
            ;;
        --name)
            CONTAINER_NAME="${2:?错误: $1 需要一个参数}"
            shift 2
            ;;
        --timeout)
            SSH_TIMEOUT="${2:?错误: $1 需要一个参数}"
            shift 2
            ;;
        start|stop|restart|status) ACTION="$1"; shift ;;
        *)
            log_err "未知参数: $1"; usage; exit "$E_INVALID_ARG"
            ;;
    esac
done

# 加载环境配置（验证失败则退出）
ENV_FILE="${SCRIPT_DIR}/docker_env.sh"
if [[ ! -f "${ENV_FILE}" ]]; then
    log_err "环境配置文件未找到: ${ENV_FILE}"
    exit 1
fi
# shellcheck source=./docker_env.sh
source "${ENV_FILE}"

# 验证必要环境变量（所有动作都需要）
: "${NODES_FILE:?环境变量 NODES_FILE 未设置}"
: "${CONTAINER_NAME:?环境变量 CONTAINER_NAME 未设置}"
: "${IMAGE_NAME:?环境变量 IMAGE_NAME 未设置}"

# ------------------------------------------
# 前置依赖检查
# ------------------------------------------
require_cmds ssh awk

if [[ ! -f "$NODES_FILE" ]]; then
    log_err "节点列表文件未找到: $NODES_FILE"
    exit "$E_NOT_FOUND"
fi

if [[ "$ACTION" != "stop" && "$ACTION" != "status" ]]; then
    : "${IMAGE_TAR:?环境变量 IMAGE_TAR 未设置}"
    : "${RUN_CONTAINER_SCRIPT:?环境变量 RUN_CONTAINER_SCRIPT 未设置}"
    if [[ ! -f "$IMAGE_TAR" ]]; then
        log_err "镜像文件未找到: $IMAGE_TAR"
        exit "$E_NOT_FOUND"
    fi
    if [[ ! -f "$RUN_CONTAINER_SCRIPT" ]]; then
        log_err "启动脚本未找到: $RUN_CONTAINER_SCRIPT"
        exit "$E_NOT_FOUND"
    fi
fi

# ------------------------------------------
# 远程执行函数
# 注意: 下列 _remote_ 前缀的函数通过 declare -f 序列化后发送到远端节点执行
# ------------------------------------------

_remote_ensure_docker_running() {
    if ! command -v docker >/dev/null 2>&1; then
        echo "[ERROR] docker command not found" >&2
        return 127
    fi
    if docker info >/dev/null 2>&1; then
        return 0
    fi
    echo "[INFO] Docker service not running, attempting to start..."
    systemctl daemon-reload || true
    systemctl start docker
    # 重试等待 Docker 就绪（最多 30 秒，每 2 秒检查一次）
    local attempt=1 max_attempts=15
    while [[ $attempt -le $max_attempts ]]; do
        if docker info >/dev/null 2>&1; then
            return 0
        fi
        sleep 2
        attempt=$((attempt + 1))
    done
    echo "[ERROR] Failed to start docker service after ${max_attempts} attempts" >&2
    return 1
}

_remote_cleanup_containers() {
    local container_name="${1:-}"
    if [[ -n "$container_name" ]]; then
        if docker ps -aq -f "name=^/${container_name}$" | grep -q .; then
            echo "[INFO] Stopping existing container: ${container_name}"
            docker stop -t 10 "${container_name}" 2>/dev/null || true
            docker rm "${container_name}" 2>/dev/null || true
        fi
    else
        echo "[INFO] Stopping and removing all existing containers..."
        local containers
        containers="$(docker ps -aq 2>/dev/null)"
        [[ -z "$containers" ]] && return 0
        echo "$containers" | xargs docker stop -t 10 2>/dev/null || true
        echo "$containers" | xargs docker rm 2>/dev/null || true
    fi
}

_remote_load_and_run() {
    local image_name="$1" image_tar="$2" run_container_script="$3" container_name="$4"

    if ! docker image inspect "${image_name}" >/dev/null 2>&1; then
        if [[ ! -f "${image_tar}" ]]; then
            echo "[ERROR] Image tar not found: ${image_tar}" >&2
            return 2
        fi
        # 校验镜像 tar 完整性（若存在 .sha256 校验文件）
        local sha256_file="${image_tar}.sha256"
        if [[ -f "${sha256_file}" ]]; then
            echo "[INFO] Verifying image tar checksum..."
            if ! sha256sum -c "${sha256_file}" --status 2>/dev/null; then
                echo "[ERROR] Image tar checksum mismatch: ${image_tar}" >&2
                echo "  期望: $(cat "${sha256_file}")" >&2
                echo "  实际: $(sha256sum "${image_tar}" | awk '{print $1}')" >&2
                return 2
            fi
            echo "[INFO] Checksum verified"
        fi
        echo "[INFO] Loading image from ${image_tar}..."
        docker load -i "${image_tar}"
        echo "[INFO] Image loaded successfully"
    else
        echo "[INFO] Image already exists: ${image_name}"
    fi

    if [[ ! -f "${run_container_script}" ]]; then
        echo "[ERROR] Run script not found: ${run_container_script}" >&2
        return 2
    fi

    _remote_cleanup_containers "${container_name}"

    echo "[INFO] Starting container: ${container_name}"
    # CONTAINER_RUN_EXTRA_ARGS 经分词传递多个 flag（如 --daemon），有意不加引号
    # shellcheck disable=SC2086
    IMAGE_NAME="${image_name}" CONTAINER_NAME="${container_name}" bash "${run_container_script}" ${CONTAINER_RUN_EXTRA_ARGS}

    # 使用 docker inspect 验证容器状态（比 sleep + docker ps 更可靠，无竞态）
    if docker inspect -f '{{.State.Running}}' "${container_name}" 2>/dev/null | grep -q "true"; then
        echo "[INFO] Container ready: ${container_name}"
        echo "  进入容器: docker exec -it ${container_name} bash"
        echo "  查看日志: docker logs ${container_name}"
        echo "  停止容器: docker stop ${container_name} && docker rm ${container_name}"
    else
        echo "[ERROR] Failed to start container: ${container_name}" >&2
        docker logs "${container_name}" 2>&1 | tail -10 || true
        return 1
    fi
}

_remote_check_status() {
    local container_name="${1:-}"
    echo "=== 容器状态: ${container_name} ==="
    if docker inspect -f '{{.State.Status}}' "$container_name" 2>/dev/null; then
        echo "  镜像:   $(docker inspect -f '{{.Config.Image}}' "$container_name" 2>/dev/null)"
        echo "  PID:    $(docker inspect -f '{{.State.Pid}}' "$container_name" 2>/dev/null)"
        echo "  启动:   $(docker inspect -f '{{.State.StartedAt}}' "$container_name" 2>/dev/null)"
        echo "  运行中: $(docker inspect -f '{{.State.Running}}' "$container_name" 2>/dev/null)"
    else
        echo "  容器 '${container_name}' 不存在"
        return 1
    fi
}

_remote_prepare_node() {
    local image_name="$1" image_tar="$2" run_container_script="$3" container_name="$4"
    local action="${5:-start}"

    set -eo pipefail
    _remote_ensure_docker_running

    case "$action" in
        stop)
            _remote_cleanup_containers
            echo "[INFO] All containers stopped"
            ;;
        restart)
            _remote_cleanup_containers
            _remote_load_and_run "${image_name}" "${image_tar}" "${run_container_script}" "${container_name}"
            ;;
        start)
            _remote_load_and_run "${image_name}" "${image_tar}" "${run_container_script}" "${container_name}"
            ;;
    esac
}

# ------------------------------------------
# status 动作: 查询集群各节点容器状态
# ------------------------------------------
if [[ "$ACTION" == "status" ]]; then
    nodes="$(read_nodes "$NODES_FILE")"
    if [[ -z "$nodes" ]]; then
        log_err "NODES_FILE 中未找到任何节点信息"
        exit "$E_NOT_FOUND"
    fi
    mapfile -t NODE_ARRAY <<< "$nodes"
    NODE_COUNT=${#NODE_ARRAY[@]}

    log_info "============================================"
    log_info " 容器: ${CONTAINER_NAME}"
    log_info " 动作: status"
    log_info " 节点数: ${NODE_COUNT}"
    log_info "============================================"
    log_info ""

    check_node_status() {
        local node="$1"
        local func_code call_code b64
        func_code="$(declare -f _remote_check_status)"
        printf -v call_code '_remote_check_status %q' "${CONTAINER_NAME}"
        b64="$(printf '%s\n%s\n' "${func_code}" "${call_code}" | base64 | tr -d '\n')"

        if ! ssh_run_timeout 30 "$node" "echo '${b64}' | base64 -d | bash -l"; then
            echo "[${node}] SSH 连接失败或超时"
            return 1
        fi
    }

    for node in "${NODE_ARRAY[@]}"; do
        limit_jobs "$MAX_PARALLEL"
        echo ""
        log_info "[${node}] 查询中..."
        check_node_status "$node" &
    done
    wait

    log_info ""
    log_info "=== 状态查询完成 ==="
    exit 0
fi

# ------------------------------------------
# 主控: 节点调度 (start/stop/restart)
# ------------------------------------------

prepare_node() {
    local node="$1"
    log_info "[${node}] 开始环境准备..."

    local func_code call_code b64
    func_code="$(declare -f _remote_ensure_docker_running _remote_cleanup_containers \
        _remote_load_and_run _remote_prepare_node)"
    printf -v call_code '_remote_prepare_node %q %q %q %q %q' \
        "${IMAGE_NAME}" "${IMAGE_TAR}" "${RUN_CONTAINER_SCRIPT}" "${CONTAINER_NAME}" "${ACTION}"
    b64="$(printf '%s\n%s\n' "${func_code}" "${call_code}" | base64 | tr -d '\n')"

    if ! ssh_run_timeout "$SSH_TIMEOUT" "$node" "echo '${b64}' | base64 -d | bash -l"; then
        log_err "[${node}] 环境准备失败（超时或执行错误）"
        return 1
    fi
    log_info "[${node}] 环境准备完成"
}

# ------------------------------------------
# 主流程入口
# ------------------------------------------

nodes="$(read_nodes "$NODES_FILE")"
if [[ -z "$nodes" ]]; then
    log_err "NODES_FILE 中未找到任何节点信息"
    exit "$E_NOT_FOUND"
fi

mapfile -t NODE_ARRAY <<< "$nodes"
NODE_COUNT=${#NODE_ARRAY[@]}
MAX_PARALLEL="${PARALLELISM:-8}"
LOG_DIR=$(mktemp -d "/tmp/docker_deploy.XXXXXX")

if [[ "$KEEP_LOGS" -eq 0 ]]; then
    trap 'rm -rf "$LOG_DIR"' EXIT
fi

START_TIME=$(date +%s)

log_info "============================================"
log_info " 容器: ${CONTAINER_NAME}"
log_info " 镜像: ${IMAGE_NAME}"
log_info " 动作: ${ACTION}"
log_info " 节点数: ${NODE_COUNT}"
log_info " 并行度: ${MAX_PARALLEL}"
log_info " 重试次数: ${RETRIES}"
log_info " 日志目录: ${LOG_DIR}"
log_info "============================================"
log_info ""
log_info "=== 开始并行准备节点 ==="

run_node_with_retry() {
    local node="$1"
    local logfile="${LOG_DIR}/${node}.log"
    local attempt=1
    local rc=1

    while [[ $attempt -le $RETRIES ]]; do
        if [[ $attempt -gt 1 ]]; then
            echo "[RETRY] 第 ${attempt}/${RETRIES} 次重试" >> "$logfile"
            sleep 3
        fi
        prepare_node "$node" >> "$logfile" 2>&1
        rc=$?
        [[ $rc -eq 0 ]] && break
        attempt=$((attempt + 1))
    done

    echo "$rc" > "${LOG_DIR}/${node}.rc"
    return $rc
}

for node in "${NODE_ARRAY[@]}"; do
    limit_jobs "$MAX_PARALLEL"
    run_node_with_retry "$node" &
done

wait

# ------------------------------------------
# 结果汇总
# ------------------------------------------
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
ELAPSED_MIN=$((ELAPSED / 60))
ELAPSED_SEC=$((ELAPSED % 60))

log_info ""
log_info "=== 执行结果汇总 (耗时: ${ELAPSED_MIN}m${ELAPSED_SEC}s) ==="

failed=0
succeeded=0
FAILED_NODES=()

for node in "${NODE_ARRAY[@]}"; do
    rc_file="${LOG_DIR}/${node}.rc"
    logfile="${LOG_DIR}/${node}.log"
    rc=1
    [[ -f "$rc_file" ]] && rc=$(cat "$rc_file")
    if [[ "$rc" -eq 0 ]]; then
        log_info "[${node}] ✓ 成功"
        succeeded=$((succeeded + 1))
    else
        log_err "[${node}] ✗ 失败"
        failed=$((failed + 1))
        FAILED_NODES+=("$node")
        if [[ -f "$logfile" ]]; then
            log_err "[${node}] 错误日志:"
            sed 's/^/    /' "$logfile" >&2
        fi
    fi
done

log_info ""
log_info "成功: ${succeeded}/${NODE_COUNT}  失败: ${failed}/${NODE_COUNT}  耗时: ${ELAPSED_MIN}m${ELAPSED_SEC}s"

if [[ "$failed" -gt 0 ]]; then
    log_err "失败节点: ${FAILED_NODES[*]}"
    log_err "日志目录: ${LOG_DIR}"
    KEEP_LOGS=1
    trap - EXIT
    exit 1
fi

log_info "=== 所有节点准备完成 ==="
