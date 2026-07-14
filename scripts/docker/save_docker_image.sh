#!/bin/bash
# save_docker_image.sh — 将 Docker 镜像导出为 tar 文件，可选分发到集群节点。
#
# 用法:
#   bash save_docker_image.sh -i <镜像名> [OPTIONS]
#
# Options:
#   -h, --help          显示帮助信息
#   -i, --image <NAME>  Docker 镜像名称（必填，也可通过环境变量 IMAGE_NAME 设置）
#   -o, --output <FILE> 输出 tar 文件路径（默认: ${IMAGE_DIR}/<镜像名>.tar）
#   -z, --gzip          使用 gzip 压缩（生成 .tar.gz）
#   -f, --file <FILE>   节点列表文件，提供后将 tar 分发到各节点
#   --no-cleanup        分发后保留本地 tar 文件（默认分发后删除）
#
# 环境变量 (均可外部覆盖):
#   IMAGE_NAME, IMAGE_TAR, IMAGE_DIR

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 加载共享库（日志、SSH、节点解析）
# shellcheck source=../common.sh
source "${SCRIPT_DIR}/../common.sh"

# 加载默认镜像配置（IMAGE_NAME / IMAGE_TAR / IMAGE_DIR）
ENV_FILE="${SCRIPT_DIR}/docker_env.sh"
if [[ -f "${ENV_FILE}" ]]; then
    # shellcheck source=./docker_env.sh
    source "${ENV_FILE}"
fi

# ------------------------------------------
# 默认值（环境变量或 docker_env.sh 已设置的会被保留）
# ------------------------------------------
IMAGE_DIR="${IMAGE_DIR:-/home/jianzhnie/llmtuner/hfhub/docker/image}"

# ------------------------------------------
# 帮助信息
# ------------------------------------------
usage() {
    cat <<'USAGE'
Usage:
  bash save_docker_image.sh -i <IMAGE> [OPTIONS]

将 Docker 镜像导出为 tar 文件，可选分发到集群节点。

Options:
  -h, --help          显示帮助信息
  -i, --image <NAME>  Docker 镜像名称（必填，也可通过 IMAGE_NAME 环境变量设置）
  -o, --output <FILE> 输出 tar 文件路径（默认: ${IMAGE_DIR}/<镜像名>.tar）
  -z, --gzip          使用 gzip 压缩（生成 .tar.gz 文件）
  -f, --file <FILE>   节点列表文件，提供后将 tar 分发到各节点
  --no-cleanup        分发后保留本地 tar 文件

Examples:
  # 使用 docker_env.sh 中的默认镜像名导出
  bash save_docker_image.sh

  # 指定镜像导出
  bash save_docker_image.sh -i quay.io/ascend/vllm-ascend:v0.22.1rc1-a3

  # 导出并压缩
  bash save_docker_image.sh -i myimage:latest -z

  # 导出并分发到集群所有节点
  bash save_docker_image.sh -i myimage:latest -f node_list.txt
USAGE
}

# ------------------------------------------
# 参数解析
# ------------------------------------------
OUTPUT_FILE=""
USE_GZIP=0
NODES_FILE=""
NO_CLEANUP=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        -i|--image)
            IMAGE_NAME="${2:?错误: $1 需要一个参数}"
            shift 2
            ;;
        -o|--output)
            OUTPUT_FILE="${2:?错误: $1 需要一个参数}"
            shift 2
            ;;
        -z|--gzip) USE_GZIP=1; shift ;;
        -f|--file)
            NODES_FILE="${2:?错误: $1 需要一个参数}"
            shift 2
            ;;
        --no-cleanup) NO_CLEANUP=1; shift ;;
        *)
            log_err "未知参数: $1"
            usage
            exit "$E_INVALID_ARG"
            ;;
    esac
done

# 验证镜像名称（可能已通过 docker_env.sh 或 -i 设置）
if [[ -z "${IMAGE_NAME:-}" ]]; then
    log_err "镜像名称未指定，请使用 -i/--image 或设置 IMAGE_NAME 环境变量"
    usage
    exit "$E_INVALID_ARG"
fi

# 确定输出路径（未通过 -o 指定时自动生成）
if [[ -z "${OUTPUT_FILE}" ]]; then
    safe_name="${IMAGE_NAME//\//_}"
    safe_name="${safe_name//:/_}"
    if [[ "${USE_GZIP}" -eq 1 ]]; then
        OUTPUT_FILE="${IMAGE_DIR}/${safe_name}.tar.gz"
    else
        OUTPUT_FILE="${IMAGE_DIR}/${safe_name}.tar"
    fi
fi

# 验证节点列表文件（指定了但不存在）
if [[ -n "${NODES_FILE}" && ! -f "${NODES_FILE}" ]]; then
    log_err "节点列表文件未找到: ${NODES_FILE}"
    exit "$E_NOT_FOUND"
fi

# ------------------------------------------
# 前置检查
# ------------------------------------------
require_cmd docker

if ! docker image inspect "${IMAGE_NAME}" >/dev/null 2>&1; then
    log_err "Docker 镜像不存在: ${IMAGE_NAME}"
    log_info "可用镜像列表:"
    docker images --format '  {{.Repository}}:{{.Tag}}' || true
    exit "$E_NOT_FOUND"
fi

# ------------------------------------------
# 导出镜像
# ------------------------------------------
save_image() {
    local image="$1" output="$2" use_gzip="$3" size

    log_info "正在导出镜像: ${image}"
    mkdir -p "$(dirname "${output}")"

    if [[ "${use_gzip}" -eq 1 ]]; then
        log_info "导出并压缩到: ${output}"
        docker save "${image}" | gzip > "${output}"
    else
        log_info "导出到: ${output}"
        docker save -o "${output}" "${image}"
    fi

    size="$(du -h "${output}" | cut -f1)"
    log_info "导出完成: ${output} (${size})"
}

# ------------------------------------------
# 分发到集群节点
# ------------------------------------------
distribute_to_nodes() {
    local tar_file="$1" nodes_file="$2" nodes failed remote_dir

    require_cmd scp
    nodes="$(read_nodes "${nodes_file}")"
    if [[ -z "${nodes}" ]]; then
        log_err "节点列表为空: ${nodes_file}"
        return 1
    fi

    log_info "目标节点: ${nodes}"
    log_info "开始分发镜像文件..."

    failed=0
    for node in ${nodes}; do
        log_info "[${node}] 正在传输 ${tar_file}..."
        remote_dir="$(dirname "${tar_file}")"
        ssh_run "${node}" "mkdir -p ${remote_dir}"
        # SSH_OPTS 通过词分割传递，SCP 复用相同的 SSH 选项
        # shellcheck disable=SC2086
        if scp ${SSH_OPTS:-} "${tar_file}" "$(ssh_target "${node}"):${tar_file}"; then
            log_info "[${node}] 传输完成"
        else
            log_err "[${node}] 传输失败"
            ((failed++)) || true
        fi
    done

    if [[ "${failed}" -gt 0 ]]; then
        log_err "部分节点传输失败 (${failed} 个)"
        return 1
    fi
    log_info "所有节点分发完成"
}

# ------------------------------------------
# 主流程
# ------------------------------------------
main() {
    save_image "${IMAGE_NAME}" "${OUTPUT_FILE}" "${USE_GZIP}"

    if [[ -n "${NODES_FILE}" ]]; then
        distribute_to_nodes "${OUTPUT_FILE}" "${NODES_FILE}"
        if [[ "${NO_CLEANUP}" -ne 1 ]]; then
            log_info "分发完成，清理本地 tar 文件: ${OUTPUT_FILE}"
            rm -f "${OUTPUT_FILE}"
        fi
    fi

    log_info "完成"
}

main
