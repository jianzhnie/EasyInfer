#!/usr/bin/env bash

# ------------------------------------------
# 容器与镜像配置
# ------------------------------------------
export IMAGE_NAME="${IMAGE_NAME:-quay.io/ascend/vllm-ascend:main-a3}"
export IMAGE_TAR="${IMAGE_TAR:-/llm_workspace_1P/robin/hfhub/docker/image/vllm-ascend.main-a3.tar}"
export RUN_CONTAINER_SCRIPT="${RUN_CONTAINER_SCRIPT:-${SCRIPT_DIR}/ascend_infer_docker_run.sh}"
export CONTAINER_NAME="${CONTAINER_NAME:-vllm-ascend-env-a3}"

# ------------------------------------------
# Ascend NPU 与底层环境配置
# ------------------------------------------
# 注意: 下列 source 命令通常在容器内生效
# 由于第三方脚本（如 Ascend 的 set_env.sh）可能存在未绑定变量，临时关闭 set -u 检查
set +u

# 加载 Ascend Toolkit 环境
if [ -f "/usr/local/Ascend/ascend-toolkit/set_env.sh" ]; then
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi

# 加载 ATB 环境（如果存在）
# 尝试多个可能的路径
_ATB_PATHS=(
    "/usr/local/Ascend/nnal/atb/set_env.sh"
    "/llm_workspace_1P/expert_monitor/ATB/ascend-transformer-boost-master/output/atb/set_env.sh"
    )

for _atb_path in "${_ATB_PATHS[@]}"; do
    if [ -f "$_atb_path" ]; then
        source "$_atb_path"
        break
    fi
done
