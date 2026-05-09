#!/usr/bin/env bash

# ==========================================
# 环境变量配置 (set_env.sh)
# 包含 Ray 集群部署、vLLM 推理及 Ascend NPU 的相关配置
# ==========================================

# ------------------------------------------
# 1. 部署与节点配置
# ------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export NODES_FILE="${NODES_FILE:-${SCRIPT_DIR}/node_list.txt}"
export MASTER_NODE="${MASTER_NODE:-}"

export SSH_USER_HOST_PREFIX="${SSH_USER_HOST_PREFIX:-}"
export SSH_OPTS="${SSH_OPTS:--o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10}"
export PARALLELISM="${PARALLELISM:-8}"

# ------------------------------------------
# 2. 容器与镜像配置
# ------------------------------------------
# export IMAGE_NAME="${IMAGE_NAME:-quay.io/ascend/vllm-ascend:main-a3}"
# export IMAGE_TAR="${IMAGE_TAR:-/llm_workspace_1P/robin/hfhub/docker/image/vllm-ascend.main-a3.tar}"
# export RUN_CONTAINER_SCRIPT="${RUN_CONTAINER_SCRIPT:-${SCRIPT_DIR}/ascend_infer_docker_run.sh}"
# export CONTAINER_NAME="${CONTAINER_NAME:-vllm-ascend-env-a3}"

# 训练镜像
# export IMAGE_NAME="cis-pengcheng.cmecloud.cn/ascendhub/mindspeed-llm:openeuler22.03-mindspeed-llm-2.3.0-a3-arm"
# export IMAGE_TAR="${IMAGE_TAR:-/llm_workspace_1P/robin/hfhub/docker/image/mindspeed-llm-2.3.0-a3-arm.tar}"
# export RUN_CONTAINER_SCRIPT="${RUN_CONTAINER_SCRIPT:-${SCRIPT_DIR}/ascend_train_docker_run.sh}"
# export CONTAINER_NAME="${CONTAINER_NAME:-mindspeed-llm-env}"
# export VLLM_START_SCRIPT="${VLLM_START_SCRIPT:-${SCRIPT_DIR}/vllm_model_server.sh}"

# 推理镜像
export IMAGE_NAME="ascend910c-cann8.5.1-torch2.9.0-vllm0.18.0:latest"
export IMAGE_TAR="${IMAGE_TAR:-/llm_workspace_1P/robin/hfhub/docker/image/ascend910c-cann8.5.1-torch2.9.0-vllm0.18.0.tar}"
export RUN_CONTAINER_SCRIPT="${RUN_CONTAINER_SCRIPT:-${SCRIPT_DIR}/ascend_infer_docker_run.sh}"
export CONTAINER_NAME="${CONTAINER_NAME:-vllm-ascend-npuslim-env}"
export VLLM_START_SCRIPT="${VLLM_START_SCRIPT:-${SCRIPT_DIR}/vllm_model_server.sh}"

# ------------------------------------------
# 3. Ray 与 vLLM 配置
# ------------------------------------------
# 基本端口配置
export RAY_PORT="${RAY_PORT:-6379}"
export RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8266}"
export VLLM_HOST="${VLLM_HOST:-0.0.0.0}"
export VLLM_PORT="${VLLM_PORT:-8000}"

# Ray 启动超时配置
export RAY_START_TIMEOUT="${RAY_START_TIMEOUT:-120}"
export RAY_CONNECT_TIMEOUT="${RAY_CONNECT_TIMEOUT:-60}"

# ------------------------------------------
# 4. Ascend NPU 与底层环境配置
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
    "/llm_workspace_1P/expert_monitor/ATB/ascend-transformer-boost-master/output/atb/set_env.sh"
    "/usr/local/Ascend/nnal/atb/set_env.sh"
)

for _atb_path in "${_ATB_PATHS[@]}"; do
    if [ -f "$_atb_path" ]; then
        source "$_atb_path"
        break
    fi
done

# 恢复 set -u 检查
set -u

# Ray 与 Ascend 兼容性配置
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-enp66s0f0}"
export HCCL_SOCKET_IFNAME="${HCCL_SOCKET_IFNAME:-enp66s0f0}"
export HCCL_P2P_DISABLE=1
export ACLNN_ALLOW_DTYPE_CONVERT=1

# ------------------------------------------
# 5. 网络配置 - 自动获取业务网卡 IP
# ------------------------------------------
# 自动获取业务网卡 IP 并统一绑定，确保 Ray 和 vLLM 互相识别的节点 IP 完全一致
_get_node_ip() {
    local interface="${1:-$GLOO_SOCKET_IFNAME}"
    local node_ip=""
    
    if command -v ip >/dev/null 2>&1; then
        node_ip=$(ip -4 addr show "${interface}" 2>/dev/null | awk '/inet / {print $2}' | cut -d/ -f1 | head -n 1)
    elif command -v ifconfig >/dev/null 2>&1; then
        node_ip=$(ifconfig "${interface}" 2>/dev/null | awk '/inet / {print $2}' | head -n 1)
    fi
    
    echo "$node_ip"
}

# 设置节点 IP
_NODE_IP=$(_get_node_ip)
    if [ -n "$_NODE_IP" ]; then
        export RAY_NODE_IP_ADDRESS="$_NODE_IP"
        export VLLM_HOST_IP="$_NODE_IP"
    fi
