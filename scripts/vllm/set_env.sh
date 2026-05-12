#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ------------------------------------------
# 容器与镜像配置
# ------------------------------------------
export IMAGE_NAME="${IMAGE_NAME:-quay.io/ascend/vllm-ascend:main-a3}"
export IMAGE_TAR="${IMAGE_TAR:-/llm_workspace_1P/robin/hfhub/docker/image/vllm-ascend.main-a3.tar}"
export RUN_CONTAINER_SCRIPT="${RUN_CONTAINER_SCRIPT:-EasyInfer/scripts/docker/ascend_infer_docker_run.sh}"
export CONTAINER_NAME="${CONTAINER_NAME:-vllm-ascend-0.18-env}"

# ------------------------------------------
# 网络及Ascend配置
# 注意: 网卡名称需与实际机器匹配，可通过 ip link show 查看
# 多机部署时各节点可能不同，应通过环境变量覆盖
# ------------------------------------------
export HCCL_P2P_DISABLE=1
export ACLNN_ALLOW_DTYPE_CONVERT=1
export TP_SOCKET_IFNAME="${TP_SOCKET_IFNAME:-enp66s0f5}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-enp66s0f5}"
export HCCL_SOCKET_IFNAME="${HCCL_SOCKET_IFNAME:-enp66s0f5}"
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

# To reduce memory fragmentation and avoid out of memory
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_BUFFSIZE=1024
export TASK_QUEUE_ENABLE=1

# 当前节点在 Ray 集群中的 IP 地址
# 默认取 TP_SOCKET_IFNAME 接口的 IPv4 地址（即 HCCL/TP 通信网卡的 IP）
# 多网卡节点若自动取值不对，请显式设置此变量
#
# 检测方法（按优先级）:
#   1. ip 命令查 TP_SOCKET_IFNAME 指定的网卡
#   2. Python fcntl 读取同一网卡的 IPv4 地址（不依赖 iproute2）
#   3. Python socket 连接至 223.5.5.5 获取出口 IP（避开 8.0.0.0/8 接口）
_detect_host_ip() {
    local ifname="${1:-}"
    local detected=""

    # Method 1: iproute2（最快、最直接）
    if [ -n "$ifname" ] && command -v ip >/dev/null 2>&1; then
        detected=$(ip -4 addr show "$ifname" 2>/dev/null | awk '/inet / {print $2}' | cut -d/ -f1 | head -1)
        [ -n "$detected" ] && echo "$detected" && return 0
    fi

    # Method 2: Python fcntl 读取指定网卡（容器中 iproute2 可能缺失，但 Python 一定可用）
    if [ -n "$ifname" ] && command -v python3 >/dev/null 2>&1; then
        detected=$(python3 -c "
import socket, fcntl, struct, sys
try:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    ifreq = struct.pack('256s', sys.argv[1][:15].encode('utf-8'))
    ip = socket.inet_ntoa(fcntl.ioctl(s.fileno(), 0x8915, ifreq)[20:24])
    print(ip)
except Exception:
    pass
" "$ifname" 2>/dev/null)
        [ -n "$detected" ] && echo "$detected" && return 0
    fi

    # Method 3: Python socket connect 探路（不依赖网卡名）
    # 注意: 不用 8.8.8.8，因为控制面网卡可能在 8.0.0.0/8 子网中
    if command -v python3 >/dev/null 2>&1; then
        detected=$(python3 -c "
import socket
try:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.connect(('223.5.5.5', 53))
        print(s.getsockname()[0])
except Exception:
    pass
" 2>/dev/null)
        [ -n "$detected" ] && echo "$detected" && return 0
    fi

    return 1
}

_detected_ip=$(_detect_host_ip "${TP_SOCKET_IFNAME}") || true
export VLLM_HOST_IP="${VLLM_HOST_IP:-${_detected_ip}}"

# Ray 启动超时配置
export RAY_START_TIMEOUT="${RAY_START_TIMEOUT:-120}"
export RAY_CONNECT_TIMEOUT="${RAY_CONNECT_TIMEOUT:-60}"

# ------------------------------------------
# Ray 配置
# ------------------------------------------
export NPUS_PER_NODE="${NPUS_PER_NODE:-8}"
export MASTER_PORT="${MASTER_PORT:-29500}"
export DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8266}"
export WAIT_TIME="${WAIT_TIME:-1}"

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
if [ -f "/usr/local/Ascend/nnal/atb/set_env.sh" ]; then
    source /usr/local/Ascend/nnal/atb/set_env.sh
fi

# 恢复 set -u 检查
set -u