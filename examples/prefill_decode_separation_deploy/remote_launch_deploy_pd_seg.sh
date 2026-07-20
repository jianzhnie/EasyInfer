#!/bin/bash
# ==============================================================================
# remote_launch_deploy_pd_seg.sh — 批量远程 PD 分离部署入口脚本
# ==============================================================================
# 支持模型: GLM-5.2 (MODEL_TYPE=glm52), DeepSeek-V4-Pro (MODEL_TYPE=deepseek-v4-pro)
# 模型类型在 remote_deploy.conf 中配置。
#
# 用法:
#   ./remote_launch_deploy_pd_seg.sh [子命令] [--config <配置文件>]
#
# 子命令:
#   deploy        一键全流程部署（默认）
#   status        检查所有节点 + Proxy 状态
#   stop          停止所有节点 + Proxy
#   stop-pnode [N]    停止 PNode（可选索引 N 停单个节点）
#   stop-dnode [N]    停止 DNode（可选索引 N 停单个节点）
#   restart       一键重启（stop + deploy）
#   restart-docker 一键重启所有 Docker 容器（stop + start）
#   distribute    仅分发脚本到所有节点
#   start-docker  仅启动所有 Docker 容器
#   start-pnode [N]  启动 PNode（可选索引 N 启动单个节点）
#   start-dnode [N]  启动 DNode（可选索引 N 启动单个节点）
#   start-proxy   仅启动 Proxy\n#   stop-proxy    仅停止 Proxy
#   stop-docker   停止所有 Docker 容器
#   clean         停止所有进程并清理远程脚本目录
#
# 示例:
#   ./remote_launch_deploy_pd_seg.sh                          # 一键部署
#   ./remote_launch_deploy_pd_seg.sh deploy                   # 同上（显式）
#   ./remote_launch_deploy_pd_seg.sh status                   # 查看状态
#   ./remote_launch_deploy_pd_seg.sh stop                     # 停止所有
#   ./remote_launch_deploy_pd_seg.sh stop-pnode 2             # 仅停止 PNode 2
#   ./remote_launch_deploy_pd_seg.sh stop-dnode 1             # 仅停止 DNode 1
#   ./remote_launch_deploy_pd_seg.sh restart                  # 重启全线
#   ./remote_launch_deploy_pd_seg.sh start-pnode 0            # 仅启动 PNode 0
#   ./remote_launch_deploy_pd_seg.sh start-dnode 3            # 仅启动 DNode 3
#   ./remote_launch_deploy_pd_seg.sh --config my.conf deploy  # 指定配置
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_SCRIPT="${SCRIPT_DIR}/remote_launch_deploy_pd_seg.py"
DEFAULT_CONF="${SCRIPT_DIR}/remote_deploy.conf"

if [ ! -f "$PY_SCRIPT" ]; then
    echo "错误: 找不到 ${PY_SCRIPT}"
    exit 1
fi

# 默认配置文件检查
CONFIG_ARG=""
SUBCMD="deploy"

# 解析参数（收集所有剩余非选项参数拼成子命令，如 "start-dnode 1"）
ARGS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --config)
            CONFIG_ARG="--config $2"
            shift 2
            ;;
        -h|--help)
            head -20 "$0" | tail -16
            exit 0
            ;;
        *)
            ARGS+=("$1")
            shift
            ;;
    esac
done

if [ ${#ARGS[@]} -gt 0 ]; then
    SUBCMD="${ARGS[*]}"
fi

if [ -z "$CONFIG_ARG" ] && [ ! -f "$DEFAULT_CONF" ]; then
    echo "错误: 默认配置文件不存在: $DEFAULT_CONF"
    echo "请先编辑 remote_deploy.conf 填入你的环境参数"
    exit 1
fi

exec python3 "$PY_SCRIPT" $CONFIG_ARG "$SUBCMD"
