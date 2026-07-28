#!/bin/bash
# =============================================================================
# GLM-5.2 W8A8 1M（方案 C）— 远程一键启动
# =============================================================================
# 在 $NODE 的 vllm-ascend-env 容器内后台启动部署，日志写入仓库根目录 glm5.2.log
# Usage:
#   bash examples/glm-5.2_w8a8/run.sh                # 默认 10.42.11.194
#   NODE=10.42.11.130 bash examples/glm-5.2_w8a8/run.sh
# 查看日志:
#   ssh $NODE "docker exec vllm-ascend-env tail -f $REPO/glm5.2.log"
# =============================================================================
set -euo pipefail

NODE="${NODE:-10.42.11.194}"
REPO=/home/jianzhnie/llmtuner/llm/EasyInfer

ssh "$NODE" "docker exec -d vllm-ascend-env bash -c 'cd $REPO && bash examples/glm-5.2_w8a8/vllm/run.sh > glm5.2.log 2>&1'"
echo "[OK] 已在 $NODE 容器内后台启动，日志: $REPO/glm5.2.log"
echo "     跟踪日志: ssh $NODE \"docker exec vllm-ascend-env tail -f $REPO/glm5.2.log\""
