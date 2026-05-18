#!/bin/bash
# SessionStart hook: 输出项目上下文信息
BRANCH=$(git -C "${CLAUDE_PROJECT_DIR}" branch --show-current 2>/dev/null || echo "unknown")
CHANGED=$(git -C "${CLAUDE_PROJECT_DIR}" status --porcelain 2>/dev/null | wc -l | tr -d ' ')

cat <<EOF
当前分支: $BRANCH
未提交更改: $CHANGED 个文件
项目: EasyInfer - Ascend NPU 集群 LLM 推理部署工具包
EOF

exit 0
