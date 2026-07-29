#!/bin/bash
# =============================================================================
# GLM-5.2 W8A8 — Claude Code / Agent 工具接入环境
# =============================================================================
# vLLM 提供 Anthropic Messages API，Claude Code 通过 ANTHROPIC_BASE_URL
# 直连本地 vLLM 服务。工具调用由服务端 glm47 parser 处理（run_vllm.sh /
# run_vllm_1m.sh 已内置 --enable-auto-tool-choice --tool-call-parser glm47
# --reasoning-parser glm45），客户端无需额外配置。
#
# Usage:
#   source agent_api.sh && claude                      # 本机服务（默认 127.0.0.1:8007）
#   HOST=10.42.11.196 source agent_api.sh && claude    # 远程服务
#   PORT=9000 MODEL_NAME=glm-5.2 source agent_api.sh && claude
#
# Reference: docs/claude-code-vllm-setup.md
# =============================================================================

# 必须用 source 方式加载（需要 export 到当前 shell）
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "请用 source 方式加载: source $0 && claude"
    exit 1
fi

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8007}"
MODEL_NAME="${MODEL_NAME:-glm-5.2}"

export ANTHROPIC_BASE_URL="http://${HOST}:${PORT}"
# vLLM 默认不鉴权，任意非空值即可；已设置的值会被保留
export ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-dummy}"
export ANTHROPIC_AUTH_TOKEN="${ANTHROPIC_AUTH_TOKEN:-dummy}"
# 模型名必须与 run_vllm.sh 的 --served-model-name 一致，且不含 '/'
# ANTHROPIC_MODEL 优先级高于 DEFAULT_*，必须一并覆盖（防止沿用外部残留值）
export ANTHROPIC_MODEL="$MODEL_NAME"
export ANTHROPIC_DEFAULT_SONNET_MODEL="$MODEL_NAME"
export ANTHROPIC_DEFAULT_HAIKU_MODEL="$MODEL_NAME"
export ANTHROPIC_DEFAULT_OPUS_MODEL="$MODEL_NAME"

# Claude Code 注入的 attribution header 会破坏 prefix caching 前缀复用，
# 导致推理明显变慢。新版 vLLM 已自动处理；如遇变慢可取消注释：
# export CLAUDE_CODE_ATTRIBUTION_HEADER=0

# 服务健康检查
if curl -sf --max-time 5 "${ANTHROPIC_BASE_URL}/v1/models" >/dev/null 2>&1; then
    echo "[OK] vLLM 服务可达: ${ANTHROPIC_BASE_URL} (model: ${MODEL_NAME})"
else
    echo "[WARN] 无法连接 ${ANTHROPIC_BASE_URL}，请先启动服务：" >&2
    echo "  bash examples/glm-5.2_w8a8/vllm/run_vllm.sh    # 32K 上下文" >&2
    echo "  bash examples/glm-5.2_w8a8/vllm/run_vllm_1m.sh # 1M 上下文" >&2
fi
