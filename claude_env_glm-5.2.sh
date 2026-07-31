# Claude Code 环境变量 — GLM-5.2 W8A8 共部署 dp1m（昇腾 A2 集群）
# 服务: http://10.42.11.194:8007  模型名: glm-5.2
# 部署: node_list3 (10.42.11.194-201) DP=8 TP=8 EP=64 DCP=8 + MTP, 1M 上下文
# 用法: source claude_env_glm-5.2.sh && claude

# ---- GLM-5.2 本地集群（默认启用）----
export ANTHROPIC_BASE_URL=http://10.42.11.138:8007
export ANTHROPIC_API_KEY=dummy
export ANTHROPIC_AUTH_TOKEN=dummy
export ANTHROPIC_MODEL=glm-5.2
export ANTHROPIC_DEFAULT_OPUS_MODEL=glm-5.2
export ANTHROPIC_DEFAULT_SONNET_MODEL=glm-5.2
export ANTHROPIC_DEFAULT_HAIKU_MODEL=glm-5.2
export CLAUDE_CODE_SUBAGENT_MODEL=glm-5.2
export CLAUDE_CODE_EFFORT_LEVEL=max
# Claude Code 默认在系统提示注入每请求变化的 attribution hash, 会破坏 vLLM
# 前缀缓存(每轮重算 ~10-20K tokens prefill); 置 0 关闭注入
export CLAUDE_CODE_ATTRIBUTION_HEADER=0

# 内网服务直连，避免 http_proxy 把集群内请求截走
# 注意: curl 不识别 no_proxy 里的 CIDR 网段和 "10." 后缀写法, 必须给精确 IP
export no_proxy="localhost,127.0.0.1,10.42.11.138${no_proxy:+,$no_proxy}"
export NO_PROXY="$no_proxy"
