#!/bin/bash
# GLM-5.1 W4A8 — 动态分块流水线并行验证
# 状态: ❌ 不适用 — GLM-5.1 架构不支持 Pipeline Parallelism (PP)
# 与 GLM-5 相同，GLM-5.1 的 GlmMoeDsaForCausalLM 缺少 SupportsPP 接口
#
# 替代方案: 使用大 TP 跨节点
#   TP=16 PP=1 MAX_MODEL_LEN=202752 bash run_vllm.sh
#
# 参考:
#   https://docs.vllm.ai/projects/ascend/zh-cn/releases-v0.20.2rc/tutorials/features/dynamic_chunked_pipeline_parallel.html

echo "============================================"
echo "[INFO] Dynamic Chunked Pipeline Parallel"
echo "[STATUS] ❌ 不适用"
echo "[REASON] GLM-5.1 架构不支持 Pipeline Parallelism"
echo "[ALT] 使用大 TP 跨节点: TP=16 PP=1 bash run_vllm.sh"
echo "============================================"
echo ""
echo "GLM-5.1 的 GlmMoeDsaForCausalLM 架构缺少 SupportsPP 接口。"
echo "Dynamic Chunked PP 需要 PP > 1 + --enable-chunked-prefill。"
echo ""
echo "多节点扩展请使用大 TP 方案:"
echo "  TP=16 PP=1 MAX_MODEL_LEN=202752 bash run_vllm.sh"
echo ""
echo "支持 Dynamic Chunked PP 的模型: Kimi-K2.6, MiniMax-M2.7"
