#!/bin/bash
# GLM-5 W4A8 — 动态分块流水线并行验证
# 状态: ❌ 不适用 — GLM-5 架构不支持 Pipeline Parallelism (PP)
#
# GLM-5/GLM-5.1 基于 GlmMoeDsaForCausalLM 架构，缺少 SupportsPP 接口，
# 无法使用 Pipeline Parallelism，因此动态分块流水线并行 (Dynamic Chunked PP) 不适用。
#
# 替代方案: 使用大 TP 跨节点 (TP=16/32) 实现多节点扩展
#
# 参考:
#   - Dynamic Chunked PP: https://docs.vllm.ai/projects/ascend/zh-cn/releases-v0.20.2rc/tutorials/features/dynamic_chunked_pipeline_parallel.html
#   - GLM-5 部署: https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/GLM5.html

echo "============================================"
echo "[INFO] Dynamic Chunked Pipeline Parallel"
echo "[STATUS] ❌ 不适用"
echo "[REASON] GLM-5/GLM-5.1 架构不支持 Pipeline Parallelism"
echo "[ALT] 使用大 TP 跨节点: TP=16 PP=1 bash run_vllm.sh"
echo "============================================"
echo ""
echo "GLM-5 的 GlmMoeDsaForCausalLM 架构缺少 SupportsPP 接口，"
echo "无法启用 --pipeline-parallel-size > 1。"
echo ""
echo "Dynamic Chunked PP 需要 PP > 1 + --enable-chunked-prefill，"
echo "因此该功能对 GLM-5/GLM-5.1 不适用。"
echo ""
echo "多节点扩展请使用大 TP 方案:"
echo "  TP=16 PP=1 MAX_MODEL_LEN=202752 bash run_vllm.sh"
echo ""
echo "支持 Dynamic Chunked PP 的模型: Kimi-K2.6, MiniMax-M2.7"
