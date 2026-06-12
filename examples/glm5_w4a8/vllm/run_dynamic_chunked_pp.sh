#!/bin/bash
# =============================================================================
# GLM-5 W4A8 — Dynamic Chunked Pipeline Parallel
# =============================================================================
# Status: Not applicable — GLM-5 architecture does not support Pipeline
#         Parallelism (GlmMoeDsaForCausalLM lacks the SupportsPP interface).
#
# Alternative: Use large TP across nodes.
#   TP=16 PP=1 MAX_MODEL_LEN=202752 bash run_vllm.sh
#
# Reference:
#   https://docs.vllm.ai/projects/ascend/zh-cn/releases-v0.20.2rc/tutorials/features/dynamic_chunked_pipeline_parallel.html
# =============================================================================
set -euo pipefail

cat <<EOF
============================================
[INFO] Dynamic Chunked Pipeline Parallel
[STATUS] Not applicable
[REASON] GLM-5/GLM-5.1 architecture does not support Pipeline Parallelism
[ALT] Use large TP across nodes:
      TP=16 PP=1 MAX_MODEL_LEN=202752 bash run_vllm.sh
============================================

GLM-5 uses the GlmMoeDsaForCausalLM architecture, which lacks the SupportsPP
interface. Dynamic Chunked PP requires PP > 1 and --enable-chunked-prefill.

Supported models for Dynamic Chunked PP: Kimi-K2.6, MiniMax-M2.7
EOF
