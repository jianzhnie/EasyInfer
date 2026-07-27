#!/bin/bash
# =============================================================================
# LongCat-Flash-Chat — 最大上下文 (131072 / 128K) 部署
# =============================================================================
# 薄封装：设置长上下文默认值后 exec run_vllm.sh（所有参数仍可环境变量覆盖）。
#
# 参考: vllm-ascend GLM-5.2 1M 教程
#   https://docs.vllm.ai/projects/ascend/zh-cn/latest/tutorials/models/GLM5.2.html
# 与 GLM-5.2 1M 方案的差异:
#   - LongCat 原生上下文为 131072 (128K)，非 1M
#   - 不支持 MTP -> 无 --speculative-config
#   - 无 DSA 稀疏注意力 -> additional-config 中的 dsa/sparse 选项不适用
#   - 未启用 PCP/DCP 上下文并行 (MLA latent KV 本身体积小, 128K 不需要)
#
# 长上下文关键点:
#   1. MAX_MODEL_LEN=131072 (模型原生上限, max_position_embeddings)
#   2. CHUNKED_PREFILL=1 + MAX_NUM_BATCHED_TOKENS=16384: 128K prompt 分块
#      喂入, 避免整吞时 ALLGATHER comm 缓冲 + prefill 尖峰 OOM (实测
#      132096 整吞在 64G 卡 OOM, 见 logs/vllm_longcat_20260727_041823.log)
#   3. MAX_NUM_SEQS 压低到 8: MLA latent KV 约 32KB/token/seq (28 层合计,
#      按 PP stage 分摊), 8 并发 128K 已占 ~8.5GB/rank (PP=4)
#   4. GPU_MEM_UTIL 提高到 0.92 给 KV cache 让出空间
#
# Usage:
#   bash run_vllm_long-context.sh                 # 128K, 沿用当前集群并行配置
#   PP=4 TP=32 EP=1 bash run_vllm_long-context.sh # 16 节点
#   MAX_NUM_SEQS=4 bash run_vllm_long-context.sh  # KV 不足时降并发
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_DIR

# 长上下文核心参数
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
# 整吞方案 (batched_tokens >= 131072) 在 64G 卡上 OOM: ALLGATHER comm 持久
# 缓冲 (~tokens×topk×hidden, 132096 时 ~30G) + 单步 prefill 尖峰 18.1G 超出容量。
# 改为 chunked prefill 分块喂入 (GLM-5.2 1M 同款: 16384)。
# 注: "chunked prefill 与 EP dispatch 冲突" 是 MC2 comm 下的结论,
#     ALLGATHER 下已实测正常 (2026-07-27, curl_test 回归 + 130K 大海捞针
#     命中, 见 logs/vllm_longcat_20260727_043057.log)。
export CHUNKED_PREFILL="${CHUNKED_PREFILL:-1}"
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-16384}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
export GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.92}"

# 长上下文环境优化 (参照 GLM-5.2 1M 教程)
export TASK_QUEUE_ENABLE=1
export VLLM_ASCEND_ENABLE_NZ="${VLLM_ASCEND_ENABLE_NZ:-1}"
export HCCL_BUFFSIZE="${HCCL_BUFFSIZE:-768}"
# 128K 单步 prefill 计算量大, 拉长 HCCL/引擎超时
export HCCL_EXEC_TIMEOUT="${HCCL_EXEC_TIMEOUT:-3600}"
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-1800}"

exec bash "${SCRIPT_DIR}/run_vllm.sh" "$@"
