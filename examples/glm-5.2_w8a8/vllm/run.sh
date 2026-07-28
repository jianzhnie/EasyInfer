#!/bin/bash
# =============================================================================
# GLM-5.2 W8A8 1M — 方案 C 冒烟: TP=16 PP=1 DP=1 EP=16 DCP=16 + MTP
# =============================================================================
# 资源: 2 节点 × 8 卡 = 16 NPU（单实例，Ray 跨 2 节点）
# 前提: Ray 集群已启动（head: 10.42.11.194:6379）
#
# 注意: TP=16 DP>1 不可行——vLLM v1 要求每个 DP rank 的 TP×PP ≤ 单节点卡数
#   （engine core 按 local_dp_rank×world_size 枚举本机设备，A2 只有 8 卡）。
#   要多实例横向扩展：每 2 节点起一个 TP=16 DP=1 实例 + 负载均衡代理。
#
# MAX_NUM_SEQS=16 是并发上限；KV 池约支持 ~6 条全 1M 序列，
#   超出部分会排队/重算，不会 OOM，但全 1M 高压时建议降到 6~8。
# =============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export RAY_ADDRESS="${RAY_ADDRESS:-10.42.11.194:6379}"
export NIC_NAME="${NIC_NAME:-enp66s0f0}"
export HCCL_IF_IP="${HCCL_IF_IP:-10.42.11.194}"

TP=16 DP=1 ENABLE_MTP=1 \
  MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}" GPU_MEM_UTIL=0.85 \
  bash "${SCRIPT_DIR}/run_vllm_1m.sh" "$@"
