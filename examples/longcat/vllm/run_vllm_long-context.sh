#!/bin/bash
# =============================================================================
# LongCat-Flash-Chat — 最大上下文 (131072 / 128K) 吞吐优化部署
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
#   2. CHUNKED_PREFILL=1 + MAX_NUM_BATCHED_TOKENS: 128K prompt 分块喂入,
#      避免整吞时 ALLGATHER comm 缓冲 + prefill 尖峰 OOM (实测 132096 整吞
#      在 64G 卡 OOM, 见 logs/vllm_longcat_20260727_041823.log)
#   3. MAX_NUM_SEQS: MLA latent KV 很小 (~1.05GB/rank/128K seq, PP=4),
#      实测 KV 池 2.7M tokens 可支撑 20.7 个 128K 请求
#   4. GPU_MEM_UTIL 提高到 0.92 给 KV cache 让出空间
#
# =============================================================================
# 吞吐优化策略 (2026-07-29)
# =============================================================================
#
# 本模型的吞吐瓶颈排序（影响从大到小）：
#
# ❶ CUDA Graph (decode 阶段) — ENFORCE_EAGER=0
#   eager 模式下每次 decode step 都重新编译 kernel，图模式可消除此开销。
#   预期 decode 吞吐提升 20-50%。当前默认 eager (兼容性优先)，
#   设 ENFORCE_EAGER=0 开启图模式。
#   代价: 需要 VLLM_ASCEND_ENABLE_FLASHCOMM1=0 (SP pass 与 graph 冲突)。
#
# ❷ Prefill 分块大小 — MAX_NUM_BATCHED_TOKENS
#   当前默认 16384，对 64G 卡保守。若显存充足可提升到 32768~49152，
#   每块更大 = 分块数更少 = prefill 更快完成。
#
# ❸ KV Cache 容量 — KV_CACHE_DTYPE=fp8 / MAX_NUM_SEQS / GPU_MEM_UTIL
#   - KV_CACHE_DTYPE=fp8 (若模型支持): KV cache 容量翻倍 → 并发翻倍
#   - GPU_MEM_UTIL=0.95: 极限挤压权重显存给 KV cache（不稳定则降回 0.92）
#   - MAX_NUM_SEQS: 长上下文纯流量设 16~32, 混合长短流量设 64~128
#
# ❹ Prefix Caching — 去掉 --no-enable-prefix-caching
#   多轮对话 / 共享 system prompt / 批量相同前缀评测时，APCache 可大幅减少
#   重复 prefill。代价: 少量显存开销（hash table）。
#
# ❺ HCCL 通信 — HCCL_BUFFSIZE
#   多节点 all-reduce 场景下增大 HCCL buffer 可提升通信带宽利用率。
#   当前 768 偏保守，可尝试 1024~2048。
#
# ❻ Pipeline Parallelism — PP
#   PP 分摊权重显存，腾出空间给 KV cache 或更大的 batched_tokens。
#   PP=2 (8节点) 或 PP=4 (16节点) 推荐用于长上下文。
#
# =============================================================================
# 快速参考：按场景选参数
# =============================================================================
#
#  场景                          | 推荐命令
#  ──────────────────────────────┼────────────────────────────────────────────
#  最大吞吐 (128K, 16 节点)      | PP=4 TP=32 EP=1 bash run_vllm_long-context.sh
#  均衡吞吐 (128K, 16 节点)      | PP=4 TP=32 EP=1 MAX_NUM_SEQS=32 bash run_vllm_long-context.sh
#  低延迟短上下文 (4K, 8 节点)   | 直接用 run_vllm.sh (TP=64 EP=64 PP=1)
#  混合长短流量                  | MAX_NUM_SEQS=64 MAX_MODEL_LEN=65536 bash run_vllm_long-context.sh
#  极限并发 (需 FP8 KV cache)    | KV_CACHE_DTYPE=fp8 MAX_NUM_SEQS=64 bash run_vllm_long-context.sh
#  开启图模式 (提升吞吐)         | ENFORCE_EAGER=0 bash run_vllm_long-context.sh
#  图模式排障                    | ENFORCE_EAGER=0 VLLM_DEBUG_DUMP=/tmp/dump bash run_vllm_long-context.sh
#
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_DIR

# =============================================================================
# 核心上下文参数
# =============================================================================
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"

# ---------------------------------------------------------------------------
# Prefill: 分块喂入
# ---------------------------------------------------------------------------
# 整吞方案 (batched_tokens >= 131072) 在 64G 卡上 OOM: ALLGATHER comm 持久
# 缓冲 (~tokens×topk×hidden, 132096 时 ~30G) + 单步 prefill 尖峰 18.1G 超出容量。
# 改为 chunked prefill 分块喂入 (GLM-5.2 1M 同款: 16384)。
# 注: "chunked prefill 与 EP dispatch 冲突" 是 MC2 comm 下的结论,
#     ALLGATHER 下已实测正常 (2026-07-27, curl_test 回归 + 130K 大海捞针
#     命中, 见 logs/vllm_longcat_20260727_043057.log)。
export CHUNKED_PREFILL="${CHUNKED_PREFILL:-1}"
# 每块大小：16384 是保守安全值；显存充足时可提升到 32768/49152 加速 prefill
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-16384}"

# ---------------------------------------------------------------------------
# 并发: MAX_NUM_SEQS
# ---------------------------------------------------------------------------
# 实测 KV 池 2.7M tokens = 20.7 个 128K 请求。
# - 纯长上下文流量: 16~32 (留有安全余量, 避免 KV cache 碎片化)
# - 混合短+长流量: 64~128 (短请求 KV 占用极小, 高上限不影响)
# 默认 32，兼顾长上下文安全和短请求并发。
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"

export GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.92}"

# =============================================================================
# 吞吐优化参数
# =============================================================================

# ---------------------------------------------------------------------------
# ❶ CUDA Graph / Eager 模式
# ---------------------------------------------------------------------------
# ENFORCE_EAGER=0 → 开启 FULL_DECODE_ONLY CUDA graph, decode 提速显著
# ENFORCE_EAGER=1 (默认) → eager 模式, 逐 step 编译 (兼容性优先, 稳定路径)
#
# 图模式 (ENFORCE_EAGER=0) 的两条硬性要求 (2026-07-27 排查结论):
#   ① cudagraph_capture_sizes 必须含 TP 的倍数 — run_vllm.sh 已自动生成
#      (TP 步进、覆盖 MAX_NUM_SEQS、最多 4 档)
#   ② 必须 VLLM_ASCEND_ENABLE_FLASHCOMM1=0 — FlashComm1 的 SP pass 在 FX 图
#      里直接插入 npu_add_rms_norm_bias, 绕过 fix_layernorm_dtype 的 dtype 保护,
#      float32 激活直送 ACLNN 报 EZ1001
export ENFORCE_EAGER="${ENFORCE_EAGER:-1}"

# ---------------------------------------------------------------------------
# ❷ KV Cache 数据类型
# ---------------------------------------------------------------------------
# bfloat16 (默认): KV cache 2.7M tokens, ~20.7 个 128K 并发
# fp8:             KV cache 容量翻倍 (~5.4M tokens), ~41 个 128K 并发
# 注意: 需要模型权重 + vllm-ascend 版本支持 fp8 KV cache。
#       若启动报错 "Unsupported KV cache dtype", 回退为 "" (bfloat16)。
export KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-bfloat16}"

# ---------------------------------------------------------------------------
# ❸ Prefix Caching
# ---------------------------------------------------------------------------
# 默认开启。多轮对话 / 共享 system prompt / 批量相同前缀评测时，
# APCache 可大幅减少重复 prefill。代价: 少量显存开销（hash table）。
# 关闭: PREFIX_CACHING=0 bash run_vllm_long-context.sh
PREFIX_CACHING="${PREFIX_CACHING:-1}"

# =============================================================================
# 环境优化 (参照 GLM-5.2 1M 教程 + 吞吐调优)
# =============================================================================

# 主机侧任务队列, 大 batch / 长 prefill 下降低下发开销
export TASK_QUEUE_ENABLE=1

# MLA 算子优化
export VLLM_ASCEND_ENABLE_NZ="${VLLM_ASCEND_ENABLE_NZ:-1}"

# HCCL 通信 buffer: 增大可提升多节点 all-reduce 带宽利用率
# 768 偏保守 (兼容性优先), 吞吐优先可设 1024~2048
export HCCL_BUFFSIZE="${HCCL_BUFFSIZE:-1024}"

# 长上下文超时保护 (128K 单步 prefill 计算量大)
export HCCL_EXEC_TIMEOUT="${HCCL_EXEC_TIMEOUT:-3600}"
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-1800}"

# NPU 显存分配器: expandable_segments 减少碎片, 对长上下文 KV cache 分配有利
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"

# =============================================================================
# 启动前摘要
# =============================================================================
echo "============================================"
echo "[INFO] LongCat-Flash-Chat — 长上下文吞吐优化部署"
echo "[INFO] MAX_MODEL_LEN=$MAX_MODEL_LEN"
echo "[INFO] MAX_NUM_SEQS=$MAX_NUM_SEQS (KV 池 ~2.7M tokens, ~$((2700000 / MAX_MODEL_LEN)) 并发容量)"
echo "[INFO] MAX_NUM_BATCHED_TOKENS=$MAX_NUM_BATCHED_TOKENS"
echo "[INFO] GPU_MEM_UTIL=$GPU_MEM_UTIL"
echo "[INFO] ENFORCE_EAGER=$ENFORCE_EAGER ($(if [[ "$ENFORCE_EAGER" == "0" ]]; then echo 'CUDA graph FULL_DECODE_ONLY'; else echo 'eager 逐 step 编译'; fi))"
echo "[INFO] KV_CACHE_DTYPE=$KV_CACHE_DTYPE"
echo "[INFO] PREFIX_CACHING=$PREFIX_CACHING"
echo "[INFO] HCCL_BUFFSIZE=$HCCL_BUFFSIZE"
echo "============================================"

# 构建额外参数：追加到 vllm serve 末尾（vllm last-one-wins）
EXTRA_ARGS=()

# KV cache dtype (run_vllm.sh 不传此参数, 由这里按需注入)
if [[ -n "$KV_CACHE_DTYPE" ]]; then
    EXTRA_ARGS+=(--kv-cache-dtype "$KV_CACHE_DTYPE")
fi

# PREFIX_CACHING 已通过环境变量传递给 run_vllm.sh 原生处理

exec bash "${SCRIPT_DIR}/run_vllm.sh" "$@" "${EXTRA_ARGS[@]}"
