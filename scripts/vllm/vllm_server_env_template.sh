#!/usr/bin/env bash
# ==============================================================================
# vLLM Model Server 环境变量配置文件
# ==============================================================================
# 文件名: vllm_server_env.sh
# 用途: 存放 vLLM 模型部署的所有可配置环境变量
# 
# 使用方法:
#   1. 在启动脚本前 source 此文件: source vllm_server_env.sh
#   2. 或在命令行直接设置: MODEL_PATH=/path/to/model ./vllm_model_server.sh
#   3. 可以复制此文件为 vllm_server_env.prod.sh 用于不同环境
#
# 注意: 所有变量都有默认值，只需覆盖需要修改的变量
# ==============================================================================

# ------------------------------------------------------------------------------
# 1. 基础环境变量配置
# ------------------------------------------------------------------------------

# 模型路径: 指向 Hugging Face 模型目录
# 必须包含 config.json, tokenizer 文件和模型权重
export MODEL_PATH="${MODEL_PATH:-$HOME/hfhub/models/moonshotai/Kimi-K2-Base}"

# 服务对外暴露的模型名称
# 客户端调用 API 时使用此名称
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-kimi-k2-base}"

# 服务监听地址
# 0.0.0.0 表示监听所有网络接口，127.0.0.1 仅监听本地
export HOST="${HOST:-0.0.0.0}"

# 服务监听端口
export PORT="${PORT:-8000}"

# 日志级别: debug, info, warning, error
export LOG_LEVEL="${LOG_LEVEL:-info}"

# ------------------------------------------------------------------------------
# 2. 分布式并行配置 (核心调整)
# ------------------------------------------------------------------------------
# Kimi-K2 采用 MoE 架构，参数量巨大，需要合理配置并行策略
#
# 推荐配置参考:
#   128 NPU (16节点 * 8 NPU): TP=8, PP=16, EP=128
#   64 NPU  (8节点 * 8 NPU):  TP=8, PP=8,  EP=64
#   32 NPU  (4节点 * 8 NPU):  TP=8, PP=4,  EP=32
#   16 NPU  (2节点 * 8 NPU):  TP=8, PP=2,  EP=16
#   8 NPU   (1节点 * 8 NPU):  TP=8, PP=1,  EP=8
# ------------------------------------------------------------------------------

# 张量并行大小 (Tensor Parallel)
# 建议: 节点内 NPU 数量，通常设为 8
export TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-8}"

# 流水线并行大小 (Pipeline Parallel)
# 建议: 根据节点数设置，跨节点并行
export PIPELINE_PARALLEL_SIZE="${PIPELINE_PARALLEL_SIZE:-8}"

# 分布式执行后端
# 可选: ray, mp (多进程)
# Ray 推荐用于多节点部署
export DISTRIBUTED_EXECUTOR_BACKEND="${DISTRIBUTED_EXECUTOR_BACKEND:-ray}"

# 专家并行开关 (Expert Parallel)
# MoE 模型强烈建议启用，可显著提升性能
export ENABLE_EXPERT_PARALLEL="${ENABLE_EXPERT_PARALLEL:-1}"

# 专家并行大小
# 默认自动计算为 TP * PP，确保专家均匀分布
# Kimi-K2 有 384 个专家，建议 EP 能整除 384
export EXPERT_PARALLEL_SIZE="${EXPERT_PARALLEL_SIZE:-$((TENSOR_PARALLEL_SIZE * PIPELINE_PARALLEL_SIZE))}"

# ------------------------------------------------------------------------------
# 3. 内存与量化配置
# ------------------------------------------------------------------------------

# 模型数据类型
# 可选: float16, bfloat16, float32
# 即使权重是 FP8 量化，激活仍使用此类型
export DTYPE="${DTYPE:-bfloat16}"

# 量化方式
# 可选: fp8, awq, gptq, squeezellm, marlin, 或留空表示无
# Kimi-K2 模型配置为 FP8 量化
export QUANTIZATION="${QUANTIZATION:-fp8}"

# 模型加载格式
# 可选: safetensors, pt, auto
export LOAD_FORMAT="${LOAD_FORMAT:-safetensors}"

# GPU(NPU) 内存利用率 (0.0 - 1.0)
# 较大的值使用更多显存用于 KV Cache，提高吞吐
# 建议范围: 0.88 - 0.95
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"

# CPU 交换空间大小 (GiB)
# 用于 KV Cache 驱逐时的缓冲，MoE 模型建议设置较大值
export SWAP_SPACE="${SWAP_SPACE:-128}"

# ------------------------------------------------------------------------------
# 4. 吞吐量与序列调度优化
# ------------------------------------------------------------------------------

# 最大模型长度 (上下文窗口)
# 模型原生支持 131072，但为内存和吞吐折中，可限制为 32k-64k
# 如需更长序列，请确保有足够的 NPU 内存
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"

# 最大并发请求数
# 根据预期负载和硬件能力调整，MoE 模型吞吐量较高
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-1024}"

# 分块预填充开关 (Chunked Prefill)
# 强烈建议启用，解耦 Prefill 和 Decode 阶段，提升并发
export ENABLE_CHUNKED_PREFILL="${ENABLE_CHUNKED_PREFILL:-1}"

# 每个 step 处理的最大 token 数
# 较大的值提高吞吐量，较小的值降低延迟
# 建议: 4096 - 16384
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"

# 每个序列的最大 tokens (prefill + decode)
# 用于限制单个请求的资源占用，防止单个请求占用过多资源
export MAX_TOKENS_PER_SEQUENCE="${MAX_TOKENS_PER_SEQUENCE:-32768}"

# ------------------------------------------------------------------------------
# 5. 高级加速特性
# ------------------------------------------------------------------------------

# 前缀缓存开关 (Prefix Caching)
# 对于多轮对话或大量重复 system prompt 极其有效，强烈建议启用
export PREFIX_CACHING="${PREFIX_CACHING:-1}"

# 多步调度步数 (Multi-step Scheduling)
# 减少框架在各个 NPU 之间的调度通信开销
# 建议值: 4-8，较大的值提高吞吐但增加延迟
export NUM_SCHEDULER_STEPS="${NUM_SCHEDULER_STEPS:-8}"

# 强制 Eager 模式
# 1 = 禁用 CUDA Graph/编译图 (推荐 NPU 环境)
# 0 = 启用 CUDA Graph (如果底层支持)
export ENFORCE_EAGER="${ENFORCE_EAGER:-1}"

# CUDA Graph 捕获的最大序列长度
# 仅在 ENFORCE_EAGER=0 时有效，对于 MoE 模型建议保持较小值
export MAX_SEQ_LEN_TO_CAPTURE="${MAX_SEQ_LEN_TO_CAPTURE:-8192}"

# 自动检测 vLLM 版本支持的参数
# 1 = 自动检测，0 = 使用预设参数
export AUTO_DETECT_FLAGS="${AUTO_DETECT_FLAGS:-1}"

# ------------------------------------------------------------------------------
# 6. vLLM 内部环境变量优化
# ------------------------------------------------------------------------------

# Ray 对象存储内存 (字节)
# 0 表示自动，对于大模型可显式设置，例如 8GB: 8589934592
export VLLM_RAY_PER_NODE_OBJECT_STORE_MEMORY="${VLLM_RAY_PER_NODE_OBJECT_STORE_MEMORY:-0}"

# 注意力计算后端
# 可选: FLASH_ATTN, FLASHINFER, XFORMERS, TORCH_SDPA
# 对于 NPU，通常使用 TORCH_SDPA
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-TORCH_SDPA}"

# Worker 多进程创建方法
# spawn: 更稳定但稍慢 (推荐 NPU)
# fork: 更快但可能有兼容性问题
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"

# MoE 填充开关
# 某些实现需要启用以确保专家并行正确工作
export VLLM_MOE_PADDING="${VLLM_MOE_PADDING:-1}"

# 禁用严格的 CUDA 版本检查
# 对于 NPU 环境或其他非标准 CUDA 环境需要启用
export VLLM_SKIP_CUDA_VERSION_CHECK="${VLLM_SKIP_CUDA_VERSION_CHECK:-1}"

# 设置 vLLM 日志级别
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-$LOG_LEVEL}"

# ------------------------------------------------------------------------------
# 7. API 和监控配置
# ------------------------------------------------------------------------------

# API 密钥 (生产环境强烈建议设置)
# 留空表示不启用认证
export API_KEY="${API_KEY:-}"

# Prometheus 指标导出开关
# 1 = 启用，0 = 禁用
export ENABLE_METRICS="${ENABLE_METRICS:-1}"

# Prometheus 指标导出端口
export METRICS_PORT="${METRICS_PORT:-8001}"

# 禁用请求日志开关
# 1 = 禁用 (减少日志量)，0 = 启用
export DISABLE_LOG_REQUESTS="${DISABLE_LOG_REQUESTS:-0}"

# CORS 允许的源
# * 表示允许所有，或设置特定域名如 "https://example.com"
export ALLOWED_ORIGINS="${ALLOWED_ORIGINS:-*}"

# ------------------------------------------------------------------------------
# 8. 启动与重试配置
# ------------------------------------------------------------------------------

# 最大重试次数
# 服务崩溃后自动重启的次数
export MAX_RETRIES="${MAX_RETRIES:-3}"

# 重试间隔 (秒)
export RETRY_DELAY="${RETRY_DELAY:-10}"

# ==============================================================================
# 配置文件结束
# ==============================================================================
# 提示: 可以通过复制此文件来创建不同环境的配置
#   cp vllm_server_env.sh vllm_server_env.prod.sh  # 生产环境
#   cp vllm_server_env.sh vllm_server_env.test.sh  # 测试环境
# ==============================================================================
