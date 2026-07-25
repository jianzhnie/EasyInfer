#!/bin/bash
# =============================================================================
# LongCat-Flash 2-Layer — SGLang serve deployment (DeepEP)
# =============================================================================
# Architecture: LongcatFlashForCausalLM | 1024 Experts | MLA | BF16
# Default: TP=64 PP=1 (8 nodes × 8 NPU, DeepEP MoE backend)
#
# Usage:
#   # Set node IPs in NODE_IPS env var (space-separated)
#   NODE_IPS="10.0.0.1 10.0.0.2 ..." bash run_sglang.sh
#   MODEL_PATH=/path/to/model bash run_sglang.sh
#
# Note: Requires SGLang with DeepEP support.
#   Placeholder IPs and model path must be configured before use.
# =============================================================================
set -euo pipefail

# System tuning
echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor || true
sysctl -w vm.swappiness=0 || true
sysctl -w kernel.numa_balancing=0 || true
sysctl -w kernel.sched_migration_cost_ns=50000 || true
export SGLANG_SET_CPU_AFFINITY=1

# Load Ascend CANN environment
set +u
if [[ -f "/usr/local/Ascend/cann/set_env.sh" ]]; then
    source /usr/local/Ascend/cann/set_env.sh
elif [[ -f "/usr/local/Ascend/ascend-toolkit/set_env.sh" ]]; then
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi
if [[ -f "/usr/local/Ascend/nnal/atb/set_env.sh" ]]; then
    source /usr/local/Ascend/nnal/atb/set_env.sh
fi
set -u

export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export STREAMS_PER_DEVICE=32

# Python path (configure via PYTHONPATH_SGLANG env var)
PYTHONPATH_SGLANG="${PYTHONPATH_SGLANG:-/home/jianzhnie/llmtuner/llm/sglang/python}"
export PYTHONPATH="${PYTHONPATH_SGLANG}:${PYTHONPATH:-}"
export SGLANG_DEEPEP_BF16_DISPATCH=1

export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_BUFFSIZE="${HCCL_BUFFSIZE:-2048}"
export HCCL_SOCKET_IFNAME="${HCCL_SOCKET_IFNAME:-enp66s0f5}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-enp66s0f5}"

# Node IP configuration (space-separated, e.g. NODE_IPS="10.0.0.1 10.0.0.2 ...")
NODE_IPS="${NODE_IPS:-}"
if [[ -z "${NODE_IPS}" ]]; then
    echo "ERROR: NODE_IPS not set. Usage: NODE_IPS='10.0.0.1 10.0.0.2 ...' bash run_sglang.sh"
    exit 1
fi
read -r -a P_IP <<< "${NODE_IPS}"

MASTER_PORT="${MASTER_PORT:-5000}"
P_MASTER="${P_IP[0]}:${MASTER_PORT}"
NNODES=${#P_IP[@]}
TP_SIZE="${TP_SIZE:-64}"

MODEL_PATH="${MODEL_PATH:-}"
if [[ -z "${MODEL_PATH}" ]]; then
    echo "ERROR: MODEL_PATH not set."
    exit 1
fi

SERVER_HOST="${SERVER_HOST:-0.0.0.0}"
SERVER_PORT="${SERVER_PORT:-6677}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-longcat-flash}"

LOCAL_IPS="$(hostname -I)"
NODE_RANK=""

echo "Local IPs: ${LOCAL_IPS}"
echo "Cluster IPs: ${P_IP[*]}"
echo "Master: ${P_MASTER}"

for i in "${!P_IP[@]}"; do
  if [[ " ${LOCAL_IPS} " == *" ${P_IP[$i]} "* ]]; then
    NODE_RANK="${i}"
    break
  fi
done

if [[ -z "${NODE_RANK}" ]]; then
  echo "ERROR: local IPs [${LOCAL_IPS}] not found in P_IP=[${P_IP[*]}]"
  exit 1
fi

export MOE_ENABLE_TOPK_NEG_ONE=1
export SGLANG_DEEPEP_BF16_DISPATCH=1
export TRANSFORMERS_VERBOSITY=error

python -m sglang.launch_server \
    --trust-remote-code \
    --model-path "${MODEL_PATH}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --host "${SERVER_HOST}" \
    --port "${SERVER_PORT}" \
    --nnodes "${NNODES}" \
    --node-rank "${NODE_RANK}" \
    --dist-init-addr "${P_MASTER}" \
    --tp-size "${TP_SIZE}" \
    --mem-fraction-static 0.65 \
    --attention-backend ascend \
    --device npu \
    --max-running-requests 16 \
    --context-length 8192 \
    --disable-radix-cache \
    --chunked-prefill-size 8192 \
    --watchdog-timeout 9000 \
    --prefill-round-robin-balance \
    --moe-a2a-backend deepep \
    --deepep-mode auto
