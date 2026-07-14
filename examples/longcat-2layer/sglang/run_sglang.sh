echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
sysctl -w kernel.sched_migration_cost_ns=50000
export SGLANG_SET_CPU_AFFINITY=1
# cann
source /usr/local/Ascend/ascend-toolkit/set_env.sh

export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export STREAMS_PER_DEVICE=32

# pythonpath
export PYTHONPATH=/xxx/sglang/python:$PYTHONPATH
export PYTHONPATH=/home/jianzhnie/llmtuner/llm/sglang/python:$PYTHONPATH
export SGLANG_DEEPEP_BF16_DISPATCH=1

export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_BUFFSIZE=2048
export HCCL_SOCKET_IFNAME=enp66s0f5
export GLOO_SOCKET_IFNAME=enp66s0f5

export NODE0_IP=<node-0-ip>
export NODE1_IP=<node-1-ip>
export NODE2_IP=<node-2-ip>
export NODE3_IP=<node-3-ip>
export NODE4_IP=<node-4-ip>
export NODE5_IP=<node-5-ip>
export NODE6_IP=<node-6-ip>
export NODE7_IP=<node-7-ip>
P_IP=(
  "${NODE0_IP}"
  "${NODE1_IP}"
  "${NODE2_IP}"
  "${NODE3_IP}"
  "${NODE4_IP}"
  "${NODE5_IP}"
  "${NODE6_IP}"
  "${NODE7_IP}"
)
MASTER_PORT=5000
P_MASTER="${P_IP[0]}:${MASTER_PORT}"
NNODES=${#P_IP[@]}
TP_SIZE=64

MODEL_PATH=xxx
SERVER_HOST=0.0.0.0
SERVER_PORT=6677
SERVED_MODEL_NAME=longcat-flash

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
    --device npu  \
    --max-running-requests 16 \
    --context-length 8192 \
    --disable-radix-cache \
    --chunked-prefill-size 8192 \
    --watchdog-timeout 9000  \
    --prefill-round-robin-balance \
    --moe-a2a-backend deepep \
    --deepep-mode auto 