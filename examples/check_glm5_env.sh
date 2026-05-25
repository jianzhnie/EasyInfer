#!/bin/bash
# =============================================================================
# GLM-5/GLM-5.1 部署验证脚本
# =============================================================================
# 在启动 vLLM 前检查环境配置是否正确
#
# 用法:
#   ./check_glm5_env.sh
#   ./check_glm5_env.sh --hardware=a3 --quant=w4a8
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../scripts/common.sh
source "${SCRIPT_DIR}/../scripts/common.sh"

# 默认配置
HARDWARE="${HARDWARE:-a2}"  # a2, a3
QUANT_TYPE="${QUANT_TYPE:-w4a8}"  # w4a8, w8a8, bf16

log_info "========================================"
log_info "GLM-5/GLM-5.1 环境检查"
log_info "========================================"

# 解析参数
while [[ $# -gt 0 ]]; do
    case "$1" in
        --hardware=*)
            HARDWARE="${1#*=}"
            shift
            ;;
        --quant=*)
            QUANT_TYPE="${1#*=}"
            shift
            ;;
        *)
            shift
            ;;
    esac
done

echo "硬件类型: $HARDWARE"
echo "量化类型: $QUANT_TYPE"
echo ""

# 检查 NPU 数量
echo "[1/8] 检查 NPU 设备..."
if command -v npu-smi &> /dev/null; then
    NPU_COUNT=$(npu-smi info -l 2>/dev/null | grep -c "NPU ID" || echo "0")
    echo "  ✓ NPU 数量: $NPU_COUNT"

    case "$HARDWARE" in
        a2)
            if [[ "$NPU_COUNT" -ne 8 ]]; then
                echo "  ⚠️ 警告: A2 硬件期望 8 NPU，但检测到 $NPU_COUNT"
            fi
            ;;
        a3)
            if [[ "$NPU_COUNT" -ne 16 ]]; then
                echo "  ⚠️ 警告: A3 硬件期望 16 NPU，但检测到 $NPU_COUNT"
            fi
            ;;
    esac
else
    echo "  ✗ npu-smi 未找到，跳过 NPU 检查"
fi

# 检查量化类型与硬件兼容性
echo ""
echo "[2/8] 检查量化与硬件兼容性..."
case "$QUANT_TYPE" in
    w4a8)
        echo "  ✓ W4A8 支持 A2 (8卡) 和 A3 (16卡)"
        ;;
    w8a8)
        if [[ "$HARDWARE" != "a3" ]]; then
            echo "  ✗ W8A8 仅支持 A3 (16卡) 硬件"
            exit 1
        fi
        echo "  ✓ W8A8 仅支持 A3 硬件，检查通过"
        ;;
    bf16)
        echo "  ✓ BF16 需要多节点部署 (至少 2×16卡)"
        ;;
    *)
        echo "  ✗ 未知量化类型: $QUANT_TYPE"
        exit 1
        ;;
esac

# 检查环境变量
echo ""
echo "[3/8] 检查环境变量..."
ENV_VARS=(
    "HCCL_OP_EXPANSION_MODE"
    "OMP_PROC_BIND"
    "OMP_NUM_THREADS"
    "HCCL_BUFFSIZE"
    "PYTORCH_NPU_ALLOC_CONF"
    "VLLM_ASCEND_BALANCE_SCHEDULING"
)

for var in "${ENV_VARS[@]}"; do
    if [[ -n "${!var:-}" ]]; then
        echo "  ✓ $var=${!var}"
    else
        echo "  ⚠️ $var 未设置"
    fi
done

# 检查 W8A8 必需变量
if [[ "$QUANT_TYPE" == "w8a8" ]]; then
    if [[ -n "${VLLM_ASCEND_ENABLE_MLAPO:-}" ]]; then
        echo "  ✓ VLLM_ASCEND_ENABLE_MLAPO=$VLLM_ASCEND_ENABLE_MLAPO (W8A8必需)"
    else
        echo "  ✗ W8A8 需要 VLLM_ASCEND_ENABLE_MLAPO=1"
        exit 1
    fi
fi

# 检查 Python 和 vllm
echo ""
echo "[4/8] 检查 Python 环境..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
    echo "  ✓ Python: $PYTHON_VERSION"
else
    echo "  ✗ Python3 未找到"
    exit 1
fi

if command -v vllm &> /dev/null; then
    VLLM_VERSION=$(vllm --version 2>&1 | head -1)
    echo "  ✓ vLLM: $VLLM_VERSION"
else
    echo "  ✗ vllm 命令未找到"
    exit 1
fi

# 检查 PyTorch NPU
echo ""
echo "[5/8] 检查 PyTorch NPU..."
python3 -c "import torch; import torch_npu; print(f'  ✓ torch_npu 可用, NPU设备数: {torch.npu.device_count()}')" 2>/dev/null || {
    echo "  ✗ torch_npu 不可用或 NPU 未正确配置"
    exit 1
}

# 检查模型路径
echo ""
echo "[6/8] 检查模型路径..."
if [[ -n "${MODEL_PATH:-}" ]]; then
    MODEL_DIR="$MODEL_PATH"
else
    case "$HARDWARE" in
        a2)
            MODEL_DIR="/root/.cache/modelscope/hub/models/vllm-ascend/GLM-5-w4a8"
            ;;
        a3)
            case "$QUANT_TYPE" in
                w4a8)
                    MODEL_DIR="/root/.cache/modelscope/hub/models/vllm-ascend/GLM5-w4a8"
                    ;;
                w8a8)
                    MODEL_DIR="/root/.cache/modelscope/hub/models/vllm-ascend/GLM5-w8a8"
                    ;;
                bf16)
                    MODEL_DIR="/root/.cache/modelscope/hub/models/vllm-ascend/GLM5-bf16"
                    ;;
            esac
            ;;
    esac
fi

if [[ -d "$MODEL_DIR" ]]; then
    echo "  ✓ 模型目录存在: $MODEL_DIR"

    # 检查关键文件
    if [[ -f "$MODEL_DIR/config.json" ]]; then
        echo "  ✓ config.json 存在"
    else
        echo "  ✗ config.json 不存在"
        exit 1
    fi

    # 检查 tokenizer
    if [[ -f "$MODEL_DIR/tokenizer.json" ]] || [[ -f "$MODEL_DIR/tokenizer.model" ]]; then
        echo "  ✓ tokenizer 文件存在"
    else
        echo "  ⚠️ tokenizer 文件可能缺失"
    fi
else
    echo "  ✗ 模型目录不存在: $MODEL_DIR"
    echo "     请从官方链接下载模型权重"
    exit 1
fi

# 检查磁盘空间
echo ""
echo "[7/8] 检查磁盘空间..."
MODEL_SIZE=$(du -sh "$MODEL_DIR" 2>/dev/null | cut -f1 || echo "unknown")
echo "  模型大小: $MODEL_SIZE"

AVAILABLE=$(df -h /root/.cache 2>/dev/null | tail -1 | awk '{print $4}' || echo "unknown")
echo "  可用空间: $AVAILABLE"

# 检查端口
echo ""
echo "[8/8] 检查端口..."
PORT="${PORT:-8077}"
if command -v ss &> /dev/null && ss -tln | grep -q ":$PORT "; then
    echo "  ⚠️ 端口 $PORT 已被占用"
elif command -v netstat &> /dev/null && netstat -tln 2>/dev/null | grep -q ":$PORT "; then
    echo "  ⚠️ 端口 $PORT 已被占用"
else
    echo "  ✓ 端口 $PORT 可用"
fi

echo ""
echo "========================================"
echo "环境检查完成"
echo "========================================"
echo ""
echo "建议的启动命令:"

case "$HARDWARE" in
    a2)
        echo "  QUANT_TYPE=$QUANT_TYPE ./glm5_server.sh"
        ;;
    a3)
        case "$QUANT_TYPE" in
            w4a8)
                echo "  QUANT_TYPE=w4a8 TENSOR_PARALLEL_SIZE=16 MAX_MODEL_LEN=200000 MAX_NUM_SEQS=8 ./glm5_server.sh"
                ;;
            w8a8)
                echo "  QUANT_TYPE=w8a8 TENSOR_PARALLEL_SIZE=16 MAX_MODEL_LEN=40960 MAX_NUM_SEQS=8 ./glm5_server.sh"
                ;;
        esac
        ;;
esac

echo ""
