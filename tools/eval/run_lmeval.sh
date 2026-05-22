#!/usr/bin/env bash
# =============================================================================
# lm-evaluation-harness 评测运行脚本
# =============================================================================
# 支持本地 HuggingFace 数据集评测，可通过环境变量或 CLI 参数配置。
#
# 用法:
#   # 使用本地 HF 缓存（API 后端）
#   ./run_lmeval.sh --model-path /data/model --tasks mmlu,gsm8k
#
#   # 使用自定义 YAML 任务配置（本地 JSON/CSV 数据集）
#   ./run_lmeval.sh --model-path /data/model --task-dir /data/custom_tasks
#
#   # 离线模式（要求所有数据集已在本地缓存）
#   HF_DATASETS_OFFLINE=1 ./run_lmeval.sh --model-path /data/model --tasks mmlu
#
#   # 指定本地 HF 数据目录
#   HF_HOME=/shared/hf_cache ./run_lmeval.sh --model-path /data/model --tasks mmlu
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# 默认值
# ---------------------------------------------------------------------------
MODEL_PATH="${MODEL_PATH:-}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/lmeval}"
BACKEND="${BACKEND:-api}"
URL="${URL:-0.0.0.0}"
PORT="${PORT:-8080}"
TASKS="${TASKS:-mmlu}"
FEWSHOT="${FEWSHOT:-5}"
BATCH_SIZE="${BATCH_SIZE:-auto}"
TASK_DIR="${TASK_DIR:-}"
NUM_FEWSHOT="${NUM_FEWSHOT:-${FEWSHOT}}"

# HF 缓存与离线配置
HF_HOME="${HF_HOME:-/llm_workspace_1P/robin/hfhub}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-0}"
TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"
HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-true}"

# ---------------------------------------------------------------------------
# 参数解析
# ---------------------------------------------------------------------------
usage() {
    cat <<'EOF'
用法: run_lmeval.sh [选项]

选项:
  --model-path PATH    模型路径（本地目录或 HF Hub 名称）
  --output-dir DIR     结果输出目录 (默认: outputs/lmeval)
  --backend TYPE       评测后端: api, hf, vllm (默认: api)
  --url URL            API 服务地址 (默认: 0.0.0.0)
  --port PORT          API 服务端口 (默认: 8080)
  --tasks LIST         评测任务，逗号分隔 (默认: mmlu)
  --fewshot N          Few-shot 样本数 (默认: 5)
  --batch-size SIZE    批大小 (默认: auto)
  --task-dir DIR       自定义 YAML 任务配置目录（用于本地数据集）
  -h, --help           显示帮助信息

环境变量:
  HF_HOME              HF 缓存根目录
  HF_DATASETS_CACHE    数据集缓存目录
  HF_DATASETS_OFFLINE  设为 1 强制离线模式
  MODEL_PATH           模型路径（同 --model-path）
  BACKEND              评测后端（同 --backend）
EOF
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model-path)   MODEL_PATH="$2";   shift 2 ;;
        --output-dir)   OUTPUT_DIR="$2";   shift 2 ;;
        --backend)      BACKEND="$2";      shift 2 ;;
        --url)          URL="$2";          shift 2 ;;
        --port)         PORT="$2";         shift 2 ;;
        --tasks)        TASKS="$2";        shift 2 ;;
        --fewshot)      NUM_FEWSHOT="$2";  shift 2 ;;
        --batch-size)   BATCH_SIZE="$2";   shift 2 ;;
        --task-dir)     TASK_DIR="$2";     shift 2 ;;
        -h|--help)      usage ;;
        *) echo "未知参数: $1" >&2; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# 前置检查
# ---------------------------------------------------------------------------
if ! command -v lm_eval &>/dev/null; then
    echo "[ERROR] lm_eval 未安装。请运行: pip install lm-eval" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# 导出环境变量
# ---------------------------------------------------------------------------
export HF_HOME
export HF_DATASETS_CACHE
export HF_DATASETS_OFFLINE
export TRANSFORMERS_OFFLINE
export HF_HUB_OFFLINE

mkdir -p "$OUTPUT_DIR"

# ---------------------------------------------------------------------------
# 构建评测命令
# ---------------------------------------------------------------------------
# shellcheck disable=SC2086
CMD=(
    lm_eval
    --num_fewshot "$NUM_FEWSHOT"
    --batch_size "$BATCH_SIZE"
    --output_path "$OUTPUT_DIR"
    --log_samples
)

# 添加自定义任务目录
if [[ -n "$TASK_DIR" ]]; then
    CMD+=(--include_path "$TASK_DIR")
fi

# 按后端设置模型参数
case "$BACKEND" in
    api)
        if [[ -z "$MODEL_PATH" ]]; then
            echo "[ERROR] api 后端需要 --model-path 指定模型名称" >&2
            exit 1
        fi
        CMD+=(--model local-chat-completions)
        CMD+=(--model_args "model=${MODEL_PATH},base_url=http://${URL}:${PORT}/v1")
        ;;
    hf)
        if [[ -z "$MODEL_PATH" ]]; then
            echo "[ERROR] hf 后端需要 --model-path 指定模型路径" >&2
            exit 1
        fi
        CMD+=(--model hf)
        CMD+=(--model_args "pretrained=${MODEL_PATH},trust_remote_code=${TRUST_REMOTE_CODE}")
        ;;
    vllm)
        if [[ -z "$MODEL_PATH" ]]; then
            echo "[ERROR] vllm 后端需要 --model-path 指定模型路径" >&2
            exit 1
        fi
        CMD+=(--model vllm)
        CMD+=(--model_args "pretrained=${MODEL_PATH},trust_remote_code=${TRUST_REMOTE_CODE}")
        ;;
    *)
        echo "[ERROR] 不支持的后端: $BACKEND (支持: api, hf, vllm)" >&2
        exit 1
        ;;
esac

CMD+=(--tasks "$TASKS")

# ---------------------------------------------------------------------------
# 打印配置并执行
# ---------------------------------------------------------------------------
echo "========================================"
echo " lm-evaluation-harness"
echo "========================================"
echo " 后端:        $BACKEND"
echo " 模型:        $MODEL_PATH"
echo " 任务:        $TASKS"
echo " Few-shot:    $NUM_FEWSHOT"
echo " 批大小:      $BATCH_SIZE"
echo " 输出目录:    $OUTPUT_DIR"
echo " HF_HOME:     $HF_HOME"
echo " 数据集缓存:  $HF_DATASETS_CACHE"
echo " 离线模式:    $HF_DATASETS_OFFLINE"
if [[ -n "$TASK_DIR" ]]; then
echo " 自定义任务:  $TASK_DIR"
fi
echo "========================================"

echo "[INFO] 执行命令: ${CMD[*]}"
echo ""

"${CMD[@]}"

echo ""
echo "[INFO] 评测完成。结果保存在: $OUTPUT_DIR"
