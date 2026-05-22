set -euo pipefail

# 设置缓存目录
export HF_HOME="${HF_HOME:-/llm_workspace_1P/robin/hfhub/cache}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

bash /llm_workspace_1P/robin/npuslim/tools/eval/run_lmeval.sh \
    --model-path /llm_workspace_1P/robin/hfhub/models/meituan-longcat/expand/LongCat-Flash-Chat-1024E-512Zero-E-Topk24-v2 \
    --output-dir /llm_workspace_1P/robin/EasyInfer/output/LongCat-Flash-Chat-1024E-512Zero-E-Topk24-v2 \
    --model-name longcat-flash \
    --backend api \
    --port 8000  \
    --tasks hendrycks_math500,minerva_math500                                                                                                                               