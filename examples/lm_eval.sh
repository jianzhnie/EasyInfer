# 设置缓存目录
export HF_HOME=//llm_workspace_1P/robin/hfhub
export HF_DATASETS_CACHE=/llm_workspace_1P/robin/hfhub/datasets
export HF_DATASETS_OFFLINE=0
export TRANSFORMERS_OFFLINE=0
export HF_HUB_OFFLINE=0

bash tools/eval/run_lmeval.sh \
    --model-path /llm_workspace_1P/robin/hfhub/pcl-kimi2-stage2/kimi2-mcore2hf_step450 \
    --output-dir outputs/mcore2hf_step450 \
    --backend api \
    --url "0.0.0.0" \
    --port 8080 \
    --tasks mmlu \
    --fewshot 5
