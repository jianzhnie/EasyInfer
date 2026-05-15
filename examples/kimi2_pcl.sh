#!/bin/bash

vllm serve "/llm_workspace_1P/robin/hfhub/pcl-kimi2-stage2/kimi2-mcore2hf_step450" \
    --distributed-executor-backend ray \
    --tensor-parallel-size 64 \
    --enable-expert-parallel \
    --max-model-len 4096 \
    --trust-remote-code \
    --enable-prefix-caching \
    --enforce-eager \
    --host 0.0.0.0 \
    --port 8080 \
    --hf-overrides '{"model_type":"kimi_k2_mcore","architectures":["KimiK2MCoreV1ForCausalLM"]}'