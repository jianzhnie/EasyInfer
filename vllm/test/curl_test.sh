#!/bin/bash

curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-30b",
    "messages": [{"role": "user", "content": "你是什么模型？"}],
    "max_tokens": 200,
    "stream": false
  }'
