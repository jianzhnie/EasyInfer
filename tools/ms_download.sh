#!/bin/bash
set -uo pipefail

# 是否强制覆盖（如果设为 true，则在下载前先清理非权重文件）
FORCE_OVERWRITE=${FORCE_OVERWRITE:-false}
# 是否后台并行执行（如果设为 true，则所有下载任务并行启动）
RUN_IN_BACKGROUND=${RUN_IN_BACKGROUND:-true}
# 是否跳过权重文件下载（如果设为 true，则只下载配置文件）
SKIP_WEIGHTS=${SKIP_WEIGHTS:-true}

if ! command -v modelscope &>/dev/null; then
    echo "[ERROR] modelscope command not found. Please install it first."
    exit 1
fi

# 检查本地权重是否完整
# 返回 0 = 完整（跳过下载）, 1 = 不完整（需要下载）
check_weights_complete() {
    local local_dir=$1

    [ -d "$local_dir" ] || return 1

    python3 - "$local_dir" << 'PYEOF'
import json, os, sys, re

local_dir = sys.argv[1]

# 方式 1: 通过 safetensors index 文件验证
for fname in os.listdir(local_dir):
    if fname.endswith('.safetensors.index.json'):
        idx_path = os.path.join(local_dir, fname)
        try:
            with open(idx_path) as f:
                idx = json.load(f)
            expected = set(idx.get('weight_map', {}).values())
            local_safetensors = set(
                f for f in os.listdir(local_dir) if f.endswith('.safetensors')
            )
            missing = expected - local_safetensors
            if missing:
                print(f"[WARN] Missing {len(missing)} weight files (via index): {sorted(missing)[:5]}...")
                sys.exit(1)
            print(f"[INFO] All {len(expected)} weight files verified via index file.")
            sys.exit(0)
        except (json.JSONDecodeError, KeyError):
            pass

# 方式 2: 通过分片命名规则验证 (xxx-00001-of-NNNNN.safetensors)
safetensors_files = [f for f in os.listdir(local_dir) if f.endswith('.safetensors')]
shard_info = None
for f in safetensors_files:
    m = re.match(r'^(.+-)(\d+)-of-0*(\d+)\.safetensors$', f)
    if m:
        prefix = m.group(1)
        shard_num = int(m.group(2))
        total = int(m.group(3))
        shard_info = (prefix, total, len(m.group(2)))
        break

if shard_info:
    prefix, total, digits = shard_info
    existing = set()
    for f in safetensors_files:
        m = re.match(r'^.+-(\d+)-of-\d+\.safetensors$', f)
        if m:
            existing.add(int(m.group(1)))
    expected = set(range(1, total + 1))
    missing = expected - existing
    if missing:
        print(f"[WARN] Missing {len(missing)}/{total} shards: {sorted(missing)[:10]}...")
        sys.exit(1)
    # 也检查是否有额外/重复的分片
    extra = existing - expected
    if extra:
        print(f"[WARN] Extra shards found: {sorted(extra)[:10]}")
    print(f"[INFO] All {total} shards verified ({len(existing)} files found).")
    sys.exit(0)

# 方式 3: 兜底 - 有权重文件但无法验证完整性（非分片模型如单个 pytorch_model.bin）
has_weights = any(
    f.endswith(ext) for f in os.listdir(local_dir)
    for ext in ('.safetensors', '.bin', '.pt', '.ckpt')
)
if has_weights:
    print("[WARN] Weights found but cannot verify completeness (no index/shard pattern). Assuming complete.")
    sys.exit(0)

print("[INFO] No weight files found.")
sys.exit(1)
PYEOF
}

# 基础下载函数
download_model() {
    local repo_id=$1
    local local_dir=$2
    local log_file="${repo_id//\//_}.log"

    local -a exclude_args=()
    if [ "$SKIP_WEIGHTS" = "true" ]; then
        echo "[INFO] SKIP_WEIGHTS is true. Excluding large weight files (*.safetensors, *.bin, *.pt, *.ckpt) ..."
        exclude_args+=(--exclude "*.safetensors" --exclude "*.bin" --exclude "*.pt" --exclude "*.ckpt")
    fi

    if [ "$FORCE_OVERWRITE" = "true" ]; then
        if [ -d "$local_dir" ]; then
            echo "[INFO] Force overwrite enabled. Cleaning all files in $local_dir ..."
            rm -rf "$local_dir"/*
        fi
        # 检查权重是否已完整，完整则跳过
        # if check_weights_complete "$local_dir"; then
        #     echo "[SKIP] Weights already complete in $local_dir, skipping download."
        #     return 0
        # fi
    fi

    echo "[INFO] Starting download: $repo_id to $local_dir (Log: $log_file)"

    if [ "$RUN_IN_BACKGROUND" = "true" ]; then
        nohup modelscope download --max-workers 16 "$repo_id" --local_dir "$local_dir" "${exclude_args[@]}" > "$log_file" 2>&1 &
    else
        modelscope download --max-workers 16 "$repo_id" --local_dir "$local_dir" "${exclude_args[@]}" 2>&1 | tee "$log_file"
    fi
}


## 1. modelscope / Meituan
# download_model "meituan-longcat/LongCat-Flash-Lite" "/home/jianzhnie/llmtuner/hfhub/models/meituan-longcat/LongCat-Flash-Lite"

## 2. Quantized Models (Eco-Tech)
BASE_ECO="/home/jianzhnie/llmtuner/hfhub/models/Eco-Tech"
# download_model "Eco-Tech/GLM-5-w8a8" "$BASE_ECO/GLM-5-w8a8"
# download_model "Eco-Tech/GLM-5-w4a8" "$BASE_ECO/GLM-5-w4a8"
# download_model "Eco-Tech/GLM-5.1-w8a8" "$BASE_ECO/GLM-5.1-w8a8"
# download_model "Eco-Tech/GLM-5.1-w4a8" "$BASE_ECO/GLM-5.1-w4a8"
download_model "Eco-Tech/GLM-5.2-w8a8" "/home/jianzhnie/llmtuner/hfhub/models/ZhipuAI/GLM-5.2-w8a8"

# kimi
# download_model "Eco-Tech/Kimi-K2.6-w4a8" "$BASE_ECO/Kimi-K2.6-w4a8"
# download_model "Eco-Tech/Kimi-K2.7-Code-w4a8" "$BASE_ECO/Kimi-K2.7-Code-w4a8"

# # deepseek
# download_model "Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp" "$BASE_ECO/DeepSeek-V4-Flash-w8a8-mtp"
# download_model "Eco-Tech/DeepSeek-V4-Pro-w4a8-mtp" "$BASE_ECO/DeepSeek-V4-Pro-w4a8-mtp"

# # minimax
# download_model "Eco-Tech/MiniMax-M2.7-w8a8-QuaRot" "$BASE_ECO/MiniMax-M2.7-w8a8-QuaRot"
# download_model "Eco-Tech/MiniMax-M3-w8a8" "$BASE_ECO/MiniMax-M3-w8a8"

# step
# download_model "Eco-Tech/Step-3.7-Flash-w8a8-mtp" "$BASE_ECO/Step-3.7-Flash-w8a8-mtp"

if [ "$RUN_IN_BACKGROUND" = "true" ]; then
    echo "[INFO] All downloads launched. Waiting for completion..."
    wait
    echo "[INFO] All downloads finished."
fi
