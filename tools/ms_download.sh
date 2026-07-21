#!/bin/bash
# ModelScope batch downloader with completeness verification.
#
# Flow: verify all models -> download the incomplete ones -> verify again ->
# retry until everything passes check_weights.py or MAX_ROUNDS is reached.
# Verification (remote file list + exact size + safetensors structure) is the
# authoritative completion signal; the modelscope CLI exit code is not trusted.
#
# Downloads are surgical: the checker reports exactly which files are missing
# or bad, bad files are deleted (the modelscope client never replaces existing
# files), and only those files are fetched. A full-repo download only happens
# for fresh/incomplete-by-a-lot models (see MAX_TARGETED_FILES).
#
# Env knobs:
#   FORCE_OVERWRITE=true    wipe each model dir before downloading (DANGEROUS)
#   RUN_IN_BACKGROUND=false download sequentially instead of all in parallel
#   SKIP_WEIGHTS=true       download everything except weight files
#   CHECK_BEFORE_DOWNLOAD=false  skip the up-front verification pass
#   MAX_ROUNDS=5            max download->verify rounds
#   RETRY_DELAY=10          seconds to wait between rounds
#   MS_MAX_WORKERS=16       --max-workers passed to modelscope
#   MAX_TARGETED_FILES=500  fetch files individually when the bad list has at
#                           most this many entries; otherwise full-repo download
#   MODELS_FILE=<path>      read model entries ("repo_id|local_dir" per line,
#                           '#' comments allowed) from this file instead of the
#                           built-in MODELS table
#   LOG_DIR=<path>          where per-model logs go (default: <script_dir>/logs)
#   PYTHON_BIN=<path>       python with modelscope installed (default: derived
#                           from the modelscope CLI location)
set -uo pipefail

FORCE_OVERWRITE=${FORCE_OVERWRITE:-false}
RUN_IN_BACKGROUND=${RUN_IN_BACKGROUND:-true}
SKIP_WEIGHTS=${SKIP_WEIGHTS:-false}
CHECK_BEFORE_DOWNLOAD=${CHECK_BEFORE_DOWNLOAD:-true}
MAX_ROUNDS=${MAX_ROUNDS:-5}
RETRY_DELAY=${RETRY_DELAY:-10}
MS_MAX_WORKERS=${MS_MAX_WORKERS:-16}
MAX_TARGETED_FILES=${MAX_TARGETED_FILES:-500}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR=${LOG_DIR:-"$SCRIPT_DIR/logs"}
CHECKER="$SCRIPT_DIR/check_weights.py"

MODELS_BASE="/home/jianzhnie/llmtuner/hfhub/models"

# Model table: "repo_id|local_dir" — the single source of truth.
# Comment out entries you do not want.
MODELS=(
    ## Meituan
    # "meituan-longcat/LongCat-Flash-Lite|$MODELS_BASE/meituan-longcat/LongCat-Flash-Lite"

    ## Quantized models (Eco-Tech): GLM
    "Eco-Tech/GLM-5-w8a8|$MODELS_BASE/Eco-Tech/GLM-5-w8a8"
    "Eco-Tech/GLM-5-w4a8|$MODELS_BASE/Eco-Tech/GLM-5-w4a8"
    "Eco-Tech/GLM-5.1-w8a8|$MODELS_BASE/Eco-Tech/GLM-5.1-w8a8"
    "Eco-Tech/GLM-5.1-w4a8|$MODELS_BASE/Eco-Tech/GLM-5.1-w4a8"
    "Eco-Tech/GLM-5.2-w8a8|$MODELS_BASE/Eco-Tech/GLM-5.2-w8a8"
    "Eco-Tech/GLM-5.2-w4a8c8|$MODELS_BASE/Eco-Tech/GLM-5.2-w4a8c8"

    ## Kimi
    "Eco-Tech/Kimi-K2.6-w4a8|$MODELS_BASE/Eco-Tech/Kimi-K2.6-w4a8"
    "Eco-Tech/Kimi-K2.7-Code-w4a8|$MODELS_BASE/Eco-Tech/Kimi-K2.7-Code-w4a8"

    ## DeepSeek
    "Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp|$MODELS_BASE/Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp"
    "Eco-Tech/DeepSeek-V4-Pro-w4a8-mtp|$MODELS_BASE/Eco-Tech/DeepSeek-V4-Pro-w4a8-mtp"

    ## MiniMax
    "Eco-Tech/MiniMax-M2.7-w8a8-QuaRot|$MODELS_BASE/Eco-Tech/MiniMax-M2.7-w8a8-QuaRot"
    "Eco-Tech/MiniMax-M3-w8a8|$MODELS_BASE/Eco-Tech/MiniMax-M3-w8a8"

    ## Step
    "Eco-Tech/Step-3.7-Flash-w8a8-mtp|$MODELS_BASE/Eco-Tech/Step-3.7-Flash-w8a8-mtp"
)

# Optional override of the model table from a file.
if [ -n "${MODELS_FILE:-}" ]; then
    MODELS=()
    while IFS= read -r line; do
        [[ -z "$line" || "$line" == \#* ]] && continue
        MODELS+=("$line")
    done < "$MODELS_FILE"
fi

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# torch_npu (present in some vllm envs) crashes "import torch" unless backend
# autoload is disabled; the modelscope CLI imports torch at startup.
export TORCH_DEVICE_BACKEND_AUTOLOAD="${TORCH_DEVICE_BACKEND_AUTOLOAD:-0}"

if ! command -v modelscope &>/dev/null; then
    log "[ERROR] modelscope command not found. Please install it first."
    exit 1
fi
if [ ! -f "$CHECKER" ]; then
    log "[ERROR] checker not found: $CHECKER"
    exit 1
fi
if [ -z "${PYTHON_BIN:-}" ]; then
    PYTHON_BIN="$(dirname "$(command -v modelscope)")/python"
    [ -x "$PYTHON_BIN" ] || PYTHON_BIN="python3"
fi
mkdir -p "$LOG_DIR"

CHECK_ARGS=()
[ "$SKIP_WEIGHTS" = "true" ] && CHECK_ARGS+=(--skip-weights)

# Returns 0 only when the model is verified 100% complete.
# $3 = "quiet" to suppress the checker report.
is_complete() {
    local repo_id=$1 local_dir=$2 mode=${3:-quiet}
    if [ "$mode" = "verbose" ]; then
        "$PYTHON_BIN" "$CHECKER" "${CHECK_ARGS[@]}" "$repo_id:$local_dir"
    else
        "$PYTHON_BIN" "$CHECKER" "${CHECK_ARGS[@]}" "$repo_id:$local_dir" >/dev/null 2>&1
    fi
}

wipe_dir() {
    local local_dir=$1
    # Safety: refuse to wipe anything outside the models base dir.
    case "$local_dir" in
        "$MODELS_BASE"/*) ;;
        *) log "[ERROR] FORCE_OVERWRITE refused for path outside $MODELS_BASE: $local_dir"; return 1 ;;
    esac
    if [ -d "$local_dir" ]; then
        log "[INFO] FORCE_OVERWRITE: wiping $local_dir"
        rm -rf -- "$local_dir"
    fi
    mkdir -p "$local_dir"
}

download_model() {
    local repo_id=$1 local_dir=$2 round=$3
    local log_file="$LOG_DIR/${repo_id//\//_}.log"

    # Ask the checker which files are missing/bad; --fix deletes the bad ones
    # (the modelscope client never replaces existing files, so they must go).
    local bad_list rc
    bad_list=$(mktemp)
    "$PYTHON_BIN" "$CHECKER" "${CHECK_ARGS[@]}" --fix --list-bad "$bad_list" \
        "$repo_id:$local_dir" >/dev/null 2>&1
    rc=$?
    if [ "$rc" -eq 0 ]; then
        rm -f "$bad_list"
        log "[INFO] $repo_id already complete, nothing to download."
        return 0
    fi

    local -a dl_files=()
    if [ "$rc" -eq 1 ] && [ -s "$bad_list" ]; then
        mapfile -t dl_files < "$bad_list"
    fi
    rm -f "$bad_list"

    local -a cmd=(modelscope download --max-workers "$MS_MAX_WORKERS" "$repo_id")
    local mode
    if [ "${#dl_files[@]}" -gt 0 ] && [ "${#dl_files[@]}" -le "$MAX_TARGETED_FILES" ]; then
        # Surgical: fetch only the missing/bad files (positional file args).
        cmd+=("${dl_files[@]}")
        mode="${#dl_files[@]} file(s)"
    else
        # Full-repo download (checker errored, or too many files to list).
        if [ "$SKIP_WEIGHTS" = "true" ]; then
            cmd+=(--exclude "*.safetensors" --exclude "*.bin" --exclude "*.pt" --exclude "*.ckpt")
        fi
        mode="full repo"
    fi
    cmd+=(--local_dir "$local_dir")

    echo "===== round $round $(date '+%F %T') : $mode =====" >>"$log_file"
    log "[INFO] round $round: downloading $repo_id ($mode; log: $log_file)"
    if [ "$RUN_IN_BACKGROUND" = "true" ]; then
        nohup "${cmd[@]}" >>"$log_file" 2>&1 &
    else
        "${cmd[@]}" 2>&1 | tee -a "$log_file"
    fi
}

# --- Up-front verification: only incomplete models enter the download loop ---
pending=()
declare -A FINAL_STATUS=()
if [ "$CHECK_BEFORE_DOWNLOAD" = "true" ] && [ "$FORCE_OVERWRITE" != "true" ]; then
    log "[INFO] Verifying existing downloads (${#MODELS[@]} models)..."
    for entry in "${MODELS[@]}"; do
        repo_id=${entry%%|*}; local_dir=${entry#*|}
        if is_complete "$repo_id" "$local_dir" verbose; then
            FINAL_STATUS[$repo_id]="OK (already complete)"
        else
            pending+=("$entry")
        fi
    done
else
    for entry in "${MODELS[@]}"; do
        repo_id=${entry%%|*}; local_dir=${entry#*|}
        if [ "$FORCE_OVERWRITE" = "true" ]; then
            if ! wipe_dir "$local_dir"; then
                FINAL_STATUS[$repo_id]="INCOMPLETE"
                continue
            fi
        fi
        pending+=("$entry")
    done
fi
log "[INFO] ${#FINAL_STATUS[@]} already complete, ${#pending[@]} to download."

# --- Download -> verify -> retry loop ---
round=1
while [ "${#pending[@]}" -gt 0 ] && [ "$round" -le "$MAX_ROUNDS" ]; do
    log "[INFO] === Round $round/$MAX_ROUNDS: ${#pending[@]} model(s) ==="
    for entry in "${pending[@]}"; do
        download_model "${entry%%|*}" "${entry#*|}" "$round"
    done
    if [ "$RUN_IN_BACKGROUND" = "true" ]; then
        log "[INFO] Waiting for downloads to finish..."
        wait
    fi

    next_pending=()
    for entry in "${pending[@]}"; do
        repo_id=${entry%%|*}; local_dir=${entry#*|}
        if is_complete "$repo_id" "$local_dir"; then
            FINAL_STATUS[$repo_id]="OK (round $round)"
            log "[INFO] $repo_id verified complete."
        else
            next_pending+=("$entry")
            log "[WARN] $repo_id still incomplete after round $round."
        fi
    done
    if [ "${#next_pending[@]}" -gt 0 ]; then
        pending=("${next_pending[@]}")
    else
        pending=()
    fi
    if [ "${#pending[@]}" -gt 0 ] && [ "$round" -lt "$MAX_ROUNDS" ]; then
        sleep "$RETRY_DELAY"
    fi
    round=$((round + 1))
done

# --- Summary ---
echo
log "================ SUMMARY ================"
for entry in "${pending[@]}"; do
    FINAL_STATUS[${entry%%|*}]="INCOMPLETE"
done
fail=0
for entry in "${MODELS[@]}"; do
    repo_id=${entry%%|*}
    status=${FINAL_STATUS[$repo_id]:-SKIPPED}
    printf "  %-40s %s\n" "$repo_id" "$status"
    [ "$status" = "INCOMPLETE" ] && fail=1
done
if [ "$fail" -eq 0 ]; then
    log "[INFO] All models verified 100% complete."
else
    log "[ERROR] Some models are still incomplete after $MAX_ROUNDS rounds. Re-run this script to resume."
fi
exit "$fail"
