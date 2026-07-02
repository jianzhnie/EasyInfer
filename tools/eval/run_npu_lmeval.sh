#!/bin/bash

# ==============================================================================
# LM-Evaluation-Harness Wrapper Script
# ==============================================================================
# Usage: bash run_lmeval.sh [MODEL_PATH] [OPTIONS]
# ==============================================================================
# Backends:
#   - vllm: Direct vLLM loading (fastest, requires GPU/NPU)
#   - hf:   HuggingFace backend (for comparison)
#   - api:  OpenAI-compatible API (flexible, supports remote servers)
# ==============================================================================

# Resolve paths independent of working directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../utils/common.sh
source "${SCRIPT_DIR}/../utils/common.sh"
PROJECT_ROOT=$(get_project_root)

# ------------------------------------------------------------------------------
# Help
# ------------------------------------------------------------------------------
usage() {
    cat << 'EOF'
Usage: run_lmeval.sh [MODEL_PATH] [OPTIONS]

Run lm-evaluation-harness benchmarks with multiple backend options.

Arguments:
  MODEL_PATH               Path or name of model (required, can be positional)

Backend Options:
  --backend TYPE              Backend type: 'vllm', 'hf', or 'api' (default: vllm)
                              - vllm: Direct vLLM loading (fastest)
                              - hf:   HuggingFace backend
                              - api:  OpenAI-compatible API (requires running server)

Evaluation Options:
  --tasks LIST                Comma- or space-separated tasks (default: wikitext)
  --fewshot N                 Number of few-shot examples (default: 0)
  --batch-size SIZE           Batch size, 'auto', or 'auto:N' (default: auto)
                              Note: API backend uses batch_size=1 internally (default
                              in lm-eval TemplateAPI), but the --batch_size flag is still
                              passed through. Do not rely on changing it for API mode.
  --max-batch-size N          Max batch size when --batch-size auto (default: 64)
  --output-dir DIR            Output directory (default: outputs/benchmark/lmeval)
  --limit N                   Limit number of samples per task (default: all)
  --log-samples               Save model outputs for debugging
  --seed SEED                 Random seed(s): single int or comma-separated 4 values
                              (seed,numpy_seed,torch_seed,fewshot_seed).
                              vllm/hf default: 0,1234,1234,1234 (lm-eval internal).
                              api: not set by default, uses lm-eval's internal default.

Model & Hardware Options (vllm/hf backends):
  -d, --devices DEVICES       Device IDs (default: 0)
  -t, --tp SIZE               Tensor parallel size (default: 1)
  --device TYPE               Override device type: cuda, npu, cpu (auto-detected).
                              Can also specify with index, e.g. cuda:0, npu:1.
  --max-model-len LEN         Max context length (input + output, default: 4096)
                              Controls KV cache size for vLLM; token limit for HF/API.

Generation Options (vllm / api backends):
  --max-gen-toks N            Max tokens to generate per sample (no default)
                              Passes to model_args max_gen_toks for vllm / api.
                              For HF backend use --gen-kwargs instead.
  --gen-kwargs KWARGS         Additional generation kwargs passed to lm_eval
                              Example: --gen-kwargs 'temperature=0.8,max_gen_toks=512'

vLLM Backend Options:
  --hccl-port PORT            HCCL base port for NPU (default: 60000).
                              Only effective on NPU devices.
  --gpu-memory UTIL           GPU memory utilization (default: 0.8)
  -q, --quantization [TYPE]   Quantization method (auto-set on NPU)
  -ep, --enable-expert-parallel
                              Enable expert parallelism for MoE models
  --compilation-config CONFIG
                              Compilation config (e.g., '{"cudagraph_mode": "FULL_DECODE_ONLY"}').
                              NPU-specific; may not apply to GPU.
  --enforce-eager             Use eager execution mode (disable graph capture)

HuggingFace Backend Options:
  -d, --devices DEVICES       Device IDs (default: 0)
  -t, --tp SIZE               Tensor parallel size (default: 1)

API Backend Options:
  --url URL                   API endpoint URL (default: http://127.0.0.1:PORT/v1/completions)
  --port PORT                 Server port (default: 8080)
  --model-name NAME           Model name sent to API (default: MODEL_PATH).
                              When different from MODEL_PATH, MODEL_PATH is used
                              as the local tokenizer path and MODEL_NAME is sent
                              in API requests.
  --num-concurrent N          Concurrent API requests (default: 1)
  --chat                      Use /v1/chat/completions endpoint with local-chat-completions model
                              Required for generative tasks (e.g. mmlu_generative) on chat-tuned models
  --apply-chat-template [TEMPLATE]
                              Apply model's chat template via HuggingFace tokenizer.
                              Optionally specify template name.
                              Note: incompatible with loglikelihood-based tasks (e.g. mmlu) —
                              use --chat + mmlu_generative instead.

Authentication & Network:
  --api-key-file PATH          Read API key from file (600 permissions recommended).
                              Prefer setting OPENAI_API_KEY env var instead.
  --verify-certificate        Verify SSL certificates (default: true)
  --no-verify-certificate     Disable SSL certificate verification
  --timeout SECONDS           Request timeout in seconds (default: 300)

Caching & Offline:
  --use-cache PATH            Cache model responses to avoid repeated inference
  --offline                   Offline mode (use cached datasets/models only)

Other Options:
  --trust-remote-code         Allow executing remote code from HuggingFace Hub
  -h, --help                  Show this help message

Examples:
  # vLLM backend (fastest, direct loading)
  run_lmeval.sh outputs/qwen-int8 --backend vllm --tasks wikitext -d 0
  run_lmeval.sh outputs/model --backend vllm \
      --tasks arc_challenge,arc_easy,boolq,hellaswag,openbookqa,piqa,winogrande \
      -d 0,1 -t 2

  # HuggingFace backend
  run_lmeval.sh outputs/qwen-int8 --backend hf --tasks wikitext -d 0

  # API backend (requires running server)
  bash tools/serve/deploy_vllm.sh outputs/qwen-int8 -d 0 -t 1
  run_lmeval.sh outputs/qwen-int8 --backend api --tasks wikitext

  # API backend with chat completions (for generative tasks)
  run_lmeval.sh outputs/qwen-int8 --backend api --tasks mmlu_generative \
      --chat --apply-chat-template --fewshot 5

  # Remote API (e.g., DeepSeek) — set key via env var (recommended)
  export OPENAI_API_KEY=sk-xxx
  run_lmeval.sh deepseek-chat --backend api \
      --url https://api.deepseek.com/v1/completions

  # Or read key from file
  run_lmeval.sh deepseek-chat --backend api \
      --url https://api.deepseek.com/v1/completions \
      --api-key-file /path/to/api_key

  # With generation kwargs and response caching
  run_lmeval.sh outputs/model --backend vllm --tasks gsm8k --fewshot 5 \
      --max-gen-toks 512 --use-cache .eval_cache/
EOF
}

# ------------------------------------------------------------------------------
# Default Configuration
# ------------------------------------------------------------------------------
MODEL_PATH=""
BACKEND="vllm"
TASKS="wikitext"
FEWSHOT=0
BATCH_SIZE="auto"
MAX_BATCH_SIZE=""
OUTPUT_DIR="outputs/benchmark/lmeval"
LIMIT=""
LOG_SAMPLES=false
OFFLINE=false
SEED=""

# Hardware options (vllm/hf backends)
DEVICES="0"
TP_SIZE=1
HCCL_PORT=""
MEM_UTIL=0.8
MAX_MODEL_LEN=4096
QUANT_METHOD=""
COMPILATION_CONFIG=""
ENABLE_EP=false
ENFORCE_EAGER=false
DEVICE_TYPE=""
EXPLICIT_DEVICE=""

# Generation options
MAX_GEN_TOKS=""
GEN_KWARGS=""

# API options (api backend)
API_URL=""
API_PORT=8080
MODEL_NAME=""
API_CHAT=false
APPLY_CHAT_TEMPLATE=""
NUM_CONCURRENT=""
MAX_RETRIES=""
VERIFY_CERTIFICATE=""
API_TIMEOUT=""

# Caching
USE_CACHE=""

# Other
TRUST_REMOTE_CODE=""

POSITIONAL_ARGS=()

# ------------------------------------------------------------------------------
# Argument Parsing
# ------------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case $1 in
        --backend)              BACKEND="$2"; shift 2 ;;
        --tasks)                TASKS="$2"; shift 2 ;;
        --fewshot|--num-fewshot) FEWSHOT="$2"; shift 2 ;;
        --batch-size)           BATCH_SIZE="$2"; shift 2 ;;
        --max-batch-size)       MAX_BATCH_SIZE="$2"; shift 2 ;;
        --output-dir)           OUTPUT_DIR="$2"; shift 2 ;;
        --limit)                LIMIT="$2"; shift 2 ;;
        --log-samples)          LOG_SAMPLES=true; shift 1 ;;
        --seed)                 SEED="$2"; shift 2 ;;
        # Model & Hardware options
        -d|--devices)           DEVICES="$2"; shift 2 ;;
        -t|--tp)                TP_SIZE="$2"; shift 2 ;;
        --device)               EXPLICIT_DEVICE="$2"; shift 2 ;;
        --hccl-port)            HCCL_PORT="$2"; shift 2 ;;
        --gpu-memory)           MEM_UTIL="$2"; shift 2 ;;
        --max-model-len)        MAX_MODEL_LEN="$2"; shift 2 ;;
        -q|--quantization)
            if [[ -n "${2:-}" && ! "$2" =~ ^- ]]; then
                QUANT_METHOD="$2"; shift 2
            else
                QUANT_METHOD="ascend"; shift 1
            fi ;;
        -ep|--enable-expert-parallel) ENABLE_EP=true; shift 1 ;;
        --compilation-config)   COMPILATION_CONFIG="$2"; shift 2 ;;
        --enforce-eager)        ENFORCE_EAGER=true; shift 1 ;;
        # Generation options
        --max-gen-toks)         MAX_GEN_TOKS="$2"; shift 2 ;;
        --gen-kwargs)           GEN_KWARGS="$2"; shift 2 ;;
        # API options
        --url)                  API_URL="$2"; shift 2 ;;
        --port)                 API_PORT="$2"; shift 2 ;;
        --model-name)           MODEL_NAME="$2"; shift 2 ;;
        --model-path)           MODEL_PATH="$2"; shift 2 ;;
        --num-concurrent)       NUM_CONCURRENT="$2"; shift 2 ;;
        --max-retries)          MAX_RETRIES="$2"; shift 2 ;;
        --chat)                 API_CHAT=true; shift 1 ;;
        --apply-chat-template)
            if [[ -n "${2:-}" && ! "$2" =~ ^- ]]; then
                APPLY_CHAT_TEMPLATE="$2"; shift 2
            else
                APPLY_CHAT_TEMPLATE="true"; shift 1
            fi ;;
        --api-key-file)
            if [[ -f "$2" ]]; then
                OPENAI_API_KEY=$(cat "$2")
                export OPENAI_API_KEY
            else
                log_warn "API key file not found: $2"
            fi
            shift 2 ;;
        --verify-certificate)   VERIFY_CERTIFICATE="true"; shift 1 ;;
        --no-verify-certificate) VERIFY_CERTIFICATE="false"; shift 1 ;;
        --timeout)              API_TIMEOUT="$2"; shift 2 ;;
        # Caching
        --use-cache)            USE_CACHE="$2"; shift 2 ;;
        # Other
        --trust-remote-code)    TRUST_REMOTE_CODE="true"; shift 1 ;;
        --offline)              OFFLINE=true; shift 1 ;;
        -h|--help)              usage; exit 0 ;;
        # Catch --model-path=VALUE (equals sign syntax)
        --model-path=*)         MODEL_PATH="${1#*=}"; shift 1 ;;
        *)                      POSITIONAL_ARGS+=("$1"); shift ;;
    esac
done

# Handle positional MODEL_PATH
if [[ -z "$MODEL_PATH" && ${#POSITIONAL_ARGS[@]} -gt 0 ]]; then
    MODEL_PATH="${POSITIONAL_ARGS[0]}"
fi

# Warn about extra positional arguments
if [[ ${#POSITIONAL_ARGS[@]} -gt 1 ]]; then
    log_warn "Extra positional arguments ignored: ${POSITIONAL_ARGS[*]:1}"
fi

# ------------------------------------------------------------------------------
# Validation
# ------------------------------------------------------------------------------
if [[ -z "$MODEL_PATH" ]]; then
    # Show usage first since log_error calls exit and won't return
    echo ""
    log_warn "Model path is required. Usage:"
    echo ""
    usage
    exit 1
fi

# Validate backend
case "$BACKEND" in
    vllm|hf|api) ;;
    *)
        log_warn "Invalid backend '$BACKEND'. Must be: vllm, hf, or api"
        echo ""
        usage
        exit 1
        ;;
esac

# Derive model name if not provided (for api backend)
if [[ -z "$MODEL_NAME" ]]; then
    MODEL_NAME="$MODEL_PATH"
fi

# ------------------------------------------------------------------------------
# Environment Setup (vllm/hf backends only)
# ------------------------------------------------------------------------------
if [[ "$BACKEND" != "api" ]]; then
    if [[ -n "$EXPLICIT_DEVICE" ]]; then
        DEVICE_TYPE="$EXPLICIT_DEVICE"
    elif [[ -z "$DEVICE_TYPE" ]]; then
        DEVICE_TYPE=$(detect_device)
    fi
    HCCL_PORT="${HCCL_PORT:-60000}"
    setup_env "$DEVICE_TYPE" "$DEVICES" "$TP_SIZE" "$HCCL_PORT"

    # Avoid OpenMP thread pool conflicts in multi-process vLLM environment
    export OMP_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1
fi

# vLLM defaults to fork workers, which is unsafe once the parent process
# has already touched CUDA/NPU. Force spawn unless the user explicitly overrides it.
if [[ "$BACKEND" == "vllm" && -z "${VLLM_WORKER_MULTIPROC_METHOD:-}" ]]; then
    export VLLM_WORKER_MULTIPROC_METHOD=spawn
fi

# ------------------------------------------------------------------------------
# Build Output Path
# ------------------------------------------------------------------------------
TIMESTAMP=$(get_timestamp)
ensure_dir "$OUTPUT_DIR"
OUTPUT_FILE="${OUTPUT_DIR}/${TASKS//,/_}_${TIMESTAMP}"

# ------------------------------------------------------------------------------
# Build Model Args (Unified key=value format)
# ------------------------------------------------------------------------------
# All model_args are built into a single comma-separated string passed to
# lm_eval's --model_args. Each backend has its own set of supported args.
# Reference: lm_eval API model classes (TemplateAPI, HFLM, VLLM)

if [[ "$BACKEND" == "api" ]]; then
    # ==============================
    # API Backend (local-completions / local-chat-completions)
    # Supported model_args: model, pretrained, base_url, tokenizer,
    #   tokenizer_backend, max_gen_toks, max_length, tokenized_requests,
    #   trust_remote_code, batch_size, num_concurrent, max_retries, seed,
    #   verify_certificate, ca_cert_path, auth_token, timeout, truncate,
    #   add_bos_token, custom_prefix_token_id, revision, use_fast_tokenizer,
    #   eos_string, header, max_images
    # ==============================

    # Build base_url if not explicitly provided
    if [[ -z "$API_URL" ]]; then
        if [[ "$API_CHAT" == true ]]; then
            API_URL="http://127.0.0.1:${API_PORT}/v1/chat/completions"
        else
            API_URL="http://127.0.0.1:${API_PORT}/v1/completions"
        fi
    fi

    # When --model-name differs from --model-path, separate API model ID from tokenizer:
    #   model/pretrained = API model name (sent in requests)
    #   tokenizer        = local path (for tokenization)
    if [[ "$MODEL_NAME" != "$MODEL_PATH" ]]; then
        MODEL_ARGS="model=${MODEL_NAME},tokenizer=${MODEL_PATH}"
    else
        MODEL_ARGS="pretrained=${MODEL_PATH}"
    fi
    MODEL_ARGS+=",base_url=${API_URL}"
    MODEL_ARGS+=",max_length=${MAX_MODEL_LEN}"
    if [[ "$TRUST_REMOTE_CODE" == "true" ]]; then
        MODEL_ARGS+=",trust_remote_code=True"
    fi
    # max_gen_toks: only set when user explicitly specifies, otherwise
    # let lm_eval use its own default (TemplateAPI: 256)
    [[ -n "$MAX_GEN_TOKS" ]] && MODEL_ARGS+=",max_gen_toks=${MAX_GEN_TOKS}"

    # tokenized_requests must be True when using --apply-chat-template,
    # otherwise apply_chat_template returns JsonChatStr which breaks _encode_pair
    if [[ -n "$APPLY_CHAT_TEMPLATE" ]]; then
        MODEL_ARGS+=",tokenized_requests=True"
    else
        MODEL_ARGS+=",tokenized_requests=False"
    fi

    # Optional API args — only append non-empty values
    [[ -n "$NUM_CONCURRENT" ]]   && MODEL_ARGS+=",num_concurrent=${NUM_CONCURRENT}"
    [[ -n "$MAX_RETRIES" ]]      && MODEL_ARGS+=",max_retries=${MAX_RETRIES}"
    [[ -n "$VERIFY_CERTIFICATE" ]] && MODEL_ARGS+=",verify_certificate=${VERIFY_CERTIFICATE}"
    [[ -n "$API_TIMEOUT" ]]      && MODEL_ARGS+=",timeout=${API_TIMEOUT}"

    if [[ "$API_CHAT" == true ]]; then
        LM_EVAL_MODEL="local-chat-completions"
    else
        LM_EVAL_MODEL="local-completions"
    fi

elif [[ "$BACKEND" == "vllm" ]]; then
    # ==============================
    # vLLM Backend
    # Supported model_args: pretrained, dtype, revision, trust_remote_code,
    #   tokenizer, tokenizer_mode, tokenizer_revision, tensor_parallel_size,
    #   quantization, max_gen_toks, max_length, max_model_len, seed,
    #   batch_size, max_batch_size, data_parallel_size, add_bos_token,
    #   prefix_token_id, lora_local_path, max_lora_rank, truncation_side,
    #   enable_thinking, chat_template_args, think_end_token
    #   Plus any extra kwargs passed directly to vllm.LLM (e.g. gpu_memory_utilization,
    #   enforce_eager, enable_expert_parallel, compilation_config)
    # ==============================

    MODEL_ARGS="pretrained=${MODEL_PATH}"
    if [[ "$TRUST_REMOTE_CODE" == "true" ]]; then
        MODEL_ARGS+=",trust_remote_code=True"
    fi
    MODEL_ARGS+=",dtype=auto"
    MODEL_ARGS+=",tensor_parallel_size=${TP_SIZE}"
    MODEL_ARGS+=",max_model_len=${MAX_MODEL_LEN}"
    MODEL_ARGS+=",gpu_memory_utilization=${MEM_UTIL}"
    # max_gen_toks: only set when user explicitly specifies, otherwise
    # let lm_eval/vLLM use its own default (VLLM model: 256)
    [[ -n "$MAX_GEN_TOKS" ]] && MODEL_ARGS+=",max_gen_toks=${MAX_GEN_TOKS}"

    if [[ -n "$QUANT_METHOD" ]]; then
        MODEL_ARGS+=",quantization=${QUANT_METHOD}"
    fi
    if [[ "$ENABLE_EP" == true ]]; then
        MODEL_ARGS+=",enable_expert_parallel=True"
    fi
    if [[ -n "$COMPILATION_CONFIG" ]]; then
        MODEL_ARGS+=",compilation_config=${COMPILATION_CONFIG}"
    fi
    if [[ "$ENFORCE_EAGER" == true ]]; then
        MODEL_ARGS+=",enforce_eager=True"
    fi
    LM_EVAL_MODEL="vllm"

elif [[ "$BACKEND" == "hf" ]]; then
    # ==============================
    # HuggingFace Backend
    # Supported model_args: pretrained, backend, revision, subfolder,
    #   tokenizer, truncation, logits_cache, max_length, device, dtype,
    #   softmax_dtype, mixed_precision_dtype, batch_size, max_batch_size,
    #   trust_remote_code, use_fast_tokenizer, add_bos_token, prefix_token_id,
    #   parallelize, max_memory_per_gpu, max_cpu_memory, offload_folder,
    #   tp_plan, peft, delta, autogptq, gptqmodel, gguf_file,
    #   think_end_token, enable_thinking, chat_template_args
    # ==============================

    MODEL_ARGS="pretrained=${MODEL_PATH}"
    if [[ "$TRUST_REMOTE_CODE" == "true" ]]; then
        MODEL_ARGS+=",trust_remote_code=True"
    fi
    MODEL_ARGS+=",max_length=${MAX_MODEL_LEN}"
    MODEL_ARGS+=",dtype=auto"
    # HFLM does not support max_gen_toks as a model_arg.
    # Use --gen-kwargs 'max_gen_toks=N' or rely on task defaults instead.

    # Device: use explicit --device, or auto-detect from DEVICE_TYPE
    if [[ -n "$EXPLICIT_DEVICE" ]]; then
        MODEL_ARGS+=",device=${EXPLICIT_DEVICE}"
    elif [[ "$DEVICE_TYPE" == "npu" ]]; then
        MODEL_ARGS+=",device=npu:${DEVICES##*,}"
    else
        MODEL_ARGS+=",device=${DEVICE_TYPE}:${DEVICES##*,}"
    fi

    if [[ "$TP_SIZE" -gt 1 ]]; then
        MODEL_ARGS+=",parallelize=True"
    fi
    LM_EVAL_MODEL="hf"
fi

# ------------------------------------------------------------------------------
# Display Configuration
# ------------------------------------------------------------------------------
log_header "LM-Evaluation-Harness"
log_info "Model" "$MODEL_PATH"
log_info "Backend" "$BACKEND"

if [[ "$BACKEND" == "vllm" ]]; then
    log_info "Device" "${DEVICE_TYPE^^} ($DEVICES)"
    log_info "TP Size" "$TP_SIZE"
    log_info "MP Method" "${VLLM_WORKER_MULTIPROC_METHOD:-default}"
    [[ -n "$QUANT_METHOD" ]] && log_info "Quant" "$QUANT_METHOD"
    log_info "Max Model Len" "$MAX_MODEL_LEN"
elif [[ "$BACKEND" == "hf" ]]; then
    log_info "Device" "${DEVICE_TYPE^^} ($DEVICES)"
    log_info "Max Length" "$MAX_MODEL_LEN"
elif [[ "$BACKEND" == "api" ]]; then
    log_info "API URL" "$API_URL"
    log_info "Max Length" "$MAX_MODEL_LEN"
    [[ "$API_CHAT" == true ]] && log_info "Chat Mode" "enabled"
fi

log_info "Tasks" "$TASKS"
log_info "Fewshot" "$FEWSHOT"
[[ -n "$MAX_GEN_TOKS" ]] && log_info "Max Gen Toks" "$MAX_GEN_TOKS"
log_info "Batch Size" "$BATCH_SIZE"
log_info "Output" "$OUTPUT_FILE"
[[ -n "$LIMIT" ]] && log_info "Limit" "$LIMIT samples per task"
[[ "$LOG_SAMPLES" == true ]] && log_info "Log Samples" "enabled"
[[ -n "$APPLY_CHAT_TEMPLATE" ]] && log_info "Chat Template" "$APPLY_CHAT_TEMPLATE"
[[ -n "$GEN_KWARGS" ]] && log_info "Gen Kwargs" "$GEN_KWARGS"
[[ -n "$USE_CACHE" ]] && log_info "Cache" "$USE_CACHE"
[[ -n "$SEED" ]] && log_info "Seed" "$SEED"

# ------------------------------------------------------------------------------
# Verify Dependencies
# ------------------------------------------------------------------------------
require_command "lm-eval" \
    "'lm-evaluation-harness' not found. Install with: pip install lm-eval[api]"

# ------------------------------------------------------------------------------
# Server Health Check (api backend only)
# ------------------------------------------------------------------------------
# Note: check_vllm_server calls log_error (which exits) when the server is
# unreachable, so we don't need an explicit exit check afterwards. The || exit 1
# is a defensive fallback in case common.sh behavior changes.
if [[ "$BACKEND" == "api" ]]; then
    log_info "Checking" "Server connectivity and model identity..."
    check_vllm_server --port "$API_PORT" --model "$MODEL_PATH" || exit 1
fi

# ------------------------------------------------------------------------------
# Execute Evaluation
# ------------------------------------------------------------------------------
log_header "Launching lm_eval..."

# Disable torch extension autoload in API mode to avoid torch_npu errors.
# In vllm/hf modes, the plugin system needs to be active.
if [[ "$BACKEND" == "api" ]]; then
    export TORCH_DEVICE_BACKEND_AUTOLOAD=0
fi

# Offline mode: use cached datasets/models only, no network access
if [[ "$OFFLINE" == true ]]; then
    export HF_DATASETS_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
fi

# Build optional arguments array (avoids set -e issues with conditional substitution)
OPTIONAL_ARGS=()
[[ -n "$LIMIT" ]]            && OPTIONAL_ARGS+=(--limit "$LIMIT")
[[ "$LOG_SAMPLES" == true ]] && OPTIONAL_ARGS+=(--log_samples)
# --apply_chat_template: nargs='?' with const=True.
# Bare flag → pass without value to trigger const=True.
# With template name → pass the template name as value.
if [[ -n "$APPLY_CHAT_TEMPLATE" ]]; then
    if [[ "$APPLY_CHAT_TEMPLATE" == "true" ]]; then
        OPTIONAL_ARGS+=(--apply_chat_template)
    else
        OPTIONAL_ARGS+=(--apply_chat_template "$APPLY_CHAT_TEMPLATE")
    fi
fi
[[ -n "$GEN_KWARGS" ]]       && OPTIONAL_ARGS+=(--gen_kwargs "$GEN_KWARGS")
[[ -n "$SEED" ]]             && OPTIONAL_ARGS+=(--seed "$SEED")
[[ -n "$MAX_BATCH_SIZE" ]]   && OPTIONAL_ARGS+=(--max_batch_size "$MAX_BATCH_SIZE")
[[ -n "$USE_CACHE" ]]        && OPTIONAL_ARGS+=(--use_cache "$USE_CACHE")
[[ "$TRUST_REMOTE_CODE" == "true" ]] && OPTIONAL_ARGS+=(--trust_remote_code)

# Use lm-eval run subcommand (new CLI) with legacy fallback
# The 'run' subcommand is the canonical interface in lm_eval >= 0.4.x
PYTHONUNBUFFERED=1 lm-eval run \
    --model "$LM_EVAL_MODEL" \
    --model_args "$MODEL_ARGS" \
    --tasks "$TASKS" \
    --batch_size "$BATCH_SIZE" \
    --num_fewshot "$FEWSHOT" \
    --output_path "$OUTPUT_FILE" \
    "${OPTIONAL_ARGS[@]}"

log_success "Evaluation completed"
log_info "Results" "$(dirname "$OUTPUT_FILE")"
