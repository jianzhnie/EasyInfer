#!/usr/bin/env python3
"""ModelScope batch downloader with completeness verification.

Flow: verify all models (in parallel) -> surgically download missing/bad
files -> verify again -> retry until everything passes check_weights or
MAX_ROUNDS is reached. Verification (remote file list + exact size +
safetensors structure, optionally + sha256) is the authoritative completion
signal; the modelscope CLI exit code is not trusted.

Downloads are surgical: the checker reports exactly which files are missing
or bad, bad files are deleted (the modelscope client never replaces existing
files), and only those files are fetched. A full-repo download only happens
for fresh/incomplete-by-a-lot models (see MAX_TARGETED_FILES).

This is the implementation; tools/ms_download.sh is a thin wrapper that just
execs this script with the right interpreter.

Env knobs:
  FORCE_OVERWRITE=true    wipe each model dir before downloading (DANGEROUS)
  RUN_IN_BACKGROUND=false download sequentially instead of all in parallel
  SKIP_WEIGHTS=true       download everything except weight files
  CHECK_BEFORE_DOWNLOAD=false  skip the up-front verification pass
  VERIFY_SHA256=true      include full SHA-256 in every verification (slow:
                          reads all data; makes pre-download checks slow too)
  CLEAN_TEMP=true         delete a model's ._____temp dir once it verifies
                          complete (frees space; kills resume data)
  MAX_ROUNDS=5            max download->verify rounds (0 = verify only)
  RETRY_DELAY=10          seconds to wait between rounds
  MS_MAX_WORKERS=16       --max-workers passed to modelscope
  MAX_TARGETED_FILES=500  fetch files individually when the bad list has at
                          most this many entries; otherwise full-repo download
  MODELS_FILE=<path>      read model entries ("repo_id|local_dir" per line,
                          '#' comments allowed) instead of the built-in table
  LOG_DIR=<path>          where per-model logs go (default: <script_dir>/logs)

Exit code: 0 = all models verified complete, 1 = some incomplete, 130 = interrupted.
"""

import os
import shutil
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import check_weights  # noqa: E402  local module (also sets TORCH_DEVICE_BACKEND_AUTOLOAD=0)


def env_bool(name, default=False):
    return os.environ.get(name, str(default)).lower() in ("1", "true", "yes")


FORCE_OVERWRITE = env_bool("FORCE_OVERWRITE")
RUN_IN_BACKGROUND = env_bool("RUN_IN_BACKGROUND", True)
SKIP_WEIGHTS = env_bool("SKIP_WEIGHTS")
CHECK_BEFORE_DOWNLOAD = env_bool("CHECK_BEFORE_DOWNLOAD", True)
VERIFY_SHA256 = env_bool("VERIFY_SHA256")
CLEAN_TEMP = env_bool("CLEAN_TEMP")
MAX_ROUNDS = int(os.environ.get("MAX_ROUNDS", "5"))
RETRY_DELAY = int(os.environ.get("RETRY_DELAY", "10"))
MS_MAX_WORKERS = os.environ.get("MS_MAX_WORKERS", "16")
MAX_TARGETED_FILES = int(os.environ.get("MAX_TARGETED_FILES", "500"))
LOG_DIR = Path(os.environ.get("LOG_DIR", str(SCRIPT_DIR / "logs")))

LOCAL_DIR_PREFIX = "/home/jianzhnie/llmtuner/hfhub/models"

# Model table: (repo_id, local_dir) — the single source of truth.
# Comment out entries you do not want.
MODELS = [
    ## Meituan
    "meituan-longcat/LongCat-Flash-Lite",
    ## Quantized models (Eco-Tech): GLM
    "Eco-Tech/GLM-5-w8a8",
    "Eco-Tech/GLM-5-w4a8",
    "Eco-Tech/GLM-5.1-w4a8",
    "Eco-Tech/GLM-5.2-w8a8",
    "Eco-Tech/GLM-5.2-w4a8c8",
    ## Kimi
    "Eco-Tech/Kimi-K2.6-w4a8",
    "Eco-Tech/Kimi-K2.7-Code-w4a8",
    ## DeepSeek
    "Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp",
    "Eco-Tech/DeepSeek-V4-Pro-w4a8-mtp",
    ## MiniMax
    "Eco-Tech/MiniMax-M2.7-w8a8-QuaRot",
    "Eco-Tech/MiniMax-M3-w8a8",
    ## Step 
    "Eco-Tech/Step-3.7-Flash-w8a8-mtp",
]

WEIGHT_EXCLUDES = ["--exclude", "*.safetensors", "--exclude", "*.bin",
                   "--exclude", "*.pt", "--exclude", "*.ckpt"]

CHILDREN = []  # live subprocess.Popen objects, killed on interrupt


def log(msg, level="INFO"):
    print(f"[{time.strftime('%H:%M:%S')}] [{level}] {msg}", flush=True)


def load_models():
    path = os.environ.get("MODELS_FILE")
    if not path:
        return MODELS
    models = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        repo, sep, local = line.partition("|")
        if not sep:
            log(f"MODELS_FILE: ignoring malformed line: {line!r}", "WARN")
            continue
        models.append((repo.strip(), local.strip()))
    return models


def checker_ns(fix=False):
    return SimpleNamespace(offline=False, sha256=VERIFY_SHA256, skip_weights=SKIP_WEIGHTS,
                           workers=8, show=10, fix=fix, list_bad=None)


def verify(repo_id, local_dir, fix=False):
    """Returns (status, lines, bad_files) — see check_weights.check_model."""
    try:
        return check_weights.check_model(repo_id, local_dir, checker_ns(fix=fix))
    except Exception as exc:
        return "error", [f"  ERROR unexpected: {exc!r}"], []


def clean_temp(local_dir):
    temp = Path(local_dir) / "._____temp"
    if not local_dir.startswith(MODELS_BASE + "/") or not temp.is_dir():
        return
    log(f"CLEAN_TEMP: removing {temp}")
    shutil.rmtree(temp, ignore_errors=True)


def wipe_dir(local_dir):
    """FORCE_OVERWRITE support; refuses paths outside MODELS_BASE."""
    if not local_dir.startswith(MODELS_BASE + "/"):
        log(f"FORCE_OVERWRITE refused for path outside {MODELS_BASE}: {local_dir}", "ERROR")
        return False
    if Path(local_dir).is_dir():
        log(f"FORCE_OVERWRITE: wiping {local_dir}")
        shutil.rmtree(local_dir)
    Path(local_dir).mkdir(parents=True, exist_ok=True)
    return True


def download_model(repo_id, local_dir, round_no):
    """Verify-then-download. Returns a Popen in background mode, else None."""
    log_file = LOG_DIR / f"{repo_id.replace('/', '_')}.log"

    # Ask the checker which files are missing/bad; fix=True deletes the bad
    # ones (the modelscope client never replaces existing files, so they must go).
    status, _, bad = verify(repo_id, local_dir, fix=True)
    if status == "ok":
        log(f"{repo_id} already complete, nothing to download.")
        return None

    cmd = ["modelscope", "download", "--max-workers", MS_MAX_WORKERS, repo_id]
    if status == "incomplete" and 0 < len(bad) <= MAX_TARGETED_FILES:
        cmd += bad  # surgical: fetch only the missing/bad files (positional args)
        mode = f"{len(bad)} file(s)"
    else:
        # Full-repo download (checker errored, or too many files to list).
        if SKIP_WEIGHTS:
            cmd += WEIGHT_EXCLUDES
        mode = "full repo"
    cmd += ["--local_dir", local_dir]

    log(f"round {round_no}: downloading {repo_id} ({mode}; log: {log_file})")
    lf = open(log_file, "ab")
    lf.write(f"===== round {round_no} {time.strftime('%F %T')} : {mode} =====\n".encode())
    lf.flush()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            start_new_session=True)
    CHILDREN.append(proc)
    if RUN_IN_BACKGROUND:
        # Pump output to the log only; the console stays readable.
        def _drain():
            for chunk in iter(lambda: proc.stdout.read(4096), b""):
                lf.write(chunk)
            lf.close()

        import threading
        threading.Thread(target=_drain, daemon=True).start()
        return proc

    # Sequential mode: stream to both console and log (tee-like).
    for chunk in iter(lambda: proc.stdout.read(4096), b""):
        lf.write(chunk)
        sys.stdout.buffer.write(chunk)
        sys.stdout.buffer.flush()
    lf.close()
    proc.wait()
    CHILDREN.remove(proc)
    return None


def on_signal(signum, _frame):
    log(f"interrupted (signal {signum}) — stopping background jobs", "WARN")
    for proc in CHILDREN:
        try:
            os.killpg(proc.pid, signal.SIGTERM)  # start_new_session => pgid == pid
        except (ProcessLookupError, PermissionError):
            pass
    sys.exit(130)


def main():
    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    if not shutil.which("modelscope"):
        log("modelscope command not found. Please install it first.", "ERROR")
        return 1
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    models = load_models()

    # --- Up-front verification: only incomplete models enter the download loop ---
    pending = []
    final_status = {}
    if CHECK_BEFORE_DOWNLOAD and not FORCE_OVERWRITE:
        log(f"Verifying existing downloads ({len(models)} models, parallel)...")
        with ThreadPoolExecutor(max_workers=len(models)) as pool:
            reports = list(pool.map(lambda m: verify(*m), models))
        for (repo_id, local_dir), (status, lines, _) in zip(models, reports):
            print(f">>> {repo_id} -> {local_dir}")
            for line in lines:
                print(line)
            if status == "ok":
                final_status[repo_id] = "OK (already complete)"
                if CLEAN_TEMP:
                    clean_temp(local_dir)
            else:
                pending.append((repo_id, local_dir))
    else:
        for repo_id, local_dir in models:
            if FORCE_OVERWRITE and not wipe_dir(local_dir):
                final_status[repo_id] = "INCOMPLETE"
                continue
            pending.append((repo_id, local_dir))
    log(f"{len(final_status)} already complete, {len(pending)} to download.")

    # --- Download -> verify -> retry loop ---
    round_no = 1
    while pending and round_no <= MAX_ROUNDS:
        log(f"=== Round {round_no}/{MAX_ROUNDS}: {len(pending)} model(s) ===")
        procs = [p for m in pending if (p := download_model(*m, round_no)) is not None]
        for proc in procs:  # background mode only; sequential mode already waited
            proc.wait()
            CHILDREN.remove(proc)

        still = []
        for repo_id, local_dir in pending:
            status, _, _ = verify(repo_id, local_dir)
            if status == "ok":
                final_status[repo_id] = f"OK (round {round_no})"
                log(f"{repo_id} verified complete.")
                if CLEAN_TEMP:
                    clean_temp(local_dir)
            else:
                still.append((repo_id, local_dir))
                log(f"{repo_id} still incomplete after round {round_no}.", "WARN")
        pending = still
        if pending and round_no < MAX_ROUNDS:
            time.sleep(RETRY_DELAY)
        round_no += 1

    # --- Summary ---
    print()
    log("================ SUMMARY ================")
    for repo_id, _ in pending:
        final_status[repo_id] = "INCOMPLETE"
    fail = False
    for repo_id, _ in models:
        status = final_status.get(repo_id, "SKIPPED")
        print(f"  {repo_id:<40} {status}")
        fail = fail or status == "INCOMPLETE"
    if fail:
        log(f"Some models are still incomplete after {MAX_ROUNDS} rounds. "
            "Re-run this script to resume.", "ERROR")
        return 1
    log("All models verified 100% complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
