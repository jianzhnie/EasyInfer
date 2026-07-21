#!/usr/bin/env python3
"""ModelScope batch downloader with completeness verification.

Flow: verify all models (in parallel) -> surgically download missing/bad
files -> verify again -> retry until everything passes check_weights or
--max-rounds is reached. Verification (remote file list + exact size +
safetensors structure, optionally + sha256) is the authoritative completion
signal; the modelscope CLI exit code is not trusted.

Downloads are surgical: the checker reports exactly which files are missing
or bad, bad files are deleted (the modelscope client never replaces existing
files), and only those files are fetched. A full-repo download only happens
for fresh/incomplete-by-a-lot models (see --max-targeted-files).

Usage:
    python tools/ms_download.py [OPTIONS]
    bash  tools/ms_download.sh [OPTIONS]     # wrapper picks the right python

Every option can also be set via the env var shown in its --help text;
command-line arguments take precedence over environment variables.

Examples:
    python tools/ms_download.py --max-rounds 0          # verify only, no download
    python tools/ms_download.py --skip-weights          # configs only
    MAX_ROUNDS=0 bash tools/ms_download.sh              # env-var style still works

Exit code: 0 = all models verified complete, 1 = some incomplete, 130 = interrupted.
"""

import argparse
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

DEFAULT_LOCAL_DIR_PREFIX = "/home/jianzhnie/llmtuner/hfhub/models"

# Model table: repo ids. Local dir = <local-dir-prefix>/<repo_id>.
# Comment out entries you do not want.
MODEL_REPOS = [
    ## Meituan
    # "meituan-longcat/LongCat-Flash-Lite",

    ## Quantized models (Eco-Tech): GLM
    "Eco-Tech/GLM-5-w8a8",
    "Eco-Tech/GLM-5-w4a8",
    "Eco-Tech/GLM-5.1-w8a8",
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


def env_bool(name, default=False):
    return os.environ.get(name, str(default)).lower() in ("1", "true", "yes")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force-overwrite", action="store_true", default=env_bool("FORCE_OVERWRITE"),
                    help="wipe each model dir before downloading (DANGEROUS) [env FORCE_OVERWRITE]")
    ap.add_argument("--sequential", dest="run_in_background", action="store_false",
                    default=env_bool("RUN_IN_BACKGROUND", True),
                    help="download sequentially instead of all in parallel [env RUN_IN_BACKGROUND=false]")
    ap.add_argument("--skip-weights", action="store_true", default=env_bool("SKIP_WEIGHTS"),
                    help="download everything except weight files [env SKIP_WEIGHTS]")
    ap.add_argument("--no-check-first", dest="check_before_download", action="store_false",
                    default=env_bool("CHECK_BEFORE_DOWNLOAD", True),
                    help="skip the up-front verification pass [env CHECK_BEFORE_DOWNLOAD=false]")
    ap.add_argument("--sha256", dest="verify_sha256", action="store_true", default=env_bool("VERIFY_SHA256"),
                    help="include full SHA-256 in every verification (slow: reads all data; "
                         "makes pre-download checks slow too) [env VERIFY_SHA256]")
    ap.add_argument("--clean-temp", action="store_true", default=env_bool("CLEAN_TEMP"),
                    help="delete a model's ._____temp dir once it verifies complete [env CLEAN_TEMP]")
    ap.add_argument("--max-rounds", type=int, default=int(os.environ.get("MAX_ROUNDS", "5")),
                    help="max download->verify rounds; 0 = verify only [env MAX_ROUNDS, default 5]")
    ap.add_argument("--retry-delay", type=int, default=int(os.environ.get("RETRY_DELAY", "10")),
                    help="seconds to wait between rounds [env RETRY_DELAY, default 10]")
    ap.add_argument("--max-workers", default=os.environ.get("MS_MAX_WORKERS", "16"),
                    help="--max-workers passed to modelscope [env MS_MAX_WORKERS, default 16]")
    ap.add_argument("--max-targeted-files", type=int,
                    default=int(os.environ.get("MAX_TARGETED_FILES", "500")),
                    help="fetch files individually when the bad list has at most this many "
                         "entries; otherwise full-repo download [env MAX_TARGETED_FILES, default 500]")
    ap.add_argument("--log-dir", default=os.environ.get("LOG_DIR", str(SCRIPT_DIR / "logs")),
                    help="where per-model logs go [env LOG_DIR, default <script_dir>/logs]")
    ap.add_argument("--local-dir-prefix", default=os.environ.get("LOCAL_DIR_PREFIX", DEFAULT_LOCAL_DIR_PREFIX),
                    help="root dir for built-in table local dirs "
                         f"[env LOCAL_DIR_PREFIX, default {DEFAULT_LOCAL_DIR_PREFIX}]")
    return ap.parse_args(argv)


def load_models(cfg):
    """Returns [(repo_id, local_dir)] from the built-in table; local dirs are
    derived from --local-dir-prefix."""
    return [(repo, f"{cfg.local_dir_prefix}/{repo}") for repo in MODEL_REPOS]


def checker_ns(cfg, fix=False):
    return SimpleNamespace(offline=False, sha256=cfg.verify_sha256, skip_weights=cfg.skip_weights,
                           workers=8, show=10, fix=fix, list_bad=None)


def verify(cfg, repo_id, local_dir, fix=False):
    """Returns (status, lines, bad_files) — see check_weights.check_model."""
    try:
        return check_weights.check_model(repo_id, local_dir, checker_ns(cfg, fix=fix))
    except Exception as exc:
        return "error", [f"  ERROR unexpected: {exc!r}"], []


def clean_temp(cfg, local_dir):
    temp = Path(local_dir) / "._____temp"
    if not local_dir.startswith(cfg.local_dir_prefix + "/") or not temp.is_dir():
        return
    log(f"clean-temp: removing {temp}")
    shutil.rmtree(temp, ignore_errors=True)


def wipe_dir(cfg, local_dir):
    """--force-overwrite support; refuses paths outside --local-dir-prefix."""
    if not local_dir.startswith(cfg.local_dir_prefix + "/"):
        log(f"force-overwrite refused for path outside {cfg.local_dir_prefix}: {local_dir}", "ERROR")
        return False
    if Path(local_dir).is_dir():
        log(f"force-overwrite: wiping {local_dir}")
        shutil.rmtree(local_dir)
    Path(local_dir).mkdir(parents=True, exist_ok=True)
    return True


def download_model(cfg, repo_id, local_dir, round_no):
    """Verify-then-download. Returns a Popen in background mode, else None."""
    log_file = Path(cfg.log_dir) / f"{repo_id.replace('/', '_')}.log"

    # Ask the checker which files are missing/bad; fix=True deletes the bad
    # ones (the modelscope client never replaces existing files, so they must go).
    status, _, bad = verify(cfg, repo_id, local_dir, fix=True)
    if status == "ok":
        log(f"{repo_id} already complete, nothing to download.")
        return None

    cmd = ["modelscope", "download", "--max-workers", cfg.max_workers, repo_id]
    if status == "incomplete" and 0 < len(bad) <= cfg.max_targeted_files:
        cmd += bad  # surgical: fetch only the missing/bad files (positional args)
        mode = f"{len(bad)} file(s)"
    else:
        # Full-repo download (checker errored, or too many files to list).
        if cfg.skip_weights:
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
    if cfg.run_in_background:
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


def main(argv=None):
    cfg = parse_args(argv)
    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    if not shutil.which("modelscope"):
        log("modelscope command not found. Please install it first.", "ERROR")
        return 1
    Path(cfg.log_dir).mkdir(parents=True, exist_ok=True)
    models = load_models(cfg)

    # --- Up-front verification: only incomplete models enter the download loop ---
    pending = []
    final_status = {}
    if cfg.check_before_download and not cfg.force_overwrite:
        log(f"Verifying existing downloads ({len(models)} models, parallel)...")
        with ThreadPoolExecutor(max_workers=len(models)) as pool:
            reports = list(pool.map(lambda m: verify(cfg, *m), models))
        for (repo_id, local_dir), (status, lines, _) in zip(models, reports):
            print(f">>> {repo_id} -> {local_dir}")
            for line in lines:
                print(line)
            if status == "ok":
                final_status[repo_id] = "OK (already complete)"
                if cfg.clean_temp:
                    clean_temp(cfg, local_dir)
            else:
                pending.append((repo_id, local_dir))
    else:
        for repo_id, local_dir in models:
            if cfg.force_overwrite and not wipe_dir(cfg, local_dir):
                final_status[repo_id] = "INCOMPLETE"
                continue
            pending.append((repo_id, local_dir))
    log(f"{len(final_status)} already complete, {len(pending)} to download.")

    # --- Download -> verify -> retry loop ---
    round_no = 1
    while pending and round_no <= cfg.max_rounds:
        log(f"=== Round {round_no}/{cfg.max_rounds}: {len(pending)} model(s) ===")
        procs = [p for m in pending if (p := download_model(cfg, *m, round_no)) is not None]
        for proc in procs:  # background mode only; sequential mode already waited
            proc.wait()
            CHILDREN.remove(proc)

        still = []
        for repo_id, local_dir in pending:
            status, _, _ = verify(cfg, repo_id, local_dir)
            if status == "ok":
                final_status[repo_id] = f"OK (round {round_no})"
                log(f"{repo_id} verified complete.")
                if cfg.clean_temp:
                    clean_temp(cfg, local_dir)
            else:
                still.append((repo_id, local_dir))
                log(f"{repo_id} still incomplete after round {round_no}.", "WARN")
        pending = still
        if pending and round_no < cfg.max_rounds:
            time.sleep(cfg.retry_delay)
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
        log(f"Some models are still incomplete after {cfg.max_rounds} rounds. "
            "Re-run this script to resume.", "ERROR")
        return 1
    log("All models verified 100% complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
