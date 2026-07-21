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
    python tools/ms_download.py [OPTIONS]    # run with a python that has modelscope installed

Every option can also be set via the env var shown in its --help text;
command-line arguments take precedence over environment variables.

Examples:
    python tools/ms_download.py --max-rounds 0          # verify only, no download
    python tools/ms_download.py --skip-weights          # configs only
    MAX_ROUNDS=0 python tools/ms_download.py            # env-var style also works

Exit code: 0 = all models verified complete, 1 = some incomplete, 130 = interrupted.
"""

import argparse
import contextlib
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

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

WEIGHT_EXCLUDES = [
    "--exclude",
    "*.safetensors",
    "--exclude",
    "*.bin",
    "--exclude",
    "*.pt",
    "--exclude",
    "*.ckpt",
]

# CLI options: (flag, dest, kind, env var, default, help text).
# kind is "store_true"/"store_false" for booleans, or a type (int/str) for
# valued options. dest=None derives the attribute name from the flag.
OPTIONS = [
    (
        "--force-overwrite",
        None,
        "store_true",
        "FORCE_OVERWRITE",
        False,
        "wipe each model dir before downloading (DANGEROUS)",
    ),
    (
        "--sequential",
        "run_in_background",
        "store_false",
        "RUN_IN_BACKGROUND",
        True,
        "download sequentially instead of all in parallel",
    ),
    (
        "--skip-weights",
        None,
        "store_true",
        "SKIP_WEIGHTS",
        False,
        "download everything except weight files",
    ),
    (
        "--no-check-first",
        "check_before_download",
        "store_false",
        "CHECK_BEFORE_DOWNLOAD",
        True,
        "skip the up-front verification pass",
    ),
    (
        "--sha256",
        "verify_sha256",
        "store_true",
        "VERIFY_SHA256",
        False,
        "include full SHA-256 in every verification (slow: reads all data; "
        "makes pre-download checks slow too)",
    ),
    (
        "--clean-temp",
        None,
        "store_true",
        "CLEAN_TEMP",
        False,
        "delete a model's ._____temp dir once it verifies complete",
    ),
    (
        "--max-rounds",
        None,
        int,
        "MAX_ROUNDS",
        5,
        "max download->verify rounds; 0 = verify only",
    ),
    ("--retry-delay", None, int, "RETRY_DELAY", 10, "seconds to wait between rounds"),
    (
        "--max-workers",
        None,
        str,
        "MS_MAX_WORKERS",
        "16",
        "--max-workers passed to modelscope",
    ),
    (
        "--max-targeted-files",
        None,
        int,
        "MAX_TARGETED_FILES",
        500,
        "fetch files individually when the bad list has at most this many entries; "
        "otherwise full-repo download",
    ),
    (
        "--log-dir",
        None,
        str,
        "LOG_DIR",
        str(SCRIPT_DIR / "logs"),
        "where per-model logs go",
    ),
    (
        "--local-dir-prefix",
        None,
        str,
        "LOCAL_DIR_PREFIX",
        DEFAULT_LOCAL_DIR_PREFIX,
        "root dir for built-in table local dirs (local dir = prefix/repo_id)",
    ),
]

CHILDREN = []  # live subprocess.Popen objects, killed on interrupt


def log(msg, level="INFO"):
    print(f"[{time.strftime('%H:%M:%S')}] [{level}] {msg}", flush=True)


def env_bool(name, default=False):
    return os.environ.get(name, str(default)).lower() in ("1", "true", "yes")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    for flag, dest, kind, env, default, help_text in OPTIONS:
        kwargs = {"help": f"{help_text} [env {env}, default {default}]"}
        if kind in ("store_true", "store_false"):
            kwargs.update(action=kind, default=env_bool(env, default))
        else:
            kwargs.update(type=kind, default=kind(os.environ.get(env, default)))
        if dest:
            kwargs["dest"] = dest
        parser.add_argument(flag, **kwargs)
    return parser.parse_args(argv)


def load_models(cfg):
    """[(repo_id, local_dir)] from the built-in table; local dirs are
    derived from --local-dir-prefix."""
    return [(repo, f"{cfg.local_dir_prefix}/{repo}") for repo in MODEL_REPOS]


def verify(cfg, repo_id, local_dir, fix=False):
    """Returns (status, lines, bad_files) — see check_weights.check_model."""
    try:
        return check_weights.check_model(
            repo_id,
            local_dir,
            sha256=cfg.verify_sha256,
            skip_weights=cfg.skip_weights,
            fix=fix,
        )
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
        log(
            f"force-overwrite refused for path outside {cfg.local_dir_prefix}: {local_dir}",
            "ERROR",
        )
        return False
    if Path(local_dir).is_dir():
        log(f"force-overwrite: wiping {local_dir}")
        shutil.rmtree(local_dir)
    Path(local_dir).mkdir(parents=True, exist_ok=True)
    return True


def _drain(proc, log_handle):
    """Pump a subprocess' output into the log file (background downloads)."""
    for chunk in iter(lambda: proc.stdout.read(4096), b""):
        log_handle.write(chunk)
    log_handle.close()


def download_model(cfg, repo_id, local_dir, round_no):
    """Verify-then-download. Returns a Popen in background mode, else None."""
    log_file = Path(cfg.log_dir) / f"{repo_id.replace('/', '_')}.log"

    # Ask the checker which files are missing/bad; fix=True deletes the bad
    # ones (the modelscope client never replaces existing files, so they must go).
    status, _, bad_files = verify(cfg, repo_id, local_dir, fix=True)
    if status == "ok":
        log(f"{repo_id} already complete, nothing to download.")
        return None

    cmd = ["modelscope", "download", "--max-workers", cfg.max_workers, repo_id]
    if status == "incomplete" and 0 < len(bad_files) <= cfg.max_targeted_files:
        cmd += bad_files  # surgical: fetch only the missing/bad files (positional args)
        mode = f"{len(bad_files)} file(s)"
    else:
        # Full-repo download (checker errored, or too many files to list).
        if cfg.skip_weights:
            cmd += WEIGHT_EXCLUDES
        mode = "full repo"
    cmd += ["--local_dir", local_dir]

    log(f"round {round_no}: downloading {repo_id} ({mode}; log: {log_file})")
    # Long-lived handle: closed by drain thread (background) or after streaming (sequential).
    lf = open(log_file, "ab")  # noqa: SIM115
    lf.write(
        f"===== round {round_no} {time.strftime('%F %T')} : {mode} =====\n".encode()
    )
    lf.flush()
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True
    )
    CHILDREN.append(proc)
    if cfg.run_in_background:  # log only; the console stays readable
        threading.Thread(target=_drain, args=(proc, lf), daemon=True).start()
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
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(proc.pid, signal.SIGTERM)  # start_new_session => pgid == pid
    sys.exit(130)


def mark_complete(cfg, final_status, repo_id, local_dir, note):
    final_status[repo_id] = note
    if cfg.clean_temp:
        clean_temp(cfg, local_dir)


def print_summary(models, final_status, max_rounds):
    """Print the per-model summary table; returns True if anything is incomplete."""
    print()
    log("================ SUMMARY ================")
    incomplete = False
    for repo_id, _ in models:
        status = final_status.get(repo_id, "SKIPPED")
        print(f"  {repo_id:<40} {status}")
        incomplete = incomplete or status == "INCOMPLETE"
    if incomplete:
        log(
            f"Some models are still incomplete after {max_rounds} rounds. "
            "Re-run this script to resume.",
            "ERROR",
        )
    else:
        log("All models verified 100% complete.")
    return incomplete


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
        for (repo_id, local_dir), (status, lines, _) in zip(
            models, reports, strict=False
        ):
            print(f">>> {repo_id} -> {local_dir}")
            for line in lines:
                print(line)
            if status == "ok":
                mark_complete(
                    cfg, final_status, repo_id, local_dir, "OK (already complete)"
                )
            else:
                pending.append((repo_id, local_dir))
    else:
        for repo_id, local_dir in models:
            if cfg.force_overwrite and not wipe_dir(cfg, local_dir):
                final_status[repo_id] = "INCOMPLETE"
            else:
                pending.append((repo_id, local_dir))
    log(f"{len(final_status)} already complete, {len(pending)} to download.")

    # --- Download -> verify -> retry loop ---
    round_no = 1
    while pending and round_no <= cfg.max_rounds:
        log(f"=== Round {round_no}/{cfg.max_rounds}: {len(pending)} model(s) ===")
        procs = []
        for model in pending:
            proc = download_model(cfg, *model, round_no)
            if proc is not None:  # background mode; sequential mode already waited
                procs.append(proc)
        for proc in procs:
            proc.wait()
            CHILDREN.remove(proc)

        still_pending = []
        for repo_id, local_dir in pending:
            status, _, _ = verify(cfg, repo_id, local_dir)
            if status == "ok":
                mark_complete(
                    cfg, final_status, repo_id, local_dir, f"OK (round {round_no})"
                )
                log(f"{repo_id} verified complete.")
            else:
                still_pending.append((repo_id, local_dir))
                log(f"{repo_id} still incomplete after round {round_no}.", "WARN")
        pending = still_pending
        if pending and round_no < cfg.max_rounds:
            time.sleep(cfg.retry_delay)
        round_no += 1

    # --- Summary ---
    for repo_id, _ in pending:
        final_status[repo_id] = "INCOMPLETE"
    return 1 if print_summary(models, final_status, cfg.max_rounds) else 0


if __name__ == "__main__":
    sys.exit(main())
