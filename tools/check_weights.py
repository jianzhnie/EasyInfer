#!/usr/bin/env python3
"""Verify that ModelScope model downloads are 100% complete.

For each model (repo_id:local_dir pair) the checker verifies, against the
remote ModelScope repository:

  1. Every remote file exists locally (recursively, incl. subdirs).
  2. Every local file size matches the remote size exactly.
  3. Every *.safetensors file is structurally complete: a parseable header
     whose tensor data_offsets end exactly at the file size (catches
     truncated/corrupted shards even when the size happens to look right).
  4. Optionally (--sha256) the full SHA-256 of every file matches the remote
     LFS checksum. Slow (reads all data) but cryptographically conclusive.

With --offline (or when no repo id is given) the remote comparison is
skipped and completeness is derived from the safetensors index file /
shard naming pattern plus the structural validation.

Leftover "._____temp" directories (modelscope resume data for interrupted
downloads) are reported as warnings; keep them if you plan to resume.

Usage:
    check_weights.py [OPTIONS] REPO_ID:LOCAL_DIR [REPO_ID:LOCAL_DIR ...]
    check_weights.py --offline LOCAL_DIR [LOCAL_DIR ...]

Exit code: 0 = all models complete, 1 = at least one incomplete, 2 = check error.
"""

import argparse
import hashlib
import json
import os
import re
import socket
import struct
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor

warnings.filterwarnings("ignore")
# torch_npu (present in some vllm envs) breaks "import torch" unless backend
# autoload is disabled; modelscope imports torch at module import time.
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
# Bound every blocking socket op (the modelscope API client issues requests
# without an explicit timeout, so a stalled connection would hang forever).
socket.setdefaulttimeout(60)

WEIGHT_EXTS = (".safetensors", ".bin", ".pt", ".ckpt")
TEMP_DIR_NAME = "._____temp"
# modelscope client-side metadata files. Some repos accidentally have them
# committed, but the client never downloads/overwrites them, so comparing
# them would always fail. They are not needed to load a model.
IGNORE_FILES = {".msc", ".mv"}
HASH_CHUNK = 64 * 1024 * 1024

# Problem categories in report order: (category, label). Files in FIXABLE
# categories exist locally but failed verification, so --fix deletes them
# ("missing" files have nothing to delete).
BAD_CATEGORIES = [("missing", "missing"),
                  ("bad_size", "size mismatch"),
                  ("corrupt", "corrupt safetensors"),
                  ("bad_hash", "sha256 mismatch")]
FIXABLE = ("bad_size", "corrupt", "bad_hash")


def human_size(n):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024
    return f"{n} B"


def validate_safetensors(path):
    """Return None if the file is a structurally complete safetensors file,
    otherwise a string describing the problem."""
    try:
        fsize = os.path.getsize(path)
    except OSError as exc:
        return f"stat failed: {exc}"
    if fsize < 8:
        return f"too small ({fsize} bytes)"
    try:
        with open(path, "rb") as f:
            (header_len,) = struct.unpack("<Q", f.read(8))
            if header_len <= 0 or 8 + header_len > fsize:
                return f"bad header length {header_len} (file size {fsize})"
            header = json.loads(f.read(header_len))
    except (OSError, json.JSONDecodeError, struct.error) as exc:
        return f"unreadable header: {exc}"
    max_end = 0
    for name, info in header.items():
        if name == "__metadata__":
            continue
        try:
            begin, end = info["data_offsets"]
        except (KeyError, TypeError, ValueError):
            return f"tensor {name!r}: missing/invalid data_offsets"
        if not (0 <= begin <= end):
            return f"tensor {name!r}: invalid data_offsets [{begin}, {end})"
        max_end = max(max_end, end)
    expected_size = 8 + header_len + max_end
    if expected_size != fsize:
        return f"truncated/corrupt: header implies {expected_size} bytes, actual {fsize}"
    return None


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(HASH_CHUNK)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def fetch_remote_files(repo_id):
    """Return {relative_path: {"size": int, "sha256": str}} for all blobs."""
    from modelscope.hub.api import HubApi

    entries = HubApi().get_model_files(repo_id, recursive=True)
    return {
        e["Path"]: {"size": int(e.get("Size") or 0), "sha256": e.get("Sha256") or ""}
        for e in entries
        if e.get("Type") == "blob"
    }


def expected_files_offline(local_dir):
    """Best-effort expected weight-file list without network access.
    Returns (relative_paths_or_None, note); None means "no index/shard
    pattern found" and the caller falls back to all local weight files."""
    for fname in sorted(os.listdir(local_dir)):
        if not fname.endswith(".safetensors.index.json"):
            continue
        try:
            with open(os.path.join(local_dir, fname)) as f:
                expected = sorted(set(json.load(f).get("weight_map", {}).values()))
        except (OSError, json.JSONDecodeError):
            continue
        if expected:
            return expected, f"index file {fname} ({len(expected)} shards)"
    for fname in sorted(f for f in os.listdir(local_dir) if f.endswith(".safetensors")):
        m = re.match(r"^(.+-)(\d+)-of-0*(\d+)\.safetensors$", fname)
        if m:
            prefix, num, total_s = m.groups()
            total, digits = int(total_s), len(num)
            expected = [f"{prefix}{i:0{digits}d}-of-{total:0{digits}d}.safetensors"
                        for i in range(1, total + 1)]
            return expected, f"shard pattern ({total} shards)"
    return None, "no index/shard pattern (size checks unavailable offline)"


def temp_dir_info(local_dir):
    """(nfiles, total_bytes) of leftover modelscope resume data, else None."""
    temp = os.path.join(local_dir, TEMP_DIR_NAME)
    if not os.path.isdir(temp):
        return None
    nfiles, total = 0, 0
    for root, _, files in os.walk(temp):
        for fname in files:
            nfiles += 1
            try:
                total += os.path.getsize(os.path.join(root, fname))
            except OSError:
                pass
    return (nfiles, total) if nfiles else None


def check_one_file(local_dir, rel, info, want_sha256):
    """Worker: size + structure (+ optional hash) for one file.
    info = {"size": int|None, "sha256": str}. Returns (category, detail)."""
    path = os.path.join(local_dir, rel)
    if not os.path.isfile(path):
        return "missing", ""
    try:
        actual = os.path.getsize(path)
    except OSError as exc:
        return "missing", f"stat failed: {exc}"
    if info["size"] is not None and actual != info["size"]:
        return "bad_size", f"local={actual} expected={info['size']}"
    if rel.endswith(".safetensors"):
        problem = validate_safetensors(path)
        if problem:
            return "corrupt", problem
    if want_sha256 and info["sha256"]:
        try:
            if sha256_file(path) != info["sha256"]:
                return "bad_hash", "sha256 mismatch"
        except OSError as exc:
            return "bad_hash", f"read failed: {exc}"
    return "ok", ""


def check_model(repo_id, local_dir, *, offline=False, sha256=False, skip_weights=False,
                workers=8, show=10, fix=False, list_bad=None):
    """Verify one model. Returns (status, lines, bad_files):
    status in ok/incomplete/error; lines = human-readable report;
    bad_files = relative paths needing (re)download (empty when ok/error)."""
    if not os.path.isdir(local_dir):
        return "incomplete", [f"  FAIL directory does not exist: {local_dir}"], []

    # 1) Determine the expected file set: remote repo listing, or offline
    #    heuristics (index file / shard pattern / all local weight files).
    if repo_id and not offline:
        try:
            expected = fetch_remote_files(repo_id)
        except Exception as exc:  # network/API failure: cannot verify online
            return "error", [f"  ERROR failed to fetch remote file list for {repo_id}: {exc}"], []
        note = f"{len(expected)} remote files"
    else:
        rels, note = expected_files_offline(local_dir)
        if rels is None:  # offline fallback: verify every local weight file structurally
            rels = [os.path.relpath(os.path.join(root, f), local_dir)
                    for root, _, files in os.walk(local_dir)
                    if TEMP_DIR_NAME not in root.split(os.sep)
                    for f in files
                    if f.endswith(WEIGHT_EXTS)]
        expected = {r: {"size": None, "sha256": ""} for r in sorted(rels)}

    expected = {r: v for r, v in expected.items()
                if os.path.basename(r) not in IGNORE_FILES
                and not (skip_weights and r.endswith(WEIGHT_EXTS))}

    # 2) Check every expected file in parallel (size + structure, + optional sha256).
    results = {cat: [] for cat, _ in BAD_CATEGORIES}
    results["ok"] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(check_one_file, local_dir, rel, info, sha256)
                   for rel, info in expected.items()]
        for (rel, _), future in zip(expected.items(), futures):
            cat, detail = future.result()
            results[cat].append((rel, detail))

    bad_files = sorted(rel for cat, _ in BAD_CATEGORIES for rel, _ in results[cat])
    lines = []
    for cat, label in BAD_CATEGORIES:
        items = sorted(results[cat])
        if not items:
            continue
        lines.append(f"  FAIL {label}: {len(items)} file(s)")
        lines += [f"    - {rel}" + (f" ({detail})" if detail else "")
                  for rel, detail in items[:show]]
        if len(items) > show:
            lines.append(f"    ... and {len(items) - show} more")

    # 3) Optional fix: delete existing-but-bad files so the next download run
    #    re-fetches them (the modelscope client never replaces existing files).
    if fix:
        deleted = 0
        for cat in FIXABLE:
            for rel, _ in results[cat]:
                try:
                    os.remove(os.path.join(local_dir, rel))
                    deleted += 1
                except OSError:
                    pass
        if deleted:
            lines.append(f"  FIX deleted {deleted} bad file(s); re-download will re-fetch them")

    if list_bad:  # machine-readable list of files needing (re)download
        try:
            with open(list_bad, "w") as f:
                f.writelines(f"{rel}\n" for rel in bad_files)
        except OSError as exc:
            lines.append(f"  WARN could not write bad-file list: {exc}")

    temp = temp_dir_info(local_dir)
    if temp:
        lines.append(f"  WARN temp leftovers: {TEMP_DIR_NAME}/ "
                     f"({temp[0]} files, {human_size(temp[1])}) - resumable partial downloads")

    if bad_files:
        parts = [f"{len(results[cat])} {label}" for cat, label in BAD_CATEGORIES if results[cat]]
        lines.append(f"  => INCOMPLETE ({note}): {', '.join(parts)}")
        return "incomplete", lines, bad_files
    mode = "sizes + safetensors structure" + (" + sha256" if sha256 else "")
    lines.append(f"  => OK: {len(results['ok'])}/{len(expected)} files verified ({mode}; {note})")
    return "ok", lines, []


def parse_pair(pair, offline):
    """Split 'REPO_ID:LOCAL_DIR'. With offline a plain LOCAL_DIR gives
    (None, pair); anything else invalid gives (None, None)."""
    if ":" in pair and not pair.startswith("/"):
        return pair.split(":", 1)
    return (None, pair) if offline else (None, None)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pairs", nargs="+",
                    help="REPO_ID:LOCAL_DIR pairs, or plain LOCAL_DIR with --offline")
    ap.add_argument("--offline", action="store_true",
                    help="no network: verify via index/shard pattern + structure only")
    ap.add_argument("--sha256", action="store_true",
                    help="also verify full SHA-256 of every file (slow, reads all data)")
    ap.add_argument("--fix", action="store_true",
                    help="delete files that fail verification (size/structure/hash) so a "
                         "re-download re-fetches them; missing files need no deletion")
    ap.add_argument("--list-bad", metavar="PATH",
                    help="write the files needing (re)download to PATH, one per line")
    ap.add_argument("--skip-weights", action="store_true",
                    help="exclude weight files (*.safetensors/*.bin/*.pt/*.ckpt) from the check")
    ap.add_argument("--workers", type=int, default=8, help="per-model file check threads (default 8)")
    ap.add_argument("--show", type=int, default=10, help="max problem files to list per category")
    args = ap.parse_args()

    statuses = {}
    for i, pair in enumerate(args.pairs, 1):
        repo_id, local_dir = parse_pair(pair, args.offline)
        if local_dir is None:
            print(f"[{i}/{len(args.pairs)}] SKIP {pair!r}: not a REPO_ID:LOCAL_DIR pair "
                  f"(use --offline for local-only checks)")
            statuses[pair] = "error"
            continue
        print(f"[{i}/{len(args.pairs)}] {repo_id or 'local'} -> {local_dir}")
        try:
            status, lines, _ = check_model(repo_id, local_dir, offline=args.offline,
                                           sha256=args.sha256, skip_weights=args.skip_weights,
                                           workers=args.workers, show=args.show,
                                           fix=args.fix, list_bad=args.list_bad)
        except Exception as exc:
            status, lines = "error", [f"  ERROR unexpected: {exc!r}"]
        for line in lines:
            print(line)
        statuses[pair] = status

    print("=" * 60)
    print("SUMMARY")
    by_status = {s: [p for p, v in statuses.items() if v == s]
                 for s in ("ok", "incomplete", "error")}
    print(f"  OK:         {len(by_status['ok'])}")
    print(f"  INCOMPLETE: {len(by_status['incomplete'])}")
    for p in by_status["incomplete"]:
        print(f"    - {p}")
    print(f"  ERROR:      {len(by_status['error'])}")
    for p in by_status["error"]:
        print(f"    - {p}")
    sys.exit(2 if by_status["error"] else 1 if by_status["incomplete"] else 0)


if __name__ == "__main__":
    main()
