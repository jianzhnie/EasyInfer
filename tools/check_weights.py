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
import struct
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings("ignore")
# torch_npu (present in some vllm envs) breaks "import torch" unless backend
# autoload is disabled; modelscope imports torch at module import time.
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

WEIGHT_EXTS = (".safetensors", ".bin", ".pt", ".ckpt")
TEMP_DIR_NAME = "._____temp"
# modelscope client-side metadata files. Some repos accidentally have them
# committed, but the client never downloads/overwrites them, so comparing
# them would always fail. They are not needed to load a model.
IGNORE_FILES = {".msc", ".mv"}
HASH_CHUNK = 64 * 1024 * 1024


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


def local_expected_files_offline(local_dir):
    """Best-effort expected weight-file set without network access.
    Returns (expected_relative_paths_or_None, note)."""
    for fname in sorted(os.listdir(local_dir)):
        if fname.endswith(".safetensors.index.json"):
            try:
                with open(os.path.join(local_dir, fname)) as f:
                    idx = json.load(f)
                expected = sorted(set(idx.get("weight_map", {}).values()))
                if expected:
                    return expected, f"index file {fname} ({len(expected)} shards)"
            except (OSError, json.JSONDecodeError):
                pass
    shards = [f for f in os.listdir(local_dir) if f.endswith(".safetensors")]
    for f in sorted(shards):
        m = re.match(r"^(.+-)(\d+)-of-0*(\d+)\.safetensors$", f)
        if m:
            prefix, _, total, digits = m.group(1), m.group(2), int(m.group(3)), len(m.group(2))
            expected = [f"{prefix}{i:0{digits}d}-of-{total:0{digits}d}.safetensors"
                        for i in range(1, total + 1)]
            return expected, f"shard pattern ({total} shards)"
    return None, "no index/shard pattern (size checks unavailable offline)"


def temp_dir_info(local_dir):
    temp = os.path.join(local_dir, TEMP_DIR_NAME)
    if not os.path.isdir(temp):
        return None
    nfiles, total = 0, 0
    for root, _, files in os.walk(temp):
        for f in files:
            nfiles += 1
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    if nfiles == 0:
        return None
    return nfiles, total


def check_one_file(local_dir, rel, expected_size, want_sha256, remote_sha256):
    """Worker: size + structure (+ optional hash) for one file.
    Returns (category, rel, detail) with category in ok/missing/bad_size/corrupt/bad_hash."""
    path = os.path.join(local_dir, rel)
    if not os.path.isfile(path):
        return ("missing", rel, "")
    try:
        actual = os.path.getsize(path)
    except OSError as exc:
        return ("missing", rel, f"stat failed: {exc}")
    if expected_size is not None and actual != expected_size:
        return ("bad_size", rel, f"local={actual} expected={expected_size}")
    if rel.endswith(".safetensors"):
        problem = validate_safetensors(path)
        if problem:
            return ("corrupt", rel, problem)
    if want_sha256 and remote_sha256:
        try:
            if sha256_file(path) != remote_sha256:
                return ("bad_hash", rel, "sha256 mismatch")
        except OSError as exc:
            return ("bad_hash", rel, f"read failed: {exc}")
    return ("ok", rel, "")


def check_model(repo_id, local_dir, args):
    """Returns (status, lines, bad_files):
    status in ok/incomplete/error; lines = human-readable report;
    bad_files = relative paths needing (re)download (empty when ok/error)."""
    lines = []
    if not os.path.isdir(local_dir):
        return "incomplete", [f"  FAIL directory does not exist: {local_dir}"], []

    expected = None  # rel -> {"size","sha256"}
    note = ""
    if repo_id and not args.offline:
        try:
            expected = fetch_remote_files(repo_id)
            note = f"{len(expected)} remote files"
        except Exception as exc:  # network/API failure: cannot verify online
            return "error", [f"  ERROR failed to fetch remote file list for {repo_id}: {exc}"], []
    else:
        rels, note = local_expected_files_offline(local_dir)
        if rels is not None:
            expected = {r: {"size": None, "sha256": ""} for r in rels}
        else:
            # Offline fallback: verify every local weight file structurally.
            rels = []
            for root, _, files in os.walk(local_dir):
                if TEMP_DIR_NAME in root.split(os.sep):
                    continue
                for f in files:
                    if f.endswith(WEIGHT_EXTS):
                        rels.append(os.path.relpath(os.path.join(root, f), local_dir))
            expected = {r: {"size": None, "sha256": ""} for r in sorted(rels)}

    if args.skip_weights:
        expected = {r: v for r, v in expected.items() if not r.endswith(WEIGHT_EXTS)}
    expected = {r: v for r, v in expected.items()
                if os.path.basename(r) not in IGNORE_FILES}

    results = {"ok": [], "missing": [], "bad_size": [], "corrupt": [], "bad_hash": []}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(check_one_file, local_dir, rel, v["size"], args.sha256, v["sha256"])
            for rel, v in expected.items()
        ]
        for fut in as_completed(futures):
            cat, rel, detail = fut.result()
            results[cat].append((rel, detail))

    n_bad = sum(len(results[c]) for c in ("missing", "bad_size", "corrupt", "bad_hash"))
    bad_files = sorted(rel for cat in ("missing", "bad_size", "corrupt", "bad_hash")
                       for rel, _ in results[cat])
    labels = {
        "missing": "missing",
        "bad_size": "size mismatch",
        "corrupt": "corrupt safetensors",
        "bad_hash": "sha256 mismatch",
    }
    for cat, label in labels.items():
        items = sorted(results[cat])
        if not items:
            continue
        lines.append(f"  FAIL {label}: {len(items)} file(s)")
        for rel, detail in items[: args.show]:
            lines.append(f"    - {rel}" + (f" ({detail})" if detail else ""))
        if len(items) > args.show:
            lines.append(f"    ... and {len(items) - args.show} more")

    if args.fix and n_bad:
        # Delete files that failed verification so the next download run
        # re-fetches them (the modelscope client skips existing files and
        # would otherwise never replace a corrupt/stale one).
        fixed = 0
        for cat in ("bad_size", "corrupt", "bad_hash"):
            for rel, _ in results[cat]:
                try:
                    os.remove(os.path.join(local_dir, rel))
                    fixed += 1
                except OSError:
                    pass
        if fixed:
            lines.append(f"  FIX deleted {fixed} bad file(s); re-download will re-fetch them")
            # They are now missing; the model stays incomplete until re-downloaded.

    if args.list_bad:
        # Machine-readable list of files that need (re)downloading, one per line.
        try:
            with open(args.list_bad, "w") as f:
                for cat in ("missing", "bad_size", "corrupt", "bad_hash"):
                    for rel, _ in sorted(results[cat]):
                        f.write(rel + "\n")
        except OSError as exc:
            lines.append(f"  WARN could not write bad-file list: {exc}")

    temp = temp_dir_info(local_dir)
    if temp:
        lines.append(f"  WARN temp leftovers: {TEMP_DIR_NAME}/ "
                     f"({temp[0]} files, {human_size(temp[1])}) - resumable partial downloads")

    if n_bad:
        parts = [f"{len(results[c])} {labels[c]}" for c in labels if results[c]]
        lines.append(f"  => INCOMPLETE ({note}): {', '.join(parts)}")
        return "incomplete", lines, bad_files
    mode = "sizes + safetensors structure" + (" + sha256" if args.sha256 else "")
    lines.append(f"  => OK: {len(results['ok'])}/{len(expected)} files verified ({mode}; {note})")
    return "ok", lines, []


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
    exit_code = 0
    for i, pair in enumerate(args.pairs, 1):
        if ":" in pair and not pair.startswith("/"):
            repo_id, local_dir = pair.split(":", 1)
        elif args.offline:
            repo_id, local_dir = None, pair
        else:
            print(f"[{i}/{len(args.pairs)}] SKIP {pair!r}: not a REPO_ID:LOCAL_DIR pair "
                  f"(use --offline for local-only checks)")
            exit_code = 2
            continue
        print(f"[{i}/{len(args.pairs)}] {repo_id or 'local'} -> {local_dir}")
        try:
            status, lines, _ = check_model(repo_id, local_dir, args)
        except Exception as exc:
            status, lines = "error", [f"  ERROR unexpected: {exc!r}"]
        for line in lines:
            print(line)
        statuses[pair] = status
        if status == "incomplete" and exit_code == 0:
            exit_code = 1
        elif status == "error":
            exit_code = 2

    print("=" * 60)
    print("SUMMARY")
    ok = [p for p, s in statuses.items() if s == "ok"]
    bad = [p for p, s in statuses.items() if s == "incomplete"]
    err = [p for p, s in statuses.items() if s == "error"]
    print(f"  OK:         {len(ok)}")
    print(f"  INCOMPLETE: {len(bad)}")
    for p in bad:
        print(f"    - {p}")
    print(f"  ERROR:      {len(err)}")
    for p in err:
        print(f"    - {p}")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
