#!/usr/bin/env python3
"""Extract first 2 layers from LongCat-Flash-Chat model.

Reads the original 28-layer model and creates a new 2-layer model with
adjusted config. Only weights for layers 0-1 plus shared weights
(embed_tokens, norm, lm_head, mtp) are copied.

Usage:
    python3 tools/extract_longcat_2layer.py
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

# --- Configuration -----------------------------------------------------------

SRC_DIR = Path(
    "/home/jianzhnie/llmtuner/hfhub/models/meituan-longcat/LongCat-Flash-Chat"
)
DST_DIR = Path(
    "/home/jianzhnie/llmtuner/hfhub/models/meituan-longcat/expand/LongCat-Flash-Chat-2layer"
)

KEEP_LAYERS = {0, 1}  # first 2 layers
NEW_NUM_LAYERS = 2

# Files to copy verbatim (not safetensors)
COPY_FILES = [
    "config.json",
    "configuration_longcat_flash.py",
    "modeling_longcat_flash.py",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "generation_config.json",
    "README.md",
    "LICENSE",
]


def load_index(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def filter_weights(index: dict) -> dict[str, str]:
    """Return {weight_name: file_name} for weights to keep."""
    weight_map = index["weight_map"]
    kept: dict[str, str] = {}

    for name, file_name in weight_map.items():
        if name.startswith("model.layers."):
            layer_idx = int(name.split(".")[2])
            if layer_idx in KEEP_LAYERS:
                # Renumber layer: old layer 0/1 → new layer 0/1 (unchanged)
                kept[name] = file_name
        else:
            # Shared weights (embed_tokens, norm, lm_head, mtp, etc.)
            kept[name] = file_name

    return kept


def build_new_index(
    kept_weights: dict[str, str],
    new_files: dict[str, list[str]],
) -> dict:
    """Build a new safetensors index.json structure."""
    new_weight_map: dict[str, str] = {}

    for weight_name, _old_file in kept_weights.items():
        # Find which new file this weight ended up in
        for new_file, names in new_files.items():
            if weight_name in names:
                new_weight_map[weight_name] = new_file
                break

    return {
        "metadata": {
            "total_size": 0,  # approximate; vLLM reads this lazily
        },
        "weight_map": new_weight_map,
    }


def extract_weights(
    src_dir: Path,
    dst_dir: Path,
    kept_weights: dict[str, str],
) -> dict[str, list[str]]:
    """Extract kept weights into new safetensors files.

    Groups weights by the source file they come from to minimize reads,
    then saves them into a single consolidated safetensors file.

    Returns {new_filename: [weight_names]} mapping.
    """
    dst_dir.mkdir(parents=True, exist_ok=True)

    # Group by source file
    src_to_weights: dict[str, list[str]] = {}
    for name, src_file in kept_weights.items():
        src_to_weights.setdefault(src_file, []).append(name)

    # Process each source file, collect all weights
    all_weights: dict[str, torch.Tensor] = {}
    total_src = len(src_to_weights)

    for i, (src_file, weight_names) in enumerate(sorted(src_to_weights.items())):
        src_path = src_dir / src_file
        print(
            f"[{i + 1}/{total_src}] Loading {src_file} ({len(weight_names)} weights)..."
        )
        src_tensors = load_file(str(src_path))

        for name in weight_names:
            if name in src_tensors:
                all_weights[name] = src_tensors[name]
            else:
                print(f"  WARNING: {name} not found in {src_file}")

    print(f"\nTotal extracted weights: {len(all_weights)}")

    # Save as a single safetensors file
    out_file = "model.safetensors"
    out_path = dst_dir / out_file
    print(f"Saving to {out_path} ({len(all_weights)} tensors)...")
    save_file(all_weights, str(out_path))

    return {out_file: list(all_weights.keys())}


def update_config(dst_dir: Path) -> None:
    """Update config.json for the 2-layer model."""
    config_path = dst_dir / "config.json"
    with open(config_path) as f:
        config = json.load(f)

    config["num_layers"] = NEW_NUM_LAYERS

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"Updated config: num_layers={NEW_NUM_LAYERS}")


def copy_aux_files(src_dir: Path, dst_dir: Path) -> None:
    """Copy non-weight files to the new model directory."""
    dst_dir.mkdir(parents=True, exist_ok=True)

    for filename in COPY_FILES:
        src = src_dir / filename
        dst = dst_dir / filename
        if src.exists():
            shutil.copy2(src, dst)
            print(f"Copied: {filename}")
        else:
            print(f"Skipped (not found): {filename}")


def main() -> None:
    print("=" * 60)
    print("LongCat-Flash-Chat → 2-Layer Model Extractor")
    print("=" * 60)

    if not SRC_DIR.exists():
        raise FileNotFoundError(f"Source model not found: {SRC_DIR}")

    # 1. Load weight index
    index_path = SRC_DIR / "model.safetensors.index.json"
    index = load_index(index_path)
    print(f"Loaded index: {len(index['weight_map'])} weight entries")

    # 2. Filter weights to keep
    kept = filter_weights(index)
    print(f"Weights to keep: {len(kept)}")

    # 3. Copy auxiliary files
    print("\n--- Copying auxiliary files ---")
    copy_aux_files(SRC_DIR, DST_DIR)

    # 4. Extract and save weights
    print("\n--- Extracting weights ---")
    new_files = extract_weights(SRC_DIR, DST_DIR, kept)

    # 5. Build new index
    print("\n--- Building new index ---")
    new_index = build_new_index(kept, new_files)
    new_index_path = DST_DIR / "model.safetensors.index.json"
    with open(new_index_path, "w") as f:
        json.dump(new_index, f, indent=2)
        f.write("\n")
    print(f"Written: {new_index_path}")

    # 6. Update config
    print("\n--- Updating config ---")
    update_config(DST_DIR)

    # 7. Verify
    print("\n--- Verification ---")
    verify_index = load_index(new_index_path)
    print(f"New model has {len(verify_index['weight_map'])} weight entries")

    # Check layer count in config
    with open(DST_DIR / "config.json") as f:
        new_config = json.load(f)
    print(f"num_layers in new config: {new_config.get('num_layers')}")

    # Check model files
    model_files = sorted(DST_DIR.iterdir())
    print(f"\nFiles in {DST_DIR}:")
    for f in model_files:
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  {f.name:50s} {size_mb:10.1f} MB")

    print("\n" + "=" * 60)
    print("Extraction complete!")
    print(f"New model at: {DST_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
