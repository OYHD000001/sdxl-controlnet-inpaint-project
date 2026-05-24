#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from src.training.validate import make_original_generated_pairs_once, run_inference
from src.utils.io import ensure_dir, load_config, load_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch generate aligned (human, mannequin) pairs.")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--pairs-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    run_inference(config)

    infer_cfg = config["inference"]
    generated_dir = Path(infer_cfg["output_dir"])
    metadata_path = Path(infer_cfg["metadata_path"])
    pairs_root = ensure_dir(args.pairs_dir)
    original_dir = ensure_dir(pairs_root / "original")
    generated_copy_dir = ensure_dir(pairs_root / "generated")
    records = load_jsonl(metadata_path)
    for record in records:
        stem = record.get("output_name") or Path(record["source_image"]).stem
        subdir = record.get("output_subdir")
        generated_path = generated_dir / f"{stem}_generated.png"
        if subdir:
            generated_path = generated_dir / subdir / f"{stem}_generated.png"
        if not generated_path.exists():
            continue
        shutil.copy2(record["source_image"], original_dir / f"{stem}{Path(record['source_image']).suffix}")
        shutil.copy2(generated_path, generated_copy_dir / generated_path.name)

    pair_output_dir = ensure_dir(pairs_root / "original_generated_pairs")
    make_original_generated_pairs_once(
        metadata_path=metadata_path,
        generated_dir=generated_dir,
        output_dir=pair_output_dir,
        image_width=int(config["data"].get("image_width", config["data"]["image_size"])),
        image_height=int(config["data"].get("image_height", config["data"]["image_size"])),
    )
    manifest = {
        "config": str(args.config.resolve()),
        "metadata_path": str(metadata_path.resolve()),
        "generated_dir": str(generated_dir.resolve()),
        "pairs_dir": str(pairs_root.resolve()),
        "count": len(records),
    }
    (pairs_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
