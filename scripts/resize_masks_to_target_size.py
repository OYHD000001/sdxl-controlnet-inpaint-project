#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.io import ensure_dir, load_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resize mask images to the exact target_image size.")
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--output-mask-dir", required=True, type=Path)
    parser.add_argument("--output-metadata", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_jsonl(args.metadata)
    output_mask_dir = ensure_dir(args.output_mask_dir)

    new_records: list[dict] = []
    for record in records:
        target_path = Path(record["target_image"])
        mask_path = Path(record["mask_image"])
        target = Image.open(target_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        if mask.size != target.size:
            mask = mask.resize(target.size, Image.NEAREST)

        stem = Path(record["source_image"]).stem
        resized_mask_path = output_mask_dir / f"{stem}.png"
        mask.save(resized_mask_path)

        new_record = dict(record)
        new_record["mask_image"] = str(resized_mask_path.resolve())
        new_records.append(new_record)

    if args.output_metadata is not None:
        args.output_metadata.parent.mkdir(parents=True, exist_ok=True)
        with args.output_metadata.open("w", encoding="utf-8") as f:
            for record in new_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                "metadata": str(args.metadata.resolve()),
                "output_mask_dir": str(output_mask_dir.resolve()),
                "output_metadata": str(args.output_metadata.resolve()) if args.output_metadata else None,
                "count": len(new_records),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
