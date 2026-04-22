#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.io import ensure_dir, load_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replace conditioning_image with a Canny edge map derived from the existing clothes conditioning image."
    )
    parser.add_argument("--train-metadata", required=True, type=Path)
    parser.add_argument("--val-metadata", required=True, type=Path)
    parser.add_argument("--all-metadata", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--low-threshold", type=int, default=80)
    parser.add_argument("--high-threshold", type=int, default=160)
    return parser.parse_args()


def make_canny_image(input_path: Path, output_path: Path, low_threshold: int, high_threshold: int) -> None:
    image = Image.open(input_path).convert("RGB")
    array = np.array(image)
    gray = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, low_threshold, high_threshold)
    edge_rgb = np.repeat(edges[:, :, None], 3, axis=2)
    Image.fromarray(edge_rgb).save(output_path)


def write_jsonl(records: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def transform_records(records: list[dict], output_dir: Path, low_threshold: int, high_threshold: int) -> list[dict]:
    transformed: list[dict] = []
    for record in records:
        input_path = Path(record["conditioning_image"])
        output_path = output_dir / f"{input_path.stem}_canny.png"
        if not output_path.exists():
            make_canny_image(input_path, output_path, low_threshold, high_threshold)

        new_record = dict(record)
        new_record["conditioning_image"] = str(output_path.resolve())
        transformed.append(new_record)
    return transformed


def main() -> None:
    args = parse_args()
    output_root = ensure_dir(args.output_root)
    canny_dir = ensure_dir(output_root / "conditioning_canny")

    train_records = transform_records(load_jsonl(args.train_metadata), canny_dir, args.low_threshold, args.high_threshold)
    val_records = transform_records(load_jsonl(args.val_metadata), canny_dir, args.low_threshold, args.high_threshold)
    all_records = transform_records(load_jsonl(args.all_metadata), canny_dir, args.low_threshold, args.high_threshold)

    train_out = output_root / "metadata_train.jsonl"
    val_out = output_root / "metadata_val.jsonl"
    all_out = output_root / "metadata_all.jsonl"
    write_jsonl(train_records, train_out)
    write_jsonl(val_records, val_out)
    write_jsonl(all_records, all_out)

    print(
        json.dumps(
            {
                "output_root": str(output_root.resolve()),
                "conditioning_canny_dir": str(canny_dir.resolve()),
                "train_metadata": str(train_out.resolve()),
                "val_metadata": str(val_out.resolve()),
                "all_metadata": str(all_out.resolve()),
                "train_count": len(train_records),
                "val_count": len(val_records),
                "all_count": len(all_records),
                "low_threshold": args.low_threshold,
                "high_threshold": args.high_threshold,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
