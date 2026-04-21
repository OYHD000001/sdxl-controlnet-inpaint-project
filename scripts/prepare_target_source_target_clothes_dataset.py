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
    parser = argparse.ArgumentParser(
        description="Create a derived dataset where source_image=target_image and conditioning_image is clothes cut out from target_image."
    )
    parser.add_argument("--train-metadata", required=True, type=Path)
    parser.add_argument("--val-metadata", required=True, type=Path)
    parser.add_argument("--all-metadata", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--background", choices=["black", "white"], default="black")
    return parser.parse_args()


def extract_clothes_from_target(target_path: Path, mask_path: Path, output_path: Path, background: str) -> None:
    target = Image.open(target_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")
    if mask.size != target.size:
        mask = mask.resize(target.size, Image.NEAREST)

    bg_value = 255 if background == "white" else 0
    output = Image.new("RGB", target.size, (bg_value, bg_value, bg_value))
    output.paste(target, mask=mask)
    output.save(output_path)


def write_jsonl(records: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def transform_records(records: list[dict], conditioning_dir: Path, background: str) -> list[dict]:
    transformed: list[dict] = []
    for record in records:
        target_path = Path(record["target_image"])
        mask_path = Path(record["mask_image"])
        stem = target_path.stem
        conditioning_path = conditioning_dir / f"{stem}_clothes.png"
        if not conditioning_path.exists():
            extract_clothes_from_target(
                target_path=target_path,
                mask_path=mask_path,
                output_path=conditioning_path,
                background=background,
            )

        new_record = dict(record)
        new_record["source_image"] = str(target_path.resolve())
        new_record["conditioning_image"] = str(conditioning_path.resolve())
        transformed.append(new_record)
    return transformed


def main() -> None:
    args = parse_args()
    output_root = ensure_dir(args.output_root)
    conditioning_dir = ensure_dir(output_root / "conditioning_images")

    train_records = load_jsonl(args.train_metadata)
    val_records = load_jsonl(args.val_metadata)
    all_records = load_jsonl(args.all_metadata)

    new_train = transform_records(train_records, conditioning_dir, args.background)
    new_val = transform_records(val_records, conditioning_dir, args.background)
    new_all = transform_records(all_records, conditioning_dir, args.background)

    train_out = output_root / "metadata_train.jsonl"
    val_out = output_root / "metadata_val.jsonl"
    all_out = output_root / "metadata_all.jsonl"

    write_jsonl(new_train, train_out)
    write_jsonl(new_val, val_out)
    write_jsonl(new_all, all_out)

    print(
        json.dumps(
            {
                "output_root": str(output_root.resolve()),
                "conditioning_dir": str(conditioning_dir.resolve()),
                "train_metadata": str(train_out.resolve()),
                "val_metadata": str(val_out.resolve()),
                "all_metadata": str(all_out.resolve()),
                "train_count": len(new_train),
                "val_count": len(new_val),
                "all_count": len(new_all),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
