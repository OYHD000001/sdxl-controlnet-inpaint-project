#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path


VALID_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare stable inpainting dual-control metadata from flux mannequin images with source_image set to clothes cutout."
    )
    parser.add_argument("--target-dir", required=True, type=Path)
    parser.add_argument("--mask-dir", required=True, type=Path)
    parser.add_argument("--source-cutout-dir", required=True, type=Path)
    parser.add_argument("--conditioning-dir", required=True, type=Path)
    parser.add_argument("--pose-dir", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--output-subdir", default="image2_2000_flux模特图", type=str)
    parser.add_argument("--prompt", required=True, type=str)
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--preview-count", type=int, default=4)
    return parser.parse_args()


def collect_stems(directory: Path) -> dict[str, Path]:
    items: dict[str, Path] = {}
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in VALID_EXTS:
            items[path.stem] = path
    return items


def write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()

    target_map = collect_stems(args.target_dir)
    mask_map = collect_stems(args.mask_dir)
    source_cutout_map = collect_stems(args.source_cutout_dir)
    conditioning_map = collect_stems(args.conditioning_dir)
    pose_map = collect_stems(args.pose_dir)

    common_stems = sorted(
        set(target_map) & set(mask_map) & set(source_cutout_map) & set(conditioning_map) & set(pose_map)
    )

    records: list[dict] = []
    for stem in common_stems:
        records.append(
            {
                "target_image": str(target_map[stem].resolve()),
                "source_image": str(source_cutout_map[stem].resolve()),
                "mask_image": str(mask_map[stem].resolve()),
                "conditioning_image": str(conditioning_map[stem].resolve()),
                "conditioning_image_2": str(pose_map[stem].resolve()),
                "text": args.prompt,
                "output_name": stem,
                "output_subdir": args.output_subdir,
            }
        )

    train_count = int(len(records) * args.train_ratio)
    train_records = records[:train_count]
    val_records = records[train_count:]
    preview_records = train_records[: min(args.preview_count, len(train_records))]

    output_root = args.output_root
    write_jsonl(records, output_root / "metadata_all.jsonl")
    write_jsonl(train_records, output_root / "metadata_train.jsonl")
    write_jsonl(val_records, output_root / "metadata_val.jsonl")
    write_jsonl(preview_records, output_root / "metadata_train_preview4.jsonl")

    print(
        json.dumps(
            {
                "output_root": str(output_root.resolve()),
                "all_count": len(records),
                "train_count": len(train_records),
                "val_count": len(val_records),
                "preview_count": len(preview_records),
                "source_cutout_dir": str(args.source_cutout_dir.resolve()),
                "conditioning_dir": str(args.conditioning_dir.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
