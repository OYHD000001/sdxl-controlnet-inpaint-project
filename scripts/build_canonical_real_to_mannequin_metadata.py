#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from preprocess.extract import build_metadata_record
from src.utils.io import ensure_dir


DEFAULT_PROMPT = (
    "replace the human model with a smooth glossy white plastic retail mannequin, "
    "clearly non-human mannequin body, rigid mannequin limbs, no human skin, no flesh, "
    "no realistic face, no realistic hands, keep the exact clothes, garment texture, "
    "garment silhouette, pose, composition, and studio background"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build canonical source->mannequin inpaint metadata.")
    parser.add_argument("--source-dir", "--human-dir", dest="source_dir", type=Path, required=True)
    parser.add_argument("--target-dir", type=Path, required=True)
    parser.add_argument("--mask-dir", type=Path, required=True)
    parser.add_argument("--pose-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-subdir", type=str, default="image2-2000-realhuman")
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--train-ratio", type=float, default=0.9)
    return parser.parse_args()


def write_jsonl(path: Path, records: list[dict]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    source_files = {p.stem: p for p in sorted(args.source_dir.iterdir()) if p.is_file()}
    target_candidates = sorted(p for p in args.target_dir.iterdir() if p.is_file())
    target_map = {}
    for path in target_candidates:
        stem = path.stem
        for source_stem in source_files:
            if stem.startswith(source_stem):
                target_map[source_stem] = path
                break

    records = []
    for source_stem, source_path in sorted(source_files.items()):
        target_path = target_map.get(source_stem)
        mask_path = args.mask_dir / f"{source_stem}.png"
        pose_path = args.pose_dir / f"{source_stem}.jpg"
        if target_path is None or not mask_path.exists() or not pose_path.exists():
            continue
        records.append(
            build_metadata_record(
                target_image=target_path,
                source_image=source_path,
                mask_image=mask_path,
                pose_image=pose_path,
                prompt=args.prompt,
                output_name=source_stem,
                output_subdir=args.output_subdir,
            )
        )

    split_index = int(len(records) * args.train_ratio)
    train_records = records[:split_index]
    val_records = records[split_index:]
    preview_records = train_records[:4]

    ensure_dir(args.output_dir)
    write_jsonl(args.output_dir / "metadata_all.jsonl", records)
    write_jsonl(args.output_dir / "metadata_train.jsonl", train_records)
    write_jsonl(args.output_dir / "metadata_val.jsonl", val_records)
    write_jsonl(args.output_dir / "metadata_train_preview4.jsonl", preview_records)
    print(
        json.dumps(
            {
                "all": len(records),
                "train": len(train_records),
                "val": len(val_records),
                "preview": len(preview_records),
                "output_dir": str(args.output_dir.resolve()),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
