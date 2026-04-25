#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.io import ensure_dir, load_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create dual-control metadata by attaching a second conditioning image directory."
    )
    parser.add_argument("--train-metadata", required=True, type=Path)
    parser.add_argument("--val-metadata", required=True, type=Path)
    parser.add_argument("--all-metadata", required=True, type=Path)
    parser.add_argument("--pose-dir", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser.parse_args()


def write_jsonl(records: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def resolve_pose_path(target_stem: str, pose_dir: Path) -> Path | None:
    for suffix in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
        candidate = pose_dir / f"{target_stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def attach_pose(records: list[dict], pose_dir: Path) -> list[dict]:
    transformed = []
    missing = []
    for record in records:
        target_path = Path(record["target_image"])
        pose_path = resolve_pose_path(target_path.stem, pose_dir)
        if pose_path is None:
            missing.append(str(pose_dir / f"{target_path.stem}.png"))
            continue
        new_record = dict(record)
        new_record["conditioning_image_2"] = str(pose_path.resolve())
        transformed.append(new_record)
    if missing:
        preview = "\n".join(missing[:10])
        raise FileNotFoundError(f"Missing pose images for {len(missing)} samples. First few:\n{preview}")
    return transformed


def main() -> None:
    args = parse_args()
    output_root = ensure_dir(args.output_root)

    train_records = attach_pose(load_jsonl(args.train_metadata), args.pose_dir)
    val_records = attach_pose(load_jsonl(args.val_metadata), args.pose_dir)
    all_records = attach_pose(load_jsonl(args.all_metadata), args.pose_dir)

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
                "pose_dir": str(args.pose_dir.resolve()),
                "train_metadata": str(train_out.resolve()),
                "val_metadata": str(val_out.resolve()),
                "all_metadata": str(all_out.resolve()),
                "train_count": len(train_records),
                "val_count": len(val_records),
                "all_count": len(all_records),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
