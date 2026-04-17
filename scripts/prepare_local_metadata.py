#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare local JSONL metadata for mannequin editing training.")
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--target-dir", required=True, type=Path)
    parser.add_argument("--mask-dirs", required=True, nargs="+", type=Path)
    parser.add_argument(
        "--conditioning-mode",
        default="same_as_mask",
        choices=["same_as_mask", "external_dir"],
    )
    parser.add_argument("--conditioning-dir", type=Path, default=None)
    parser.add_argument("--train-output", required=True, type=Path)
    parser.add_argument("--val-output", required=True, type=Path)
    parser.add_argument("--all-output", required=True, type=Path)
    parser.add_argument("--val-count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--source-glob", default="*")
    parser.add_argument("--target-suffix", default="_00001_.png")
    parser.add_argument("--conditioning-suffix", default="")
    parser.add_argument("--sample-list", type=Path, default=None)
    parser.add_argument(
        "--prompt",
        default=(
            "replace only the visible human skin, head, neck, hands, and other non-clothing body parts ，with a smooth matte white retail mannequin while preserving all clothing pixels, garment texture, ，garment silhouette, pose, composition, and studio background"
        ),
    )
    return parser.parse_args()


def resolve_mask(stem: str, mask_dirs: list[Path]) -> Path:
    for mask_dir in mask_dirs:
        candidate = mask_dir / f"{stem}.png"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Mask not found for {stem}")


def resolve_conditioning_path(stem: str, args: argparse.Namespace, mask_path: Path) -> Path:
    if args.conditioning_mode == "same_as_mask":
        return mask_path

    if args.conditioning_mode == "external_dir":
        if args.conditioning_dir is None:
            raise ValueError("--conditioning-dir is required when conditioning-mode=external_dir")
        candidate = args.conditioning_dir / f"{stem}{args.conditioning_suffix}"
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"Conditioning image missing for {stem}: {candidate}")

    raise ValueError(f"Unsupported conditioning mode: {args.conditioning_mode}")


def main() -> None:
    args = parse_args()
    source_paths = sorted(
        [p for p in args.source_dir.glob(args.source_glob) if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}]
    )
    if args.sample_list is not None:
        wanted_stems = {line.strip() for line in args.sample_list.read_text(encoding="utf-8").splitlines() if line.strip()}
        source_paths = [path for path in source_paths if path.stem in wanted_stems]

    records = []
    for source_path in source_paths:
        stem = source_path.stem
        target_path = args.target_dir / f"{stem}{args.target_suffix}"
        if not target_path.exists():
            raise FileNotFoundError(f"Target image missing for {stem}: {target_path}")

        mask_path = resolve_mask(stem, args.mask_dirs)
        conditioning_path = resolve_conditioning_path(stem, args, mask_path)

        records.append(
            {
                "target_image": str(target_path.resolve()),
                "source_image": str(source_path.resolve()),
                "mask_image": str(mask_path.resolve()),
                "conditioning_image": str(conditioning_path.resolve()),
                "text": args.prompt,
            }
        )

    rng = random.Random(args.seed)
    rng.shuffle(records)
    val_count = min(args.val_count, len(records))
    val_records = records[:val_count]
    train_records = records[val_count:]

    args.train_output.parent.mkdir(parents=True, exist_ok=True)
    args.val_output.parent.mkdir(parents=True, exist_ok=True)
    args.all_output.parent.mkdir(parents=True, exist_ok=True)

    with args.train_output.open("w", encoding="utf-8") as f:
        for record in train_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    with args.val_output.open("w", encoding="utf-8") as f:
        for record in val_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    with args.all_output.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                "source_count": len(source_paths),
                "train_count": len(train_records),
                "val_count": len(val_records),
                "all_count": len(records),
                "train_output": str(args.train_output),
                "val_output": str(args.val_output),
                "all_output": str(args.all_output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
