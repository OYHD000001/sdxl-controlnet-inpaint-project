#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build external inference metadata using clothes masks as inpaint masks and clothes canny as ControlNet input."
    )
    parser.add_argument("--input-metadata", required=True, type=Path)
    parser.add_argument("--clothes-mask-dir", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--output-subdir", required=True)
    parser.add_argument("--output-name-prefix", default="")
    parser.add_argument("--low-threshold", type=int, default=80)
    parser.add_argument("--high-threshold", type=int, default=160)
    parser.add_argument("--black-background-threshold", type=int, default=8)
    return parser.parse_args()


def resolve_mask(stem: str, mask_dir: Path) -> Path:
    for suffix in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
        path = mask_dir / f"{stem}{suffix}"
        if path.exists():
            return path
    raise FileNotFoundError(f"Missing clothes mask for {stem} in {mask_dir}")


def build_clothes_canny(
    image_path: Path,
    clothes_mask_path: Path,
    output_path: Path,
    low_threshold: int,
    high_threshold: int,
    black_background_threshold: int,
) -> None:
    if output_path.exists():
        return

    image = Image.open(image_path).convert("RGB")
    clothes_mask = Image.open(clothes_mask_path).convert("L")
    if clothes_mask.size != image.size:
        clothes_mask = clothes_mask.resize(image.size, Image.NEAREST)

    image_np = np.array(image)
    mask_np = (np.array(clothes_mask) > 127).astype(np.uint8)

    clothes_only = np.zeros_like(image_np)
    clothes_only[mask_np > 0] = image_np[mask_np > 0]

    gray = cv2.cvtColor(clothes_only, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, low_threshold, high_threshold)

    # Keep only non-background edges within clothes to avoid border noise.
    clothes_gray = cv2.cvtColor(clothes_only, cv2.COLOR_RGB2GRAY)
    edges[clothes_gray <= black_background_threshold] = 0

    edge_rgb = np.repeat(edges[:, :, None], 3, axis=2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(edge_rgb).save(output_path)


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    conditioning_dir = args.output_root / "conditioning_canny"
    conditioning_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = args.output_root / "metadata_all.jsonl"

    records = []
    for line in args.input_metadata.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        source_path = Path(record["source_image"])
        stem = source_path.stem
        clothes_mask_path = resolve_mask(stem, args.clothes_mask_dir)
        conditioning_path = conditioning_dir / f"{stem}_clothes_canny.png"
        build_clothes_canny(
            image_path=source_path,
            clothes_mask_path=clothes_mask_path,
            output_path=conditioning_path,
            low_threshold=args.low_threshold,
            high_threshold=args.high_threshold,
            black_background_threshold=args.black_background_threshold,
        )

        new_record = dict(record)
        new_record["target_image"] = str(source_path.resolve())
        new_record["mask_image"] = str(clothes_mask_path.resolve())
        new_record["conditioning_image"] = str(conditioning_path.resolve())
        new_record["output_subdir"] = args.output_subdir
        if args.output_name_prefix:
            new_record["output_name"] = f"{args.output_name_prefix}{record.get('output_name', stem)}"
        records.append(new_record)

    with metadata_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                "input_metadata": str(args.input_metadata.resolve()),
                "clothes_mask_dir": str(args.clothes_mask_dir.resolve()),
                "metadata_path": str(metadata_path.resolve()),
                "conditioning_dir": str(conditioning_dir.resolve()),
                "count": len(records),
                "output_subdir": args.output_subdir,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
