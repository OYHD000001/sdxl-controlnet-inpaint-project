#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Crop original clothes masks to match a 384x512 center-cropped dataset and extract clothes cutouts."
    )
    parser.add_argument("--cropped-image-dir", required=True, type=Path)
    parser.add_argument("--original-mask-dir", required=True, type=Path)
    parser.add_argument("--output-mask-dir", required=True, type=Path)
    parser.add_argument("--output-clothes-dir", required=True, type=Path)
    parser.add_argument("--target-width", type=int, default=384)
    parser.add_argument("--target-height", type=int, default=512)
    return parser.parse_args()


def compute_center_crop_box(width: int, height: int, target_width: int, target_height: int) -> tuple[int, int, int, int]:
    target_ratio = target_width / target_height
    source_ratio = width / height

    if source_ratio > target_ratio:
        crop_height = height
        crop_width = round(height * target_ratio)
    else:
        crop_width = width
        crop_height = round(width / target_ratio)

    left = max(0, (width - crop_width) // 2)
    top = max(0, (height - crop_height) // 2)
    right = left + crop_width
    bottom = top + crop_height
    return left, top, right, bottom


def main() -> None:
    args = parse_args()
    args.output_mask_dir.mkdir(parents=True, exist_ok=True)
    args.output_clothes_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for image_path in sorted(args.cropped_image_dir.glob("*.png")):
        mask_path = args.original_mask_dir / image_path.name
        if not mask_path.exists():
            raise FileNotFoundError(f"Missing original mask for {image_path.name}: {mask_path}")

        with Image.open(image_path) as cropped_image:
            cropped_image = cropped_image.convert("RGB")

        with Image.open(mask_path) as original_mask:
            original_mask = original_mask.convert("L")
            box = compute_center_crop_box(
                original_mask.width,
                original_mask.height,
                args.target_width,
                args.target_height,
            )
            cropped_mask = original_mask.crop(box).resize((args.target_width, args.target_height), Image.NEAREST)

        cropped_mask.save(args.output_mask_dir / image_path.name)

        image_array = np.array(cropped_image)
        mask_array = (np.array(cropped_mask) > 127).astype(np.uint8)
        clothes_array = image_array * mask_array[:, :, None]
        Image.fromarray(clothes_array).save(args.output_clothes_dir / image_path.name)
        count += 1

    print(count)


if __name__ == "__main__":
    main()
