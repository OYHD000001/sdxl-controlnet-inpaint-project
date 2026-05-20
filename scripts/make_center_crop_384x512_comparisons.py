#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create center-crop 384x512 previews with comparisons.")
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--count", type=int, default=20)
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


def add_label(image: Image.Image, label: str, label_height: int = 36) -> Image.Image:
    canvas = Image.new("RGB", (image.width, image.height + label_height), (255, 255, 255))
    canvas.paste(image, (0, label_height))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 10), label, fill=(0, 0, 0))
    return canvas


def draw_crop_box(image: Image.Image, box: tuple[int, int, int, int]) -> Image.Image:
    preview = image.copy().convert("RGB")
    draw = ImageDraw.Draw(preview)
    draw.rectangle(box, outline=(255, 64, 64), width=8)
    return preview


def build_triptych(original: Image.Image, boxed: Image.Image, cropped: Image.Image) -> Image.Image:
    panels = [
        add_label(original, "original"),
        add_label(boxed, "crop_box"),
        add_label(cropped, "center_crop_384x512"),
    ]
    width = sum(panel.width for panel in panels)
    height = max(panel.height for panel in panels)
    canvas = Image.new("RGB", (width, height), (245, 245, 245))
    x = 0
    for panel in panels:
        canvas.paste(panel, (x, 0))
        x += panel.width
    return canvas


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cropped_dir = args.output_dir / "cropped_384x512"
    compare_dir = args.output_dir / "comparisons"
    cropped_dir.mkdir(parents=True, exist_ok=True)
    compare_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        path
        for path in args.input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    )[: args.count]

    for path in image_paths:
        with Image.open(path) as image:
            image = image.convert("RGB")
            box = compute_center_crop_box(image.width, image.height, args.target_width, args.target_height)
            boxed = draw_crop_box(image, box)
            cropped = image.crop(box).resize((args.target_width, args.target_height), Image.BILINEAR)

            cropped.save(cropped_dir / path.name)
            compare = build_triptych(image, boxed, cropped)
            compare.save(compare_dir / f"{path.stem}_crop_compare.png")

    print(compare_dir.resolve())


if __name__ == "__main__":
    main()
