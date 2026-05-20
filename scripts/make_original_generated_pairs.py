#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.io import ensure_dir, load_jsonl, load_pil_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create original vs generated side-by-side comparison images.")
    parser.add_argument("--metadata-path", required=True, type=Path)
    parser.add_argument("--generated-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--image-width", type=int, default=384)
    parser.add_argument("--image-height", type=int, default=512)
    parser.add_argument("--interval", type=int, default=20)
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def add_label(image: Image.Image, label: str, label_height: int = 36) -> Image.Image:
    canvas = Image.new("RGB", (image.width, image.height + label_height), (255, 255, 255))
    canvas.paste(image, (0, label_height))
    ImageDraw.Draw(canvas).text((10, 10), label, fill=(0, 0, 0))
    return canvas


def resize_panel(image: Image.Image, width: int, height: int) -> Image.Image:
    return image.resize((width, height), Image.BILINEAR)


def build_pair(original: Image.Image, generated: Image.Image) -> Image.Image:
    left = add_label(original, "original_human")
    right = add_label(generated, "generated")
    canvas = Image.new("RGB", (left.width + right.width, max(left.height, right.height)), (245, 245, 245))
    canvas.paste(left, (0, 0))
    canvas.paste(right, (left.width, 0))
    return canvas


def process_once(metadata_path: Path, generated_dir: Path, output_dir: Path, image_width: int, image_height: int) -> int:
    made = 0
    for record in load_jsonl(metadata_path):
        source_path = Path(record["source_image"])
        stem = record.get("output_name") or source_path.stem
        generated_path = generated_dir
        if record.get("output_subdir"):
            generated_path = generated_path / record["output_subdir"]
        generated_path = generated_path / f"{stem}_generated.png"
        out_path = output_dir / f"{stem}_original_generated.png"
        if out_path.exists() or not generated_path.exists():
            continue
        original = resize_panel(load_pil_image(source_path).convert("RGB"), image_width, image_height)
        generated = resize_panel(load_pil_image(generated_path).convert("RGB"), image_width, image_height)
        build_pair(original, generated).save(out_path)
        made += 1
    return made


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    if args.once:
        made = process_once(args.metadata_path, args.generated_dir, output_dir, args.image_width, args.image_height)
        print(output_dir.resolve())
        print(f"made={made}")
        return

    import time

    while True:
        made = process_once(args.metadata_path, args.generated_dir, output_dir, args.image_width, args.image_height)
        print(f"made={made}")
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
