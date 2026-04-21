#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.io import ensure_dir, load_config, load_jsonl, load_pil_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paste original clothes back onto generated images.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--metadata-path", type=Path, default=None)
    parser.add_argument("--generated-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--comparison-dir", type=Path, default=None)
    parser.add_argument("--threshold", type=int, default=8, help="RGB threshold for clothes matte extraction.")
    return parser.parse_args()


def build_clothes_alpha(conditioning_image: Image.Image, threshold: int) -> Image.Image:
    rgb = conditioning_image.convert("RGB")
    alpha = Image.new("L", rgb.size, 0)
    src = rgb.load()
    dst = alpha.load()
    width, height = rgb.size
    for y in range(height):
        for x in range(width):
            r, g, b = src[x, y]
            dst[x, y] = 255 if max(r, g, b) > threshold else 0
    return alpha


def add_label(image: Image.Image, label: str, label_height: int = 36) -> Image.Image:
    canvas = Image.new("RGB", (image.width, image.height + label_height), (255, 255, 255))
    canvas.paste(image, (0, label_height))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 10), label, fill=(0, 0, 0))
    return canvas


def build_comparison(images: list[tuple[str, Image.Image]]) -> Image.Image:
    labeled = [add_label(image, label) for label, image in images]
    width = sum(image.width for image in labeled)
    height = max(image.height for image in labeled)
    canvas = Image.new("RGB", (width, height), (245, 245, 245))
    x = 0
    for image in labeled:
        canvas.paste(image, (x, 0))
        x += image.width
    return canvas


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    infer_cfg = config["inference"]

    metadata_path = args.metadata_path or Path(infer_cfg["metadata_path"])
    generated_dir = args.generated_dir or Path(infer_cfg["output_dir"])
    output_dir = ensure_dir(args.output_dir or (Path(generated_dir).parent / f"{Path(generated_dir).name}_pasted_clothes"))
    comparison_dir = ensure_dir(
        args.comparison_dir or (Path(generated_dir).parent / f"{Path(generated_dir).name}_pasted_clothes_comparisons")
    )

    records = load_jsonl(metadata_path)
    for record in records:
        stem = Path(record["source_image"]).stem
        generated_path = generated_dir / f"{stem}_generated.png"
        if not generated_path.exists():
            continue

        generated = load_pil_image(generated_path).convert("RGB")
        source = load_pil_image(record["source_image"]).convert("RGB")
        conditioning = load_pil_image(record["conditioning_image"]).convert("RGB")
        target = load_pil_image(record["target_image"]).convert("RGB") if record.get("target_image") else None

        if source.size != generated.size:
            source = source.resize(generated.size, Image.BILINEAR)
        if conditioning.size != generated.size:
            conditioning = conditioning.resize(generated.size, Image.BILINEAR)
        if target is not None and target.size != generated.size:
            target = target.resize(generated.size, Image.BILINEAR)

        clothes_alpha = build_clothes_alpha(conditioning, threshold=args.threshold)

        pasted = generated.copy()
        pasted.paste(conditioning, mask=clothes_alpha)
        pasted.save(output_dir / f"{stem}_pasted.png")

        comparison = build_comparison(
            [
                ("1_source(original)", source),
                ("2_conditioning(clothes)", conditioning),
                ("3_generated", generated),
                ("4_pasted_original_clothes", pasted),
                ("5_target", target if target is not None else source),
            ]
        )
        comparison.save(comparison_dir / f"{stem}_comparison.png")

    print(output_dir.resolve())
    print(comparison_dir.resolve())


if __name__ == "__main__":
    main()
