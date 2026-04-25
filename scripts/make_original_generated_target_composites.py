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
    parser = argparse.ArgumentParser(description="Create original vs generated vs target composite images.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--metadata-path", type=Path, default=None)
    parser.add_argument("--infer-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--original-dir", type=Path, default=Path("/data/ouyanghaodong/oyhd_20260318_extracted/oyhd/ComfyUI/input/image2"))
    return parser.parse_args()


def add_label(image: Image.Image, label: str, label_height: int = 36) -> Image.Image:
    canvas = Image.new("RGB", (image.width, image.height + label_height), (255, 255, 255))
    canvas.paste(image, (0, label_height))
    ImageDraw.Draw(canvas).text((10, 10), label, fill=(0, 0, 0))
    return canvas


def resize_panel(image: Image.Image, width: int, height: int) -> Image.Image:
    return image.resize((width, height), Image.BILINEAR)


def build_composite(images: list[tuple[str, Image.Image]]) -> Image.Image:
    labeled = [add_label(image, label) for label, image in images]
    width = sum(image.width for image in labeled)
    height = max(image.height for image in labeled)
    canvas = Image.new("RGB", (width, height), (245, 245, 245))
    x = 0
    for image in labeled:
        canvas.paste(image, (x, 0))
        x += image.width
    return canvas


def find_original_image(original_dir: Path, target_stem: str) -> Path | None:
    base_stem = target_stem.split("_00001_")[0]
    for ext in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
        candidate = original_dir / f"{base_stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    infer_cfg = config["inference"]
    data_cfg = config["data"]
    image_width = int(data_cfg.get("image_width", data_cfg["image_size"]))
    image_height = int(data_cfg.get("image_height", data_cfg["image_size"]))

    metadata_path = args.metadata_path or Path(infer_cfg["metadata_path"])
    infer_dir = args.infer_dir or Path(infer_cfg["output_dir"])
    output_dir = ensure_dir(args.output_dir or (Path(infer_dir).parent / "original_generated_target_composites"))
    original_dir = args.original_dir

    made = 0
    for record in load_jsonl(metadata_path):
        target_path = Path(record["target_image"])
        target_stem = target_path.stem
        generated_path = infer_dir / f"{target_stem}_generated.png"
        original_path = find_original_image(original_dir, target_stem)
        if not generated_path.exists() or original_path is None:
            continue

        original = resize_panel(load_pil_image(original_path).convert("RGB"), image_width, image_height)
        generated = resize_panel(load_pil_image(generated_path).convert("RGB"), image_width, image_height)
        target = resize_panel(load_pil_image(target_path).convert("RGB"), image_width, image_height)

        composite = build_composite(
            [
                ("original_human", original),
                ("generated", generated),
                ("target_white_plastic", target),
            ]
        )
        composite.save(output_dir / f"{target_stem}_original_generated_target.png")
        made += 1

    print(output_dir.resolve())
    print(f"made={made}")


if __name__ == "__main__":
    main()
