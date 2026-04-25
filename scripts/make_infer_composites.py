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
from src.utils.masks import invert_binary_mask, invert_conditioning_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create side-by-side inference composite images.")
    parser.add_argument("--config", required=True, type=Path, help="Training/inference config path.")
    parser.add_argument("--metadata-path", type=Path, default=None, help="Optional metadata override.")
    parser.add_argument("--infer-dir", type=Path, default=None, help="Optional generated image directory override.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Optional composite output directory override.")
    return parser.parse_args()


def add_label(image: Image.Image, label: str, label_height: int = 36) -> Image.Image:
    canvas = Image.new("RGB", (image.width, image.height + label_height), (255, 255, 255))
    canvas.paste(image, (0, label_height))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 10), label, fill=(0, 0, 0))
    return canvas


def to_rgb_mask(mask: Image.Image) -> Image.Image:
    return mask.convert("RGB")


def resize_panel(image: Image.Image, width: int, height: int, is_mask: bool) -> Image.Image:
    interpolation = Image.NEAREST if is_mask else Image.BILINEAR
    return image.resize((width, height), interpolation)


def resize_conditioning_panel(
    image: Image.Image,
    width: int,
    height: int,
    resize_modes: list[str],
    index: int,
) -> Image.Image:
    mode = resize_modes[index] if index < len(resize_modes) else "bilinear"
    interpolation = Image.NEAREST if mode.lower() == "nearest" else Image.BILINEAR
    return image.resize((width, height), interpolation)


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


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    infer_cfg = config["inference"]
    data_cfg = config["data"]
    invert_mask = bool(infer_cfg.get("invert_mask", data_cfg.get("invert_mask", False)))
    image_width = int(data_cfg.get("image_width", data_cfg["image_size"]))
    image_height = int(data_cfg.get("image_height", data_cfg["image_size"]))
    conditioning_resize_modes = data_cfg.get("conditioning_resize_modes", ["bilinear", "bilinear"])

    metadata_path = args.metadata_path or Path(infer_cfg["metadata_path"])
    infer_dir = args.infer_dir or Path(infer_cfg["output_dir"])
    output_dir = ensure_dir(args.output_dir or (Path(infer_dir).parent / f"{Path(infer_dir).name}_composites_resized"))

    records = load_jsonl(metadata_path)
    for record in records:
        stem = Path(record["source_image"]).stem
        generated_path = infer_dir / f"{stem}_generated.png"
        if not generated_path.exists():
            continue

        source_image = load_pil_image(record["source_image"]).convert("RGB")
        mask_image = load_pil_image(record["mask_image"]).convert("L")
        conditioning_image = load_pil_image(record["conditioning_image"]).convert("RGB")
        conditioning_image_2 = (
            load_pil_image(record["conditioning_image_2"]).convert("RGB")
            if record.get("conditioning_image_2")
            else None
        )
        generated_image = load_pil_image(generated_path).convert("RGB")
        target_image = load_pil_image(record["target_image"]).convert("RGB") if record.get("target_image") else None

        if invert_mask:
            mask_image = invert_binary_mask(mask_image)
            if record["conditioning_image"] == record["mask_image"]:
                conditioning_image = invert_conditioning_image(conditioning_image)

        source_image = resize_panel(source_image, image_width, image_height, is_mask=False)
        mask_image = resize_panel(mask_image, image_width, image_height, is_mask=True)
        conditioning_image = resize_conditioning_panel(
            conditioning_image,
            image_width,
            image_height,
            conditioning_resize_modes,
            0,
        )
        if conditioning_image_2 is not None:
            conditioning_image_2 = resize_conditioning_panel(
                conditioning_image_2,
                image_width,
                image_height,
                conditioning_resize_modes,
                1,
            )
        generated_image = resize_panel(generated_image, image_width, image_height, is_mask=False)
        if target_image is not None:
            target_image = resize_panel(target_image, image_width, image_height, is_mask=False)

        panel_images: list[tuple[str, Image.Image]] = [
            ("source", source_image),
            ("mask", to_rgb_mask(mask_image)),
            ("conditioning", conditioning_image),
        ]
        if conditioning_image_2 is not None:
            panel_images.append(("conditioning_2", conditioning_image_2))
        panel_images.append(("generated", generated_image))
        if target_image is not None:
            panel_images.append(("target", target_image))

        composite = build_composite(panel_images)
        composite.save(output_dir / f"{stem}_composite.png")

    print(output_dir.resolve())


if __name__ == "__main__":
    main()
