from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image
from PIL import ImageDraw

from preprocess.extract import prepare_inference_inputs
from src.utils.io import ensure_dir, load_config, load_jsonl, load_pil_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check canonical inpaint mask polarity.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--index", type=int, default=0, help="Sample index inside metadata.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output image path. Defaults to outputs/mask_polarity_check.png beside config output dir.",
    )
    return parser.parse_args()


def add_label(image: Image.Image, label: str, label_height: int = 36) -> Image.Image:
    canvas = Image.new("RGB", (image.width, image.height + label_height), (255, 255, 255))
    canvas.paste(image.convert("RGB"), (0, label_height))
    ImageDraw.Draw(canvas).text((10, 10), label, fill=(0, 0, 0))
    return canvas


def build_grid(panels: list[tuple[str, Image.Image]]) -> Image.Image:
    labeled: list[Image.Image] = []
    for label, image in panels:
        panel = add_label(image, label)
        labeled.append(panel)
    canvas = Image.new("RGB", (sum(p.width for p in labeled), max(p.height for p in labeled)), (245, 245, 245))
    x = 0
    for panel in labeled:
        canvas.paste(panel, (x, 0))
        x += panel.width
    return canvas


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    metadata_path = Path(config["inference"]["metadata_path"])
    records = load_jsonl(metadata_path)
    if not records:
        raise RuntimeError(f"No records found in {metadata_path}")
    if args.index < 0 or args.index >= len(records):
        raise IndexError(f"Index {args.index} out of range for {metadata_path} ({len(records)} records)")
    record = records[args.index]

    image_width = int(config["data"]["image_width"])
    image_height = int(config["data"]["image_height"])
    source_background_value = int(config["data"].get("source_background_value", 0))
    conditioning_resize_modes = config["data"].get("conditioning_resize_modes") or ["nearest"]
    conditioning_resize_mode = conditioning_resize_modes[0]

    source_image = load_pil_image(record["source_image"]).convert("RGB")
    clothes_mask = load_pil_image(record["mask_image"]).convert("L")
    pose_image = load_pil_image(record["conditioning_image"]).convert("RGB")

    canonical = prepare_inference_inputs(
        source_image=source_image,
        clothes_mask=clothes_mask,
        pose_image=pose_image,
        image_height=image_height,
        image_width=image_width,
        source_background_value=source_background_value,
        conditioning_resize_mode=conditioning_resize_mode,
    )

    mask_image = canonical["mask_image"].convert("L")
    mask_min, mask_max = mask_image.getextrema()
    histogram = mask_image.histogram()
    total_pixels = max(sum(histogram), 1)
    keep_pixels = histogram[0]
    redraw_pixels = total_pixels - keep_pixels

    print(f"record: {record.get('output_name') or Path(record['source_image']).stem}")
    print(f"mask extrema: min={mask_min} max={mask_max}")
    print(f"mask keep pixels (0): {keep_pixels} ({keep_pixels / total_pixels:.4%})")
    print(f"mask redraw pixels (255): {redraw_pixels} ({redraw_pixels / total_pixels:.4%})")
    print("expected polarity: clothes keep region -> 0, mannequin/body/background redraw region -> 255")

    output_path = args.output
    if output_path is None:
        project_output = Path(config["project"]["output_dir"])
        output_path = project_output / "mask_polarity_check.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mask_rgb = mask_image.convert("RGB")
    grid = build_grid(
        [
            ("source", canonical["source_image"]),
            ("masked_source", canonical["masked_source_image"]),
            ("mask", mask_rgb),
            ("pose", canonical["conditioning_image"]),
        ]
    )
    grid.save(output_path)
    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
