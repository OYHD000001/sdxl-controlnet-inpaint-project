#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.io import ensure_dir, load_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare direct clothing paste and Poisson blending on generated images.")
    parser.add_argument("--metadata-path", required=True, type=Path)
    parser.add_argument("--generated-dir", required=True, type=Path)
    parser.add_argument("--original-dir", type=Path, default=None)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--comparison-dir", required=True, type=Path)
    parser.add_argument("--mask-dilate", type=int, default=5)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument(
        "--source-field",
        choices=["original_dir", "target_image", "source_image"],
        default="original_dir",
        help="Image source used for the clothes pixels that will be pasted/blended onto the generated image.",
    )
    parser.add_argument(
        "--comparison-layout",
        choices=["full", "triptych"],
        default="full",
        help="`full` keeps the debug panels; `triptych` only keeps direct paste, poisson blend, and target.",
    )
    return parser.parse_args()


def base_name_from_target(target_path: Path) -> str:
    name = target_path.name
    if "_00001_" in name:
        return name.split("_00001_")[0]
    return target_path.stem


def add_label(image: Image.Image, label: str, label_h: int = 34) -> Image.Image:
    canvas = Image.new("RGB", (image.width, image.height + label_h), "white")
    canvas.paste(image.convert("RGB"), (0, label_h))
    ImageDraw.Draw(canvas).text((8, 10), label, fill=(0, 0, 0))
    return canvas


def concat(panels: list[tuple[str, Image.Image]]) -> Image.Image:
    labeled = [add_label(image, label) for label, image in panels]
    canvas = Image.new("RGB", (sum(p.width for p in labeled), max(p.height for p in labeled)), (245, 245, 245))
    x = 0
    for panel in labeled:
        canvas.paste(panel, (x, 0))
        x += panel.width
    return canvas


def poisson_blend(source: Image.Image, target: Image.Image, mask: Image.Image) -> Image.Image:
    src_bgr = cv2.cvtColor(np.array(source.convert("RGB")), cv2.COLOR_RGB2BGR)
    dst_bgr = cv2.cvtColor(np.array(target.convert("RGB")), cv2.COLOR_RGB2BGR)
    mask_u8 = np.array(mask.convert("L"))
    ys, xs = np.where(mask_u8 > 0)
    if len(xs) == 0:
        return target.copy()
    center = (int((xs.min() + xs.max()) / 2), int((ys.min() + ys.max()) / 2))
    blended_bgr = cv2.seamlessClone(src_bgr, dst_bgr, mask_u8, center, cv2.NORMAL_CLONE)
    return Image.fromarray(cv2.cvtColor(blended_bgr, cv2.COLOR_BGR2RGB))


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    comparison_dir = ensure_dir(args.comparison_dir)
    records = load_jsonl(args.metadata_path)
    if args.max_images is not None:
        records = records[: args.max_images]

    made = 0
    for record in records:
        target_path = Path(record["target_image"])
        generated_stem = record.get("output_name") or target_path.stem
        generated_path = args.generated_dir
        if record.get("output_subdir"):
            generated_path = generated_path / record["output_subdir"]
        generated_path = generated_path / f"{generated_stem}_generated.png"
        base = base_name_from_target(target_path)
        if args.source_field == "original_dir":
            if args.original_dir is None:
                raise ValueError("--original-dir is required when --source-field=original_dir")
            source_path = args.original_dir / f"{base}.jpg"
        else:
            source_path = Path(record[args.source_field])

        if not generated_path.exists() or not source_path.exists():
            continue

        generated = Image.open(generated_path).convert("RGB")
        source_image = Image.open(source_path).convert("RGB").resize(generated.size, Image.BILINEAR)
        target = Image.open(target_path).convert("RGB").resize(generated.size, Image.BILINEAR)
        clothes_mask = Image.open(record["mask_image"]).convert("L").resize(generated.size, Image.NEAREST)
        mask_np = (np.array(clothes_mask) > 127).astype(np.uint8) * 255
        if args.mask_dilate > 0:
            k = np.ones((args.mask_dilate, args.mask_dilate), np.uint8)
            mask_np = cv2.dilate(mask_np, k, iterations=1)
            mask_np = cv2.morphologyEx(mask_np, cv2.MORPH_CLOSE, k)
        paste_mask = Image.fromarray(mask_np, mode="L")

        direct = generated.copy()
        direct.paste(source_image, mask=paste_mask)
        poisson = poisson_blend(source_image, generated, paste_mask)

        direct.save(output_dir / f"{base}_direct_paste.png")
        poisson.save(output_dir / f"{base}_poisson.png")
        if args.comparison_layout == "triptych":
            panels = [
                ("direct_paste_to_generated", direct),
                ("poisson_blend_to_generated", poisson),
                ("target_white_plastic", target),
            ]
        else:
            panels = [
                (f"{args.source_field}", source_image),
                ("generated", generated),
                ("clothes_mask", paste_mask.convert("RGB")),
                ("direct_paste_to_generated", direct),
                ("poisson_blend_to_generated", poisson),
                ("target_white_plastic", target),
            ]
        concat(panels).save(comparison_dir / f"{base}_paste_vs_poisson.png")
        made += 1

    print(json.dumps({"made": made, "output_dir": str(output_dir), "comparison_dir": str(comparison_dir)}, indent=2))


if __name__ == "__main__":
    main()
