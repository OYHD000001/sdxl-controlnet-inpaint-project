#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
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
    parser = argparse.ArgumentParser(description="Prepare generated images and head/hand masks for Flux inpaint.")
    parser.add_argument("--metadata-path", required=True, type=Path)
    parser.add_argument("--generated-dir", required=True, type=Path)
    parser.add_argument("--original-dir", required=True, type=Path)
    parser.add_argument("--comfy-input-root", required=True, type=Path)
    parser.add_argument("--comparison-dir", required=True, type=Path)
    parser.add_argument("--max-images", type=int, default=None)
    return parser.parse_args()


def base_name_from_target(target_path: Path) -> str:
    name = target_path.name
    if "_00001_" in name:
        return name.split("_00001_")[0]
    return target_path.stem


def skin_mask_from_original(original_bgr: np.ndarray, clothes_mask: np.ndarray) -> np.ndarray:
    h, w = original_bgr.shape[:2]
    ycrcb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    hsv = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2HSV)
    hue, sat, val = cv2.split(hsv)

    ycrcb_skin = (y > 45) & (cr > 130) & (cr < 185) & (cb > 70) & (cb < 145)
    hsv_skin = (((hue < 30) | (hue > 165)) & (sat > 18) & (sat < 180) & (val > 55))
    skin = (ycrcb_skin | hsv_skin).astype(np.uint8) * 255

    clothes_dilated = cv2.dilate((clothes_mask > 127).astype(np.uint8) * 255, np.ones((11, 11), np.uint8), iterations=1)
    skin[clothes_dilated > 0] = 0

    skin = cv2.morphologyEx(skin, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    skin = cv2.morphologyEx(skin, cv2.MORPH_CLOSE, np.ones((13, 13), np.uint8))

    ys, xs = np.where(clothes_mask > 127)
    if len(xs) == 0:
        clothing_bottom = int(h * 0.75)
    else:
        clothing_bottom = int(np.percentile(ys, 98))

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats((skin > 0).astype(np.uint8), connectivity=8)
    selected = np.zeros((h, w), np.uint8)
    min_area = max(180, int(h * w * 0.00025))
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        cx, cy = centroids[label]
        # Keep face/neck/hands in the upper and middle body; reject exposed legs when present.
        if cy > clothing_bottom + 0.08 * h:
            continue
        if cy > 0.92 * h:
            continue
        selected[labels == label] = 255

    selected = cv2.dilate(selected, np.ones((31, 31), np.uint8), iterations=1)
    selected = cv2.morphologyEx(selected, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    selected[clothes_dilated > 0] = 0
    return selected


def make_comparison(generated: Image.Image, original: Image.Image, clothes_mask: Image.Image, head_hand_mask: Image.Image) -> Image.Image:
    panels = [
        ("generated", generated),
        ("original_human", original),
        ("clothes_mask", clothes_mask.convert("RGB")),
        ("head_hand_mask", head_hand_mask.convert("RGB")),
    ]
    width, height = generated.size
    label_h = 34
    canvas = Image.new("RGB", (width * len(panels), height + label_h), (245, 245, 245))
    draw = ImageDraw.Draw(canvas)
    for index, (label, image) in enumerate(panels):
        image = image.resize((width, height), Image.NEAREST if "mask" in label else Image.BILINEAR).convert("RGB")
        x = index * width
        draw.text((x + 8, 10), label, fill=(0, 0, 0))
        canvas.paste(image, (x, label_h))
    return canvas


def main() -> None:
    args = parse_args()
    image_dir = ensure_dir(args.comfy_input_root / "image")
    mask_dir = ensure_dir(args.comfy_input_root / "mask")
    comparison_dir = ensure_dir(args.comparison_dir)

    records = load_jsonl(args.metadata_path)
    if args.max_images is not None:
        records = records[: args.max_images]

    made = 0
    for record in records:
        target_path = Path(record["target_image"])
        generated_path = args.generated_dir / f"{target_path.stem}_generated.png"
        base = base_name_from_target(target_path)
        original_path = args.original_dir / f"{base}.jpg"
        if not generated_path.exists() or not original_path.exists():
            continue

        generated = Image.open(generated_path).convert("RGB")
        original = Image.open(original_path).convert("RGB").resize(generated.size, Image.BILINEAR)
        clothes_mask = Image.open(record["mask_image"]).convert("L").resize(generated.size, Image.NEAREST)

        original_bgr = cv2.cvtColor(np.array(original), cv2.COLOR_RGB2BGR)
        clothes_np = np.array(clothes_mask)
        head_hand_np = skin_mask_from_original(original_bgr, clothes_np)
        head_hand_mask = Image.fromarray(head_hand_np, mode="L")

        shutil.copy2(generated_path, image_dir / generated_path.name)
        head_hand_mask.save(mask_dir / generated_path.name)
        make_comparison(generated, original, clothes_mask, head_hand_mask).save(comparison_dir / f"{base}_head_hand_mask_debug.png")
        made += 1

    print(json.dumps({"made": made, "image_dir": str(image_dir), "mask_dir": str(mask_dir), "comparison_dir": str(comparison_dir)}, indent=2))


if __name__ == "__main__":
    main()
