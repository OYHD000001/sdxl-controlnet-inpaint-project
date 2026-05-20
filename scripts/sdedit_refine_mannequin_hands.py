#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from diffusers import StableDiffusionXLInpaintPipeline
from PIL import Image, ImageDraw
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.io import ensure_dir, load_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SDEdit-style local hand refinement for mannequin generations.")
    parser.add_argument("--metadata-path", required=True, type=Path)
    parser.add_argument("--generated-dir", required=True, type=Path)
    parser.add_argument("--reference-dir", required=True, type=Path)
    parser.add_argument("--mask-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--comparison-dir", required=True, type=Path)
    parser.add_argument("--pretrained-model-name-or-path", required=True, type=Path)
    parser.add_argument("--lora-dir", required=True, type=Path)
    parser.add_argument("--variant", default="fp16")
    parser.add_argument("--prompt", default="smooth matte white mannequin hands with clear fingers, clean palm silhouette, crisp wrist shape, preserve all clothing, sleeves, pose, body, composition, and background")
    parser.add_argument("--negative-prompt", default="blurry hands, malformed hands, extra fingers, fused fingers, broken fingers, dark skin, human skin texture, distorted wrists, changed clothes, changed sleeves")
    parser.add_argument("--num-inference-steps", type=int, default=40)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--strength", type=float, default=0.5)
    parser.add_argument("--dilate", type=int, default=11)
    parser.add_argument("--max-images", type=int, default=None)
    return parser.parse_args()


def load_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def load_l(path: Path) -> Image.Image:
    return Image.open(path).convert("L")


def _touches_border(stats_row: np.ndarray, width: int, height: int) -> bool:
    x = int(stats_row[cv2.CC_STAT_LEFT])
    y = int(stats_row[cv2.CC_STAT_TOP])
    w = int(stats_row[cv2.CC_STAT_WIDTH])
    h = int(stats_row[cv2.CC_STAT_HEIGHT])
    return x <= 0 or y <= 0 or (x + w) >= width or (y + h) >= height


def mannequin_hand_mask(reference_rgb: Image.Image, clothes_mask_l: Image.Image, dilate: int) -> Image.Image:
    rgb = np.array(reference_rgb)
    clothes = np.array(clothes_mask_l)
    h, w = clothes.shape

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    _, sat, val = cv2.split(hsv)

    corners = np.concatenate(
        [rgb[:20, :20], rgb[:20, -20:], rgb[-20:, :20], rgb[-20:, -20:]],
        axis=0,
    ).reshape(-1, 3)
    bg_rgb = np.median(corners, axis=0).astype(np.float32)
    bg_val = float(np.median(cv2.cvtColor(corners.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV)[:, 0, 2]))
    dist_from_bg = np.linalg.norm(rgb.astype(np.float32) - bg_rgb[None, None, :], axis=2)

    # White mannequin is close to the pale background in color space, so use
    # both low-saturation brightness and distance-from-background cues.
    white_like = (
        (dist_from_bg > 18.0)
        | ((sat < 36) & (val > 175) & (val > bg_val + 6.0))
    ).astype(np.uint8) * 255

    clothes_margin = max(3, dilate // 3)
    if clothes_margin % 2 == 0:
        clothes_margin += 1
    clothes_dilated = cv2.dilate(
        (clothes > 127).astype(np.uint8) * 255,
        np.ones((clothes_margin, clothes_margin), np.uint8),
        iterations=1,
    )
    white_like[clothes_dilated > 0] = 0

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats((white_like > 0).astype(np.uint8), connectivity=8)
    selected = np.zeros((h, w), dtype=np.uint8)

    min_area = max(40, int(h * w * 0.00018))
    max_area = int(h * w * 0.03)
    left_bound = int(w * 0.34)
    right_bound = int(w * 0.66)
    upper_y = int(h * 0.43)
    lower_y = int(h * 0.97)

    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area or area > max_area:
            continue
        if _touches_border(stats[label], w, h):
            continue
        cx, cy = centroids[label]
        if cy < upper_y:
            continue
        if cy > lower_y:
            continue
        if left_bound <= cx <= right_bound:
            continue
        selected[labels == label] = 255

    if selected.max() > 0:
        kernel = np.ones((max(3, dilate), max(3, dilate)), np.uint8)
        selected = cv2.dilate(selected, kernel, iterations=1)
        selected = cv2.morphologyEx(selected, cv2.MORPH_CLOSE, kernel)

    return Image.fromarray(selected, mode="L")


def add_label(image: Image.Image, label: str, label_height: int = 34) -> Image.Image:
    canvas = Image.new("RGB", (image.width, image.height + label_height), (255, 255, 255))
    canvas.paste(image, (0, label_height))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 9), label, fill=(0, 0, 0))
    return canvas


def build_comparison(generated: Image.Image, reference: Image.Image, hand_mask: Image.Image, refined: Image.Image) -> Image.Image:
    width, height = generated.size
    panels = [
        ("generated", generated),
        ("reference", reference),
        ("hand_mask", hand_mask.convert("RGB")),
        ("refined", refined),
    ]
    labeled = []
    for label, image in panels:
        image = image.resize((width, height), Image.NEAREST if "mask" in label else Image.BILINEAR).convert("RGB")
        labeled.append(add_label(image, label))
    canvas = Image.new("RGB", (sum(panel.width for panel in labeled), max(panel.height for panel in labeled)), (245, 245, 245))
    x = 0
    for panel in labeled:
        canvas.paste(panel, (x, 0))
        x += panel.width
    return canvas


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    comparison_dir = ensure_dir(args.comparison_dir)
    hand_mask_dir = ensure_dir(output_dir / "hand_masks")

    records = load_jsonl(args.metadata_path)
    if args.max_images is not None:
        records = records[: args.max_images]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32

    pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        variant=args.variant,
        torch_dtype=torch_dtype,
    )
    pipe.unet.load_lora_adapter(
        args.lora_dir,
        adapter_name="default",
        prefix=None,
        weight_name="pytorch_lora_weights.safetensors",
    )
    if hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()
    if torch.cuda.is_available():
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)
    pipe.set_progress_bar_config(disable=False)

    made = 0
    for record in tqdm(records, desc="sdedit-hand-refine"):
        stem = Path(record["source_image"]).stem
        generated_path = args.generated_dir / f"{stem}_generated.png"
        refined_path = output_dir / f"{stem}_generated.png"
        hand_mask_path = hand_mask_dir / f"{stem}_hand_mask.png"
        comparison_path = comparison_dir / f"{stem}_hand_sdedit_compare.png"

        if refined_path.exists() and comparison_path.exists() and hand_mask_path.exists():
            continue
        if not generated_path.exists():
            continue

        generated = load_rgb(generated_path)
        reference = load_rgb(args.reference_dir / Path(record["source_image"]).name).resize(generated.size, Image.BILINEAR)
        clothes_mask = load_l(args.mask_dir / Path(record["mask_image"]).name).resize(generated.size, Image.NEAREST)
        hand_mask = mannequin_hand_mask(reference, clothes_mask, args.dilate)
        hand_mask_path.parent.mkdir(parents=True, exist_ok=True)
        hand_mask.save(hand_mask_path)

        if np.array(hand_mask).max() == 0:
            generated.save(refined_path)
            build_comparison(generated, reference, hand_mask, generated).save(comparison_path)
            continue

        with torch.inference_mode():
            result = pipe(
                prompt=args.prompt,
                negative_prompt=args.negative_prompt,
                image=generated,
                mask_image=hand_mask,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                strength=args.strength,
            )
        refined = result.images[0].convert("RGB")
        if refined.size != generated.size:
            refined = refined.resize(generated.size, Image.BILINEAR)
        refined.save(refined_path)
        build_comparison(generated, reference, hand_mask, refined).save(comparison_path)
        made += 1

    print(json.dumps(
        {
            "made": made,
            "output_dir": str(output_dir),
            "comparison_dir": str(comparison_dir),
            "hand_mask_dir": str(hand_mask_dir),
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
