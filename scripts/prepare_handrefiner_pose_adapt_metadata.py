#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.io import ensure_dir, load_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare HandRefiner-style metadata using pose-derived hand masks.")
    parser.add_argument("--input-metadata", required=True, type=Path)
    parser.add_argument("--generated-dir", required=True, type=Path)
    parser.add_argument("--output-metadata", required=True, type=Path)
    parser.add_argument("--mask-dir", required=True, type=Path)
    parser.add_argument("--dilate", type=int, default=21)
    parser.add_argument("--min-area", type=int, default=12)
    return parser.parse_args()


def build_hand_mask_from_pose(pose_path: Path, dilate: int, min_area: int) -> Image.Image:
    pose = np.array(Image.open(pose_path).convert("RGB"))

    # DWPose hand landmarks are rendered in magenta / pink tones.
    magenta = (
        (pose[:, :, 0] > 120)
        & (pose[:, :, 2] > 120)
        & (pose[:, :, 1] < 170)
        & ((pose[:, :, 0].astype(np.int16) + pose[:, :, 2].astype(np.int16) - pose[:, :, 1].astype(np.int16)) > 180)
    ).astype(np.uint8) * 255

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((magenta > 0).astype(np.uint8), connectivity=8)
    selected = np.zeros_like(magenta)
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        selected[labels == label] = 255

    if selected.max() > 0:
        kernel_size = max(3, dilate)
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        selected = cv2.dilate(selected, kernel, iterations=1)
        selected = cv2.morphologyEx(selected, cv2.MORPH_CLOSE, kernel)

    return Image.fromarray(selected, mode="L")


def main() -> None:
    args = parse_args()
    records = load_jsonl(args.input_metadata)
    ensure_dir(args.mask_dir)
    args.output_metadata.parent.mkdir(parents=True, exist_ok=True)

    prompt = (
        "refine only the mannequin hands so they have clear fingers, clean palm silhouettes, and crisp wrists, "
        "while preserving the exact clothes, sleeves, garment texture, pose, body, face, composition, and background"
    )

    output_records = []
    for record in records:
        stem = Path(record["source_image"]).stem
        generated_path = args.generated_dir / f"{stem}_generated.png"
        if not generated_path.exists():
            continue

        pose_path = Path(record["conditioning_image_2"])
        hand_mask = build_hand_mask_from_pose(pose_path, args.dilate, args.min_area)
        mask_path = args.mask_dir / f"{stem}_hand_mask.png"
        hand_mask.save(mask_path)

        output_records.append(
            {
                "source_image": str(generated_path),
                "target_image": record.get("target_image"),
                "mask_image": str(mask_path),
                "conditioning_image": record["conditioning_image"],
                "conditioning_image_2": record.get("conditioning_image_2"),
                "text": prompt,
                "output_name": stem,
            }
        )

    with args.output_metadata.open("w", encoding="utf-8") as f:
        for record in output_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(json.dumps({"count": len(output_records), "output_metadata": str(args.output_metadata), "mask_dir": str(args.mask_dir)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
