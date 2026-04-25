#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm.auto import tqdm


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build inference metadata and canny conditioning images for external mannequin datasets.")
    parser.add_argument("--image-dir", required=True, type=Path)
    parser.add_argument("--mask-dir", required=True, type=Path)
    parser.add_argument("--pose-dir", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--dataset-tag", required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--low-threshold", type=int, default=80)
    parser.add_argument("--high-threshold", type=int, default=160)
    parser.add_argument("--head-margin", type=int, default=8)
    parser.add_argument(
        "--prompt",
        default=(
            "replace only the visible human skin, head, neck, hands, and other non-clothing body parts "
            "with a smooth matte white retail mannequin while preserving all clothing pixels, garment texture, "
            "garment silhouette, pose, composition, and studio background"
        ),
    )
    return parser.parse_args()


def list_source_paths(image_dir: Path) -> list[Path]:
    return sorted(path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def resolve_file(stem: str, directory: Path) -> Path | None:
    for suffix in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
        candidate = directory / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def build_headless_person_canny(
    image_path: Path,
    mask_path: Path,
    pose_path: Path,
    output_path: Path,
    low_threshold: int,
    high_threshold: int,
    head_margin: int,
) -> None:
    if output_path.exists():
        return

    image = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")
    pose = Image.open(pose_path).convert("RGB")
    if mask.size != image.size:
        mask = mask.resize(image.size, Image.NEAREST)
    if pose.size != image.size:
        pose = pose.resize(image.size, Image.NEAREST)

    image_np = np.array(image)
    mask_np = (np.array(mask) > 127).astype(np.uint8)
    pose_np = np.array(pose)
    pose_mask = pose_np.max(axis=2) > 8

    person_only = np.zeros_like(image_np)
    person_only[mask_np > 0] = image_np[mask_np > 0]

    if pose_mask.any():
        top_y = max(0, int(np.where(pose_mask)[0].min()) - head_margin)
        person_only[:top_y, :, :] = 0

    gray = cv2.cvtColor(person_only, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, low_threshold, high_threshold)
    edge_rgb = np.repeat(edges[:, :, None], 3, axis=2)
    Image.fromarray(edge_rgb).save(output_path)


def process_one(
    image_path: Path,
    mask_path: Path,
    pose_path: Path,
    conditioning_path: Path,
    low_threshold: int,
    high_threshold: int,
    head_margin: int,
) -> str:
    if conditioning_path.exists():
        return "skipped"
    build_headless_person_canny(
        image_path=image_path,
        mask_path=mask_path,
        pose_path=pose_path,
        output_path=conditioning_path,
        low_threshold=low_threshold,
        high_threshold=high_threshold,
        head_margin=head_margin,
    )
    return "written"


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    conditioning_dir = args.output_root / "conditioning_canny"
    conditioning_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = args.output_root / "metadata_all.jsonl"
    safe_tag = args.dataset_tag.replace("/", "__")

    records: list[dict] = []
    jobs = []
    for image_path in list_source_paths(args.image_dir):
        stem = image_path.stem
        mask_path = resolve_file(stem, args.mask_dir)
        pose_path = resolve_file(stem, args.pose_dir)
        if mask_path is None:
            raise FileNotFoundError(f"Missing mask for {stem} in {args.mask_dir}")
        if pose_path is None:
            raise FileNotFoundError(f"Missing pose for {stem} in {args.pose_dir}")
        conditioning_path = conditioning_dir / f"{stem}_headless_person_canny.png"
        jobs.append((image_path, mask_path, pose_path, conditioning_path))
        records.append(
            {
                "target_image": str(image_path.resolve()),
                "source_image": str(image_path.resolve()),
                "mask_image": str(mask_path.resolve()),
                "conditioning_image": str(conditioning_path.resolve()),
                "conditioning_image_2": str(pose_path.resolve()),
                "text": args.prompt,
                "output_name": f"{safe_tag}__{stem}",
                "output_subdir": args.dataset_tag,
            }
        )

    written = 0
    skipped = 0
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                process_one,
                image_path,
                mask_path,
                pose_path,
                conditioning_path,
                args.low_threshold,
                args.high_threshold,
                args.head_margin,
            )
            for image_path, mask_path, pose_path, conditioning_path in jobs
        ]
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"canny:{args.dataset_tag}"):
            result = future.result()
            if result == "written":
                written += 1
            else:
                skipped += 1

    with metadata_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                "dataset_tag": args.dataset_tag,
                "metadata_path": str(metadata_path.resolve()),
                "conditioning_dir": str(conditioning_dir.resolve()),
                "count": len(records),
                "conditioning_written": written,
                "conditioning_skipped": skipped,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
