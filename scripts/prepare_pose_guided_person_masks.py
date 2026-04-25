#!/usr/bin/env python3

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm.auto import tqdm


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create person silhouette masks from images and rendered pose maps.")
    parser.add_argument("--image-dir", required=True, type=Path)
    parser.add_argument("--pose-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--border-width", type=int, default=8)
    parser.add_argument("--pose-dilate", type=int, default=18)
    parser.add_argument("--rect-pad", type=float, default=0.18)
    parser.add_argument("--grabcut-iters", type=int, default=5)
    return parser.parse_args()


def list_source_paths(image_dir: Path) -> list[Path]:
    return sorted(path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def resolve_pose_path(stem: str, pose_dir: Path) -> Path | None:
    for suffix in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
        candidate = pose_dir / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def largest_component(binary: np.ndarray) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), connectivity=8)
    if count <= 1:
        return binary
    areas = stats[1:, cv2.CC_STAT_AREA]
    best = 1 + int(np.argmax(areas))
    return (labels == best).astype(np.uint8)


def build_pose_seed(pose_rgb: np.ndarray, dilate: int) -> tuple[np.ndarray, tuple[int, int, int, int] | None]:
    pose_mask = (pose_rgb.max(axis=2) > 8).astype(np.uint8)
    if pose_mask.sum() == 0:
        return pose_mask, None
    kernel_size = max(3, dilate | 1)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    pose_fg = cv2.dilate(pose_mask, kernel, iterations=1)
    ys, xs = np.where(pose_mask > 0)
    return pose_fg, (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))


def build_grabcut_mask(
    image_rgb: np.ndarray,
    pose_fg: np.ndarray,
    pose_bbox: tuple[int, int, int, int] | None,
    border_width: int,
    rect_pad: float,
    grabcut_iters: int,
) -> np.ndarray:
    height, width = image_rgb.shape[:2]
    mask = np.full((height, width), cv2.GC_PR_BGD, dtype=np.uint8)

    border = np.zeros((height, width), dtype=bool)
    border[:border_width, :] = True
    border[-border_width:, :] = True
    border[:, :border_width] = True
    border[:, -border_width:] = True

    border_pixels = image_rgb[border]
    bg_color = np.median(border_pixels, axis=0)
    bg_dist = np.linalg.norm(image_rgb.astype(np.float32) - bg_color.astype(np.float32), axis=2)
    mask[bg_dist < 24.0] = cv2.GC_BGD
    mask[border] = cv2.GC_BGD

    if pose_bbox is not None:
        x0, y0, x1, y1 = pose_bbox
        pad_x = int((x1 - x0 + 1) * rect_pad)
        pad_y = int((y1 - y0 + 1) * rect_pad)
        rx0 = max(0, x0 - pad_x)
        ry0 = max(0, y0 - pad_y)
        rx1 = min(width - 1, x1 + pad_x)
        ry1 = min(height - 1, y1 + pad_y)
        mask[ry0 : ry1 + 1, rx0 : rx1 + 1] = cv2.GC_PR_FGD
    else:
        rx0, ry0, rx1, ry1 = int(width * 0.2), int(height * 0.05), int(width * 0.8), int(height * 0.98)
        mask[ry0 : ry1 + 1, rx0 : rx1 + 1] = cv2.GC_PR_FGD

    if pose_fg.sum() > 0:
        mask[pose_fg > 0] = cv2.GC_FGD

    def build_fallback() -> np.ndarray:
        fallback = np.zeros((height, width), dtype=np.uint8)
        if pose_fg.sum() > 0:
            fallback = pose_fg.copy()
            fallback = cv2.dilate(fallback, np.ones((11, 11), np.uint8), iterations=1)
        elif pose_bbox is not None:
            x0, y0, x1, y1 = pose_bbox
            fallback[y0 : y1 + 1, x0 : x1 + 1] = 1
        else:
            fallback[bg_dist >= 24.0] = 1
        return fallback

    bg_model = np.zeros((1, 65), np.float64)
    fg_model = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(image_rgb, mask, None, bg_model, fg_model, grabcut_iters, cv2.GC_INIT_WITH_MASK)
        person = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0).astype(np.uint8)
    except cv2.error:
        person = build_fallback()

    if person.sum() == 0 and pose_fg.sum() > 0:
        person = pose_fg.copy()
    elif person.sum() == 0:
        person = build_fallback()

    person = largest_component(person)
    close_kernel = np.ones((7, 7), np.uint8)
    person = cv2.morphologyEx(person, cv2.MORPH_CLOSE, close_kernel)
    person = cv2.morphologyEx(person, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return person


def process_one(
    image_path: Path,
    pose_path: Path,
    output_path: Path,
    border_width: int,
    pose_dilate: int,
    rect_pad: float,
    grabcut_iters: int,
) -> str:
    if output_path.exists():
        return "skipped"

    image_rgb = np.array(Image.open(image_path).convert("RGB"))
    pose_rgb = np.array(Image.open(pose_path).convert("RGB").resize((image_rgb.shape[1], image_rgb.shape[0]), Image.NEAREST))
    pose_fg, pose_bbox = build_pose_seed(pose_rgb, dilate=pose_dilate)
    person = build_grabcut_mask(
        image_rgb=image_rgb,
        pose_fg=pose_fg,
        pose_bbox=pose_bbox,
        border_width=border_width,
        rect_pad=rect_pad,
        grabcut_iters=grabcut_iters,
    )
    Image.fromarray((person * 255).astype(np.uint8)).save(output_path)
    return "written"


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    jobs: list[tuple[Path, Path, Path]] = []
    missing_pose: list[str] = []
    for image_path in list_source_paths(args.image_dir):
        pose_path = resolve_pose_path(image_path.stem, args.pose_dir)
        if pose_path is None:
            missing_pose.append(image_path.stem)
            continue
        jobs.append((image_path, pose_path, args.output_dir / f"{image_path.stem}.png"))

    if missing_pose:
        preview = ", ".join(missing_pose[:20])
        raise FileNotFoundError(f"Missing pose files for {len(missing_pose)} images. First few: {preview}")

    written = 0
    skipped = 0
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                process_one,
                image_path,
                pose_path,
                output_path,
                args.border_width,
                args.pose_dilate,
                args.rect_pad,
                args.grabcut_iters,
            )
            for image_path, pose_path, output_path in jobs
        ]
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"mask:{args.image_dir.name}"):
            result = future.result()
            if result == "written":
                written += 1
            else:
                skipped += 1

    print({"image_dir": str(args.image_dir), "output_dir": str(args.output_dir), "written": written, "skipped": skipped, "total": len(jobs)})


if __name__ == "__main__":
    main()
