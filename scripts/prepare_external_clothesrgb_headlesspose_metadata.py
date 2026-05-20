#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image


VALID_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare external inference metadata for clothes-RGB + headless-pose dual-control inference."
    )
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--mask-dir", required=True, type=Path)
    parser.add_argument("--cutout-dir", required=True, type=Path)
    parser.add_argument("--pose-dir", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--output-subdir", required=True, type=str)
    parser.add_argument("--prompt", required=True, type=str)
    parser.add_argument("--overwrite-conditioning", action="store_true")
    return parser.parse_args()


def collect_stems(directory: Path) -> dict[str, Path]:
    items: dict[str, Path] = {}
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in VALID_EXTS:
            items[path.stem] = path
    return items


def rgba_to_black_rgb(src_path: Path, dst_path: Path) -> None:
    if dst_path.exists():
        return

    image = Image.open(src_path)
    if image.mode == "RGBA":
        black_bg = Image.new("RGBA", image.size, (0, 0, 0, 255))
        image = Image.alpha_composite(black_bg, image).convert("RGB")
    else:
        image = image.convert("RGB")

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(dst_path)


def main() -> None:
    args = parse_args()

    source_map = collect_stems(args.source_dir)
    mask_map = collect_stems(args.mask_dir)
    cutout_map = collect_stems(args.cutout_dir)
    pose_map = collect_stems(args.pose_dir)

    common_stems = sorted(set(source_map) & set(mask_map) & set(cutout_map) & set(pose_map))

    conditioning_dir = args.output_root / "conditioning_images"
    metadata_dir = args.output_root / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = metadata_dir / "metadata_all.jsonl"

    records = []
    for stem in common_stems:
        conditioning_path = conditioning_dir / f"{stem}__clothes.png"
        if args.overwrite_conditioning and conditioning_path.exists():
            conditioning_path.unlink()
        rgba_to_black_rgb(cutout_map[stem], conditioning_path)

        records.append(
            {
                "target_image": str(source_map[stem].resolve()),
                "source_image": str(source_map[stem].resolve()),
                "mask_image": str(mask_map[stem].resolve()),
                "conditioning_image": str(conditioning_path.resolve()),
                "conditioning_image_2": str(pose_map[stem].resolve()),
                "text": args.prompt,
                "output_name": stem,
                "output_subdir": args.output_subdir,
            }
        )

    with metadata_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                "source_dir": str(args.source_dir.resolve()),
                "mask_dir": str(args.mask_dir.resolve()),
                "cutout_dir": str(args.cutout_dir.resolve()),
                "pose_dir": str(args.pose_dir.resolve()),
                "conditioning_dir": str(conditioning_dir.resolve()),
                "metadata_path": str(metadata_path.resolve()),
                "count": len(records),
                "output_subdir": args.output_subdir,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
