#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build person-minus-clothes edit masks from binary person and clothes masks.")
    parser.add_argument("--person-mask-dir", required=True, type=Path)
    parser.add_argument("--clothes-mask-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--samples", nargs="*", default=None)
    parser.add_argument("--person-suffix", default=".png")
    parser.add_argument("--clothes-suffix", default="_00001_.png")
    parser.add_argument("--dilate", type=int, default=0)
    return parser.parse_args()


def load_mask(path: Path, size: tuple[int, int] | None = None) -> np.ndarray:
    mask = Image.open(path).convert("L")
    if size is not None and mask.size != size:
        mask = mask.resize(size, Image.NEAREST)
    return (np.array(mask) > 127).astype(np.uint8)


def maybe_dilate(mask: np.ndarray, iterations: int) -> np.ndarray:
    if iterations <= 0:
        return mask

    current = mask.copy()
    for _ in range(iterations):
        padded = np.pad(current, ((1, 1), (1, 1)), mode="constant")
        neighbors = []
        for dy in range(3):
            for dx in range(3):
                neighbors.append(padded[dy : dy + current.shape[0], dx : dx + current.shape[1]])
        current = np.maximum.reduce(neighbors)
    return current


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.samples:
        stems = args.samples
    else:
        stems = sorted(path.stem.replace(args.person_suffix.replace(".", ""), "") for path in args.person_mask_dir.glob(f"*{args.person_suffix}"))

    written = 0
    for stem in stems:
        person_path = args.person_mask_dir / f"{stem}{args.person_suffix}"
        clothes_path = args.clothes_mask_dir / f"{stem}{args.clothes_suffix}"
        if not person_path.exists():
            raise FileNotFoundError(f"Missing person mask: {person_path}")
        if not clothes_path.exists():
            raise FileNotFoundError(f"Missing clothes mask: {clothes_path}")

        person_image = Image.open(person_path).convert("L")
        person_mask = (np.array(person_image) > 127).astype(np.uint8)
        clothes_mask = load_mask(clothes_path, size=person_image.size)
        clothes_mask = maybe_dilate(clothes_mask, args.dilate)

        edit_mask = np.clip(person_mask - clothes_mask, 0, 1).astype(np.uint8) * 255
        Image.fromarray(edit_mask, mode="L").save(args.output_dir / f"{stem}.png")
        written += 1

    print({"written": written, "output_dir": str(args.output_dir)})


if __name__ == "__main__":
    main()
