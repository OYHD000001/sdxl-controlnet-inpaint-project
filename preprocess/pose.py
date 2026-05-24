from __future__ import annotations

from pathlib import Path

from PIL import Image

from src.utils.io import load_pil_image


def load_pose_image(path: str | Path) -> Image.Image:
    return load_pil_image(path).convert("RGB")


def extract_pose_from_image(*args, **kwargs) -> Image.Image:
    raise NotImplementedError(
        "Pose extraction is project-specific. Plug your OpenPose / DWPose / MMPose extractor here, "
        "then keep training and inference on the same exported pose image format."
    )
