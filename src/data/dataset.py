from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from src.data.transforms import build_train_transforms
from src.utils.io import load_jsonl, load_pil_image
from src.utils.masks import apply_binary_mask_to_image, ensure_mask_is_single_channel
from src.utils.prompts import maybe_drop_prompt


@dataclass
class SamplePaths:
    target_image: str
    source_image: str
    mask_image: str
    conditioning_image: str
    text: str
    masked_source_image: str | None = None


class SDXLControlNetInpaintDataset(Dataset):
    """
    Minimal local JSONL dataset for SDXL + ControlNet + inpaint training.

    Expected fields per record:
    - target_image
    - source_image
    - mask_image
    - conditioning_image
    - text
    - masked_source_image (optional)

    TODO:
    - add dataset-level validation for real project data
    - support multiple metadata schemas if your dataset evolves
    - add optional prompt templates / overrides for mannequin editing
    """

    def __init__(
        self,
        metadata_path: str | Path,
        image_height: int,
        image_width: int,
        center_crop: bool = False,
        random_flip: bool = False,
        prompt_dropout: float = 0.0,
    ) -> None:
        self.metadata_path = Path(metadata_path)
        self.records = load_jsonl(self.metadata_path)
        self.transforms = build_train_transforms(
            image_height=image_height,
            image_width=image_width,
            center_crop=center_crop,
            random_flip=random_flip,
        )
        self.prompt_dropout = prompt_dropout

    def __len__(self) -> int:
        return len(self.records)

    def _parse_record(self, record: dict[str, Any]) -> SamplePaths:
        return SamplePaths(
            target_image=record["target_image"],
            source_image=record["source_image"],
            mask_image=record["mask_image"],
            conditioning_image=record["conditioning_image"],
            text=record["text"],
            masked_source_image=record.get("masked_source_image"),
        )

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self._parse_record(self.records[index])

        target_image = load_pil_image(sample.target_image).convert("RGB")
        source_image = load_pil_image(sample.source_image).convert("RGB")
        mask_image = ensure_mask_is_single_channel(load_pil_image(sample.mask_image))
        conditioning_image = load_pil_image(sample.conditioning_image).convert("RGB")

        if sample.masked_source_image:
            masked_source_image = load_pil_image(sample.masked_source_image).convert("RGB")
        else:
            masked_source_image = apply_binary_mask_to_image(source_image, mask_image)

        transformed = self.transforms(
            target_image=target_image,
            source_image=source_image,
            mask_image=mask_image,
            masked_source_image=masked_source_image,
            conditioning_image=conditioning_image,
        )

        text = maybe_drop_prompt(sample.text, self.prompt_dropout)

        batch = {
            "target_image": transformed["target_image"],
            "source_image": transformed["source_image"],
            "mask_image": transformed["mask_image"],
            "masked_source_image": transformed["masked_source_image"],
            "conditioning_image": transformed["conditioning_image"],
            "text": text,
            "metadata": self.records[index],
        }
        return batch


def collate_fn(samples: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "target_image": torch.stack([sample["target_image"] for sample in samples], dim=0),
        "source_image": torch.stack([sample["source_image"] for sample in samples], dim=0),
        "mask_image": torch.stack([sample["mask_image"] for sample in samples], dim=0),
        "masked_source_image": torch.stack([sample["masked_source_image"] for sample in samples], dim=0),
        "conditioning_image": torch.stack([sample["conditioning_image"] for sample in samples], dim=0),
        "text": [sample["text"] for sample in samples],
        "metadata": [sample["metadata"] for sample in samples],
    }
