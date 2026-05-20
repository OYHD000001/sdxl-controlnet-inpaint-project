from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset
from PIL import Image

from src.data.transforms import build_train_transforms
from src.utils.io import load_jsonl, load_pil_image
from src.utils.masks import (
    apply_binary_mask_to_image,
    ensure_mask_is_single_channel,
    invert_binary_mask,
    invert_conditioning_image,
)
from src.utils.prompts import maybe_drop_prompt


@dataclass
class SamplePaths:
    target_image: str
    source_image: str
    mask_image: str
    conditioning_image: str
    conditioning_image_2: str | None
    text: str
    masked_source_image: str | None = None
    target_latents: str | None = None
    masked_source_latents: str | None = None
    prompt_embeds: str | None = None
    pooled_prompt_embeds: str | None = None


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
        base_mode: str = "inpaint",
        center_crop: bool = False,
        random_flip: bool = False,
        prompt_dropout: float = 0.0,
        invert_mask: bool = False,
        conditioning_resize_modes: list[str] | None = None,
    ) -> None:
        self.metadata_path = Path(metadata_path)
        self.records = load_jsonl(self.metadata_path)
        self.transforms = build_train_transforms(
            image_height=image_height,
            image_width=image_width,
            center_crop=center_crop,
            random_flip=random_flip,
            conditioning_resize_modes=conditioning_resize_modes,
        )
        self.base_mode = base_mode
        self.prompt_dropout = prompt_dropout
        self.invert_mask = invert_mask

    def __len__(self) -> int:
        return len(self.records)

    def _parse_record(self, record: dict[str, Any]) -> SamplePaths:
        return SamplePaths(
            target_image=record["target_image"],
            source_image=record["source_image"],
            mask_image=record["mask_image"],
            conditioning_image=record["conditioning_image"],
            conditioning_image_2=record.get("conditioning_image_2"),
            text=record["text"],
            masked_source_image=record.get("masked_source_image"),
            target_latents=record.get("target_latents"),
            masked_source_latents=record.get("masked_source_latents"),
            prompt_embeds=record.get("prompt_embeds"),
            pooled_prompt_embeds=record.get("pooled_prompt_embeds"),
        )

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self._parse_record(self.records[index])

        mask_image = ensure_mask_is_single_channel(load_pil_image(sample.mask_image))
        conditioning_image = load_pil_image(sample.conditioning_image).convert("RGB")
        conditioning_image_2 = None
        if sample.conditioning_image_2:
            conditioning_image_2 = load_pil_image(sample.conditioning_image_2).convert("RGB")

        has_cached_prompt = bool(sample.prompt_embeds and sample.pooled_prompt_embeds)
        has_cached_target_latents = bool(sample.target_latents)
        has_cached_masked_source_latents = bool(sample.masked_source_latents)
        use_cached_t2i_latents = self.base_mode == "t2i" and has_cached_target_latents and has_cached_prompt
        use_cached_inpaint_latents = (
            self.base_mode == "inpaint"
            and has_cached_target_latents
            and has_cached_masked_source_latents
            and has_cached_prompt
        )

        if use_cached_t2i_latents or use_cached_inpaint_latents:
            placeholder_size = conditioning_image.size
            target_image = Image.new("RGB", placeholder_size, color=0)
            source_image = Image.new("RGB", placeholder_size, color=0)
        else:
            target_image = load_pil_image(sample.target_image).convert("RGB")
            source_image = load_pil_image(sample.source_image).convert("RGB")

        if self.invert_mask and self.base_mode == "inpaint":
            mask_image = invert_binary_mask(mask_image)
            if Path(sample.conditioning_image) == Path(sample.mask_image):
                conditioning_image = invert_conditioning_image(conditioning_image)

        masked_source_image = None
        if self.base_mode == "inpaint":
            if use_cached_inpaint_latents:
                masked_source_image = None
            elif sample.masked_source_image:
                masked_source_image = load_pil_image(sample.masked_source_image).convert("RGB")
            else:
                masked_source_image = apply_binary_mask_to_image(source_image, mask_image)

        transformed = self.transforms(
            target_image=target_image,
            source_image=source_image,
            mask_image=mask_image,
            masked_source_image=masked_source_image,
            conditioning_image=conditioning_image,
            conditioning_image_2=conditioning_image_2,
        )

        text = maybe_drop_prompt(sample.text, self.prompt_dropout)

        batch = {
            "target_image": transformed["target_image"],
            "source_image": transformed["source_image"],
            "mask_image": transformed["mask_image"],
            "conditioning_image": transformed["conditioning_image"],
            "conditioning_image_2": transformed.get("conditioning_image_2"),
            "text": text,
            "metadata": self.records[index],
        }
        if "masked_source_image" in transformed:
            batch["masked_source_image"] = transformed["masked_source_image"]
        if sample.target_latents and sample.prompt_embeds and sample.pooled_prompt_embeds:
            batch["target_latents"] = torch.load(sample.target_latents, map_location="cpu", weights_only=True)
            batch["prompt_embeds"] = torch.load(sample.prompt_embeds, map_location="cpu", weights_only=True)
            batch["pooled_prompt_embeds"] = torch.load(sample.pooled_prompt_embeds, map_location="cpu", weights_only=True)
            if sample.masked_source_latents:
                batch["masked_source_latents"] = torch.load(sample.masked_source_latents, map_location="cpu", weights_only=True)
        return batch


def collate_fn(samples: list[dict[str, Any]]) -> dict[str, Any]:
    batch = {
        "target_image": torch.stack([sample["target_image"] for sample in samples], dim=0),
        "source_image": torch.stack([sample["source_image"] for sample in samples], dim=0),
        "mask_image": torch.stack([sample["mask_image"] for sample in samples], dim=0),
        "conditioning_image": torch.stack([sample["conditioning_image"] for sample in samples], dim=0),
        "text": [sample["text"] for sample in samples],
        "metadata": [sample["metadata"] for sample in samples],
    }
    if all(sample.get("masked_source_image") is not None for sample in samples):
        batch["masked_source_image"] = torch.stack([sample["masked_source_image"] for sample in samples], dim=0)
    if all(sample.get("conditioning_image_2") is not None for sample in samples):
        batch["conditioning_image_2"] = torch.stack([sample["conditioning_image_2"] for sample in samples], dim=0)
    if all(sample.get("target_latents") is not None for sample in samples):
        batch["target_latents"] = torch.stack([sample["target_latents"] for sample in samples], dim=0)
        batch["prompt_embeds"] = torch.stack([sample["prompt_embeds"] for sample in samples], dim=0)
        batch["pooled_prompt_embeds"] = torch.stack([sample["pooled_prompt_embeds"] for sample in samples], dim=0)
    if all(sample.get("masked_source_latents") is not None for sample in samples):
        batch["masked_source_latents"] = torch.stack([sample["masked_source_latents"] for sample in samples], dim=0)
    return batch
