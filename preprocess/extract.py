from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageEnhance, ImageFilter

from src.utils.io import load_pil_image
from src.utils.masks import apply_binary_mask_to_image, ensure_mask_is_single_channel, invert_binary_mask


def build_inpaint_mask_from_clothes_mask(clothes_mask: Image.Image) -> Image.Image:
    """Convert clothes mask to inpaint mask: clothes=0, redraw-region=1."""
    return invert_binary_mask(clothes_mask)


def build_masked_clothes_image(
    source_image: Image.Image,
    inpaint_mask: Image.Image,
    fill_value: int = 0,
) -> Image.Image:
    """Keep only clothes pixels using the already-inverted inpaint mask."""
    return apply_binary_mask_to_image(source_image, inpaint_mask, fill_value=fill_value)


def resize_pil(image: Image.Image, width: int, height: int, *, is_mask: bool = False, mode: str = "bilinear") -> Image.Image:
    interpolation = Image.NEAREST if is_mask or mode.lower() == "nearest" else Image.BILINEAR
    return image.resize((width, height), interpolation)


def to_image_tensor(image: Image.Image) -> torch.Tensor:
    tensor = TF.to_tensor(image.convert("RGB"))
    return tensor * 2.0 - 1.0


def to_conditioning_tensor(image: Image.Image) -> torch.Tensor:
    return TF.to_tensor(image.convert("RGB"))


def to_mask_tensor(mask: Image.Image) -> torch.Tensor:
    tensor = TF.to_tensor(ensure_mask_is_single_channel(mask))
    if tensor.shape[0] > 1:
        tensor = tensor[:1]
    return (tensor > 0.5).float()


def _maybe_affine(
    image: Image.Image,
    angle: float,
    translate_xy: tuple[int, int],
    scale: float,
    *,
    is_mask: bool,
) -> Image.Image:
    interpolation = Image.NEAREST if is_mask else Image.BILINEAR
    fill = 0
    return TF.affine(
        image,
        angle=angle,
        translate=list(translate_xy),
        scale=scale,
        shear=[0.0, 0.0],
        interpolation=interpolation,
        fill=fill,
    )


def maybe_augment_condition_inputs(
    clothes_image: Image.Image,
    clothes_mask: Image.Image,
    pose_image: Image.Image | None,
    augment_cfg: dict[str, Any] | None,
) -> tuple[Image.Image, Image.Image, Image.Image | None]:
    """
    Lightweight domain-gap augmentation applied only to condition-side inputs.

    This intentionally leaves x0 / target image untouched.
    """
    if not augment_cfg or not augment_cfg.get("enabled", False):
        return clothes_image, clothes_mask, pose_image

    clothes_image = clothes_image.convert("RGB")
    clothes_mask = ensure_mask_is_single_channel(clothes_mask)
    pose_image = pose_image.convert("RGB") if pose_image is not None else None

    max_translate = int(augment_cfg.get("max_translate_px", 0))
    max_angle = float(augment_cfg.get("max_rotate_deg", 0.0))
    min_scale = float(augment_cfg.get("min_scale", 1.0))
    max_scale = float(augment_cfg.get("max_scale", 1.0))
    if max_translate > 0 or max_angle > 0 or min_scale != 1.0 or max_scale != 1.0:
        angle = random.uniform(-max_angle, max_angle) if max_angle > 0 else 0.0
        translate_xy = (
            random.randint(-max_translate, max_translate) if max_translate > 0 else 0,
            random.randint(-max_translate, max_translate) if max_translate > 0 else 0,
        )
        scale = random.uniform(min_scale, max_scale) if min_scale != 1.0 or max_scale != 1.0 else 1.0
        clothes_image = _maybe_affine(clothes_image, angle, translate_xy, scale, is_mask=False)
        clothes_mask = _maybe_affine(clothes_mask, angle, translate_xy, scale, is_mask=True)

    jitter_strength = float(augment_cfg.get("color_jitter_strength", 0.0))
    if jitter_strength > 0:
        brightness = 1.0 + random.uniform(-jitter_strength, jitter_strength)
        contrast = 1.0 + random.uniform(-jitter_strength, jitter_strength)
        saturation = 1.0 + random.uniform(-jitter_strength, jitter_strength)
        clothes_image = ImageEnhance.Brightness(clothes_image).enhance(brightness)
        clothes_image = ImageEnhance.Contrast(clothes_image).enhance(contrast)
        clothes_image = ImageEnhance.Color(clothes_image).enhance(saturation)

    morphology_px = int(augment_cfg.get("mask_morphology_px", 0))
    if morphology_px > 0:
        kernel = morphology_px * 2 + 1
        choice = random.choice(["dilate", "erode", "none"])
        if choice == "dilate":
            clothes_mask = clothes_mask.filter(ImageFilter.MaxFilter(kernel))
        elif choice == "erode":
            clothes_mask = clothes_mask.filter(ImageFilter.MinFilter(kernel))

    pose_translate = int(augment_cfg.get("pose_max_translate_px", 0))
    pose_angle = float(augment_cfg.get("pose_max_rotate_deg", 0.0))
    if pose_image is not None and (pose_translate > 0 or pose_angle > 0):
        pose_image = _maybe_affine(
            pose_image,
            random.uniform(-pose_angle, pose_angle) if pose_angle > 0 else 0.0,
            (
                random.randint(-pose_translate, pose_translate) if pose_translate > 0 else 0,
                random.randint(-pose_translate, pose_translate) if pose_translate > 0 else 0,
            ),
            1.0,
            is_mask=False,
        )

    return clothes_image, clothes_mask, pose_image


def prepare_pil_views(
    *,
    target_image: Image.Image,
    source_image: Image.Image,
    clothes_mask: Image.Image,
    pose_image: Image.Image | None,
    image_height: int,
    image_width: int,
    source_background_value: int = 0,
    conditioning_resize_mode: str = "nearest",
    apply_condition_augs: bool = False,
    condition_augmentation_cfg: dict[str, Any] | None = None,
) -> dict[str, Image.Image]:
    """
    Shared canonical preprocessing used by both training and inference.

    Returns:
    - target_image: clean mannequin x0
    - source_image: resized source image
    - mask_image: resized inpaint mask (clothes kept, body/background redrawn)
    - masked_source_image: clothes-only image used for the 4 inpaint conditioning channels
    - conditioning_image: resized pose image
    """
    clothes_mask = ensure_mask_is_single_channel(clothes_mask)
    inpaint_mask = build_inpaint_mask_from_clothes_mask(clothes_mask)
    clothes_only = build_masked_clothes_image(source_image, inpaint_mask, fill_value=source_background_value)

    if apply_condition_augs:
        clothes_only, clothes_mask, pose_image = maybe_augment_condition_inputs(
            clothes_only,
            clothes_mask,
            pose_image,
            condition_augmentation_cfg,
        )
        inpaint_mask = build_inpaint_mask_from_clothes_mask(clothes_mask)

    resized_target = resize_pil(target_image.convert("RGB"), image_width, image_height, is_mask=False)
    resized_source = resize_pil(source_image.convert("RGB"), image_width, image_height, is_mask=False)
    resized_mask = resize_pil(inpaint_mask, image_width, image_height, is_mask=True)
    resized_masked_source = resize_pil(clothes_only.convert("RGB"), image_width, image_height, is_mask=False)
    resized_pose = None
    if pose_image is not None:
        resized_pose = resize_pil(
            pose_image.convert("RGB"),
            image_width,
            image_height,
            is_mask=False,
            mode=conditioning_resize_mode,
        )

    return {
        "target_image": resized_target,
        "source_image": resized_source,
        "mask_image": resized_mask,
        "masked_source_image": resized_masked_source,
        "conditioning_image": resized_pose,
    }


def prepare_training_tensors(
    *,
    target_image: Image.Image,
    source_image: Image.Image,
    clothes_mask: Image.Image,
    pose_image: Image.Image | None,
    image_height: int,
    image_width: int,
    source_background_value: int = 0,
    conditioning_resize_mode: str = "nearest",
    condition_augmentation_cfg: dict[str, Any] | None = None,
) -> dict[str, torch.Tensor]:
    views = prepare_pil_views(
        target_image=target_image,
        source_image=source_image,
        clothes_mask=clothes_mask,
        pose_image=pose_image,
        image_height=image_height,
        image_width=image_width,
        source_background_value=source_background_value,
        conditioning_resize_mode=conditioning_resize_mode,
        apply_condition_augs=bool(condition_augmentation_cfg and condition_augmentation_cfg.get("enabled", False)),
        condition_augmentation_cfg=condition_augmentation_cfg,
    )
    payload = {
        "target_image": to_image_tensor(views["target_image"]),
        "source_image": to_image_tensor(views["source_image"]),
        "mask_image": to_mask_tensor(views["mask_image"]),
        "masked_source_image": to_image_tensor(views["masked_source_image"]),
    }
    if views["conditioning_image"] is not None:
        payload["conditioning_image"] = to_conditioning_tensor(views["conditioning_image"])
    return payload


def prepare_inference_inputs(
    *,
    source_image: Image.Image,
    clothes_mask: Image.Image,
    pose_image: Image.Image | None,
    image_height: int,
    image_width: int,
    source_background_value: int = 0,
    conditioning_resize_mode: str = "nearest",
) -> dict[str, Image.Image]:
    blank_target = Image.new("RGB", source_image.size, (0, 0, 0))
    return prepare_pil_views(
        target_image=blank_target,
        source_image=source_image,
        clothes_mask=clothes_mask,
        pose_image=pose_image,
        image_height=image_height,
        image_width=image_width,
        source_background_value=source_background_value,
        conditioning_resize_mode=conditioning_resize_mode,
        apply_condition_augs=False,
        condition_augmentation_cfg=None,
    )


def assert_pipeline_consistency(
    record: dict[str, Any],
    *,
    image_height: int,
    image_width: int,
    source_background_value: int = 0,
    conditioning_resize_mode: str = "nearest",
) -> None:
    source_image = load_pil_image(record["source_image"]).convert("RGB")
    target_image = load_pil_image(record["target_image"]).convert("RGB")
    clothes_mask = load_pil_image(record["mask_image"]).convert("L")
    pose_image = load_pil_image(record["conditioning_image"]).convert("RGB")

    train_payload = prepare_training_tensors(
        target_image=target_image,
        source_image=source_image,
        clothes_mask=clothes_mask,
        pose_image=pose_image,
        image_height=image_height,
        image_width=image_width,
        source_background_value=source_background_value,
        conditioning_resize_mode=conditioning_resize_mode,
        condition_augmentation_cfg=None,
    )
    infer_payload = prepare_inference_inputs(
        source_image=source_image,
        clothes_mask=clothes_mask,
        pose_image=pose_image,
        image_height=image_height,
        image_width=image_width,
        source_background_value=source_background_value,
        conditioning_resize_mode=conditioning_resize_mode,
    )
    infer_mask = to_mask_tensor(infer_payload["mask_image"])
    infer_masked_source = to_image_tensor(infer_payload["masked_source_image"])
    infer_pose = to_conditioning_tensor(infer_payload["conditioning_image"])

    if not torch.equal(train_payload["mask_image"], infer_mask):
        raise AssertionError("mask preprocessing mismatch between training and inference paths")
    if not torch.equal(train_payload["masked_source_image"], infer_masked_source):
        raise AssertionError("masked-image preprocessing mismatch between training and inference paths")
    if not torch.equal(train_payload["conditioning_image"], infer_pose):
        raise AssertionError("pose preprocessing mismatch between training and inference paths")


def build_metadata_record(
    *,
    target_image: str | Path,
    source_image: str | Path,
    mask_image: str | Path,
    pose_image: str | Path,
    prompt: str,
    output_name: str | None = None,
    output_subdir: str | None = None,
) -> dict[str, Any]:
    return {
        "target_image": str(Path(target_image)),
        "source_image": str(Path(source_image)),
        "mask_image": str(Path(mask_image)),
        "conditioning_image": str(Path(pose_image)),
        "text": prompt,
        "output_name": output_name or Path(source_image).stem,
        "output_subdir": output_subdir,
    }
