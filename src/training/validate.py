from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from diffusers import ControlNetModel, StableDiffusionXLControlNetInpaintPipeline
from PIL import Image
from tqdm.auto import tqdm
from src.utils.io import ensure_dir, load_config, load_jsonl, load_pil_image
from src.utils.masks import invert_binary_mask, invert_conditioning_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validation and inference entrypoint.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--mode", default="validate", choices=["validate", "infer"])
    return parser.parse_args()


def run_validation(config: dict[str, Any]) -> None:
    """
    Lightweight validation placeholder.

    TODO:
    - build a proper SDXL ControlNet inpaint validation pipeline
    - log side-by-side grids for source / mask / condition / prediction / target
    - restore checkpoints instead of only using base pretrained weights
    """
    print("Validation skipped for scaffold training. Use infer mode after checkpoint export.")


def _resize_for_inference(
    source_image: Image.Image,
    mask_image: Image.Image,
    conditioning_image: Image.Image,
    conditioning_image_2: Image.Image | None,
    image_height: int,
    image_width: int,
    conditioning_resize_modes: list[str] | None = None,
) -> tuple[Image.Image, Image.Image, Image.Image, Image.Image | None]:
    conditioning_resize_modes = conditioning_resize_modes or ["bilinear", "bilinear"]
    conditioning_interpolation_0 = Image.NEAREST if conditioning_resize_modes[0].lower() == "nearest" else Image.BILINEAR
    conditioning_interpolation_1 = (
        Image.NEAREST
        if len(conditioning_resize_modes) > 1 and conditioning_resize_modes[1].lower() == "nearest"
        else Image.BILINEAR
    )
    resized_source = source_image.resize((image_width, image_height), Image.BILINEAR)
    resized_mask = mask_image.resize((image_width, image_height), Image.NEAREST)
    resized_conditioning = conditioning_image.resize((image_width, image_height), conditioning_interpolation_0)
    resized_conditioning_2 = None
    if conditioning_image_2 is not None:
        resized_conditioning_2 = conditioning_image_2.resize((image_width, image_height), conditioning_interpolation_1)
    return resized_source, resized_mask, resized_conditioning, resized_conditioning_2


def run_inference(config: dict[str, Any]) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cfg = config["model"]
    infer_cfg = config["inference"]
    invert_mask = bool(infer_cfg.get("invert_mask", config["data"].get("invert_mask", False)))

    checkpoint_dir = Path(infer_cfg["checkpoint_dir"])
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"ControlNet checkpoint not found: {checkpoint_dir}")

    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    controlnet_dirs = []
    if (checkpoint_dir / "controlnet_0").exists():
        index = 0
        while (checkpoint_dir / f"controlnet_{index}").exists():
            controlnet_dirs.append(checkpoint_dir / f"controlnet_{index}")
            index += 1
    else:
        controlnet_dirs.append(checkpoint_dir)

    controlnets = [ControlNetModel.from_pretrained(path, torch_dtype=torch_dtype) for path in controlnet_dirs]
    controlnet = controlnets[0] if len(controlnets) == 1 else controlnets
    pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
        model_cfg["pretrained_model_name_or_path"],
        controlnet=controlnet,
        variant=model_cfg.get("variant"),
        torch_dtype=torch_dtype,
    )
    pipe.to(device)
    pipe.set_progress_bar_config(disable=False)
    image_height = int(config["data"].get("image_height", config["data"]["image_size"]))
    image_width = int(config["data"].get("image_width", config["data"]["image_size"]))

    metadata_path = Path(infer_cfg["metadata_path"])
    records = load_jsonl(metadata_path)
    output_dir = ensure_dir(infer_cfg["output_dir"])
    for idx, record in enumerate(tqdm(records, desc="infer"), start=1):
        stem = Path(record["source_image"]).stem
        save_path = output_dir / f"{stem}_generated.png"
        if save_path.exists():
            continue

        source_image = load_pil_image(record["source_image"]).convert("RGB")
        mask_image = load_pil_image(record["mask_image"]).convert("L")
        conditioning_image = load_pil_image(record["conditioning_image"]).convert("RGB")
        conditioning_image_2 = None
        if record.get("conditioning_image_2"):
            conditioning_image_2 = load_pil_image(record["conditioning_image_2"]).convert("RGB")
        if invert_mask:
            mask_image = invert_binary_mask(mask_image)
            if record["conditioning_image"] == record["mask_image"]:
                conditioning_image = invert_conditioning_image(conditioning_image)
        original_size = source_image.size
        source_image, mask_image, conditioning_image, conditioning_image_2 = _resize_for_inference(
            source_image=source_image,
            mask_image=mask_image,
            conditioning_image=conditioning_image,
            conditioning_image_2=conditioning_image_2,
            image_height=image_height,
            image_width=image_width,
            conditioning_resize_modes=config["data"].get("conditioning_resize_modes"),
        )
        prompt = infer_cfg.get("prompt") or record["text"]
        control_image = conditioning_image if conditioning_image_2 is None else [conditioning_image, conditioning_image_2]
        control_scales = infer_cfg.get("controlnet_conditioning_scale", 1.0)
        with torch.inference_mode():
            result = pipe(
                prompt=prompt,
                image=source_image,
                mask_image=mask_image,
                control_image=control_image,
                negative_prompt=infer_cfg.get("negative_prompt"),
                num_inference_steps=int(infer_cfg["num_inference_steps"]),
                guidance_scale=float(infer_cfg["guidance_scale"]),
                controlnet_conditioning_scale=control_scales,
                strength=float(infer_cfg["strength"]),
            )
        image = result.images[0]
        if image.size != original_size:
            image = image.resize(original_size, Image.BILINEAR)
        image.save(save_path)
        if idx % 10 == 0 or idx == len(records):
            print(f"[{idx}/{len(records)}] generated")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.mode == "validate":
        run_validation(config)
    else:
        run_inference(config)


if __name__ == "__main__":
    main()
