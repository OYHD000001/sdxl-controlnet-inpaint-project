from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from diffusers import ControlNetModel, StableDiffusionXLControlNetInpaintPipeline
from PIL import Image
from tqdm.auto import tqdm
from src.utils.io import ensure_dir, load_config, load_jsonl, load_pil_image


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
    image_height: int,
    image_width: int,
) -> tuple[Image.Image, Image.Image, Image.Image]:
    resized_source = source_image.resize((image_width, image_height), Image.BILINEAR)
    resized_mask = mask_image.resize((image_width, image_height), Image.NEAREST)
    resized_conditioning = conditioning_image.resize((image_width, image_height), Image.BILINEAR)
    return resized_source, resized_mask, resized_conditioning


def run_inference(config: dict[str, Any]) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cfg = config["model"]
    infer_cfg = config["inference"]

    checkpoint_dir = Path(infer_cfg["checkpoint_dir"])
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"ControlNet checkpoint not found: {checkpoint_dir}")

    controlnet = ControlNetModel.from_pretrained(checkpoint_dir, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
    pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
        model_cfg["pretrained_model_name_or_path"],
        controlnet=controlnet,
        variant=model_cfg.get("variant"),
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
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
        original_size = source_image.size
        source_image, mask_image, conditioning_image = _resize_for_inference(
            source_image=source_image,
            mask_image=mask_image,
            conditioning_image=conditioning_image,
            image_height=image_height,
            image_width=image_width,
        )
        prompt = infer_cfg.get("prompt") or record["text"]
        with torch.inference_mode():
            result = pipe(
                prompt=prompt,
                image=source_image,
                mask_image=mask_image,
                control_image=conditioning_image,
                negative_prompt=infer_cfg.get("negative_prompt"),
                num_inference_steps=int(infer_cfg["num_inference_steps"]),
                guidance_scale=float(infer_cfg["guidance_scale"]),
                controlnet_conditioning_scale=float(infer_cfg["controlnet_conditioning_scale"]),
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
