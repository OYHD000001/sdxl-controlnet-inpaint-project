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
    parser = argparse.ArgumentParser(description="Inference with frozen ControlNets + UNet LoRA.")
    parser.add_argument("--config", required=True, type=Path)
    return parser.parse_args()


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


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    model_cfg = config["model"]
    infer_cfg = config["inference"]
    data_cfg = config["data"]
    invert_mask = bool(infer_cfg.get("invert_mask", data_cfg.get("invert_mask", False)))

    checkpoint_dir = Path(infer_cfg["checkpoint_dir"])
    lora_dir = checkpoint_dir / "unet_lora"
    if not lora_dir.exists():
        raise FileNotFoundError(f"LoRA adapter not found: {lora_dir}")

    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    controlnets = [ControlNetModel.from_pretrained(path, torch_dtype=torch_dtype) for path in model_cfg["controlnet_model_name_or_paths"]]
    controlnet = controlnets[0] if len(controlnets) == 1 else controlnets
    pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
        model_cfg["pretrained_model_name_or_path"],
        controlnet=controlnet,
        variant=model_cfg.get("variant"),
        torch_dtype=torch_dtype,
    )
    pipe.unet.load_lora_adapter(
        lora_dir,
        adapter_name="default",
        prefix=None,
        weight_name="pytorch_lora_weights.safetensors",
    )
    if hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()
    if infer_cfg.get("enable_model_cpu_offload", False) and torch.cuda.is_available():
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda" if torch.cuda.is_available() else "cpu")

    image_height = int(data_cfg.get("image_height", data_cfg["image_size"]))
    image_width = int(data_cfg.get("image_width", data_cfg["image_size"]))
    metadata_path = Path(infer_cfg["metadata_path"])
    output_dir = ensure_dir(infer_cfg["output_dir"])
    records = load_jsonl(metadata_path)

    for idx, record in enumerate(tqdm(records, desc="infer"), start=1):
        stem = record.get("output_name") or Path(record["source_image"]).stem
        subdir = record.get("output_subdir")
        save_dir = output_dir
        if subdir:
            save_dir = ensure_dir(output_dir / subdir)
        save_path = save_dir / f"{stem}_generated.png"
        if save_path.exists():
            continue

        source_image = load_pil_image(record["source_image"]).convert("RGB")
        mask_image = load_pil_image(record["mask_image"]).convert("L")
        conditioning_image = load_pil_image(record["conditioning_image"]).convert("RGB")
        conditioning_image_2 = load_pil_image(record["conditioning_image_2"]).convert("RGB") if record.get("conditioning_image_2") else None

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
            conditioning_resize_modes=data_cfg.get("conditioning_resize_modes"),
        )
        prompt = infer_cfg.get("prompt") or record["text"]
        control_image = conditioning_image if conditioning_image_2 is None else [conditioning_image, conditioning_image_2]
        with torch.inference_mode():
            result = pipe(
                prompt=prompt,
                image=source_image,
                mask_image=mask_image,
                control_image=control_image,
                negative_prompt=infer_cfg.get("negative_prompt"),
                num_inference_steps=int(infer_cfg["num_inference_steps"]),
                guidance_scale=float(infer_cfg["guidance_scale"]),
                controlnet_conditioning_scale=infer_cfg.get("controlnet_conditioning_scale", 1.0),
                strength=float(infer_cfg["strength"]),
            )
        image = result.images[0]
        if image.size != original_size:
            image = image.resize(original_size, Image.BILINEAR)
        image.save(save_path)
        if idx % 10 == 0 or idx == len(records):
            print(f"[{idx}/{len(records)}] generated")


if __name__ == "__main__":
    main()
