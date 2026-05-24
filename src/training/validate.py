from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from diffusers import ControlNetModel, StableDiffusionXLControlNetInpaintPipeline, StableDiffusionXLControlNetPipeline
from PIL import Image, ImageDraw
from tqdm.auto import tqdm
from preprocess.extract import assert_pipeline_consistency, prepare_inference_inputs
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


def add_label(image: Image.Image, label: str, label_height: int = 36) -> Image.Image:
    canvas = Image.new("RGB", (image.width, image.height + label_height), (255, 255, 255))
    canvas.paste(image, (0, label_height))
    ImageDraw.Draw(canvas).text((10, 10), label, fill=(0, 0, 0))
    return canvas


def resize_panel(image: Image.Image, width: int, height: int) -> Image.Image:
    return image.resize((width, height), Image.BILINEAR)


def build_pair(original: Image.Image, generated: Image.Image) -> Image.Image:
    left = add_label(original, "original_human")
    right = add_label(generated, "generated")
    canvas = Image.new("RGB", (left.width + right.width, max(left.height, right.height)), (245, 245, 245))
    canvas.paste(left, (0, 0))
    canvas.paste(right, (left.width, 0))
    return canvas


def make_original_generated_pairs_once(
    metadata_path: Path,
    generated_dir: Path,
    output_dir: Path,
    image_width: int,
    image_height: int,
) -> int:
    output_dir = ensure_dir(output_dir)
    made = 0
    for record in load_jsonl(metadata_path):
        source_path = Path(record["source_image"])
        stem = record.get("output_name") or source_path.stem
        generated_path = generated_dir
        if record.get("output_subdir"):
            generated_path = generated_path / record["output_subdir"]
        generated_path = generated_path / f"{stem}_generated.png"
        out_path = output_dir / f"{stem}_original_generated.png"
        if out_path.exists() or not generated_path.exists():
            continue
        original = resize_panel(load_pil_image(source_path).convert("RGB"), image_width, image_height)
        generated = resize_panel(load_pil_image(generated_path).convert("RGB"), image_width, image_height)
        build_pair(original, generated).save(out_path)
        made += 1
    return made


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


def _resize_conditioning_for_t2i(
    conditioning_image: Image.Image,
    conditioning_image_2: Image.Image | None,
    image_height: int,
    image_width: int,
    conditioning_resize_modes: list[str] | None = None,
) -> tuple[Image.Image, Image.Image | None]:
    conditioning_resize_modes = conditioning_resize_modes or ["bilinear", "bilinear"]
    conditioning_interpolation_0 = Image.NEAREST if conditioning_resize_modes[0].lower() == "nearest" else Image.BILINEAR
    conditioning_interpolation_1 = (
        Image.NEAREST
        if len(conditioning_resize_modes) > 1 and conditioning_resize_modes[1].lower() == "nearest"
        else Image.BILINEAR
    )
    resized_conditioning = conditioning_image.resize((image_width, image_height), conditioning_interpolation_0)
    resized_conditioning_2 = None
    if conditioning_image_2 is not None:
        resized_conditioning_2 = conditioning_image_2.resize((image_width, image_height), conditioning_interpolation_1)
    return resized_conditioning, resized_conditioning_2


def run_inference(config: dict[str, Any]) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cfg = config["model"]
    infer_cfg = config["inference"]
    base_mode = str(model_cfg.get("base_mode", "inpaint")).lower()
    invert_mask = bool(infer_cfg.get("invert_mask", config["data"].get("invert_mask", False)))
    canonical_pose_inpaint = bool(config.get("project", {}).get("canonical_pose_inpaint", False))

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
    elif (checkpoint_dir / "controlnet").exists():
        controlnet_dirs.append(checkpoint_dir / "controlnet")
    else:
        controlnet_dirs.append(checkpoint_dir)

    print(f"[infer] base_mode={base_mode} device={device} checkpoint_dir={checkpoint_dir}", flush=True)
    print(f"[infer] loading controlnets from: {[str(path) for path in controlnet_dirs]}", flush=True)
    controlnets = [ControlNetModel.from_pretrained(path, torch_dtype=torch_dtype) for path in controlnet_dirs]
    controlnet = controlnets[0] if len(controlnets) == 1 else controlnets
    print(f"[infer] loading base pipeline from: {model_cfg['pretrained_model_name_or_path']}", flush=True)
    if base_mode == "t2i":
        pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            model_cfg["pretrained_model_name_or_path"],
            controlnet=controlnet,
            variant=model_cfg.get("variant"),
            torch_dtype=torch_dtype,
        )
    else:
        pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
            model_cfg["pretrained_model_name_or_path"],
            controlnet=controlnet,
            variant=model_cfg.get("variant"),
            torch_dtype=torch_dtype,
        )
    lora_dir = checkpoint_dir / "unet_lora"
    if lora_dir.exists():
        pipe.unet.load_lora_adapter(lora_dir, adapter_name="default")
        if hasattr(pipe.unet, "set_adapters"):
            pipe.unet.set_adapters(["default"], adapter_weights=[1.0])
        elif hasattr(pipe.unet, "set_adapter"):
            pipe.unet.set_adapter("default")
        lora_params = sum(parameter.numel() for name, parameter in pipe.unet.named_parameters() if "lora_" in name)
        print(f"[infer] loaded UNet LoRA adapter from {lora_dir} (params={lora_params})", flush=True)
    print("[infer] pipeline loaded", flush=True)
    if hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()
    if infer_cfg.get("enable_model_cpu_offload", False) and torch.cuda.is_available():
        print("[infer] enabling model cpu offload", flush=True)
        pipe.enable_model_cpu_offload()
    else:
        print(f"[infer] moving pipeline to device: {device}", flush=True)
        pipe.to(device)
    pipe.set_progress_bar_config(disable=False)
    image_height = int(config["data"].get("image_height", config["data"]["image_size"]))
    image_width = int(config["data"].get("image_width", config["data"]["image_size"]))
    source_background_value = int(config["data"].get("source_background_value", 0))

    metadata_path = Path(infer_cfg["metadata_path"])
    records = load_jsonl(metadata_path)
    output_dir = ensure_dir(infer_cfg["output_dir"])
    print(f"[infer] metadata={metadata_path} records={len(records)} output_dir={output_dir}", flush=True)
    if canonical_pose_inpaint:
        strength = float(infer_cfg["strength"])
        if abs(strength - 1.0) > 1e-6:
            raise AssertionError("Canonical pose-only inpaint inference requires strength == 1.0")
        if records:
            assert_pipeline_consistency(
                records[0],
                image_height=image_height,
                image_width=image_width,
                source_background_value=source_background_value,
                conditioning_resize_mode=(config["data"].get("conditioning_resize_modes") or ["nearest"])[0],
            )
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
        conditioning_image = load_pil_image(record["conditioning_image"]).convert("RGB")
        conditioning_image_2 = None
        if record.get("conditioning_image_2"):
            conditioning_image_2 = load_pil_image(record["conditioning_image_2"]).convert("RGB")
        original_size = source_image.size
        prompt = infer_cfg.get("prompt") or record["text"]
        control_scales = infer_cfg.get("controlnet_conditioning_scale", 1.0)
        if isinstance(control_scales, list) and len(control_scales) == 1:
            control_scales = float(control_scales[0])
        control_image = None
        if base_mode == "t2i":
            conditioning_image, conditioning_image_2 = _resize_conditioning_for_t2i(
                conditioning_image=conditioning_image,
                conditioning_image_2=conditioning_image_2,
                image_height=image_height,
                image_width=image_width,
                conditioning_resize_modes=config["data"].get("conditioning_resize_modes"),
            )
            control_image = conditioning_image if conditioning_image_2 is None else [conditioning_image, conditioning_image_2]
        else:
            if canonical_pose_inpaint:
                canonical = prepare_inference_inputs(
                    source_image=source_image,
                    clothes_mask=load_pil_image(record["mask_image"]).convert("L"),
                    pose_image=conditioning_image,
                    image_height=image_height,
                    image_width=image_width,
                    source_background_value=source_background_value,
                    conditioning_resize_mode=(config["data"].get("conditioning_resize_modes") or ["nearest"])[0],
                )
                source_image = canonical["masked_source_image"]
                mask_image = canonical["mask_image"]
                conditioning_image = canonical["conditioning_image"]
            else:
                mask_image = load_pil_image(record["mask_image"]).convert("L")
                if invert_mask:
                    mask_image = invert_binary_mask(mask_image)
                    if record["conditioning_image"] == record["mask_image"]:
                        conditioning_image = invert_conditioning_image(conditioning_image)
                source_image, mask_image, conditioning_image, conditioning_image_2 = _resize_for_inference(
                    source_image=source_image,
                    mask_image=mask_image,
                    conditioning_image=conditioning_image,
                    conditioning_image_2=conditioning_image_2,
                    image_height=image_height,
                    image_width=image_width,
                    conditioning_resize_modes=config["data"].get("conditioning_resize_modes"),
                )
            control_image = conditioning_image if conditioning_image_2 is None else [conditioning_image, conditioning_image_2]
        with torch.inference_mode():
            if base_mode == "t2i":
                result = pipe(
                    prompt=prompt,
                    image=control_image,
                    negative_prompt=infer_cfg.get("negative_prompt"),
                    num_inference_steps=int(infer_cfg["num_inference_steps"]),
                    guidance_scale=float(infer_cfg["guidance_scale"]),
                    controlnet_conditioning_scale=control_scales,
                    height=image_height,
                    width=image_width,
                )
            else:
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

    if infer_cfg.get("generate_pairs", False):
        pair_output_dir = infer_cfg.get("pair_output_dir")
        if pair_output_dir:
            made = make_original_generated_pairs_once(
                metadata_path=metadata_path,
                generated_dir=output_dir,
                output_dir=Path(pair_output_dir),
                image_width=image_width,
                image_height=image_height,
            )
            print(f"generated original/generated pairs: made={made}")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.mode == "validate":
        run_validation(config)
    else:
        run_inference(config)


if __name__ == "__main__":
    main()
