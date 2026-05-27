from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
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


def build_debug_grid(panels: list[tuple[str, Image.Image]], image_width: int, image_height: int) -> Image.Image:
    labeled = []
    for label, image in panels:
        labeled.append(add_label(resize_panel(image.convert("RGB"), image_width, image_height), label))
    canvas = Image.new(
        "RGB",
        (sum(panel.width for panel in labeled), max(panel.height for panel in labeled)),
        (245, 245, 245),
    )
    offset = 0
    for panel in labeled:
        canvas.paste(panel, (offset, 0))
        offset += panel.width
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


def make_inference_debug_composites_once(
    metadata_path: Path,
    generated_dir: Path,
    output_dir: Path,
    *,
    image_width: int,
    image_height: int,
    source_background_value: int,
    conditioning_resize_mode: str,
    canonical_pose_inpaint: bool,
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
        out_path = output_dir / f"{stem}_debug.png"
        if out_path.exists() or not generated_path.exists():
            continue

        source_image = load_pil_image(record["source_image"]).convert("RGB")
        target_image = load_pil_image(record["target_image"]).convert("RGB")
        clothes_mask = load_pil_image(record["mask_image"]).convert("L")
        pose_image = load_pil_image(record["conditioning_image"]).convert("RGB")
        generated_image = load_pil_image(generated_path).convert("RGB")

        if canonical_pose_inpaint:
            canonical = prepare_inference_inputs(
                source_image=source_image,
                clothes_mask=clothes_mask,
                pose_image=pose_image,
                image_height=image_height,
                image_width=image_width,
                source_background_value=source_background_value,
                conditioning_resize_mode=conditioning_resize_mode,
            )
            masked_source_image = canonical["masked_source_image"]
            mask_image = canonical["mask_image"]
            pose_view = canonical["conditioning_image"]
        else:
            masked_source_image = source_image
            mask_image = clothes_mask.convert("RGB")
            pose_view = pose_image

        panels = [
            ("source", source_image),
            ("masked_source", masked_source_image),
            ("mask", mask_image.convert("RGB") if mask_image.mode != "RGB" else mask_image),
            ("pose", pose_view),
            ("generated", generated_image),
            ("target", target_image),
        ]
        build_debug_grid(panels, image_width=image_width, image_height=image_height).save(out_path)
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


def _encode_image_to_latents_for_blend(
    pipe: StableDiffusionXLControlNetInpaintPipeline,
    image: Image.Image,
    device: torch.device,
) -> torch.Tensor:
    image_tensor = TF.to_tensor(image.convert("RGB")).unsqueeze(0).to(device)
    image_tensor = image_tensor * 2.0 - 1.0
    vae_dtype = getattr(pipe.vae, "dtype", torch.float32)
    image_tensor = image_tensor.to(dtype=vae_dtype)
    if torch.cuda.is_available():
        pipe.vae.to(device)
    with torch.no_grad():
        encoded = pipe.vae.encode(image_tensor)
        latents = encoded.latent_dist.sample() * pipe.vae.config.scaling_factor
    return latents.to(device=device, dtype=pipe.unet.dtype)


def make_blend_callback(
    *,
    image_latents: torch.Tensor,
    scheduler: Any,
    debug_prefix: str | None = None,
):
    printed = {"done": False}

    def callback(pipe_self, step_index, timestep, callback_kwargs):
        latents = callback_kwargs["latents"]
        mask = callback_kwargs["mask"]
        bsz = latents.shape[0]
        if mask.shape[0] != bsz:
            mask = mask[:bsz]
        if mask.shape[-2:] != latents.shape[-2:]:
            mask = F.interpolate(mask.float(), size=latents.shape[-2:], mode="nearest")
        mask = mask.to(device=latents.device, dtype=latents.dtype)
        img_lat = image_latents.to(device=latents.device, dtype=latents.dtype)

        timesteps = scheduler.timesteps
        if step_index < len(timesteps) - 1:
            t_next = timesteps[step_index + 1]
            noise = torch.randn_like(img_lat)
            if torch.is_tensor(t_next):
                timestep_tensor = t_next.reshape(1).to(device=img_lat.device, dtype=torch.long)
            else:
                timestep_tensor = torch.tensor([int(t_next)], device=img_lat.device, dtype=torch.long)
            init_latents_noisy = scheduler.add_noise(img_lat, noise, timestep_tensor)
        else:
            init_latents_noisy = img_lat

        if not printed["done"]:
            print(
                f"[infer-blend] {debug_prefix or ''} step={step_index} "
                f"latents.shape={tuple(latents.shape)} mask.shape={tuple(mask.shape)} "
                f"mask.dtype={mask.dtype} mask.min={float(mask.min().item()):.4f} mask.max={float(mask.max().item()):.4f}",
                flush=True,
            )
            printed["done"] = True

        callback_kwargs["latents"] = (1.0 - mask) * init_latents_noisy + mask * latents
        return callback_kwargs

    return callback


def _encode_keep_region_image_latents(
    pipe: StableDiffusionXLControlNetInpaintPipeline,
    source_image: Image.Image,
    torch_dtype: torch.dtype,
) -> torch.Tensor:
    image_tensor = TF.to_tensor(source_image.convert("RGB")).unsqueeze(0)
    image_tensor = image_tensor.mul(2.0).sub(1.0)
    target_device = pipe._execution_device if hasattr(pipe, "_execution_device") else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if hasattr(pipe.vae, "to"):
        pipe.vae.to(target_device)
    image_tensor = image_tensor.to(device=target_device, dtype=pipe.vae.dtype)
    with torch.no_grad():
        encoded = pipe.vae.encode(image_tensor)
        image_latents = encoded.latent_dist.sample() * pipe.vae.config.scaling_factor
    return image_latents.to(device=target_device, dtype=torch_dtype)


def make_keep_region_blend_callback(
    *,
    image_latents: torch.Tensor,
    scheduler: Any,
) -> Any:
    state = {"printed": False}

    def callback(pipe_self: Any, step_index: int, t: torch.Tensor, callback_kwargs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        latents = callback_kwargs["latents"]
        mask = callback_kwargs["mask"]
        if mask.shape[0] != latents.shape[0]:
            mask = mask[: latents.shape[0]]
        if mask.shape[-2:] != latents.shape[-2:]:
            mask = F.interpolate(mask.float(), size=latents.shape[-2:], mode="nearest")
        mask = mask.to(device=latents.device, dtype=latents.dtype)
        if not state["printed"]:
            print(
                f"[blend] latents={tuple(latents.shape)} mask={tuple(mask.shape)} "
                f"dtype={mask.dtype} min={float(mask.min().item()):.4f} max={float(mask.max().item()):.4f}",
                flush=True,
            )
            state["printed"] = True

        img_lat = image_latents.to(device=latents.device, dtype=latents.dtype)
        timesteps = scheduler.timesteps
        if step_index < len(timesteps) - 1:
            t_next = timesteps[step_index + 1]
            noise = torch.randn_like(img_lat)
            next_t = torch.tensor([t_next], device=img_lat.device, dtype=torch.long)
            init_latents_noisy = scheduler.add_noise(img_lat, noise, next_t)
        else:
            init_latents_noisy = img_lat
        callback_kwargs["latents"] = (1.0 - mask) * init_latents_noisy + mask * latents
        return callback_kwargs

    return callback


def run_inference(config: dict[str, Any]) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cfg = config["model"]
    infer_cfg = config["inference"]
    base_mode = str(model_cfg.get("base_mode", "inpaint")).lower()
    invert_mask = bool(infer_cfg.get("invert_mask", config["data"].get("invert_mask", False)))
    canonical_pose_inpaint = bool(config.get("project", {}).get("canonical_pose_inpaint", False))
    enforce_blend = bool(infer_cfg.get("enforce_keep_region_blend", True))

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
        weight_name = None
        if (lora_dir / "pytorch_lora_weights.safetensors").exists():
            weight_name = "pytorch_lora_weights.safetensors"
        elif (lora_dir / "pytorch_lora_weights.bin").exists():
            weight_name = "pytorch_lora_weights.bin"
        load_kwargs = {"adapter_name": "default", "prefix": None}
        if weight_name is not None:
            load_kwargs["weight_name"] = weight_name
        pipe.unet.load_lora_adapter(lora_dir, **load_kwargs)
        if hasattr(pipe.unet, "set_adapter"):
            pipe.unet.set_adapter("default")
        elif hasattr(pipe.unet, "set_adapters"):
            try:
                pipe.unet.set_adapters(["default"], adapter_weights=[1.0])
            except TypeError:
                pipe.unet.set_adapters(["default"])
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
        blend_callback = None
        if canonical_pose_inpaint and base_mode == "inpaint" and enforce_blend:
            image_latents_for_blend = _encode_image_to_latents_for_blend(pipe, source_image, device=device)
            blend_callback = make_blend_callback(
                image_latents=image_latents_for_blend,
                scheduler=pipe.scheduler,
                debug_prefix=stem,
            )
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
                pipe_call_kwargs = dict(
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
                if blend_callback is not None:
                    pipe_call_kwargs["callback_on_step_end"] = blend_callback
                    pipe_call_kwargs["callback_on_step_end_tensor_inputs"] = ["latents", "mask"]
                result = pipe(**pipe_call_kwargs)
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
    if infer_cfg.get("generate_debug_composites", False):
        debug_output_dir = infer_cfg.get("debug_output_dir")
        if debug_output_dir:
            made = make_inference_debug_composites_once(
                metadata_path=metadata_path,
                generated_dir=output_dir,
                output_dir=Path(debug_output_dir),
                image_width=image_width,
                image_height=image_height,
                source_background_value=source_background_value,
                conditioning_resize_mode=(config["data"].get("conditioning_resize_modes") or ["nearest"])[0],
                canonical_pose_inpaint=canonical_pose_inpaint,
            )
            print(f"generated inference debug composites: made={made}")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.mode == "validate":
        run_validation(config)
    else:
        run_inference(config)


if __name__ == "__main__":
    main()
