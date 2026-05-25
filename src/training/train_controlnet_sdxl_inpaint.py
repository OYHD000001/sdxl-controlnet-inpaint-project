from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import ControlNetModel
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.data.dataset import SDXLControlNetInpaintDataset, collate_fn
from src.models.pipeline import SDXLControlNetInpaintTrainerPipeline
from src.training.losses import diffusion_mse_loss
from src.training.validate import run_inference, run_validation
from src.utils.io import ensure_dir, load_config, save_json

try:
    from bitsandbytes.optim import AdamW8bit as AdamWOptim
except Exception:
    from torch.optim import AdamW as AdamWOptim


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SDXL + ControlNet + inpaint scaffold.")
    parser.add_argument("--config", required=True, type=Path)
    return parser.parse_args()


def find_resume_checkpoint(resume_value: str | None, output_dir: Path) -> Path | None:
    if not resume_value:
        return None
    if resume_value == "latest":
        checkpoint_root = output_dir / "checkpoints"
        candidates = sorted(
            [
                path
                for path in checkpoint_root.glob("step_*")
                if path.is_dir() and (path / "train_state.json").exists()
            ]
        )
        return candidates[-1] if candidates else None
    checkpoint_path = Path(resume_value)
    return checkpoint_path if checkpoint_path.exists() else None


def load_checkpoint_state(
    pipeline: SDXLControlNetInpaintTrainerPipeline,
    checkpoint_dir: Path,
) -> int:
    train_state = json.loads((checkpoint_dir / "train_state.json").read_text(encoding="utf-8"))
    resume_step = int(train_state.get("global_step", 0))

    for index, controlnet in enumerate(pipeline.components.controlnets):
        controlnet_dir = checkpoint_dir / ("controlnet" if len(pipeline.components.controlnets) == 1 else f"controlnet_{index}")
        restored = ControlNetModel.from_pretrained(controlnet_dir)
        controlnet.load_state_dict(restored.state_dict(), strict=True)
        del restored
    lora_dir = checkpoint_dir / "unet_lora"
    if lora_dir.exists():
        adapter_name = pipeline.unet_lora_adapter_name
        unet = pipeline.components.unet
        weight_name = None
        if (lora_dir / "pytorch_lora_weights.safetensors").exists():
            weight_name = "pytorch_lora_weights.safetensors"
        elif (lora_dir / "pytorch_lora_weights.bin").exists():
            weight_name = "pytorch_lora_weights.bin"
        if hasattr(unet, "delete_adapters"):
            peft_config = getattr(unet, "peft_config", None)
            if isinstance(peft_config, dict) and adapter_name in peft_config:
                try:
                    unet.delete_adapters([adapter_name])
                except Exception:
                    pass
        load_kwargs = {"adapter_name": adapter_name, "prefix": None}
        if weight_name is not None:
            load_kwargs["weight_name"] = weight_name
        unet.load_lora_adapter(lora_dir, **load_kwargs)
        for name, parameter in unet.named_parameters():
            if "lora_" in name:
                parameter.data = parameter.data.to(torch.float32)
        pipeline.unet_lora_enabled = True
    return resume_step


def normalize_loss_history_for_resume(loss_history_path: Path, resume_step: int) -> None:
    if not loss_history_path.exists():
        return
    filtered_lines = []
    for line in loss_history_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if int(payload.get("step", 0)) <= resume_step:
            filtered_lines.append(json.dumps(payload, ensure_ascii=False))
    loss_history_path.write_text("".join(f"{line}\n" for line in filtered_lines), encoding="utf-8")


def build_dataloader(config: dict[str, Any]) -> DataLoader:
    data_cfg = config["data"]
    train_cfg = config["training"]
    dataset = SDXLControlNetInpaintDataset(
        metadata_path=data_cfg["train_metadata_path"],
        image_height=int(data_cfg.get("image_height", data_cfg["image_size"])),
        image_width=int(data_cfg.get("image_width", data_cfg["image_size"])),
        base_mode=str(config.get("model", {}).get("base_mode", "inpaint")).lower(),
        center_crop=data_cfg.get("center_crop", False),
        random_flip=data_cfg.get("random_flip", False),
        prompt_dropout=data_cfg.get("prompt_dropout", 0.0),
        invert_mask=data_cfg.get("invert_mask", False),
        conditioning_resize_modes=data_cfg.get("conditioning_resize_modes"),
        prompt_override=config.get("project", {}).get("fixed_prompt"),
        source_background_value=int(data_cfg.get("source_background_value", 0)),
        canonical_pose_inpaint=bool(config.get("project", {}).get("canonical_pose_inpaint", False)),
        condition_augmentation_cfg=data_cfg.get("condition_augmentation"),
    )
    return DataLoader(
        dataset,
        batch_size=train_cfg["train_batch_size"],
        shuffle=True,
        num_workers=train_cfg["num_workers"],
        collate_fn=collate_fn,
        drop_last=True,
        pin_memory=bool(train_cfg.get("pin_memory", torch.cuda.is_available())),
        persistent_workers=bool(train_cfg.get("persistent_workers", train_cfg["num_workers"] > 0)),
        prefetch_factor=int(train_cfg.get("prefetch_factor", 4)) if train_cfg["num_workers"] > 0 else None,
    )


def build_optimizer(pipeline: SDXLControlNetInpaintTrainerPipeline, config: dict[str, Any]) -> torch.optim.Optimizer:
    train_cfg = config["training"]
    controlnet_params = [
        parameter
        for controlnet in pipeline.components.controlnets
        for parameter in controlnet.parameters()
        if parameter.requires_grad
    ]
    unet_lora_params = pipeline.get_unet_lora_trainable_parameters()
    param_groups = []
    if controlnet_params:
        param_groups.append(
            {
                "params": controlnet_params,
                "lr": float(train_cfg["learning_rate"]),
            }
        )
    if unet_lora_params:
        param_groups.append(
            {
                "params": unet_lora_params,
                "lr": float(train_cfg.get("unet_lora_learning_rate", train_cfg["learning_rate"])),
            }
        )
    if not param_groups:
        raise ValueError("No trainable parameters were found for optimizer construction.")
    optimizer = AdamWOptim(
        param_groups,
        lr=float(train_cfg["learning_rate"]),
        betas=(float(train_cfg["adam_beta1"]), float(train_cfg["adam_beta2"])),
        eps=float(train_cfg["adam_epsilon"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    return optimizer


def _tensor_to_rgb_pil(tensor: torch.Tensor) -> Image.Image:
    tensor = tensor.detach().float().cpu().clamp(-1.0, 1.0)
    tensor = ((tensor + 1.0) * 127.5).round().to(torch.uint8)
    array = tensor.permute(1, 2, 0).numpy()
    return Image.fromarray(array, mode="RGB")


def _conditioning_to_rgb_pil(tensor: torch.Tensor) -> Image.Image:
    tensor = tensor.detach().float().cpu().clamp(0.0, 1.0)
    tensor = (tensor * 255.0).round().to(torch.uint8)
    array = tensor.permute(1, 2, 0).numpy()
    return Image.fromarray(array, mode="RGB")


def _mask_to_rgb_pil(tensor: torch.Tensor) -> Image.Image:
    tensor = tensor.detach().float().cpu().clamp(0.0, 1.0)
    tensor = (tensor * 255.0).round().to(torch.uint8)
    array = tensor[0].numpy()
    return Image.fromarray(array, mode="L").convert("RGB")


def _decode_latents_to_pil(pipeline: SDXLControlNetInpaintTrainerPipeline, latents: torch.Tensor) -> Image.Image:
    with torch.no_grad():
        vae = pipeline.components.vae
        # Decode on the same device as the live VAE. In canonical training the VAE
        # may stay on GPU for dynamic masked_source encodes, so moving latents to
        # aux_device here can crash exactly at epoch-end preview time.
        latents = latents[:1].to(pipeline.vae_device, dtype=torch.float32)
        latents = latents / vae.config.scaling_factor
        image = vae.decode(latents).sample
    return _tensor_to_rgb_pil(image[0])


def _predict_x0_latents(
    pipeline: SDXLControlNetInpaintTrainerPipeline,
    noisy_latents: torch.Tensor,
    model_pred: torch.Tensor,
    timesteps: torch.Tensor,
) -> torch.Tensor:
    scheduler = pipeline.components.noise_scheduler
    prediction_type = getattr(scheduler.config, "prediction_type", "epsilon")
    alphas_cumprod = scheduler.alphas_cumprod.to(device=noisy_latents.device, dtype=noisy_latents.dtype)
    alpha_prod_t = alphas_cumprod[timesteps].view(-1, 1, 1, 1)
    beta_prod_t = 1.0 - alpha_prod_t
    if prediction_type == "epsilon":
        pred_x0 = (noisy_latents - beta_prod_t.sqrt() * model_pred) / alpha_prod_t.sqrt()
    elif prediction_type == "v_prediction":
        pred_x0 = alpha_prod_t.sqrt() * noisy_latents - beta_prod_t.sqrt() * model_pred
    elif prediction_type == "sample":
        pred_x0 = model_pred
    else:
        raise ValueError(f"Unsupported scheduler prediction_type: {prediction_type}")
    return pred_x0


def save_epoch_preview(
    pipeline: SDXLControlNetInpaintTrainerPipeline,
    batch: dict[str, Any],
    loss_inputs: dict[str, torch.Tensor],
    output_path: Path,
) -> None:
    pred_x0_latents = _predict_x0_latents(
        pipeline,
        loss_inputs["noisy_latents"],
        loss_inputs["model_pred"],
        loss_inputs["timesteps"],
    )
    target_preview = _decode_latents_to_pil(pipeline, loss_inputs["target_latents"])
    pred_preview = _decode_latents_to_pil(pipeline, pred_x0_latents)
    panels = [
        ("source", _tensor_to_rgb_pil(batch["source_image"][0])),
        ("masked_source", _tensor_to_rgb_pil(batch["masked_source_image"][0])) if "masked_source_image" in batch else None,
        ("mask", _mask_to_rgb_pil(batch["mask_image"][0])),
        ("pose", _conditioning_to_rgb_pil(batch["conditioning_image"][0])),
        ("pred_x0", pred_preview),
        ("target", target_preview),
    ]
    panels = [panel for panel in panels if panel is not None]

    label_height = 32
    width = max(image.width for _, image in panels)
    height = max(image.height for _, image in panels) + label_height
    labeled = []
    for label, image in panels:
        canvas = Image.new("RGB", (width, height), (245, 245, 245))
        canvas.paste(image.resize((width, height - label_height), Image.BILINEAR), (0, label_height))
        ImageDraw.Draw(canvas).text((10, 8), label, fill=(0, 0, 0))
        labeled.append(canvas)
    grid = Image.new("RGB", (width * len(labeled), height), (255, 255, 255))
    for index, image in enumerate(labeled):
        grid.paste(image, (index * width, 0))
    ensure_dir(output_path.parent)
    grid.save(output_path)


def train_one_step(
    accelerator: Accelerator,
    pipeline: SDXLControlNetInpaintTrainerPipeline,
    optimizer: torch.optim.Optimizer,
    batch: dict[str, Any],
    max_grad_norm: float,
    keep_region_loss_weight: float,
) -> dict[str, float]:
    with accelerator.autocast():
        loss_inputs = pipeline.forward_loss_inputs(batch)
        loss = diffusion_mse_loss(
            loss_inputs["model_pred"],
            loss_inputs["noise"],
            mask=loss_inputs["mask"],
            keep_region_weight=keep_region_loss_weight,
            mask_weight_mode=loss_inputs.get("mask_weight_mode", "keep_region"),
        )

    accelerator.backward(loss)
    params = [
        parameter
        for controlnet in pipeline.components.controlnets
        for parameter in controlnet.parameters()
        if parameter.grad is not None
    ]
    params.extend([parameter for parameter in pipeline.components.unet.parameters() if parameter.requires_grad and parameter.grad is not None])
    accelerator.clip_grad_norm_(params, max_grad_norm)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    gathered_loss = accelerator.gather_for_metrics(loss.detach()).float().mean()
    preview_inputs = {
        "target_latents": loss_inputs["target_latents"].detach(),
        "noisy_latents": loss_inputs["noisy_latents"].detach(),
        "timesteps": loss_inputs["timesteps"].detach(),
        "model_pred": loss_inputs["model_pred"].detach(),
    }
    return {"loss": float(gathered_loss.cpu().item()), "preview_inputs": preview_inputs}


def train_from_config(config: dict[str, Any]) -> None:
    train_cfg = config["training"]
    mixed_precision = str(train_cfg.get("mixed_precision", "no"))
    if mixed_precision not in {"no", "fp16", "bf16"}:
        mixed_precision = "no"
    accelerator = Accelerator(mixed_precision=mixed_precision)

    output_dir = Path(config["project"]["output_dir"])
    loss_history_path = output_dir / "logs" / "loss_history.jsonl"
    resume_checkpoint_dir = find_resume_checkpoint(train_cfg.get("resume_from_checkpoint"), output_dir)
    if accelerator.is_main_process:
        ensure_dir(output_dir)
        ensure_dir(output_dir / "checkpoints")
        ensure_dir(output_dir / "logs")
        save_json(config, output_dir / "resolved_config.json")
        if resume_checkpoint_dir is None:
            loss_history_path.write_text("", encoding="utf-8")
    accelerator.wait_for_everyone()

    seed = int(config["project"]["seed"])
    set_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    device = accelerator.device
    dataloader = build_dataloader(config)
    pipeline = SDXLControlNetInpaintTrainerPipeline.from_pretrained_config(config["model"], device=device)
    pipeline.to(device)
    pipeline.controlnet_conditioning_scales = list(train_cfg.get("controlnet_conditioning_scale", [1.0] * len(pipeline.components.controlnets)))
    pipeline.set_train(
        trainable_module_patterns=config["training"].get("trainable_module_patterns"),
        unet_lora_cfg=config["training"].get("unet_lora"),
    )
    global_step = 0
    if resume_checkpoint_dir is not None:
        global_step = load_checkpoint_state(pipeline, resume_checkpoint_dir)
        if accelerator.is_main_process:
            normalize_loss_history_for_resume(loss_history_path, global_step)
    if accelerator.is_main_process:
        stats = pipeline.get_trainable_param_stats()
        print(
            f"ControlNet trainable params: {stats['controlnet_trainable']} / {stats['controlnet_total']}",
            flush=True,
        )
        print(
            f"UNet LoRA trainable params: {stats['unet_lora_trainable']} / {stats['unet_total']}",
            flush=True,
        )
    if train_cfg.get("gradient_checkpointing", False):
        if pipeline.unet_lora_enabled and hasattr(pipeline.components.unet, "enable_input_require_grads"):
            pipeline.components.unet.enable_input_require_grads()
        pipeline.components.unet.enable_gradient_checkpointing()
        for controlnet in pipeline.components.controlnets:
            controlnet.enable_gradient_checkpointing()
    optimizer = build_optimizer(pipeline, config)
    prepare_args = [dataloader, optimizer]
    if pipeline.unet_lora_enabled:
        prepare_args.append(pipeline.components.unet)
    prepare_args.extend(pipeline.components.controlnets)
    prepared = accelerator.prepare(*prepare_args)
    dataloader = prepared[0]
    optimizer = prepared[1]
    index = 2
    if pipeline.unet_lora_enabled:
        pipeline.components.unet = prepared[index]
        index += 1
    pipeline.components.controlnets = list(prepared[index:])

    max_train_steps = int(train_cfg["max_train_steps"])
    save_every_steps = int(train_cfg["save_every_steps"])
    validate_every_steps = int(train_cfg["validate_every_steps"])
    steps_per_epoch = len(dataloader)
    preview_cfg = config.get("preview", {})
    preview_every_epochs = int(preview_cfg.get("every_epochs", 1))
    preview_output_dir = Path(preview_cfg.get("output_dir", output_dir / "epoch_previews"))

    progress_bar = tqdm(total=max_train_steps, desc="train", disable=not accelerator.is_main_process)
    if global_step > 0:
        progress_bar.update(global_step)

    # TODO:
    # Add resume-from-checkpoint support.
    # Add mixed precision support.
    # Add scheduler / warmup support.
    # Add checkpoint serialization for ControlNet and optimizer state.
    while global_step < max_train_steps:
        for batch in dataloader:
            metrics = train_one_step(
                accelerator=accelerator,
                pipeline=pipeline,
                optimizer=optimizer,
                batch=batch,
                max_grad_norm=float(train_cfg["max_grad_norm"]),
                keep_region_loss_weight=float(train_cfg.get("keep_region_loss_weight", 1.0)),
            )
            global_step += 1
            if accelerator.is_main_process:
                progress_bar.update(1)
                progress_bar.set_postfix(loss=f"{metrics['loss']:.4f}")

                with loss_history_path.open("a", encoding="utf-8") as f:
                    f.write(
                        json.dumps(
                            {
                                "step": global_step,
                                "loss": metrics["loss"],
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                if (
                    steps_per_epoch > 0
                    and preview_every_epochs > 0
                    and global_step % steps_per_epoch == 0
                    and (global_step // steps_per_epoch) % preview_every_epochs == 0
                ):
                    epoch_index = global_step // steps_per_epoch
                    preview_path = preview_output_dir / f"epoch_{epoch_index:04d}_step_{global_step:08d}.png"
                    save_epoch_preview(
                        pipeline=pipeline,
                        batch=batch,
                        loss_inputs=metrics["preview_inputs"],
                        output_path=preview_path,
                    )
                    print(f"[preview] saved: {preview_path}")

            if global_step % save_every_steps == 0:
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    checkpoint_dir = output_dir / "checkpoints" / f"step_{global_step:08d}"
                    ensure_dir(checkpoint_dir)
                    for index, controlnet in enumerate(pipeline.components.controlnets):
                        controlnet_dir = checkpoint_dir / ("controlnet" if len(pipeline.components.controlnets) == 1 else f"controlnet_{index}")
                        accelerator.unwrap_model(controlnet).save_pretrained(controlnet_dir)
                    unet_lora_dir = checkpoint_dir / "unet_lora"
                    if pipeline.unet_lora_enabled:
                        accelerator.unwrap_model(pipeline.components.unet).save_lora_adapter(
                            unet_lora_dir,
                            adapter_name=pipeline.unet_lora_adapter_name,
                        )
                    save_json(
                        {
                            "global_step": global_step,
                            "loss": metrics["loss"],
                        },
                        checkpoint_dir / "train_state.json",
                    )
                accelerator.wait_for_everyone()

            if validate_every_steps > 0 and global_step % validate_every_steps == 0:
                if accelerator.is_main_process:
                    run_validation(config)

            if global_step >= max_train_steps:
                break

    final_dir = output_dir / "checkpoints" / "final"
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        ensure_dir(final_dir)
        for index, controlnet in enumerate(pipeline.components.controlnets):
            controlnet_dir = final_dir / ("controlnet" if len(pipeline.components.controlnets) == 1 else f"controlnet_{index}")
            accelerator.unwrap_model(controlnet).save_pretrained(controlnet_dir)
        if pipeline.unet_lora_enabled:
            accelerator.unwrap_model(pipeline.components.unet).save_lora_adapter(
                final_dir / "unet_lora",
                adapter_name=pipeline.unet_lora_adapter_name,
            )
        save_json({"global_step": global_step}, final_dir / "train_state.json")
        print(f"Training scaffold completed. Final artifacts saved to: {final_dir}")
        if config.get("inference", {}).get("run_after_train", False):
            infer_config = json.loads(json.dumps(config))
            infer_config["inference"]["checkpoint_dir"] = str(final_dir.resolve())
            print("Starting automatic post-train inference...")
            run_inference(infer_config)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    train_from_config(config)


if __name__ == "__main__":
    main()
