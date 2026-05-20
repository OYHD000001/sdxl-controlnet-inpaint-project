from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from peft import LoraConfig
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.data.dataset import SDXLControlNetInpaintDataset, collate_fn
from src.models.pipeline import SDXLControlNetInpaintTrainerPipeline
from src.training.losses import diffusion_mse_loss
from src.utils.io import ensure_dir, load_config, save_json

try:
    from bitsandbytes.optim import AdamW8bit as AdamWOptim
except Exception:
    from torch.optim import AdamW as AdamWOptim


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SDXL inpaint with frozen ControlNets and UNet LoRA.")
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
        center_crop=data_cfg.get("center_crop", False),
        random_flip=data_cfg.get("random_flip", False),
        prompt_dropout=data_cfg.get("prompt_dropout", 0.0),
        invert_mask=data_cfg.get("invert_mask", False),
        conditioning_resize_modes=data_cfg.get("conditioning_resize_modes"),
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


def attach_unet_lora(pipeline: SDXLControlNetInpaintTrainerPipeline, config: dict[str, Any]) -> None:
    lora_cfg = config["training"]["lora"]
    target_modules = list(lora_cfg.get("target_modules", ["to_q", "to_k", "to_v", "to_out.0"]))
    adapter_name = str(lora_cfg.get("adapter_name", "default"))
    peft_config = LoraConfig(
        r=int(lora_cfg.get("rank", 16)),
        lora_alpha=int(lora_cfg.get("alpha", 16)),
        lora_dropout=float(lora_cfg.get("dropout", 0.0)),
        bias=str(lora_cfg.get("bias", "none")),
        target_modules=target_modules,
    )
    pipeline.components.unet.add_adapter(peft_config, adapter_name=adapter_name)


def set_lora_train_mode(pipeline: SDXLControlNetInpaintTrainerPipeline) -> None:
    pipeline.components.text_encoder_one.requires_grad_(False)
    pipeline.components.text_encoder_two.requires_grad_(False)
    pipeline.components.vae.requires_grad_(False)
    for controlnet in pipeline.components.controlnets:
        controlnet.requires_grad_(False)
        controlnet.eval()

    pipeline.components.unet.requires_grad_(False)
    for name, parameter in pipeline.components.unet.named_parameters():
        if "lora_" in name:
            parameter.requires_grad_(True)

    pipeline.components.unet.train()
    pipeline.components.text_encoder_one.eval()
    pipeline.components.text_encoder_two.eval()
    pipeline.components.vae.eval()


def get_trainable_lora_params(unet: torch.nn.Module) -> list[torch.nn.Parameter]:
    return [parameter for parameter in unet.parameters() if parameter.requires_grad]


def build_optimizer(unet: torch.nn.Module, config: dict[str, Any]) -> torch.optim.Optimizer:
    train_cfg = config["training"]
    params = get_trainable_lora_params(unet)
    return AdamWOptim(
        params,
        lr=train_cfg["learning_rate"],
        betas=(train_cfg["adam_beta1"], train_cfg["adam_beta2"]),
        eps=train_cfg["adam_epsilon"],
        weight_decay=train_cfg["weight_decay"],
    )


def save_lora_checkpoint(
    accelerator: Accelerator,
    pipeline: SDXLControlNetInpaintTrainerPipeline,
    checkpoint_dir: Path,
    global_step: int,
    metrics: dict[str, float],
) -> None:
    ensure_dir(checkpoint_dir)
    unet = accelerator.unwrap_model(pipeline.components.unet)
    lora_dir = checkpoint_dir / "unet_lora"
    ensure_dir(lora_dir)
    unet.save_lora_adapter(lora_dir, adapter_name="default")
    save_json({"global_step": global_step, **metrics}, checkpoint_dir / "train_state.json")


def load_lora_checkpoint(
    pipeline: SDXLControlNetInpaintTrainerPipeline,
    checkpoint_dir: Path,
) -> int:
    train_state = json.loads((checkpoint_dir / "train_state.json").read_text(encoding="utf-8"))
    resume_step = int(train_state.get("global_step", 0))
    pipeline.components.unet.load_lora_adapter(checkpoint_dir / "unet_lora", adapter_name="default")
    return resume_step


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
        )

    accelerator.backward(loss)
    params = [p for p in pipeline.components.unet.parameters() if p.requires_grad and p.grad is not None]
    accelerator.clip_grad_norm_(params, max_grad_norm)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    gathered_loss = accelerator.gather_for_metrics(loss.detach()).float().mean()
    return {"loss": float(gathered_loss.cpu().item())}


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
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

    set_seed(int(config["project"]["seed"]))
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    device = accelerator.device
    dataloader = build_dataloader(config)
    pipeline = SDXLControlNetInpaintTrainerPipeline.from_pretrained_config(config["model"], device=device)
    attach_unet_lora(pipeline, config)
    pipeline.to(device)
    pipeline.controlnet_conditioning_scales = list(train_cfg.get("controlnet_conditioning_scale", [1.0] * len(pipeline.components.controlnets)))
    set_lora_train_mode(pipeline)

    global_step = 0
    if resume_checkpoint_dir is not None:
        global_step = load_lora_checkpoint(pipeline, resume_checkpoint_dir)
        if accelerator.is_main_process:
            normalize_loss_history_for_resume(loss_history_path, global_step)

    if train_cfg.get("gradient_checkpointing", False):
        pipeline.components.unet.enable_gradient_checkpointing()

    optimizer = build_optimizer(pipeline.components.unet, config)
    dataloader, optimizer, unet = accelerator.prepare(dataloader, optimizer, pipeline.components.unet)
    pipeline.components.unet = unet

    max_train_steps = int(train_cfg["max_train_steps"])
    save_every_steps = int(train_cfg["save_every_steps"])
    keep_region_loss_weight = float(train_cfg.get("keep_region_loss_weight", 1.0))

    progress_bar = tqdm(total=max_train_steps, desc="lora-train", disable=not accelerator.is_main_process)
    if global_step > 0:
        progress_bar.update(global_step)

    while global_step < max_train_steps:
        for batch in dataloader:
            metrics = train_one_step(
                accelerator=accelerator,
                pipeline=pipeline,
                optimizer=optimizer,
                batch=batch,
                max_grad_norm=float(train_cfg["max_grad_norm"]),
                keep_region_loss_weight=keep_region_loss_weight,
            )
            global_step += 1
            if accelerator.is_main_process:
                progress_bar.update(1)
                progress_bar.set_postfix(loss=f"{metrics['loss']:.4f}")
                with loss_history_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({"step": global_step, "loss": metrics["loss"]}, ensure_ascii=False) + "\n")

            if global_step % save_every_steps == 0:
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    checkpoint_dir = output_dir / "checkpoints" / f"step_{global_step:08d}"
                    save_lora_checkpoint(accelerator, pipeline, checkpoint_dir, global_step, metrics)
                accelerator.wait_for_everyone()

            if global_step >= max_train_steps:
                break

    final_dir = output_dir / "checkpoints" / "final"
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_lora_checkpoint(accelerator, pipeline, final_dir, global_step, {"loss": metrics["loss"]})
        print(f"LoRA training completed. Final artifacts saved to: {final_dir}")


if __name__ == "__main__":
    main()
