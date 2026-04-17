from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.data.dataset import SDXLControlNetInpaintDataset, collate_fn
from src.models.pipeline import SDXLControlNetInpaintTrainerPipeline
from src.training.losses import diffusion_mse_loss
from src.training.validate import run_validation
from src.utils.io import ensure_dir, load_config, save_json

try:
    from bitsandbytes.optim import AdamW8bit as AdamWOptim
except Exception:
    from torch.optim import AdamW as AdamWOptim


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SDXL + ControlNet + inpaint scaffold.")
    parser.add_argument("--config", required=True, type=Path)
    return parser.parse_args()


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
    )
    return DataLoader(
        dataset,
        batch_size=train_cfg["train_batch_size"],
        shuffle=True,
        num_workers=train_cfg["num_workers"],
        collate_fn=collate_fn,
        drop_last=True,
    )


def build_optimizer(pipeline: SDXLControlNetInpaintTrainerPipeline, config: dict[str, Any]) -> torch.optim.Optimizer:
    train_cfg = config["training"]
    params = list(pipeline.components.controlnet.parameters())

    # TODO:
    # Decide whether UNet should be partially or fully trainable in your final setup.
    # For the scaffold we only optimize ControlNet to stay close to the official baseline idea.
    optimizer = AdamWOptim(
        params,
        lr=train_cfg["learning_rate"],
        betas=(train_cfg["adam_beta1"], train_cfg["adam_beta2"]),
        eps=train_cfg["adam_epsilon"],
        weight_decay=train_cfg["weight_decay"],
    )
    return optimizer


def train_one_step(
    pipeline: SDXLControlNetInpaintTrainerPipeline,
    optimizer: torch.optim.Optimizer,
    batch: dict[str, Any],
    max_grad_norm: float,
    mixed_precision: str,
    scaler: torch.cuda.amp.GradScaler | None,
) -> dict[str, float]:
    use_autocast = mixed_precision in {"fp16", "bf16"} and torch.cuda.is_available()
    autocast_dtype = torch.float16 if mixed_precision == "fp16" else torch.bfloat16

    with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=use_autocast):
        loss_inputs = pipeline.forward_loss_inputs(batch)
        loss = diffusion_mse_loss(loss_inputs["model_pred"], loss_inputs["noise"])

    if scaler is not None and use_autocast and mixed_precision == "fp16":
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(pipeline.components.controlnet.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(pipeline.components.controlnet.parameters(), max_grad_norm)
        optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return {"loss": float(loss.detach().cpu().item())}


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    train_cfg = config["training"]

    output_dir = Path(config["project"]["output_dir"])
    ensure_dir(output_dir)
    ensure_dir(output_dir / "checkpoints")
    ensure_dir(output_dir / "logs")
    save_json(config, output_dir / "resolved_config.json")

    seed = int(config["project"]["seed"])
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloader = build_dataloader(config)
    pipeline = SDXLControlNetInpaintTrainerPipeline.from_pretrained_config(config["model"], device=device)
    pipeline.to(device)
    pipeline.set_train()
    if train_cfg.get("gradient_checkpointing", False):
        pipeline.components.unet.enable_gradient_checkpointing()
        pipeline.components.controlnet.enable_gradient_checkpointing()
    optimizer = build_optimizer(pipeline, config)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available() and train_cfg.get("mixed_precision") == "fp16")

    max_train_steps = int(train_cfg["max_train_steps"])
    save_every_steps = int(train_cfg["save_every_steps"])
    validate_every_steps = int(train_cfg["validate_every_steps"])

    progress_bar = tqdm(total=max_train_steps, desc="train")
    global_step = 0

    # TODO:
    # Add resume-from-checkpoint support.
    # Add mixed precision support.
    # Add scheduler / warmup support.
    # Add checkpoint serialization for ControlNet and optimizer state.
    while global_step < max_train_steps:
        for batch in dataloader:
            metrics = train_one_step(
                pipeline=pipeline,
                optimizer=optimizer,
                batch=batch,
                max_grad_norm=float(train_cfg["max_grad_norm"]),
                mixed_precision=str(train_cfg.get("mixed_precision", "no")),
                scaler=scaler,
            )
            global_step += 1
            progress_bar.update(1)
            progress_bar.set_postfix(loss=f"{metrics['loss']:.4f}")

            if global_step % save_every_steps == 0:
                checkpoint_dir = output_dir / "checkpoints" / f"step_{global_step:08d}"
                ensure_dir(checkpoint_dir)
                pipeline.components.controlnet.save_pretrained(checkpoint_dir / "controlnet")
                save_json({"global_step": global_step, **metrics}, checkpoint_dir / "train_state.json")

            if validate_every_steps > 0 and global_step % validate_every_steps == 0:
                run_validation(config)

            if global_step >= max_train_steps:
                break

    final_dir = output_dir / "checkpoints" / "final"
    ensure_dir(final_dir)
    pipeline.components.controlnet.save_pretrained(final_dir / "controlnet")
    save_json({"global_step": global_step}, final_dir / "train_state.json")
    print(f"Training scaffold completed. Final artifacts saved to: {final_dir}")


if __name__ == "__main__":
    main()
