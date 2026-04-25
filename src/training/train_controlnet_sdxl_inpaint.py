from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import ControlNetModel
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


def build_optimizer(pipeline: SDXLControlNetInpaintTrainerPipeline, config: dict[str, Any]) -> torch.optim.Optimizer:
    train_cfg = config["training"]
    params = []
    for controlnet in pipeline.components.controlnets:
        params.extend([parameter for parameter in controlnet.parameters() if parameter.requires_grad])

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
    params = [
        p
        for controlnet in pipeline.components.controlnets
        for p in controlnet.parameters()
        if p.grad is not None
    ]
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

    seed = int(config["project"]["seed"])
    set_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    device = accelerator.device
    dataloader = build_dataloader(config)
    pipeline = SDXLControlNetInpaintTrainerPipeline.from_pretrained_config(config["model"], device=device)
    pipeline.to(device)
    pipeline.controlnet_conditioning_scales = list(train_cfg.get("controlnet_conditioning_scale", [1.0] * len(pipeline.components.controlnets)))
    global_step = 0
    if resume_checkpoint_dir is not None:
        global_step = load_checkpoint_state(pipeline, resume_checkpoint_dir)
        if accelerator.is_main_process:
            normalize_loss_history_for_resume(loss_history_path, global_step)
    pipeline.set_train(trainable_module_patterns=config["training"].get("trainable_module_patterns"))
    if train_cfg.get("gradient_checkpointing", False):
        pipeline.components.unet.enable_gradient_checkpointing()
        for controlnet in pipeline.components.controlnets:
            controlnet.enable_gradient_checkpointing()
    optimizer = build_optimizer(pipeline, config)
    prepared = accelerator.prepare(dataloader, optimizer, *pipeline.components.controlnets)
    dataloader = prepared[0]
    optimizer = prepared[1]
    pipeline.components.controlnets = list(prepared[2:])

    max_train_steps = int(train_cfg["max_train_steps"])
    save_every_steps = int(train_cfg["save_every_steps"])
    validate_every_steps = int(train_cfg["validate_every_steps"])

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

            if global_step % save_every_steps == 0:
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    checkpoint_dir = output_dir / "checkpoints" / f"step_{global_step:08d}"
                    ensure_dir(checkpoint_dir)
                    for index, controlnet in enumerate(pipeline.components.controlnets):
                        controlnet_dir = checkpoint_dir / ("controlnet" if len(pipeline.components.controlnets) == 1 else f"controlnet_{index}")
                        accelerator.unwrap_model(controlnet).save_pretrained(controlnet_dir)
                    save_json({"global_step": global_step, **metrics}, checkpoint_dir / "train_state.json")
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
        save_json({"global_step": global_step}, final_dir / "train_state.json")
        print(f"Training scaffold completed. Final artifacts saved to: {final_dir}")


if __name__ == "__main__":
    main()
