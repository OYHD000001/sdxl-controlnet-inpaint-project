from __future__ import annotations

import argparse
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed

from src.training.train_controlnet_sdxl_inpaint import build_dataloader, build_optimizer
from src.models.pipeline import SDXLControlNetInpaintTrainerPipeline
from src.utils.io import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check UNet LoRA gradient sync across distributed workers.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=10)
    return parser.parse_args()


def _first_lora_parameter(unet: torch.nn.Module) -> tuple[str, torch.nn.Parameter]:
    for name, parameter in unet.named_parameters():
        if "lora_" in name and parameter.requires_grad:
            return name, parameter
    raise RuntimeError("No trainable LoRA parameter found on UNet.")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    train_cfg = config["training"]
    mixed_precision = str(train_cfg.get("mixed_precision", "no"))
    if mixed_precision not in {"no", "fp16", "bf16"}:
        mixed_precision = "no"

    accelerator = Accelerator(mixed_precision=mixed_precision)
    set_seed(int(config["project"]["seed"]))

    dataloader = build_dataloader(config)
    pipeline = SDXLControlNetInpaintTrainerPipeline.from_pretrained_config(config["model"], device=accelerator.device)
    pipeline.to(accelerator.device)
    pipeline.controlnet_conditioning_scales = list(
        train_cfg.get("controlnet_conditioning_scale", [1.0] * len(pipeline.components.controlnets))
    )
    pipeline.set_train(
        trainable_module_patterns=train_cfg.get("trainable_module_patterns"),
        unet_lora_cfg=train_cfg.get("unet_lora"),
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

    first_name, _ = _first_lora_parameter(pipeline.components.unet)
    if accelerator.is_main_process:
        print(f"[check] tracking parameter: {first_name}", flush=True)

    step = 0
    for batch in dataloader:
        optimizer.zero_grad(set_to_none=True)
        with accelerator.autocast():
            loss_inputs = pipeline.forward_loss_inputs(batch)
            loss = (loss_inputs["model_pred"].float() - loss_inputs["noise"].float()).pow(2).mean()
        accelerator.backward(loss)

        lora_name, lora_param = _first_lora_parameter(pipeline.components.unet)
        grad = lora_param.grad.detach().float().reshape(-1)[:4].to(accelerator.device)
        values = accelerator.gather(grad)
        grad_norm = float(lora_param.grad.detach().float().norm().item()) if lora_param.grad is not None else 0.0

        if accelerator.is_main_process:
            world = accelerator.num_processes
            rows = [values[i * 4 : (i + 1) * 4].tolist() for i in range(world)]
            print(
                f"[check] step={step + 1} loss={loss.detach().float().item():.6f} "
                f"lora_grad_norm={grad_norm:.6f} gathered_first4={rows}",
                flush=True,
            )

        optimizer.step()
        step += 1
        if step >= args.steps:
            break


if __name__ == "__main__":
    main()
