#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from diffusers import AutoencoderKL
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, CLIPTextModel, CLIPTextModelWithProjection

from src.data.dataset import SDXLControlNetInpaintDataset, collate_fn
from src.utils.io import ensure_dir, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute VAE latents and SDXL text embeddings for local training.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--output-metadata", required=True, type=Path)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=1)
    return parser.parse_args()


def save_tensor(tensor: torch.Tensor, path: Path, dtype: torch.dtype | None = torch.float16) -> str:
    ensure_dir(path.parent)
    payload = tensor.detach().cpu()
    if dtype is not None:
        payload = payload.to(dtype=dtype)
    torch.save(payload, path)
    return str(path)


def encode_images_to_latents(vae: AutoencoderKL, images: torch.Tensor, device: torch.device) -> torch.Tensor:
    # Keep VAE encoding in fp32 for high-resolution SDXL latent caching.
    # fp16 VAE encode is prone to NaNs on 768x1024 samples.
    images = images.to(device=device, dtype=torch.float32)
    posterior = vae.encode(images).latent_dist
    latents = posterior.sample()
    return latents * vae.config.scaling_factor


def encode_prompt(
    texts: list[str],
    tokenizer_one: AutoTokenizer,
    tokenizer_two: AutoTokenizer,
    text_encoder_one: CLIPTextModel,
    text_encoder_two: CLIPTextModelWithProjection,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    text_inputs_one = tokenizer_one(
        texts,
        padding="max_length",
        max_length=tokenizer_one.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_inputs_two = tokenizer_two(
        texts,
        padding="max_length",
        max_length=tokenizer_two.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    enc_one = text_encoder_one(
        text_inputs_one.input_ids.to(device),
        output_hidden_states=True,
    )
    enc_two = text_encoder_two(
        text_inputs_two.input_ids.to(device),
        output_hidden_states=True,
    )
    return {
        "prompt_embeds": torch.cat([enc_one.hidden_states[-2], enc_two.hidden_states[-2]], dim=-1),
        "pooled_prompt_embeds": enc_two.text_embeds,
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    data_cfg = config["data"]
    dataset = SDXLControlNetInpaintDataset(
        metadata_path=args.metadata,
        image_height=int(data_cfg.get("image_height", data_cfg["image_size"])),
        image_width=int(data_cfg.get("image_width", data_cfg["image_size"])),
        base_mode=str(config.get("model", {}).get("base_mode", "inpaint")).lower(),
        center_crop=data_cfg.get("center_crop", False),
        random_flip=False,
        prompt_dropout=0.0,
        invert_mask=data_cfg.get("invert_mask", False),
        conditioning_resize_modes=data_cfg.get("conditioning_resize_modes"),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
        drop_last=False,
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_cfg = config["model"]
    model_name = model_cfg["pretrained_model_name_or_path"]
    revision = model_cfg.get("revision")
    variant = model_cfg.get("variant")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    tokenizer_one = AutoTokenizer.from_pretrained(model_name, subfolder="tokenizer", revision=revision, use_fast=False)
    tokenizer_two = AutoTokenizer.from_pretrained(model_name, subfolder="tokenizer_2", revision=revision, use_fast=False)
    text_encoder_one = CLIPTextModel.from_pretrained(
        model_name,
        subfolder="text_encoder",
        revision=revision,
        variant=variant,
        torch_dtype=dtype,
    ).to(device)
    text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
        model_name,
        subfolder="text_encoder_2",
        revision=revision,
        variant=variant,
        torch_dtype=dtype,
    ).to(device)
    vae_source = model_cfg.get("vae_model_name_or_path") or model_name
    vae_subfolder = None if model_cfg.get("vae_model_name_or_path") else "vae"
    vae = AutoencoderKL.from_pretrained(
        vae_source,
        subfolder=vae_subfolder,
        revision=revision,
        variant=variant,
        torch_dtype=torch.float32,
    ).to(device)
    text_encoder_one.eval()
    text_encoder_two.eval()
    vae.eval()

    ensure_dir(args.cache_dir)
    ensure_dir(args.output_metadata.parent)
    updated_records: list[dict] = []
    cursor = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"cache {args.metadata.name}"):
            target_latents = encode_images_to_latents(vae, batch["target_image"], device)
            masked_source_latents = None
            if "masked_source_image" in batch:
                masked_source_latents = encode_images_to_latents(vae, batch["masked_source_image"], device)
            prompt_data = encode_prompt(
                batch["text"],
                tokenizer_one,
                tokenizer_two,
                text_encoder_one,
                text_encoder_two,
                device,
            )

            batch_size = target_latents.shape[0]
            for item_index in range(batch_size):
                record = dict(batch["metadata"][item_index])
                cache_stem = f"{cursor:06d}"
                item_dir = args.cache_dir / cache_stem
                record["target_latents"] = save_tensor(
                    target_latents[item_index],
                    item_dir / "target_latents.pt",
                    dtype=torch.float32,
                )
                if masked_source_latents is not None:
                    record["masked_source_latents"] = save_tensor(
                        masked_source_latents[item_index],
                        item_dir / "masked_source_latents.pt",
                        dtype=torch.float32,
                    )
                record["prompt_embeds"] = save_tensor(
                    prompt_data["prompt_embeds"][item_index],
                    item_dir / "prompt_embeds.pt",
                    dtype=torch.float16,
                )
                record["pooled_prompt_embeds"] = save_tensor(
                    prompt_data["pooled_prompt_embeds"][item_index],
                    item_dir / "pooled_prompt_embeds.pt",
                    dtype=torch.float16,
                )
                updated_records.append(record)
                cursor += 1

    with args.output_metadata.open("w", encoding="utf-8") as f:
        for record in updated_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Wrote cached metadata: {args.output_metadata}")


if __name__ == "__main__":
    main()
