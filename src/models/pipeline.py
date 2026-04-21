from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, ControlNetModel, DDPMScheduler, UNet2DConditionModel
from transformers import AutoTokenizer, CLIPTextModel, CLIPTextModelWithProjection


@dataclass
class SDXLControlNetInpaintComponents:
    tokenizer_one: Any
    tokenizer_two: Any
    text_encoder_one: CLIPTextModel
    text_encoder_two: CLIPTextModelWithProjection
    vae: AutoencoderKL
    unet: UNet2DConditionModel
    controlnets: list[ControlNetModel]
    noise_scheduler: DDPMScheduler


class SDXLControlNetInpaintTrainerPipeline:
    """
    Training-time wrapper around SDXL + ControlNet + inpaint-style latent preparation.

    TODO:
    - verify the final inpaint UNet input channel contract against the exact pretrained model
    - support local diffusers checkpoints and safetensors folders cleanly
    - add xformers / gradient checkpointing / memory optimizations later
    """

    def __init__(self, components: SDXLControlNetInpaintComponents, device: torch.device) -> None:
        self.components = components
        self.device = device
        self.aux_device = torch.device("cpu")
        self.aux_dtype = torch.float32

    @staticmethod
    def _build_4ch_controlnet_from_inpaint_unet(unet: UNet2DConditionModel) -> ControlNetModel:
        transformer_layers_per_block = (
            unet.config.transformer_layers_per_block if "transformer_layers_per_block" in unet.config else 1
        )
        encoder_hid_dim = unet.config.encoder_hid_dim if "encoder_hid_dim" in unet.config else None
        encoder_hid_dim_type = unet.config.encoder_hid_dim_type if "encoder_hid_dim_type" in unet.config else None
        addition_embed_type = unet.config.addition_embed_type if "addition_embed_type" in unet.config else None
        addition_time_embed_dim = (
            unet.config.addition_time_embed_dim if "addition_time_embed_dim" in unet.config else None
        )

        controlnet = ControlNetModel(
            encoder_hid_dim=encoder_hid_dim,
            encoder_hid_dim_type=encoder_hid_dim_type,
            addition_embed_type=addition_embed_type,
            addition_time_embed_dim=addition_time_embed_dim,
            transformer_layers_per_block=transformer_layers_per_block,
            in_channels=4,
            flip_sin_to_cos=unet.config.flip_sin_to_cos,
            freq_shift=unet.config.freq_shift,
            down_block_types=unet.config.down_block_types,
            only_cross_attention=unet.config.only_cross_attention,
            block_out_channels=unet.config.block_out_channels,
            layers_per_block=unet.config.layers_per_block,
            downsample_padding=unet.config.downsample_padding,
            mid_block_scale_factor=unet.config.mid_block_scale_factor,
            act_fn=unet.config.act_fn,
            norm_num_groups=unet.config.norm_num_groups,
            norm_eps=unet.config.norm_eps,
            cross_attention_dim=unet.config.cross_attention_dim,
            attention_head_dim=unet.config.attention_head_dim,
            num_attention_heads=unet.config.num_attention_heads,
            use_linear_projection=unet.config.use_linear_projection,
            class_embed_type=unet.config.class_embed_type,
            num_class_embeds=unet.config.num_class_embeds,
            upcast_attention=unet.config.upcast_attention,
            resnet_time_scale_shift=unet.config.resnet_time_scale_shift,
            projection_class_embeddings_input_dim=unet.config.projection_class_embeddings_input_dim,
            mid_block_type=unet.config.mid_block_type,
            conditioning_channels=3,
        )

        controlnet.time_proj.load_state_dict(unet.time_proj.state_dict())
        controlnet.time_embedding.load_state_dict(unet.time_embedding.state_dict())

        if controlnet.class_embedding:
            controlnet.class_embedding.load_state_dict(unet.class_embedding.state_dict())

        if hasattr(controlnet, "add_embedding"):
            controlnet.add_embedding.load_state_dict(unet.add_embedding.state_dict())

        controlnet.down_blocks.load_state_dict(unet.down_blocks.state_dict())
        controlnet.mid_block.load_state_dict(unet.mid_block.state_dict())

        with torch.no_grad():
            controlnet.conv_in.weight.copy_(unet.conv_in.weight[:, :4])
            if controlnet.conv_in.bias is not None and unet.conv_in.bias is not None:
                controlnet.conv_in.bias.copy_(unet.conv_in.bias)

        return controlnet

    @classmethod
    def from_pretrained_config(cls, model_cfg: dict[str, Any], device: torch.device) -> "SDXLControlNetInpaintTrainerPipeline":
        model_name = model_cfg["pretrained_model_name_or_path"]
        if not model_name:
            raise ValueError("model.pretrained_model_name_or_path must be set.")

        revision = model_cfg.get("revision")
        variant = model_cfg.get("variant")

        tokenizer_one = AutoTokenizer.from_pretrained(model_name, subfolder="tokenizer", revision=revision, use_fast=False)
        tokenizer_two = AutoTokenizer.from_pretrained(model_name, subfolder="tokenizer_2", revision=revision, use_fast=False)
        text_encoder_one = CLIPTextModel.from_pretrained(
            model_name,
            subfolder="text_encoder",
            revision=revision,
            variant=variant,
        )
        text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
            model_name,
            subfolder="text_encoder_2",
            revision=revision,
            variant=variant,
        )

        vae_source = model_cfg.get("vae_model_name_or_path") or model_name
        vae_subfolder = None if model_cfg.get("vae_model_name_or_path") else "vae"
        vae = AutoencoderKL.from_pretrained(
            vae_source,
            subfolder=vae_subfolder,
            revision=revision,
            variant=variant,
        )

        unet = UNet2DConditionModel.from_pretrained(
            model_name,
            subfolder="unet",
            revision=revision,
            variant=variant,
        )

        controlnet_name = model_cfg.get("controlnet_model_name_or_path")
        controlnet_names = model_cfg.get("controlnet_model_name_or_paths")
        if controlnet_names:
            controlnets = [
                ControlNetModel.from_pretrained(
                    name,
                    revision=revision,
                    variant=variant,
                )
                for name in controlnet_names
            ]
        elif controlnet_name:
            controlnets = [
                ControlNetModel.from_pretrained(
                    controlnet_name,
                    revision=revision,
                    variant=variant,
                )
            ]
        else:
            num_controlnets = int(model_cfg.get("num_controlnets", 1))
            controlnets = [cls._build_4ch_controlnet_from_inpaint_unet(unet) for _ in range(num_controlnets)]

        noise_scheduler = DDPMScheduler.from_pretrained(model_name, subfolder="scheduler", revision=revision)

        components = SDXLControlNetInpaintComponents(
            tokenizer_one=tokenizer_one,
            tokenizer_two=tokenizer_two,
            text_encoder_one=text_encoder_one,
            text_encoder_two=text_encoder_two,
            vae=vae,
            unet=unet,
            controlnets=controlnets,
            noise_scheduler=noise_scheduler,
        )
        return cls(components=components, device=device)

    def to(self, device: torch.device) -> None:
        self.device = device
        self.aux_device = torch.device("cpu") if device.type == "cuda" else device
        self.aux_dtype = torch.float32
        self.components.text_encoder_one.to(self.aux_device)
        self.components.text_encoder_two.to(self.aux_device)
        self.components.vae.to(self.aux_device)
        self.components.text_encoder_one.float()
        self.components.text_encoder_two.float()
        self.components.vae.float()

        unet_dtype = torch.float16 if device.type == "cuda" else torch.float32
        self.components.unet.to(device=device, dtype=unet_dtype)
        if hasattr(self.components.unet, "set_attention_slice"):
            self.components.unet.set_attention_slice("max")
        # Keep trainable ControlNet weights in fp32 for stable optimization.
        # Mixed-precision savings still come from autocast during the forward pass.
        controlnet_dtype = torch.float32
        for controlnet in self.components.controlnets:
            controlnet.to(device=device, dtype=controlnet_dtype)
            if hasattr(controlnet, "set_attention_slice"):
                controlnet.set_attention_slice("max")

    def set_train(self, trainable_module_patterns: list[str] | None = None) -> None:
        self.components.text_encoder_one.requires_grad_(False)
        self.components.text_encoder_two.requires_grad_(False)
        self.components.vae.requires_grad_(False)
        self.components.unet.requires_grad_(False)
        for controlnet in self.components.controlnets:
            controlnet.requires_grad_(True)
            if trainable_module_patterns:
                controlnet.requires_grad_(False)
                for name, parameter in controlnet.named_parameters():
                    if any(pattern in name for pattern in trainable_module_patterns):
                        parameter.requires_grad_(True)
            controlnet.train()
        self.components.unet.eval()
        self.components.text_encoder_one.eval()
        self.components.text_encoder_two.eval()
        self.components.vae.eval()

    def encode_images_to_latents(self, images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            posterior = self.components.vae.encode(images.to(self.aux_device, dtype=self.aux_dtype)).latent_dist
            latents = posterior.sample()
            latents = latents * self.components.vae.config.scaling_factor
        return latents.to(self.device)

    def resize_mask_to_latent(self, mask: torch.Tensor, latent_shape: torch.Size) -> torch.Tensor:
        return F.interpolate(mask.to(self.device), size=latent_shape[-2:], mode="nearest")

    def encode_prompt(self, texts: list[str]) -> dict[str, torch.Tensor]:
        tokenizer_one = self.components.tokenizer_one
        tokenizer_two = self.components.tokenizer_two

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

        with torch.no_grad():
            enc_one = self.components.text_encoder_one(
                text_inputs_one.input_ids.to(self.aux_device),
                output_hidden_states=True,
            )
            enc_two = self.components.text_encoder_two(
                text_inputs_two.input_ids.to(self.aux_device),
                output_hidden_states=True,
            )

        prompt_embeds = torch.cat(
            [enc_one.hidden_states[-2], enc_two.hidden_states[-2]],
            dim=-1,
        )
        pooled_prompt_embeds = enc_two.text_embeds
        return {
            "prompt_embeds": prompt_embeds.to(self.device),
            "pooled_prompt_embeds": pooled_prompt_embeds.to(self.device),
        }

    def prepare_model_inputs(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        target_latents = self.encode_images_to_latents(batch["target_image"])
        masked_source_latents = self.encode_images_to_latents(batch["masked_source_image"])

        noise = torch.randn_like(target_latents)
        timesteps = torch.randint(
            0,
            self.components.noise_scheduler.config.num_train_timesteps,
            (target_latents.shape[0],),
            device=self.device,
            dtype=torch.long,
        )
        noisy_latents = self.components.noise_scheduler.add_noise(target_latents, noise, timesteps)
        resized_mask = self.resize_mask_to_latent(batch["mask_image"], noisy_latents.shape)
        model_input = torch.cat([noisy_latents, resized_mask, masked_source_latents], dim=1)

        prompt_data = self.encode_prompt(batch["text"])
        conditioning_images = []
        for index, controlnet in enumerate(self.components.controlnets):
            key = "conditioning_image" if index == 0 else f"conditioning_image_{index + 1}"
            if key not in batch:
                raise KeyError(f"Missing batch key for dual control input: {key}")
            conditioning_images.append(batch[key].to(self.device, dtype=controlnet.dtype))

        return {
            "target_latents": target_latents,
            "noise": noise,
            "timesteps": timesteps,
            "control_model_input": noisy_latents,
            "noisy_latents": noisy_latents,
            "masked_source_latents": masked_source_latents,
            "mask": resized_mask,
            "model_input": model_input,
            "conditioning_images": conditioning_images,
            "prompt_embeds": prompt_data["prompt_embeds"],
            "pooled_prompt_embeds": prompt_data["pooled_prompt_embeds"],
        }

    def forward_loss_inputs(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        prepared = self.prepare_model_inputs(batch)

        # TODO:
        # Replace the placeholder SDXL added conditioning values with the exact
        # micro-conditioning setup required by your final chosen SDXL checkpoint.
        batch_size = prepared["target_latents"].shape[0]
        height = int(batch["target_image"].shape[-2])
        width = int(batch["target_image"].shape[-1])
        time_ids = torch.tensor(
            [[height, width, 0, 0, height, width]] * batch_size,
            device=self.device,
            dtype=prepared["prompt_embeds"].dtype,
        )

        added_cond_kwargs = {
            "text_embeds": prepared["pooled_prompt_embeds"],
            "time_ids": time_ids,
        }

        down_block_res_samples = None
        mid_block_res_sample = None
        for controlnet, conditioning_image in zip(self.components.controlnets, prepared["conditioning_images"]):
            current_down_block_res_samples, current_mid_block_res_sample = controlnet(
                prepared["control_model_input"],
                prepared["timesteps"],
                encoder_hidden_states=prepared["prompt_embeds"],
                controlnet_cond=conditioning_image,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )
            if down_block_res_samples is None:
                down_block_res_samples = list(current_down_block_res_samples)
                mid_block_res_sample = current_mid_block_res_sample
            else:
                down_block_res_samples = [
                    base + extra for base, extra in zip(down_block_res_samples, current_down_block_res_samples)
                ]
                mid_block_res_sample = mid_block_res_sample + current_mid_block_res_sample

        model_pred = self.components.unet(
            prepared["model_input"],
            prepared["timesteps"],
            encoder_hidden_states=prepared["prompt_embeds"],
            down_block_additional_residuals=down_block_res_samples,
            mid_block_additional_residual=mid_block_res_sample,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False,
        )[0]

        prepared["model_pred"] = model_pred
        return prepared
