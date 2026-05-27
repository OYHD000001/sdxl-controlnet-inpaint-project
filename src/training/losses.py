from __future__ import annotations

import torch
import torch.nn.functional as F


def diffusion_mse_loss(
    model_pred: torch.Tensor,
    target_noise: torch.Tensor,
    mask: torch.Tensor | None = None,
    keep_region_weight: float = 1.0,
    mask_weight_mode: str = "keep_region",
) -> torch.Tensor:
    """
    Standard diffusion epsilon-prediction MSE loss.

    For the 9-channel SDXL inpaint UNet path, the pipeline does not perform
    explicit keep-region latent blending for us. Preservation of the clothes
    region is therefore learned through the masked-image conditioning channels.
    Keep `keep_region_weight >= 1.0` (typically 1.0-2.0) so the model has
    enough incentive to copy those channels instead of hallucinating average
    training-distribution clothing in the keep region.

    TODO:
    - add optional v_prediction support if the final scheduler/model setup uses it
    - add masked loss weighting only if experiments prove it helps
    """

    loss = F.mse_loss(model_pred.float(), target_noise.float(), reduction="none")
    if mask is not None and keep_region_weight != 1.0:
        mask = mask.float()
        if mask.shape[-2:] != loss.shape[-2:]:
            mask = F.interpolate(mask, size=loss.shape[-2:], mode="nearest")
        if mask_weight_mode == "mask_region":
            weights = mask * float(keep_region_weight) + (1.0 - mask)
        elif mask_weight_mode == "keep_region":
            keep_mask = 1.0 - mask
            weights = mask + keep_mask * float(keep_region_weight)
        else:
            raise ValueError(f"Unsupported mask_weight_mode: {mask_weight_mode}")
        loss = loss * weights
    return loss.mean()
