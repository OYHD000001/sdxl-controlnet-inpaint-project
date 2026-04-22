from __future__ import annotations

import torch
import torch.nn.functional as F


def diffusion_mse_loss(
    model_pred: torch.Tensor,
    target_noise: torch.Tensor,
    mask: torch.Tensor | None = None,
    keep_region_weight: float = 1.0,
) -> torch.Tensor:
    """
    Standard diffusion epsilon-prediction MSE loss.

    TODO:
    - add optional v_prediction support if the final scheduler/model setup uses it
    - add masked loss weighting only if experiments prove it helps
    """

    loss = F.mse_loss(model_pred.float(), target_noise.float(), reduction="none")
    if mask is not None and keep_region_weight != 1.0:
        mask = mask.float()
        if mask.shape[-2:] != loss.shape[-2:]:
            mask = F.interpolate(mask, size=loss.shape[-2:], mode="nearest")
        keep_mask = 1.0 - mask
        weights = mask + keep_mask * float(keep_region_weight)
        loss = loss * weights
    return loss.mean()
