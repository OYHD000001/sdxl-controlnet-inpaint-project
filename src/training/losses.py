from __future__ import annotations

import torch
import torch.nn.functional as F


def diffusion_mse_loss(model_pred: torch.Tensor, target_noise: torch.Tensor) -> torch.Tensor:
    """
    Standard diffusion epsilon-prediction MSE loss.

    TODO:
    - add optional v_prediction support if the final scheduler/model setup uses it
    - add masked loss weighting only if experiments prove it helps
    """

    return F.mse_loss(model_pred.float(), target_noise.float(), reduction="mean")
