"""Continuous attribute denoiser used by the small ReactGDiff experiment."""

from __future__ import annotations

import torch
from torch import nn


class ContinuousAttributeDenoiser(nn.Module):
    """Predict clean numeric/reference attribute proxies from noisy latents."""

    def __init__(
        self,
        *,
        attribute_dim: int,
        condition_dim: int,
        time_dim: int,
        context_dim: int,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        input_dim = attribute_dim + condition_dim + time_dim + context_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, attribute_dim),
        )

    def forward(
        self,
        noisy_attributes: torch.Tensor,
        condition: torch.Tensor,
        timestep_embedding: torch.Tensor,
        coupled_context: torch.Tensor,
    ) -> torch.Tensor:
        features = torch.cat(
            (noisy_attributes, condition, timestep_embedding, coupled_context),
            dim=-1,
        )
        return self.net(features)
