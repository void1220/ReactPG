"""Discrete graph-skeleton denoiser used by the small ReactGDiff experiment."""

from __future__ import annotations

import torch
from torch import nn


class DiscreteGraphDenoiser(nn.Module):
    """Predict a clean operation-skeleton proxy from a noisy latent vector.

    The MVP experiment represents the discrete graph chain with an operation
    histogram over the OpenExp action vocabulary. This is intentionally small:
    it lets the training/sampling scripts exercise the joint denoising path
    before a categorical graph transition kernel is introduced.
    """

    def __init__(
        self,
        *,
        structure_dim: int,
        condition_dim: int,
        time_dim: int,
        context_dim: int,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        input_dim = structure_dim + condition_dim + time_dim + context_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, structure_dim),
            nn.Sigmoid(),
        )

    def forward(
        self,
        noisy_structure: torch.Tensor,
        condition: torch.Tensor,
        timestep_embedding: torch.Tensor,
        coupled_context: torch.Tensor,
    ) -> torch.Tensor:
        features = torch.cat(
            (noisy_structure, condition, timestep_embedding, coupled_context),
            dim=-1,
        )
        return self.net(features)
