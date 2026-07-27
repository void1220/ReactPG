"""Coupling utilities for the small joint diffusion experiment."""

from __future__ import annotations

import math

import torch
from torch import nn


def sinusoidal_timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """Return transformer-style sinusoidal embeddings for integer timesteps."""

    if dim <= 0:
        raise ValueError("dim must be positive")
    device = timesteps.device
    half = dim // 2
    if half == 0:
        return timesteps.float().unsqueeze(-1)
    scale = math.log(10_000) / max(half - 1, 1)
    frequencies = torch.exp(torch.arange(half, device=device, dtype=torch.float32) * -scale)
    angles = timesteps.float().unsqueeze(-1) * frequencies.unsqueeze(0)
    embedding = torch.cat((torch.sin(angles), torch.cos(angles)), dim=-1)
    if dim % 2:
        embedding = torch.cat((embedding, torch.zeros_like(embedding[:, :1])), dim=-1)
    return embedding


class DiscreteContinuousCoupling(nn.Module):
    """Exchange context between the graph-skeleton and attribute denoisers."""

    def __init__(
        self,
        *,
        structure_dim: int,
        attribute_dim: int,
        condition_dim: int,
        time_dim: int,
        context_dim: int,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        input_dim = structure_dim + attribute_dim + condition_dim + time_dim
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.to_structure_context = nn.Linear(hidden_dim, context_dim)
        self.to_attribute_context = nn.Linear(hidden_dim, context_dim)

    def forward(
        self,
        noisy_structure: torch.Tensor,
        noisy_attributes: torch.Tensor,
        condition: torch.Tensor,
        timestep_embedding: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = torch.cat(
            (noisy_structure, noisy_attributes, condition, timestep_embedding),
            dim=-1,
        )
        hidden = self.shared(features)
        return self.to_structure_context(hidden), self.to_attribute_context(hidden)
