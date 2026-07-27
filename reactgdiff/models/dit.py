"""Small DiT building blocks used by ReactGDiff experiments."""

from __future__ import annotations

import torch
from torch import nn


def modulate(features: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return features * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    """Transformer block with condition-controlled normalization."""

    def __init__(self, hidden_dim: int, num_heads: int, *, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        mlp_dim = int(hidden_dim * mlp_ratio)
        self.norm_attention = nn.LayerNorm(hidden_dim)
        self.attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            batch_first=True,
        )
        self.norm_mlp = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_dim, hidden_dim),
        )
        self.adaln = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim * 6),
        )

    def forward(self, tokens: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        (
            attn_shift,
            attn_scale,
            attn_gate,
            mlp_shift,
            mlp_scale,
            mlp_gate,
        ) = self.adaln(conditioning).chunk(6, dim=-1)
        normalized_attention = modulate(self.norm_attention(tokens), attn_shift, attn_scale)
        attended, _ = self.attention(
            normalized_attention,
            normalized_attention,
            normalized_attention,
            need_weights=False,
        )
        tokens = tokens + attn_gate.unsqueeze(1) * attended
        tokens = tokens + mlp_gate.unsqueeze(1) * self.mlp(
            modulate(self.norm_mlp(tokens), mlp_shift, mlp_scale)
        )
        return tokens
