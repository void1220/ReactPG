"""Discrete procedure-graph diffusion backend."""

from __future__ import annotations

import math
import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from reactgdiff.data.action_parser import parse_action_sequence
from reactgdiff.eval.lev import text_gap
from reactgdiff.models.argument_filler import (
    ArgumentTextCodec,
    ArgumentTextFiller,
    should_fill_argument_slot,
    slot_features_from_decoded_slot,
)
from reactgdiff.models.dit import DiTBlock
from reactgdiff.models.graph_codec import GraphTargetCodec
from reactgdiff.models.graph_encoder_decoder import (
    STRUCTURE_TARGET_DIM,
    GraphDecoderOutput,
    build_balanced_sample_weights,
    build_operation_class_weights,
    build_structure_targets,
    graph_slot_loss,
)
from reactgdiff.models.shared_text_conditioning import (
    build_shared_text_inputs,
    load_skeleton_text_encoder,
    slice_shared_text_inputs,
)


@dataclass(slots=True)
class ProcedureGraphDiffusionOutput:
    slot_output: GraphDecoderOutput
    structure_logits: torch.Tensor
    skeleton_op_logits: torch.Tensor | None = None
    skeleton_structure_logits: torch.Tensor | None = None


@dataclass(slots=True)
class SharedTextContext:
    memory: torch.Tensor
    memory_padding_mask: torch.Tensor
    pooled: torch.Tensor
    material_keys: torch.Tensor
    material_mask: torch.Tensor
    numeric_candidate_keys: torch.Tensor
    numeric_candidate_mask: torch.Tensor


@dataclass(slots=True)
class NumericCandidateContext:
    keys: torch.Tensor
    mask: torch.Tensor


class ProcedureGraphDiffusion(nn.Module):
    """Discrete denoiser over OpenExp procedure graph slots.

    Node/edge slot categories are corrupted toward their empirical marginals,
    and a conditioned graph transformer predicts the clean categorical graph at
    a sampled timestep. The sparse "edge" slots are material references,
    condition references, quantity gates, and unit categories attached to each
    operation.
    """

    def __init__(
        self,
        *,
        condition_dim: int,
        action_dim: int,
        material_dim: int,
        condition_slot_dim: int,
        unit_dim: int,
        max_steps: int,
        max_material_slots: int,
        numeric_candidate_dim: int = 0,
        hidden_dim: int = 256,
        dit_depth: int = 6,
        dit_heads: int = 8,
        diffusion_steps: int = 64,
        noise_schedule: str = "cosine",
        structure_target_dim: int = STRUCTURE_TARGET_DIM,
        diffuse_quantities: bool = True,
        skeleton_conditioning: bool = False,
        skeleton_depth: int | None = None,
        shared_text_encoder: nn.Module | None = None,
        shared_encoder_dim: int = 0,
        shared_encoder_trainable: bool = False,
        numeric_candidate_feature_pointer: bool = False,
        numeric_candidate_type_dim: int = 8,
        numeric_candidate_source_dim: int = 6,
    ) -> None:
        super().__init__()
        if hidden_dim % dit_heads:
            raise ValueError("hidden_dim must be divisible by dit_heads")
        if noise_schedule not in {"cosine", "linear"}:
            raise ValueError(f"Unsupported noise_schedule: {noise_schedule}")
        self.condition_dim = condition_dim
        self.action_dim = action_dim
        self.material_dim = material_dim
        self.condition_slot_dim = condition_slot_dim
        self.unit_dim = unit_dim
        self.numeric_candidate_dim = max(0, int(numeric_candidate_dim))
        self.max_steps = max_steps
        self.max_material_slots = max_material_slots
        self.hidden_dim = hidden_dim
        self.dit_depth = dit_depth
        self.dit_heads = dit_heads
        self.diffusion_steps = diffusion_steps
        self.noise_schedule = noise_schedule
        self.structure_target_dim = structure_target_dim
        self.diffuse_quantities = bool(diffuse_quantities)
        self.skeleton_conditioning = bool(skeleton_conditioning)
        self.skeleton_depth = max(1, int(skeleton_depth or max(1, dit_depth // 2)))
        self.shared_text_encoder = shared_text_encoder
        self.shared_encoder_dim = max(0, int(shared_encoder_dim))
        self.shared_encoder_trainable = bool(shared_encoder_trainable)
        self.numeric_candidate_feature_pointer = bool(
            numeric_candidate_feature_pointer and self.numeric_candidate_dim > 0
        )
        self.numeric_candidate_type_dim = max(1, int(numeric_candidate_type_dim))
        self.numeric_candidate_source_dim = max(1, int(numeric_candidate_source_dim))
        if (self.shared_text_encoder is None) != (self.shared_encoder_dim <= 0):
            raise ValueError(
                "shared_text_encoder and a positive shared_encoder_dim must be provided together"
            )

        self.position_embedding = nn.Embedding(max_steps, hidden_dim)
        self.timestep_embedding = nn.Embedding(diffusion_steps + 1, hidden_dim)
        self.operation_embedding = nn.Embedding(action_dim, hidden_dim)
        self.material_embedding = nn.Embedding(material_dim, hidden_dim)
        self.condition_embedding = nn.Embedding(condition_slot_dim, hidden_dim)
        self.quantity_gate_embedding = nn.Embedding(2, hidden_dim)
        self.unit_embedding = nn.Embedding(unit_dim, hidden_dim)
        self.numeric_candidate_embedding = (
            nn.Embedding(self.numeric_candidate_dim, hidden_dim)
            if self.numeric_candidate_dim > 0
            else None
        )
        self.conditioning = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        if self.shared_text_encoder is not None:
            self.shared_memory_projection = nn.Linear(self.shared_encoder_dim, hidden_dim)
            self.shared_global_projection = nn.Sequential(
                nn.Linear(self.shared_encoder_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.shared_skeleton_conditioning = nn.Sequential(
                nn.Linear(self.shared_encoder_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.input_cross_attention = nn.MultiheadAttention(
                hidden_dim,
                dit_heads,
                batch_first=True,
            )
            self.input_cross_attention_norm = nn.LayerNorm(hidden_dim)
            self.material_pointer_query = nn.Linear(
                hidden_dim,
                max_material_slots * hidden_dim,
            )
            self.material_pointer_bias = nn.Parameter(torch.zeros(material_dim))
        else:
            self.shared_memory_projection = None
            self.shared_global_projection = None
            self.shared_skeleton_conditioning = None
            self.input_cross_attention = None
            self.input_cross_attention_norm = None
            self.material_pointer_query = None
            self.register_parameter("material_pointer_bias", None)
        if self.numeric_candidate_dim > 0 and (
            self.shared_text_encoder is not None
            or self.numeric_candidate_feature_pointer
        ):
            self.numeric_candidate_pointer_query = nn.Linear(
                hidden_dim,
                max_material_slots * hidden_dim,
            )
            self.numeric_candidate_pointer_bias = nn.Parameter(
                torch.zeros(self.numeric_candidate_dim)
            )
        else:
            self.numeric_candidate_pointer_query = None
            self.register_parameter("numeric_candidate_pointer_bias", None)
        if self.numeric_candidate_feature_pointer:
            self.numeric_candidate_key_unit_embedding = nn.Embedding(unit_dim, hidden_dim)
            self.numeric_candidate_key_type_embedding = nn.Embedding(
                self.numeric_candidate_type_dim,
                hidden_dim,
            )
            self.numeric_candidate_key_source_embedding = nn.Embedding(
                self.numeric_candidate_source_dim,
                hidden_dim,
            )
            self.numeric_candidate_key_position_embedding = nn.Embedding(
                self.numeric_candidate_dim,
                hidden_dim,
            )
            self.numeric_candidate_key_scalar_projection = nn.Linear(2, hidden_dim)
            self.numeric_candidate_key_norm = nn.LayerNorm(hidden_dim)
        else:
            self.numeric_candidate_key_unit_embedding = None
            self.numeric_candidate_key_type_embedding = None
            self.numeric_candidate_key_source_embedding = None
            self.numeric_candidate_key_position_embedding = None
            self.numeric_candidate_key_scalar_projection = None
            self.numeric_candidate_key_norm = None
        self.skeleton_conditioning_network = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.skeleton_seed = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.skeleton_blocks = nn.ModuleList(
            DiTBlock(hidden_dim, dit_heads) for _ in range(self.skeleton_depth)
        )
        self.skeleton_norm = nn.LayerNorm(hidden_dim)
        self.skeleton_operation_head = nn.Linear(hidden_dim, action_dim)
        self.skeleton_structure_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, structure_target_dim),
        )
        token_feature_count = 6 + int(self.skeleton_conditioning) + int(self.numeric_candidate_dim > 0)
        self.token_projection = nn.Sequential(
            nn.LayerNorm(hidden_dim * token_feature_count),
            nn.Linear(hidden_dim * token_feature_count, hidden_dim),
        )
        self.dit_blocks = nn.ModuleList(DiTBlock(hidden_dim, dit_heads) for _ in range(dit_depth))
        self.final_norm = nn.LayerNorm(hidden_dim)

        self.operation_head = nn.Linear(hidden_dim, action_dim)
        self.material_head = nn.Linear(hidden_dim, max_material_slots * material_dim)
        self.condition_head = nn.Linear(hidden_dim, condition_slot_dim)
        self.quantity_gate_head = nn.Linear(hidden_dim, max_material_slots * 2)
        self.unit_head = nn.Linear(hidden_dim, max_material_slots * unit_dim)
        self.numeric_candidate_head = (
            nn.Linear(hidden_dim, max_material_slots * self.numeric_candidate_dim)
            if self.numeric_candidate_dim > 0
            else None
        )
        self.quantity_value_head = nn.Linear(hidden_dim, max_material_slots)
        self.condition_value_head = nn.Linear(hidden_dim, 2)
        self.structure_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, structure_target_dim),
        )
        nn.init.zeros_(self.quantity_value_head.weight)
        nn.init.zeros_(self.quantity_value_head.bias)
        nn.init.zeros_(self.condition_value_head.weight)
        nn.init.zeros_(self.condition_value_head.bias)

        if noise_schedule == "cosine":
            self.register_buffer(
                "_discrete_cosine_alpha_bar",
                self._build_discrete_cosine_alpha_bar(diffusion_steps),
            )
        else:
            self.register_buffer("_discrete_cosine_alpha_bar", torch.empty(0))

        self.register_buffer("action_marginal", torch.ones(action_dim) / action_dim)
        self.register_buffer("material_marginal", torch.ones(material_dim) / material_dim)
        self.register_buffer(
            "condition_marginal",
            torch.ones(condition_slot_dim) / condition_slot_dim,
        )
        self.register_buffer("quantity_gate_marginal", torch.ones(2) / 2)
        self.register_buffer("unit_marginal", torch.ones(unit_dim) / unit_dim)
        self.register_buffer(
            "numeric_candidate_marginal",
            (
                torch.ones(self.numeric_candidate_dim) / self.numeric_candidate_dim
                if self.numeric_candidate_dim > 0
                else torch.empty(0)
            ),
        )

    def set_marginals(self, marginals: dict[str, torch.Tensor | list[float]]) -> None:
        for name in (
            "action",
            "material",
            "condition",
            "quantity_gate",
            "unit",
            "numeric_candidate",
        ):
            if name == "numeric_candidate" and self.numeric_candidate_dim <= 0:
                continue
            if name not in marginals:
                continue
            value = marginals[name]
            tensor = torch.as_tensor(value, dtype=torch.float32, device=self.action_marginal.device)
            tensor = tensor.clamp_min(1e-12)
            tensor = tensor / tensor.sum().clamp_min(1e-12)
            getattr(self, f"{name}_marginal").copy_(tensor)

    def marginal_dict(self) -> dict[str, list[float]]:
        return {
            "action": self.action_marginal.detach().cpu().tolist(),
            "material": self.material_marginal.detach().cpu().tolist(),
            "condition": self.condition_marginal.detach().cpu().tolist(),
            "quantity_gate": self.quantity_gate_marginal.detach().cpu().tolist(),
            "unit": self.unit_marginal.detach().cpu().tolist(),
            "numeric_candidate": self.numeric_candidate_marginal.detach().cpu().tolist(),
        }

    def forward(
        self,
        condition: torch.Tensor,
        timesteps: torch.Tensor,
        op_ids: torch.Tensor,
        material_ids: torch.Tensor,
        condition_ids: torch.Tensor,
        quantity_gate_ids: torch.Tensor,
        unit_ids: torch.Tensor,
        numeric_candidate_ids: torch.Tensor | None = None,
        skeleton_op_ids: torch.Tensor | None = None,
        shared_context: SharedTextContext | None = None,
        numeric_candidate_context: NumericCandidateContext | None = None,
        shared_input_ids: torch.Tensor | None = None,
        shared_attention_mask: torch.Tensor | None = None,
        material_candidate_positions: torch.Tensor | None = None,
        numeric_candidate_positions: torch.Tensor | None = None,
    ) -> ProcedureGraphDiffusionOutput:
        batch_size = condition.size(0)
        if self.shared_text_encoder is not None and shared_context is None:
            shared_context = self.prepare_shared_text_context(
                input_ids=shared_input_ids,
                attention_mask=shared_attention_mask,
                material_candidate_positions=material_candidate_positions,
                numeric_candidate_positions=numeric_candidate_positions,
            )
        skeleton_op_logits: torch.Tensor | None = None
        skeleton_structure_logits: torch.Tensor | None = None
        if self.skeleton_conditioning:
            skeleton_op_logits, skeleton_structure_logits = self.predict_skeleton(
                condition,
                shared_context=shared_context,
            )
            if skeleton_op_ids is None:
                skeleton_op_ids = skeleton_op_logits.argmax(dim=-1)
        positions = torch.arange(self.max_steps, device=condition.device)
        position_features = self.position_embedding(positions).unsqueeze(0).expand(
            batch_size,
            -1,
            -1,
        )
        material_features = self.material_embedding(material_ids).mean(dim=2)
        gate_features = self.quantity_gate_embedding(quantity_gate_ids).mean(dim=2)
        unit_features = self.unit_embedding(unit_ids).mean(dim=2)
        token_parts = [
            position_features,
            self.operation_embedding(op_ids),
            material_features,
            self.condition_embedding(condition_ids),
            gate_features,
            unit_features,
        ]
        if self.numeric_candidate_embedding is not None:
            if numeric_candidate_ids is None:
                numeric_candidate_ids = torch.zeros_like(unit_ids)
            candidate_features = self.numeric_candidate_embedding(
                numeric_candidate_ids.long().clamp(0, self.numeric_candidate_dim - 1)
            ).mean(dim=2)
            token_parts.append(candidate_features)
        if self.skeleton_conditioning:
            assert skeleton_op_ids is not None
            token_parts.append(self.operation_embedding(skeleton_op_ids.long().clamp(0, self.action_dim - 1)))
        token_features = self.token_projection(
            torch.cat(tuple(token_parts), dim=-1)
        )
        timesteps = timesteps.long().clamp(0, self.diffusion_steps)
        if shared_context is not None:
            assert self.shared_global_projection is not None
            base_conditioning = self.shared_global_projection(shared_context.pooled)
        else:
            base_conditioning = self.conditioning(condition)
        conditioning = base_conditioning + self.timestep_embedding(timesteps)
        hidden = token_features
        if shared_context is not None:
            assert self.input_cross_attention is not None
            assert self.input_cross_attention_norm is not None
            cross_attention, _ = self.input_cross_attention(
                hidden,
                shared_context.memory,
                shared_context.memory,
                key_padding_mask=shared_context.memory_padding_mask,
                need_weights=False,
            )
            hidden = self.input_cross_attention_norm(hidden + cross_attention)
        for block in self.dit_blocks:
            hidden = block(hidden, conditioning)
        hidden = self.final_norm(hidden)
        pooled = hidden.mean(dim=1)
        op_logits = self.operation_head(hidden)
        structure_logits = self.structure_head(pooled)
        if self.skeleton_conditioning and skeleton_op_logits is not None:
            op_logits = skeleton_op_logits
        if self.skeleton_conditioning and skeleton_structure_logits is not None:
            structure_logits = skeleton_structure_logits
        material_logits = self.material_head(hidden).view(
            batch_size,
            self.max_steps,
            self.max_material_slots,
            self.material_dim,
        )
        numeric_candidate_logits = (
            self.numeric_candidate_head(hidden).view(
                batch_size,
                self.max_steps,
                self.max_material_slots,
                self.numeric_candidate_dim,
            )
            if self.numeric_candidate_head is not None
            else None
        )
        if shared_context is not None:
            material_logits = self._dynamic_pointer_logits(
                hidden,
                shared_context.material_keys,
                shared_context.material_mask,
                query_projection=self.material_pointer_query,
                bias=self.material_pointer_bias,
            )
            if self.numeric_candidate_dim > 0:
                numeric_candidate_logits = self._dynamic_pointer_logits(
                    hidden,
                    shared_context.numeric_candidate_keys,
                    shared_context.numeric_candidate_mask,
                    query_projection=self.numeric_candidate_pointer_query,
                    bias=self.numeric_candidate_pointer_bias,
                )
        elif numeric_candidate_context is not None and self.numeric_candidate_dim > 0:
            numeric_candidate_logits = self._dynamic_pointer_logits(
                hidden,
                numeric_candidate_context.keys,
                numeric_candidate_context.mask,
                query_projection=self.numeric_candidate_pointer_query,
                bias=self.numeric_candidate_pointer_bias,
            )
        return ProcedureGraphDiffusionOutput(
            slot_output=GraphDecoderOutput(
                op_logits=op_logits,
                material_logits=material_logits,
                condition_logits=self.condition_head(hidden),
                quantity_gate_logits=self.quantity_gate_head(hidden).view(
                    batch_size,
                    self.max_steps,
                    self.max_material_slots,
                    2,
                ),
                unit_logits=self.unit_head(hidden).view(
                    batch_size,
                    self.max_steps,
                    self.max_material_slots,
                    self.unit_dim,
                ),
                quantity_values=self.quantity_value_head(hidden),
                condition_values=self.condition_value_head(hidden),
                numeric_candidate_logits=numeric_candidate_logits,
            ),
            structure_logits=structure_logits,
            skeleton_op_logits=skeleton_op_logits,
            skeleton_structure_logits=skeleton_structure_logits,
        )

    def predict_skeleton(
        self,
        condition: torch.Tensor,
        *,
        shared_context: SharedTextContext | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = condition.size(0)
        positions = torch.arange(self.max_steps, device=condition.device)
        position_features = self.position_embedding(positions).unsqueeze(0).expand(
            batch_size,
            -1,
            -1,
        )
        if shared_context is not None:
            assert self.shared_skeleton_conditioning is not None
            conditioning = self.shared_skeleton_conditioning(shared_context.pooled)
        else:
            conditioning = self.skeleton_conditioning_network(condition)
        hidden = self.skeleton_seed(position_features + conditioning.unsqueeze(1))
        for block in self.skeleton_blocks:
            hidden = block(hidden, conditioning)
        hidden = self.skeleton_norm(hidden)
        pooled = hidden.mean(dim=1)
        return self.skeleton_operation_head(hidden), self.skeleton_structure_head(pooled)

    def prepare_shared_text_context(
        self,
        *,
        input_ids: torch.Tensor | None,
        attention_mask: torch.Tensor | None,
        material_candidate_positions: torch.Tensor | None,
        numeric_candidate_positions: torch.Tensor | None,
    ) -> SharedTextContext:
        if self.shared_text_encoder is None:
            raise RuntimeError("No shared text encoder is configured")
        if any(
            value is None
            for value in (
                input_ids,
                attention_mask,
                material_candidate_positions,
                numeric_candidate_positions,
            )
        ):
            raise ValueError("Shared text encoder inputs and candidate positions are required")
        assert input_ids is not None
        assert attention_mask is not None
        assert material_candidate_positions is not None
        assert numeric_candidate_positions is not None
        if not self.shared_encoder_trainable:
            self.shared_text_encoder.eval()
        grad_enabled = self.shared_encoder_trainable and torch.is_grad_enabled()
        with torch.set_grad_enabled(grad_enabled):
            output = self.shared_text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
            encoder_memory = output.last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(encoder_memory.dtype)
        pooled = (encoder_memory * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        assert self.shared_memory_projection is not None
        memory = self.shared_memory_projection(encoder_memory)
        material_keys, material_mask = self._gather_candidate_keys(
            memory,
            material_candidate_positions,
        )
        numeric_keys, numeric_mask = self._gather_candidate_keys(
            memory,
            numeric_candidate_positions,
        )
        return SharedTextContext(
            memory=memory,
            memory_padding_mask=~attention_mask.bool(),
            pooled=pooled,
            material_keys=material_keys,
            material_mask=material_mask,
            numeric_candidate_keys=numeric_keys,
            numeric_candidate_mask=numeric_mask,
        )

    def prepare_numeric_candidate_context(
        self,
        *,
        values: torch.Tensor,
        confidences: torch.Tensor,
        unit_ids: torch.Tensor,
        type_ids: torch.Tensor,
        source_ids: torch.Tensor,
        mask: torch.Tensor,
    ) -> NumericCandidateContext:
        if not self.numeric_candidate_feature_pointer:
            raise RuntimeError("Lightweight numeric candidate pointer is disabled")
        assert self.numeric_candidate_key_unit_embedding is not None
        assert self.numeric_candidate_key_type_embedding is not None
        assert self.numeric_candidate_key_source_embedding is not None
        assert self.numeric_candidate_key_position_embedding is not None
        assert self.numeric_candidate_key_scalar_projection is not None
        assert self.numeric_candidate_key_norm is not None
        positions = torch.arange(self.numeric_candidate_dim, device=values.device)
        keys = (
            self.numeric_candidate_key_unit_embedding(
                unit_ids.long().clamp(0, self.unit_dim - 1)
            )
            + self.numeric_candidate_key_type_embedding(
                type_ids.long().clamp(0, self.numeric_candidate_type_dim - 1)
            )
            + self.numeric_candidate_key_source_embedding(
                source_ids.long().clamp(0, self.numeric_candidate_source_dim - 1)
            )
            + self.numeric_candidate_key_position_embedding(positions).unsqueeze(0)
            + self.numeric_candidate_key_scalar_projection(
                torch.stack((values.float(), confidences.float()), dim=-1)
            )
        )
        return NumericCandidateContext(
            keys=self.numeric_candidate_key_norm(keys),
            mask=mask.bool(),
        )

    @staticmethod
    def _gather_candidate_keys(
        memory: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        valid = positions.ge(0)
        safe_positions = positions.clamp(0, max(memory.size(1) - 1, 0))
        gather_index = safe_positions.unsqueeze(-1).expand(-1, -1, memory.size(-1))
        keys = memory.gather(1, gather_index)
        return keys, valid

    def _dynamic_pointer_logits(
        self,
        hidden: torch.Tensor,
        keys: torch.Tensor,
        candidate_mask: torch.Tensor,
        *,
        query_projection: nn.Linear | None,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        if query_projection is None:
            raise RuntimeError("Dynamic pointer query projection is unavailable")
        queries = query_projection(hidden).view(
            hidden.size(0),
            self.max_steps,
            self.max_material_slots,
            self.hidden_dim,
        )
        logits = torch.einsum("bsmh,bch->bsmc", queries, keys) / math.sqrt(
            self.hidden_dim
        )
        if bias is not None:
            logits = logits + bias.view(1, 1, 1, -1)
        return logits.masked_fill(
            ~candidate_mask[:, None, None, :],
            -1.0e4,
        )

    @staticmethod
    def _build_discrete_cosine_alpha_bar(diffusion_steps: int) -> torch.Tensor:
        diffusion_steps = max(int(diffusion_steps), 1)
        grid = torch.linspace(0.0, float(diffusion_steps), diffusion_steps + 1, dtype=torch.float32)
        tau = grid / float(diffusion_steps)
        alpha_cumprod = torch.cos(
            0.5 * math.pi * ((tau + 0.008) / 1.008)
        ).pow(2)
        alpha_cumprod = alpha_cumprod / alpha_cumprod[0].clamp_min(1e-12)
        alpha_cumprod[0] = 1.0
        alpha_cumprod[-1] = 0.0
        return alpha_cumprod.clamp(0.0, 1.0)

    def signal_probability(self, timesteps: torch.Tensor) -> torch.Tensor:
        if self.noise_schedule == "cosine":
            indices = timesteps.long().clamp(0, self.diffusion_steps)
            return self._discrete_cosine_alpha_bar.to(timesteps.device).index_select(
                0,
                indices.reshape(-1),
            ).view_as(timesteps).clamp(0.0, 1.0)
        tau = timesteps.float().clamp(0, self.diffusion_steps) / max(self.diffusion_steps, 1)
        return (1.0 - tau).clamp(0.0, 1.0)

    def noise_probability(self, timesteps: torch.Tensor) -> torch.Tensor:
        return (1.0 - self.signal_probability(timesteps)).clamp(0.0, 1.0)

    @torch.no_grad()
    def sample_output(
        self,
        condition: torch.Tensor,
        *,
        forced_op_ids: torch.Tensor | None = None,
        sample_steps: int | None = None,
        sample_mode: str = "sample",
        temperature: float = 1.0,
        sampler: str = "posterior",
        shared_input_ids: torch.Tensor | None = None,
        shared_attention_mask: torch.Tensor | None = None,
        material_candidate_positions: torch.Tensor | None = None,
        numeric_candidate_positions: torch.Tensor | None = None,
        numeric_candidate_values: torch.Tensor | None = None,
        numeric_candidate_confidences: torch.Tensor | None = None,
        numeric_candidate_unit_ids: torch.Tensor | None = None,
        numeric_candidate_type_ids: torch.Tensor | None = None,
        numeric_candidate_source_ids: torch.Tensor | None = None,
        numeric_candidate_mask: torch.Tensor | None = None,
    ) -> ProcedureGraphDiffusionOutput:
        if sample_mode not in {"argmax", "sample", "sample_argmax_final"}:
            raise ValueError(f"Unsupported sample_mode: {sample_mode}")
        if sampler not in {"single_step", "posterior", "iterative"}:
            raise ValueError(f"Unsupported graph diffusion sampler: {sampler}")
        batch_size = condition.size(0)
        device = condition.device
        shared_context = (
            self.prepare_shared_text_context(
                input_ids=shared_input_ids,
                attention_mask=shared_attention_mask,
                material_candidate_positions=material_candidate_positions,
                numeric_candidate_positions=numeric_candidate_positions,
            )
            if self.shared_text_encoder is not None
            else None
        )
        numeric_candidate_context = None
        if self.numeric_candidate_feature_pointer:
            candidate_inputs = (
                numeric_candidate_values,
                numeric_candidate_confidences,
                numeric_candidate_unit_ids,
                numeric_candidate_type_ids,
                numeric_candidate_source_ids,
                numeric_candidate_mask,
            )
            if any(value is None for value in candidate_inputs):
                raise ValueError(
                    "Lightweight numeric pointer requires all candidate feature tensors"
                )
            assert numeric_candidate_values is not None
            assert numeric_candidate_confidences is not None
            assert numeric_candidate_unit_ids is not None
            assert numeric_candidate_type_ids is not None
            assert numeric_candidate_source_ids is not None
            assert numeric_candidate_mask is not None
            numeric_candidate_context = self.prepare_numeric_candidate_context(
                values=numeric_candidate_values,
                confidences=numeric_candidate_confidences,
                unit_ids=numeric_candidate_unit_ids,
                type_ids=numeric_candidate_type_ids,
                source_ids=numeric_candidate_source_ids,
                mask=numeric_candidate_mask,
            )
        steps = int(sample_steps or self.diffusion_steps)
        steps = max(1, min(steps, self.diffusion_steps))
        skeleton_op_ids: torch.Tensor | None = None
        skeleton_op_logits: torch.Tensor | None = None
        if forced_op_ids is not None:
            skeleton_op_ids = forced_op_ids.to(device=device).long().clamp(0, self.action_dim - 1)
            skeleton_op_logits = _ids_to_logits(skeleton_op_ids, self.action_dim)
        elif self.skeleton_conditioning:
            skeleton_op_logits, _ = self.predict_skeleton(
                condition,
                shared_context=shared_context,
            )
            skeleton_op_ids = skeleton_op_logits.argmax(dim=-1)

        op_ids = _sample_marginal(self.action_marginal, (batch_size, self.max_steps))
        material_ids = _sample_marginal(
            self.material_marginal,
            (batch_size, self.max_steps, self.max_material_slots),
        )
        condition_ids = _sample_marginal(self.condition_marginal, (batch_size, self.max_steps))
        quantity_gate_ids = _sample_marginal(
            self.quantity_gate_marginal,
            (batch_size, self.max_steps, self.max_material_slots),
        )
        unit_ids = _sample_marginal(
            self.unit_marginal,
            (batch_size, self.max_steps, self.max_material_slots),
        )
        numeric_candidate_ids = (
            _sample_marginal(
                self.numeric_candidate_marginal,
                (batch_size, self.max_steps, self.max_material_slots),
            )
            if self.numeric_candidate_dim > 0
            else torch.zeros_like(unit_ids)
        )
        op_ids = op_ids.to(device)
        material_ids = material_ids.to(device)
        condition_ids = condition_ids.to(device)
        quantity_gate_ids = quantity_gate_ids.to(device)
        unit_ids = unit_ids.to(device)
        numeric_candidate_ids = numeric_candidate_ids.to(device)
        if skeleton_op_ids is not None:
            op_ids = skeleton_op_ids
        if not self.diffuse_quantities:
            quantity_gate_ids = torch.zeros_like(quantity_gate_ids)
            unit_ids = torch.zeros_like(unit_ids)
            numeric_candidate_ids = torch.zeros_like(numeric_candidate_ids)

        if sampler == "single_step":
            timestep = torch.full((batch_size,), self.diffusion_steps, dtype=torch.long, device=device)
            output = self(
                condition,
                timestep,
                op_ids,
                material_ids,
                condition_ids,
                quantity_gate_ids,
                unit_ids,
                numeric_candidate_ids,
                skeleton_op_ids=skeleton_op_ids,
                shared_context=shared_context,
                numeric_candidate_context=numeric_candidate_context,
            )
            if skeleton_op_ids is not None:
                slot_output = output.slot_output
                return output_with_sampled_categories(
                    output,
                    op_ids=skeleton_op_ids,
                    material_ids=slot_output.material_logits.argmax(dim=-1),
                    condition_ids=slot_output.condition_logits.argmax(dim=-1),
                    quantity_gate_ids=slot_output.quantity_gate_logits.argmax(dim=-1),
                    unit_ids=slot_output.unit_logits.argmax(dim=-1),
                    numeric_candidate_ids=(
                        slot_output.numeric_candidate_logits.argmax(dim=-1)
                        if slot_output.numeric_candidate_logits is not None
                        else numeric_candidate_ids
                    ),
                    action_dim=self.action_dim,
                    material_dim=self.material_dim,
                    condition_dim=self.condition_slot_dim,
                    unit_dim=self.unit_dim,
                    numeric_candidate_dim=self.numeric_candidate_dim,
                    op_logits=skeleton_op_logits,
                )
            return output

        if sampler == "posterior":
            output: ProcedureGraphDiffusionOutput | None = None
            for s_step, t_step in reverse_timestep_pairs(
                total_steps=self.diffusion_steps,
                sample_steps=steps,
                device=device,
            ):
                timestep = torch.full((batch_size,), t_step, dtype=torch.long, device=device)
                output = self(
                    condition,
                    timestep,
                    op_ids,
                    material_ids,
                    condition_ids,
                    quantity_gate_ids,
                    unit_ids,
                    numeric_candidate_ids,
                    skeleton_op_ids=skeleton_op_ids,
                    shared_context=shared_context,
                    numeric_candidate_context=numeric_candidate_context,
                )
                alpha_t = self.signal_probability(timestep)
                alpha_s = self.signal_probability(
                    torch.full((batch_size,), s_step, dtype=torch.long, device=device)
                )
                slot_output = output.slot_output
                step_sample_mode = _posterior_step_sample_mode(sample_mode, is_final=s_step == 0)
                if skeleton_op_ids is None:
                    op_ids = _select_ids_from_probs(
                        categorical_posterior_step_probabilities(
                            slot_output.op_logits,
                            op_ids,
                            self.action_marginal,
                            alpha_s=alpha_s,
                            alpha_t=alpha_t,
                            temperature=temperature,
                        ),
                        sample_mode=step_sample_mode,
                    )
                else:
                    op_ids = skeleton_op_ids
                material_ids = _select_ids_from_probs(
                    categorical_posterior_step_probabilities(
                        slot_output.material_logits,
                        material_ids,
                        self.material_marginal,
                        alpha_s=alpha_s,
                        alpha_t=alpha_t,
                        temperature=temperature,
                    ),
                    sample_mode=step_sample_mode,
                )
                condition_ids = _select_ids_from_probs(
                    categorical_posterior_step_probabilities(
                        slot_output.condition_logits,
                        condition_ids,
                        self.condition_marginal,
                        alpha_s=alpha_s,
                        alpha_t=alpha_t,
                        temperature=temperature,
                    ),
                    sample_mode=step_sample_mode,
                )
                if self.diffuse_quantities:
                    quantity_gate_ids = _select_ids_from_probs(
                        categorical_posterior_step_probabilities(
                            slot_output.quantity_gate_logits,
                            quantity_gate_ids,
                            self.quantity_gate_marginal,
                            alpha_s=alpha_s,
                            alpha_t=alpha_t,
                            temperature=temperature,
                        ),
                        sample_mode=step_sample_mode,
                    )
                    unit_ids = _select_ids_from_probs(
                        categorical_posterior_step_probabilities(
                            slot_output.unit_logits,
                            unit_ids,
                            self.unit_marginal,
                            alpha_s=alpha_s,
                            alpha_t=alpha_t,
                            temperature=temperature,
                        ),
                        sample_mode=step_sample_mode,
                    )
                    if slot_output.numeric_candidate_logits is not None:
                        numeric_candidate_ids = _select_ids_from_probs(
                            categorical_posterior_step_probabilities(
                                slot_output.numeric_candidate_logits,
                                numeric_candidate_ids,
                                self.numeric_candidate_marginal,
                                alpha_s=alpha_s,
                                alpha_t=alpha_t,
                                temperature=temperature,
                            ),
                            sample_mode=step_sample_mode,
                        )
                else:
                    quantity_gate_ids = torch.zeros_like(quantity_gate_ids)
                    unit_ids = torch.zeros_like(unit_ids)
                    numeric_candidate_ids = torch.zeros_like(numeric_candidate_ids)
            if output is None:
                raise RuntimeError("sample_output did not run any diffusion steps")
            return output_with_sampled_categories(
                output,
                op_ids=op_ids,
                material_ids=material_ids,
                condition_ids=condition_ids,
                quantity_gate_ids=quantity_gate_ids,
                unit_ids=unit_ids,
                numeric_candidate_ids=numeric_candidate_ids,
                action_dim=self.action_dim,
                material_dim=self.material_dim,
                condition_dim=self.condition_slot_dim,
                unit_dim=self.unit_dim,
                numeric_candidate_dim=self.numeric_candidate_dim,
                op_logits=skeleton_op_logits,
            )

        output: ProcedureGraphDiffusionOutput | None = None
        for step in range(steps, 0, -1):
            timestep = torch.full((batch_size,), step, dtype=torch.long, device=device)
            output = self(
                condition,
                timestep,
                op_ids,
                material_ids,
                condition_ids,
                quantity_gate_ids,
                unit_ids,
                numeric_candidate_ids,
                skeleton_op_ids=skeleton_op_ids,
                shared_context=shared_context,
                numeric_candidate_context=numeric_candidate_context,
            )
            slot_output = output.slot_output
            step_sample_mode = _posterior_step_sample_mode(sample_mode, is_final=step == 1)
            if skeleton_op_ids is None:
                op_clean = _select_ids(
                    slot_output.op_logits,
                    sample_mode=step_sample_mode,
                    temperature=temperature,
                )
            else:
                op_clean = skeleton_op_ids
            material_clean = _select_ids(
                slot_output.material_logits,
                sample_mode=step_sample_mode,
                temperature=temperature,
            )
            condition_clean = _select_ids(
                slot_output.condition_logits,
                sample_mode=step_sample_mode,
                temperature=temperature,
            )
            if self.diffuse_quantities:
                gate_clean = _select_ids(
                    slot_output.quantity_gate_logits,
                    sample_mode=step_sample_mode,
                    temperature=temperature,
                )
                unit_clean = _select_ids(
                    slot_output.unit_logits,
                    sample_mode=step_sample_mode,
                    temperature=temperature,
                )
                candidate_clean = (
                    _select_ids(
                        slot_output.numeric_candidate_logits,
                        sample_mode=step_sample_mode,
                        temperature=temperature,
                    )
                    if slot_output.numeric_candidate_logits is not None
                    else numeric_candidate_ids
                )
            else:
                gate_clean = torch.zeros_like(quantity_gate_ids)
                unit_clean = torch.zeros_like(unit_ids)
                candidate_clean = torch.zeros_like(numeric_candidate_ids)
            if step == 1:
                if skeleton_op_ids is not None:
                    return output_with_sampled_categories(
                        output,
                        op_ids=skeleton_op_ids,
                        material_ids=material_clean,
                        condition_ids=condition_clean,
                        quantity_gate_ids=gate_clean,
                        unit_ids=unit_clean,
                        numeric_candidate_ids=candidate_clean,
                        action_dim=self.action_dim,
                        material_dim=self.material_dim,
                        condition_dim=self.condition_slot_dim,
                        unit_dim=self.unit_dim,
                        numeric_candidate_dim=self.numeric_candidate_dim,
                        op_logits=skeleton_op_logits,
                    )
                return output
            next_timestep = torch.full((batch_size,), step - 1, dtype=torch.long, device=device)
            noise_probability = self.noise_probability(next_timestep)
            op_ids = corrupt_categorical(
                op_clean,
                torch.ones_like(op_clean, dtype=torch.bool),
                self.action_marginal,
                noise_probability,
            )
            if skeleton_op_ids is not None:
                op_ids = skeleton_op_ids
            material_ids = corrupt_categorical(
                material_clean,
                torch.ones_like(material_clean, dtype=torch.bool),
                self.material_marginal,
                noise_probability,
            )
            condition_ids = corrupt_categorical(
                condition_clean,
                torch.ones_like(condition_clean, dtype=torch.bool),
                self.condition_marginal,
                noise_probability,
            )
            if self.diffuse_quantities:
                quantity_gate_ids = corrupt_categorical(
                    gate_clean,
                    torch.ones_like(gate_clean, dtype=torch.bool),
                    self.quantity_gate_marginal,
                    noise_probability,
                )
                unit_ids = corrupt_categorical(
                    unit_clean,
                    torch.ones_like(unit_clean, dtype=torch.bool),
                    self.unit_marginal,
                    noise_probability,
                )
                if self.numeric_candidate_dim > 0:
                    numeric_candidate_ids = corrupt_categorical(
                        candidate_clean,
                        torch.ones_like(candidate_clean, dtype=torch.bool),
                        self.numeric_candidate_marginal,
                        noise_probability,
                    )
            else:
                quantity_gate_ids = torch.zeros_like(quantity_gate_ids)
                unit_ids = torch.zeros_like(unit_ids)
                numeric_candidate_ids = torch.zeros_like(numeric_candidate_ids)
        if output is None:
            raise RuntimeError("sample_output did not run any diffusion steps")
        return output

    def config(self) -> dict[str, Any]:
        return {
            "condition_dim": self.condition_dim,
            "action_dim": self.action_dim,
            "material_dim": self.material_dim,
            "condition_slot_dim": self.condition_slot_dim,
            "unit_dim": self.unit_dim,
            "numeric_candidate_dim": self.numeric_candidate_dim,
            "max_steps": self.max_steps,
            "max_material_slots": self.max_material_slots,
            "hidden_dim": self.hidden_dim,
            "dit_depth": self.dit_depth,
            "dit_heads": self.dit_heads,
            "diffusion_steps": self.diffusion_steps,
            "noise_schedule": self.noise_schedule,
            "structure_target_dim": self.structure_target_dim,
            "diffuse_quantities": self.diffuse_quantities,
            "skeleton_conditioning": self.skeleton_conditioning,
            "skeleton_depth": self.skeleton_depth,
            "shared_encoder_dim": self.shared_encoder_dim,
            "shared_encoder_trainable": self.shared_encoder_trainable,
            "numeric_candidate_feature_pointer": self.numeric_candidate_feature_pointer,
            "numeric_candidate_type_dim": self.numeric_candidate_type_dim,
            "numeric_candidate_source_dim": self.numeric_candidate_source_dim,
        }


def train_procedure_graph_diffusion(
    records: list[dict[str, Any]],
    *,
    condition_vectors: list[list[float]],
    codec: GraphTargetCodec,
    condition_dim: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    hidden_dim: int,
    dit_depth: int,
    dit_heads: int,
    diffusion_steps: int,
    noise_schedule: str,
    material_loss_weight: float,
    condition_loss_weight: float,
    quantity_gate_loss_weight: float,
    material_none_weight: float,
    material_present_weight: float,
    condition_none_weight: float,
    condition_present_weight: float,
    quantity_negative_weight: float,
    quantity_positive_weight: float,
    quantity_value_loss_weight: float,
    numeric_candidate_loss_weight: float,
    condition_value_loss_weight: float,
    numeric_value_clip: float,
    structure_loss_weight: float,
    sampling_strategy: str,
    sample_weight_max: float,
    skeleton_weight_max: float,
    operation_weighting: str,
    operation_weight_alpha: float,
    operation_weight_max: float,
    gradient_clip_norm: float,
    pad_loss_weight: float,
    diffuse_quantities: bool,
    skeleton_conditioning: bool,
    skeleton_loss_weight: float,
    skeleton_teacher_forcing_probability: float,
    skeleton_teacher_forcing_final_probability: float | None,
    skeleton_corruption_probability: float,
    slot_operation_loss_weight: float,
    timestep_sampling: str,
    endpoint_probability: float,
    seed: int,
    device: str | torch.device,
    shared_encoder_checkpoint: str | None = None,
    shared_encoder_mode: str = "frozen",
    shared_encoder_learning_rate: float = 1e-5,
    shared_encoder_max_length: int = 512,
    shared_encoder_prompt_style: str = "checkpoint",
    shared_encoder_include_numeric_evidence: bool = True,
    shared_encoder_numeric_evidence_include_source: bool = False,
    shared_encoder_local_files_only: bool = True,
    numeric_candidate_feature_pointer: bool = True,
    log_every: int = 1,
    validation_callback: Callable[[ProcedureGraphDiffusion, int], dict[str, float]] | None = None,
    validation_interval: int = 0,
    restore_best: bool = False,
    best_metric_key: str = "discrete_slot_score",
    best_metric_mode: str = "max",
) -> tuple[ProcedureGraphDiffusion, list[dict[str, float]]]:
    if not records:
        raise ValueError("records must not be empty")
    if shared_encoder_mode not in {"frozen", "finetune"}:
        raise ValueError(f"Unsupported shared_encoder_mode: {shared_encoder_mode}")
    if shared_encoder_prompt_style not in {"checkpoint", "compact", "reactxt"}:
        raise ValueError(
            f"Unsupported shared_encoder_prompt_style: {shared_encoder_prompt_style}"
        )
    torch.manual_seed(seed)
    tensors = codec.encode_records(records, condition_vectors, device=device)
    (
        condition,
        op_ids,
        material_ids,
        condition_ids,
        quantity_gate_ids,
        unit_ids,
        quantity_values,
        quantity_value_masks,
        condition_values,
        condition_value_masks,
        slot_mask,
    ) = tensors
    numeric_candidate_ids = torch.tensor(
        [codec.encode_record(record)["numeric_candidate_ids"] for record in records],
        dtype=torch.long,
        device=device,
    )
    numeric_candidate_features = codec.encode_numeric_candidate_features(
        records,
        device=device,
    )
    numeric_target_mask = quantity_value_masks.bool()
    numeric_candidate_target_count = int(numeric_target_mask.sum().item())
    numeric_candidate_evidence_count = int(
        ((numeric_candidate_ids > 1) & numeric_target_mask).sum().item()
    )
    numeric_candidate_evidence_rate = (
        numeric_candidate_evidence_count / max(numeric_candidate_target_count, 1)
    )
    shared_text_encoder: nn.Module | None = None
    shared_text_inputs: dict[str, torch.Tensor] | None = None
    shared_encoder_metadata: dict[str, Any] | None = None
    if shared_encoder_checkpoint:
        shared_text_encoder, shared_tokenizer, shared_encoder_metadata = (
            load_skeleton_text_encoder(
                shared_encoder_checkpoint,
                device=device,
                trainable=shared_encoder_mode == "finetune",
                local_files_only=shared_encoder_local_files_only,
            )
        )
        resolved_prompt_style = (
            str(shared_encoder_metadata.get("prompt_style", "compact"))
            if shared_encoder_prompt_style == "checkpoint"
            else shared_encoder_prompt_style
        )
        shared_text_inputs = build_shared_text_inputs(
            records,
            codec=codec,
            tokenizer=shared_tokenizer,
            prompt_style=resolved_prompt_style,
            include_numeric_evidence=shared_encoder_include_numeric_evidence,
            numeric_evidence_include_source=(
                shared_encoder_numeric_evidence_include_source
            ),
            max_length=shared_encoder_max_length,
        )
        shared_encoder_metadata.update(
            {
                "mode": shared_encoder_mode,
                "learning_rate": float(shared_encoder_learning_rate),
                "max_length": int(shared_encoder_max_length),
                "prompt_style": resolved_prompt_style,
                "include_numeric_evidence": bool(
                    shared_encoder_include_numeric_evidence
                ),
                "numeric_evidence_include_source": bool(
                    shared_encoder_numeric_evidence_include_source
                ),
            }
        )
        print(
            "共享文本编码器："
            f"checkpoint={shared_encoder_checkpoint} "
            f"mode={shared_encoder_mode} hidden={shared_encoder_metadata['hidden_size']} "
            f"max_length={shared_encoder_max_length}。",
            flush=True,
        )
    print(
        "数值候选监督覆盖："
        f"{numeric_candidate_evidence_count}/{numeric_candidate_target_count} "
        f"({numeric_candidate_evidence_rate:.2%})，"
        f"source证据={'启用' if codec.numeric_candidate_include_source else '关闭'}。",
        flush=True,
    )
    if timestep_sampling not in {"uniform", "endpoint", "endpoint_mix"}:
        raise ValueError(f"Unsupported graph diffusion timestep_sampling: {timestep_sampling}")
    if numeric_value_clip > 0:
        clip_value = float(numeric_value_clip)
        quantity_values = quantity_values.clamp(min=-clip_value, max=clip_value)
        condition_values = condition_values.clamp(min=-clip_value, max=clip_value)
    if condition_value_loss_weight <= 0:
        condition_values = torch.zeros_like(condition_values)
        condition_value_masks = torch.zeros_like(condition_value_masks)
    if not diffuse_quantities:
        quantity_gate_ids = torch.zeros_like(quantity_gate_ids)
        unit_ids = torch.zeros_like(unit_ids)
        quantity_values = torch.zeros_like(quantity_values)
        quantity_value_masks = torch.zeros_like(quantity_value_masks)
        numeric_candidate_ids = torch.zeros_like(numeric_candidate_ids)
    structure_targets = build_structure_targets(
        op_ids,
        material_ids,
        condition_ids,
        quantity_value_masks,
        slot_mask,
        codec=codec,
    )
    dataset_tensors: list[torch.Tensor] = [
        condition,
        op_ids,
        material_ids,
        condition_ids,
        quantity_gate_ids,
        unit_ids,
        numeric_candidate_ids,
        quantity_values,
        quantity_value_masks,
        condition_values,
        condition_value_masks,
        slot_mask,
        structure_targets,
        *numeric_candidate_features,
    ]
    if shared_text_inputs is not None:
        dataset_tensors.extend(
            [
                shared_text_inputs["input_ids"],
                shared_text_inputs["attention_mask"],
                shared_text_inputs["material_candidate_positions"],
                shared_text_inputs["numeric_candidate_positions"],
            ]
        )
    dataset = TensorDataset(*dataset_tensors)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    if sampling_strategy == "balanced":
        sample_weights = build_balanced_sample_weights(
            records,
            max_weight=sample_weight_max,
            skeleton_weight_max=skeleton_weight_max,
        )
        sampler = WeightedRandomSampler(
            sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
            generator=generator,
        )
        loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    elif sampling_strategy == "random":
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)
    else:
        raise ValueError(f"Unsupported sampling_strategy: {sampling_strategy}")

    op_class_weights = build_operation_class_weights(
        records,
        codec=codec,
        device=device,
        weighting=operation_weighting,
        alpha=operation_weight_alpha,
        max_weight=operation_weight_max,
    )
    if op_class_weights is None:
        op_class_weights = torch.ones(codec.action_dim, dtype=torch.float32, device=device)
    op_class_weights[codec.pad_id] = max(float(pad_loss_weight), 0.0)
    marginals = fit_category_marginals(
        op_ids=op_ids,
        material_ids=material_ids,
        condition_ids=condition_ids,
        quantity_gate_ids=quantity_gate_ids,
        unit_ids=unit_ids,
        numeric_candidate_ids=numeric_candidate_ids,
        slot_mask=slot_mask,
        codec=codec,
    )
    if not diffuse_quantities:
        marginals["quantity_gate"] = torch.tensor([1.0, 0.0], dtype=torch.float32, device=device)
        unit_marginal = torch.zeros(codec.unit_dim, dtype=torch.float32, device=device)
        unit_marginal[0] = 1.0
        marginals["unit"] = unit_marginal
        numeric_candidate_marginal = torch.zeros(
            codec.numeric_candidate_dim,
            dtype=torch.float32,
            device=device,
        )
        numeric_candidate_marginal[0] = 1.0
        marginals["numeric_candidate"] = numeric_candidate_marginal
    model = ProcedureGraphDiffusion(
        condition_dim=condition_dim,
        action_dim=codec.action_dim,
        material_dim=codec.material_dim,
        condition_slot_dim=codec.condition_dim,
        unit_dim=codec.unit_dim,
        numeric_candidate_dim=codec.numeric_candidate_dim,
        max_steps=codec.max_steps,
        max_material_slots=codec.max_material_slots,
        hidden_dim=hidden_dim,
        dit_depth=dit_depth,
        dit_heads=dit_heads,
        diffusion_steps=diffusion_steps,
        noise_schedule=noise_schedule,
        diffuse_quantities=diffuse_quantities,
        skeleton_conditioning=skeleton_conditioning,
        shared_text_encoder=shared_text_encoder,
        shared_encoder_dim=(
            int(shared_encoder_metadata["hidden_size"])
            if shared_encoder_metadata is not None
            else 0
        ),
        shared_encoder_trainable=shared_encoder_mode == "finetune",
        numeric_candidate_feature_pointer=(
            numeric_candidate_feature_pointer
            and shared_text_encoder is None
            and diffuse_quantities
        ),
        numeric_candidate_type_dim=codec.numeric_candidate_type_dim,
        numeric_candidate_source_dim=codec.numeric_candidate_source_dim,
    ).to(device)
    model.shared_encoder_metadata = shared_encoder_metadata
    model.shared_text_tokenizer = (
        shared_tokenizer if shared_encoder_metadata is not None else None
    )
    model.set_marginals(marginals)
    shared_encoder_parameter_ids = (
        {id(parameter) for parameter in model.shared_text_encoder.parameters()}
        if model.shared_text_encoder is not None
        else set()
    )
    graph_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in shared_encoder_parameter_ids
    ]
    optimizer_groups: list[dict[str, Any]] = [
        {"params": graph_parameters, "lr": learning_rate}
    ]
    if model.shared_text_encoder is not None and model.shared_encoder_trainable:
        optimizer_groups.append(
            {
                "params": [
                    parameter
                    for parameter in model.shared_text_encoder.parameters()
                    if parameter.requires_grad
                ],
                "lr": shared_encoder_learning_rate,
            }
        )
    optimizer = torch.optim.AdamW(optimizer_groups, lr=learning_rate)
    history: list[dict[str, float]] = []
    log_every = max(int(log_every), 0)
    validation_interval = max(int(validation_interval), 0)
    if best_metric_mode not in {"min", "max"}:
        raise ValueError(f"Unsupported best_metric_mode: {best_metric_mode}")
    best_metric = math.inf if best_metric_mode == "min" else -math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None

    for epoch in range(1, epochs + 1):
        model.train()
        totals = {
            "loss": 0.0,
            "slot_loss": 0.0,
            "structure_loss": 0.0,
            "skeleton_loss": 0.0,
            "grad_norm": 0.0,
            "noise_probability": 0.0,
        }
        total_items = 0
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            core_batch = batch[:19]
            (
                batch_condition,
                batch_op,
                batch_material,
                batch_condition_id,
                batch_quantity_gate,
                batch_unit,
                batch_numeric_candidate,
                batch_quantity_value,
                batch_quantity_value_mask,
                batch_condition_value,
                batch_condition_value_mask,
                batch_mask,
                batch_structure_target,
                batch_numeric_candidate_values,
                batch_numeric_candidate_confidences,
                batch_numeric_candidate_units,
                batch_numeric_candidate_types,
                batch_numeric_candidate_sources,
                batch_numeric_candidate_mask,
            ) = core_batch
            shared_batch = batch[19:] if len(batch) > 19 else ()
            if shared_batch:
                (
                    shared_input_ids,
                    shared_attention_mask,
                    material_candidate_positions,
                    numeric_candidate_positions,
                ) = (
                    tensor.to(batch_condition.device)
                    for tensor in shared_batch
                )
                shared_context = model.prepare_shared_text_context(
                    input_ids=shared_input_ids,
                    attention_mask=shared_attention_mask,
                    material_candidate_positions=material_candidate_positions,
                    numeric_candidate_positions=numeric_candidate_positions,
                )
            else:
                shared_context = None
            numeric_candidate_context = (
                model.prepare_numeric_candidate_context(
                    values=batch_numeric_candidate_values,
                    confidences=batch_numeric_candidate_confidences,
                    unit_ids=batch_numeric_candidate_units,
                    type_ids=batch_numeric_candidate_types,
                    source_ids=batch_numeric_candidate_sources,
                    mask=batch_numeric_candidate_mask,
                )
                if model.numeric_candidate_feature_pointer
                else None
            )
            batch_size_now = batch_condition.size(0)
            active_steps = batch_mask.bool()
            active_material_slots = active_steps.unsqueeze(-1).expand_as(batch_material)
            timesteps = sample_training_timesteps(
                batch_size_now,
                diffusion_steps=diffusion_steps,
                strategy=timestep_sampling,
                endpoint_probability=endpoint_probability,
                device=batch_condition.device,
            )
            noise_probability = model.noise_probability(timesteps)
            noisy_op = corrupt_categorical(
                batch_op,
                torch.ones_like(batch_op, dtype=torch.bool),
                model.action_marginal,
                noise_probability,
            )
            noisy_material = corrupt_categorical(
                batch_material,
                active_material_slots,
                model.material_marginal,
                noise_probability,
            )
            noisy_condition = corrupt_categorical(
                batch_condition_id,
                active_steps,
                model.condition_marginal,
                noise_probability,
            )
            if diffuse_quantities:
                noisy_gate = corrupt_categorical(
                    batch_quantity_gate,
                    active_material_slots,
                    model.quantity_gate_marginal,
                    noise_probability,
                )
                noisy_unit = corrupt_categorical(
                    batch_unit,
                    active_material_slots,
                    model.unit_marginal,
                    noise_probability,
                )
                noisy_numeric_candidate = corrupt_categorical(
                    batch_numeric_candidate,
                    active_material_slots,
                    model.numeric_candidate_marginal,
                    noise_probability,
                )
            else:
                noisy_gate = torch.zeros_like(batch_quantity_gate)
                noisy_unit = torch.zeros_like(batch_unit)
                noisy_numeric_candidate = torch.zeros_like(batch_numeric_candidate)
            skeleton_op_ids = None
            if model.skeleton_conditioning:
                skeleton_op_ids = batch_op
                final_teacher_probability = (
                    skeleton_teacher_forcing_probability
                    if skeleton_teacher_forcing_final_probability is None
                    else skeleton_teacher_forcing_final_probability
                )
                progress = (epoch - 1) / max(epochs - 1, 1)
                teacher_probability = max(
                    0.0,
                    min(
                        float(skeleton_teacher_forcing_probability)
                        + progress
                        * (
                            float(final_teacher_probability)
                            - float(skeleton_teacher_forcing_probability)
                        ),
                        1.0,
                    ),
                )
                use_predicted = torch.zeros(
                    (batch_size_now, 1),
                    dtype=torch.bool,
                    device=batch_condition.device,
                )
                if teacher_probability < 1.0:
                    with torch.no_grad():
                        predicted_skeleton_logits, _ = model.predict_skeleton(
                            batch_condition,
                            shared_context=shared_context,
                        )
                        predicted_skeleton = predicted_skeleton_logits.argmax(dim=-1)
                    use_predicted = (
                        torch.rand(batch_size_now, device=batch_condition.device)
                        >= teacher_probability
                    ).unsqueeze(-1)
                    skeleton_op_ids = torch.where(use_predicted, predicted_skeleton, batch_op)
                corruption_probability = max(
                    0.0,
                    min(float(skeleton_corruption_probability), 1.0),
                )
                if corruption_probability > 0:
                    corrupt_rows = use_predicted
                    corrupt_tokens = (
                        torch.rand(skeleton_op_ids.shape, device=batch_condition.device)
                        < corruption_probability
                    )
                    replacements = _sample_marginal(
                        model.action_marginal,
                        tuple(skeleton_op_ids.shape),
                    ).to(batch_condition.device)
                    skeleton_op_ids = torch.where(
                        corrupt_rows & corrupt_tokens,
                        replacements,
                        skeleton_op_ids,
                    )
            output = model(
                batch_condition,
                timesteps,
                noisy_op,
                noisy_material,
                noisy_condition,
                noisy_gate,
                noisy_unit,
                noisy_numeric_candidate,
                skeleton_op_ids=skeleton_op_ids,
                shared_context=shared_context,
                numeric_candidate_context=numeric_candidate_context,
            )
            slot_loss = graph_slot_loss(
                output.slot_output,
                batch_op,
                batch_material,
                batch_condition_id,
                batch_quantity_gate,
                batch_unit,
                batch_quantity_value,
                batch_quantity_value_mask,
                batch_condition_value,
                batch_condition_value_mask,
                batch_mask,
                pad_id=codec.pad_id,
                material_loss_weight=material_loss_weight,
                condition_loss_weight=condition_loss_weight,
                material_none_weight=material_none_weight,
                material_present_weight=material_present_weight,
                condition_none_weight=condition_none_weight,
                condition_present_weight=condition_present_weight,
                quantity_negative_weight=quantity_negative_weight,
                quantity_positive_weight=quantity_positive_weight,
                quantity_gate_loss_weight=quantity_gate_loss_weight if diffuse_quantities else 0.0,
                unit_loss_weight=0.7 if diffuse_quantities else 0.0,
                quantity_value_loss_weight=quantity_value_loss_weight if diffuse_quantities else 0.0,
                condition_value_loss_weight=condition_value_loss_weight,
                op_class_weights=op_class_weights,
                ignore_pad_operations=pad_loss_weight <= 0,
                operation_loss_weight=slot_operation_loss_weight,
                numeric_candidate_ids=batch_numeric_candidate,
                numeric_candidate_loss_weight=(
                    numeric_candidate_loss_weight if diffuse_quantities else 0.0
                ),
            )
            skeleton_loss = (
                skeleton_operation_loss(
                    output.skeleton_op_logits,
                    batch_op,
                    op_class_weights=op_class_weights,
                )
                if output.skeleton_op_logits is not None
                else slot_loss.detach() * 0.0
            )
            structure_loss = F.mse_loss(
                torch.sigmoid(output.structure_logits),
                batch_structure_target,
            )
            loss = (
                slot_loss
                + structure_loss_weight * structure_loss
                + skeleton_loss_weight * skeleton_loss
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    "Non-finite graph diffusion loss before backward "
                    f"(epoch={epoch}, slot={float(slot_loss.detach())}, "
                    f"struct={float(structure_loss.detach())})."
                )
            loss.backward()
            grad_norm = (
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=float(gradient_clip_norm),
                    error_if_nonfinite=False,
                )
                if gradient_clip_norm > 0
                else torch.tensor(0.0, device=batch_condition.device)
            )
            if not torch.isfinite(grad_norm):
                raise FloatingPointError(f"Non-finite graph diffusion gradient at epoch {epoch}.")
            optimizer.step()

            totals["loss"] += float(loss.detach()) * batch_size_now
            totals["slot_loss"] += float(slot_loss.detach()) * batch_size_now
            totals["structure_loss"] += float(structure_loss.detach()) * batch_size_now
            totals["skeleton_loss"] += float(skeleton_loss.detach()) * batch_size_now
            totals["grad_norm"] += float(grad_norm.detach()) * batch_size_now
            totals["noise_probability"] += float(noise_probability.detach().mean()) * batch_size_now
            total_items += batch_size_now
        epoch_metrics = {
            "epoch": float(epoch),
            "numeric_candidate_target_count": float(numeric_candidate_target_count),
            "numeric_candidate_evidence_count": float(numeric_candidate_evidence_count),
            "numeric_candidate_evidence_rate": numeric_candidate_evidence_rate,
            **{key: value / max(total_items, 1) for key, value in totals.items()},
        }
        should_validate = (
            validation_callback is not None
            and validation_interval > 0
            and (epoch == epochs or epoch % validation_interval == 0)
        )
        if should_validate:
            model.eval()
            print(
                f"[验证 {epoch:03d}/{epochs:03d}] 开始周期验证："
                f"目标指标={best_metric_key}，模式={best_metric_mode}",
                flush=True,
            )
            validation_metrics = validation_callback(model, epoch)
            for key, value in validation_metrics.items():
                if isinstance(value, (int, float)):
                    epoch_metrics[f"val_{key}"] = float(value)
            candidate = validation_metrics.get(best_metric_key)
            if candidate is not None:
                candidate_value = float(candidate)
                improved = (
                    candidate_value < best_metric
                    if best_metric_mode == "min"
                    else candidate_value > best_metric
                )
                if improved:
                    best_metric = candidate_value
                    best_epoch = epoch
                    best_state = copy.deepcopy(
                        {
                            name: tensor.detach().cpu()
                            for name, tensor in model.state_dict().items()
                        }
                    )
                    print(
                        f"[最佳模型] epoch={epoch:03d} 刷新最佳 "
                        f"{best_metric_key}={best_metric:.4f}，已暂存当前权重。",
                        flush=True,
                    )
                else:
                    print(
                        f"[最佳模型] epoch={epoch:03d} 未刷新；"
                        f"当前 {best_metric_key}={candidate_value:.4f}，"
                        f"最佳 {best_metric_key}={best_metric:.4f}@epoch {best_epoch:03d}。",
                        flush=True,
                    )
                epoch_metrics[f"best_{best_metric_key}"] = float(best_metric)
                epoch_metrics["best_epoch"] = float(best_epoch)
        history.append(epoch_metrics)
        if log_every and (epoch == 1 or epoch == epochs or epoch % log_every == 0):
            message = (
                "[训练 "
                f"{epoch:03d}/{epochs:03d}] "
                f"loss={epoch_metrics['loss']:.4f} "
                f"离散槽={epoch_metrics['slot_loss']:.4f} "
                f"骨架={epoch_metrics['skeleton_loss']:.4f} "
                f"结构={epoch_metrics['structure_loss']:.4f} "
                f"平均噪声={epoch_metrics['noise_probability']:.3f} "
                f"梯度范数={epoch_metrics['grad_norm']:.2f} "
                f"样本={total_items}"
            )
            if f"val_{best_metric_key}" in epoch_metrics:
                message += (
                    f" 验证_{best_metric_key}={epoch_metrics[f'val_{best_metric_key}']:.4f} "
                    f"最佳={epoch_metrics[f'best_{best_metric_key}']:.4f}"
                    f"@epoch {int(epoch_metrics['best_epoch']):03d}"
                )
            print(message, flush=True)
    if restore_best and best_state is not None:
        model.load_state_dict(best_state)
        if history:
            history[-1]["restored_best_epoch"] = float(best_epoch)
            history[-1][f"restored_best_{best_metric_key}"] = float(best_metric)
        print(
            f"[最佳模型] 已恢复 epoch {best_epoch:03d} 的最佳权重，"
            f"{best_metric_key}={best_metric:.4f}；后续 checkpoint 将保存这份权重。",
            flush=True,
        )
    elif restore_best:
        print("[最佳模型] 本次没有周期验证结果可恢复，将保存最后一轮权重。", flush=True)
    return model, history


@torch.no_grad()
def predict_procedure_graph_diffusion_records(
    model: ProcedureGraphDiffusion,
    codec: GraphTargetCodec,
    records: list[dict[str, Any]],
    *,
    condition_vectors: list[list[float]],
    argument_filler: ArgumentTextFiller | None = None,
    argument_text_codec: ArgumentTextCodec | None = None,
    include_generated_graph: bool = False,
    argument_filler_target: str | None = None,
    argument_condition_on_quantity_units: bool | None = None,
    quantity_gate_threshold: float = 0.65,
    condition_probability_threshold: float = 0.35,
    decode_quantities: bool | None = None,
    decode_quantity_values: bool = True,
    ground_numeric_slots: bool = False,
    numeric_candidate_reuse_penalty: float = 0.0,
    numeric_candidate_unit_weight: float = 0.0,
    drop_unsupported_numeric_slots: bool = False,
    sample_steps: int | None = None,
    sample_mode: str = "sample",
    sample_temperature: float = 1.0,
    sample_batch_size: int = 64,
    sampler: str = "posterior",
    skeleton_source: str = "predicted",
    skeleton_cache: dict[str, Any] | None = None,
    use_structure_length: bool = False,
    min_structure_steps: int = 2,
    seed: int = 19,
    device: str | torch.device = "cpu",
) -> list[dict[str, Any]]:
    torch.manual_seed(seed)
    model = model.to(device)
    model.eval()
    shared_text_inputs: dict[str, torch.Tensor] | None = None
    if model.shared_text_encoder is not None:
        shared_tokenizer = getattr(model, "shared_text_tokenizer", None)
        shared_metadata = getattr(model, "shared_encoder_metadata", None) or {}
        if shared_tokenizer is None:
            raise RuntimeError(
                "Shared-encoder graph model has no tokenizer attached; load it "
                "through load_procedure_graph_diffusion_checkpoint."
            )
        shared_text_inputs = build_shared_text_inputs(
            records,
            codec=codec,
            tokenizer=shared_tokenizer,
            prompt_style=str(shared_metadata.get("prompt_style", "compact")),
            include_numeric_evidence=bool(
                shared_metadata.get("include_numeric_evidence", True)
            ),
            numeric_evidence_include_source=bool(
                shared_metadata.get("numeric_evidence_include_source", False)
            ),
            max_length=int(shared_metadata.get("max_length", 512)),
        )
    if decode_quantities is None:
        decode_quantities = bool(getattr(model, "diffuse_quantities", True))
    if skeleton_source not in {"predicted", "gold", "cache", "none"}:
        raise ValueError(f"Unsupported skeleton_source: {skeleton_source}")
    if argument_filler is not None:
        argument_filler = argument_filler.to(device)
        argument_filler.eval()
        if argument_filler_target is None:
            argument_filler_target = str(getattr(argument_filler, "argument_filler_target", "all"))
        if argument_condition_on_quantity_units is None:
            argument_condition_on_quantity_units = bool(
                getattr(argument_filler, "condition_on_quantity_units", True)
            )
    if argument_filler_target is None:
        argument_filler_target = "all"
    if argument_condition_on_quantity_units is None:
        argument_condition_on_quantity_units = True
    rows: list[dict[str, Any]] = []
    batch_size = max(1, int(sample_batch_size))
    for start in range(0, len(records), batch_size):
        batch_records = records[start : start + batch_size]
        batch_vectors = condition_vectors[start : start + batch_size]
        condition = torch.tensor(batch_vectors, dtype=torch.float32, device=device)
        shared_batch = (
            slice_shared_text_inputs(
                shared_text_inputs,
                start,
                start + len(batch_records),
                device=device,
            )
            if shared_text_inputs is not None
            else {}
        )
        candidate_batch = (
            codec.encode_numeric_candidate_features(batch_records, device=device)
            if model.numeric_candidate_feature_pointer
            else ()
        )
        forced_op_ids = _forced_skeleton_ids(
            codec,
            batch_records,
            source=skeleton_source,
            cache=skeleton_cache,
            device=device,
        )
        output = model.sample_output(
            condition,
            forced_op_ids=forced_op_ids,
            sample_steps=sample_steps,
            sample_mode=sample_mode,
            temperature=sample_temperature,
            sampler=sampler,
            shared_input_ids=shared_batch.get("input_ids"),
            shared_attention_mask=shared_batch.get("attention_mask"),
            material_candidate_positions=shared_batch.get(
                "material_candidate_positions"
            ),
            numeric_candidate_positions=shared_batch.get(
                "numeric_candidate_positions"
            ),
            numeric_candidate_values=candidate_batch[0] if candidate_batch else None,
            numeric_candidate_confidences=(
                candidate_batch[1] if candidate_batch else None
            ),
            numeric_candidate_unit_ids=candidate_batch[2] if candidate_batch else None,
            numeric_candidate_type_ids=candidate_batch[3] if candidate_batch else None,
            numeric_candidate_source_ids=candidate_batch[4] if candidate_batch else None,
            numeric_candidate_mask=candidate_batch[5] if candidate_batch else None,
        )
        for offset, record in enumerate(batch_records):
            decode_candidate_pool = (
                codec.numeric_candidates_from_record(record)
                if (
                    model.numeric_candidate_feature_pointer
                    or numeric_candidate_reuse_penalty
                    or numeric_candidate_unit_weight
                    or drop_unsupported_numeric_slots
                )
                else None
            )
            forced_step_count = None
            if use_structure_length:
                structure = torch.sigmoid(output.structure_logits[offset]).detach().cpu()
                predicted_steps = int(round(float(structure[0]) * max(codec.max_steps - 1, 1)))
                forced_step_count = max(
                    int(min_structure_steps),
                    min(predicted_steps, codec.max_steps),
                )
            slots = codec.decode_logits(
                output.slot_output.op_logits[offset].cpu(),
                output.slot_output.material_logits[offset].cpu(),
                output.slot_output.condition_logits[offset].cpu(),
                output.slot_output.quantity_gate_logits[offset].cpu(),
                output.slot_output.unit_logits[offset].cpu(),
                output.slot_output.quantity_values[offset].cpu(),
                output.slot_output.condition_values[offset].cpu(),
                (
                    output.slot_output.numeric_candidate_logits[offset].cpu()
                    if output.slot_output.numeric_candidate_logits is not None
                    else None
                ),
                quantity_gate_threshold=quantity_gate_threshold,
                condition_probability_threshold=condition_probability_threshold,
                forced_step_count=forced_step_count,
                decode_quantities=decode_quantities,
                decode_quantity_values=decode_quantity_values,
                numeric_candidates=decode_candidate_pool,
                numeric_candidate_reuse_penalty=numeric_candidate_reuse_penalty,
                numeric_candidate_unit_weight=numeric_candidate_unit_weight,
                drop_unsupported_numeric_slots=drop_unsupported_numeric_slots,
            )
            if ground_numeric_slots:
                slots = codec.ground_numeric_slots(record, slots)
            if argument_filler is not None and argument_text_codec is not None:
                single_condition = condition[offset : offset + 1]
                for slot in slots:
                    if not should_fill_argument_slot(slot, target=argument_filler_target):
                        continue
                    features = slot_features_from_decoded_slot(
                        slot,
                        graph_codec=codec,
                        device=device,
                        condition_on_quantity_units=argument_condition_on_quantity_units,
                    )
                    step_id = torch.tensor(
                        [min(int(slot.get("step_id", 0)), codec.max_steps - 1)],
                        dtype=torch.long,
                        device=device,
                    )
                    argument_text = argument_filler.generate(
                        argument_text_codec,
                        single_condition,
                        step_id,
                        features["op_id"],
                        features["material_ids"],
                        features["condition_id"],
                        features["quantity_gate_ids"],
                        features["unit_ids"],
                    )
                    if argument_text:
                        slot["argument_text"] = argument_text
            graph = codec.build_generated_graph(record, slots)
            prediction = codec.decompile_generated_graph(graph)
            reference = str(record.get("actions", ""))
            gap = text_gap(prediction, reference)
            row = {
                "index": record.get("index"),
                "input_vector": batch_vectors[offset],
                "reference_actions": reference,
                "predicted_actions": prediction,
                "decoded_slots": slots,
                "text_gap": gap,
                "levenshtein_similarity": 1.0 - gap,
                "decoder_backend": "procedure_graph_diffusion",
                "skeleton_source": skeleton_source,
                "skeleton_operations": [str(slot.get("operation_type") or "") for slot in slots],
            }
            if include_generated_graph:
                row["generated_graph"] = graph
            rows.append(row)
    return rows


def _forced_skeleton_ids(
    codec: GraphTargetCodec,
    records: list[dict[str, Any]],
    *,
    source: str,
    cache: dict[str, Any] | None,
    device: str | torch.device,
) -> torch.Tensor | None:
    if source == "none":
        return None
    if source == "predicted":
        return None
    rows: list[list[int]] = []
    for record in records:
        if source == "gold":
            rows.append(codec.skeleton_ids_from_record(record))
            continue
        cache_row = _lookup_skeleton_cache(cache or {}, record)
        operations = _operations_from_skeleton_payload(cache_row)
        if not operations:
            return None
        rows.append(codec.operation_ids_from_sequence(operations))
    return torch.tensor(rows, dtype=torch.long, device=device)


def _lookup_skeleton_cache(cache: dict[str, Any], record: dict[str, Any]) -> Any:
    index = record.get("index")
    for key in (str(index), index):
        if key in cache:
            return cache[key]
    return {}


def _operations_from_skeleton_payload(payload: Any) -> list[str]:
    if isinstance(payload, list):
        return [str(item).upper() for item in payload]
    if not isinstance(payload, dict):
        return []
    for key in ("predicted_skeleton", "skeleton", "operations", "operation_types"):
        value = payload.get(key)
        if isinstance(value, list):
            return [str(item).upper() for item in value]
        if isinstance(value, str):
            return [step.operation_type for step in parse_action_sequence(value)]
    slots = payload.get("decoded_slots")
    if isinstance(slots, list):
        return [str(slot.get("operation_type") or "").upper() for slot in slots if isinstance(slot, dict)]
    for key in ("predicted_actions", "prediction", "decoded_actions", "output"):
        value = payload.get(key)
        if isinstance(value, str):
            return [step.operation_type for step in parse_action_sequence(value)]
    return []


def save_procedure_graph_diffusion_checkpoint(
    path: str | Path,
    *,
    model: ProcedureGraphDiffusion,
    codec: GraphTargetCodec,
    condition_featurizer: dict[str, Any],
    history: list[dict[str, float]],
    argument_filler: ArgumentTextFiller | None = None,
    argument_text_codec: ArgumentTextCodec | None = None,
    argument_history: list[dict[str, float]] | None = None,
    argument_filler_target: str = "all",
    argument_condition_on_quantity_units: bool = True,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "checkpoint_type": "procedure_graph_diffusion",
        "model_state": model.cpu().state_dict(),
        "model_config": model.config(),
        "marginals": model.marginal_dict(),
        "codec": codec.to_dict(),
        "condition_featurizer": condition_featurizer,
        "history": history,
        "shared_encoder": getattr(model, "shared_encoder_metadata", None),
    }
    if argument_filler is not None and argument_text_codec is not None:
        payload.update(
            {
                "argument_filler_state": argument_filler.cpu().state_dict(),
                "argument_filler_config": argument_filler.config(),
                "argument_text_codec": argument_text_codec.to_dict(),
                "argument_history": argument_history or [],
                "argument_filler_target": argument_filler_target,
                "argument_condition_on_quantity_units": argument_condition_on_quantity_units,
            }
        )
    torch.save(payload, path)


def load_procedure_graph_diffusion_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> tuple[ProcedureGraphDiffusion, GraphTargetCodec, dict[str, Any], list[dict[str, float]]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    model_config = dict(payload["model_config"])
    shared_encoder_metadata = payload.get("shared_encoder")
    shared_text_encoder = None
    shared_text_tokenizer = None
    if int(model_config.get("shared_encoder_dim", 0)) > 0:
        if not shared_encoder_metadata:
            raise ValueError("Shared-encoder checkpoint is missing encoder metadata")
        shared_text_encoder, shared_text_tokenizer, loaded_metadata = (
            load_skeleton_text_encoder(
                shared_encoder_metadata["checkpoint"],
                device=device,
                trainable=bool(model_config.get("shared_encoder_trainable", False)),
                local_files_only=bool(
                    shared_encoder_metadata.get("local_files_only", True)
                ),
            )
        )
        loaded_metadata.update(shared_encoder_metadata)
        shared_encoder_metadata = loaded_metadata
    model = ProcedureGraphDiffusion(
        **model_config,
        shared_text_encoder=shared_text_encoder,
    ).to(device)
    model.shared_encoder_metadata = shared_encoder_metadata
    model.shared_text_tokenizer = shared_text_tokenizer
    if "marginals" in payload:
        model.set_marginals(payload["marginals"])
    model.load_state_dict(payload["model_state"], strict=False)
    codec = GraphTargetCodec.from_dict(payload["codec"])
    return model, codec, payload["condition_featurizer"], payload.get("history", [])


def fit_category_marginals(
    *,
    op_ids: torch.Tensor,
    material_ids: torch.Tensor,
    condition_ids: torch.Tensor,
    quantity_gate_ids: torch.Tensor,
    unit_ids: torch.Tensor,
    numeric_candidate_ids: torch.Tensor | None = None,
    slot_mask: torch.Tensor,
    codec: GraphTargetCodec,
) -> dict[str, torch.Tensor]:
    active_steps = slot_mask.bool()
    active_material_slots = active_steps.unsqueeze(-1).expand_as(material_ids)
    action = _bincount_distribution(op_ids, codec.action_dim)
    material = _bincount_distribution(material_ids[active_material_slots], codec.material_dim)
    condition = _bincount_distribution(condition_ids[active_steps], codec.condition_dim)
    quantity_gate = _bincount_distribution(quantity_gate_ids[active_material_slots], 2)
    unit = _bincount_distribution(unit_ids[active_material_slots], codec.unit_dim)
    numeric_candidate = (
        _bincount_distribution(
            numeric_candidate_ids[active_material_slots],
            codec.numeric_candidate_dim,
        )
        if numeric_candidate_ids is not None
        else torch.ones(codec.numeric_candidate_dim, device=op_ids.device)
        / codec.numeric_candidate_dim
    )
    return {
        "action": action,
        "material": material,
        "condition": condition,
        "quantity_gate": quantity_gate,
        "unit": unit,
        "numeric_candidate": numeric_candidate,
    }


def corrupt_categorical(
    clean_ids: torch.Tensor,
    mask: torch.Tensor,
    marginal: torch.Tensor,
    noise_probability: torch.Tensor,
) -> torch.Tensor:
    marginal = marginal.to(clean_ids.device)
    sampled = _sample_marginal(marginal, tuple(clean_ids.shape)).to(clean_ids.device)
    probability = noise_probability.to(clean_ids.device)
    while probability.dim() < clean_ids.dim():
        probability = probability.unsqueeze(-1)
    replace = (torch.rand(clean_ids.shape, device=clean_ids.device) < probability) & mask
    return torch.where(replace, sampled, clean_ids)


def skeleton_operation_loss(
    logits: torch.Tensor | None,
    targets: torch.Tensor,
    *,
    op_class_weights: torch.Tensor | None,
) -> torch.Tensor:
    if logits is None:
        return targets.float().sum() * 0.0
    flat_logits = logits.reshape(-1, logits.size(-1))
    flat_targets = targets.reshape(-1)
    return F.cross_entropy(flat_logits, flat_targets, weight=op_class_weights)


def sample_training_timesteps(
    batch_size: int,
    *,
    diffusion_steps: int,
    strategy: str,
    endpoint_probability: float,
    device: str | torch.device,
) -> torch.Tensor:
    if strategy == "endpoint":
        return torch.full((batch_size,), diffusion_steps, dtype=torch.long, device=device)
    timesteps = torch.randint(
        0,
        diffusion_steps + 1,
        (batch_size,),
        dtype=torch.long,
        device=device,
    )
    if strategy == "endpoint_mix":
        probability = max(0.0, min(float(endpoint_probability), 1.0))
        endpoint_mask = torch.rand(batch_size, device=device) < probability
        timesteps = torch.where(
            endpoint_mask,
            torch.full_like(timesteps, diffusion_steps),
            timesteps,
        )
    return timesteps


def _sample_marginal(marginal: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    flat_count = math.prod(shape)
    sampled = torch.multinomial(marginal.detach(), flat_count, replacement=True)
    return sampled.view(*shape)


def reverse_timestep_pairs(
    *,
    total_steps: int,
    sample_steps: int,
    device: torch.device | str,
) -> list[tuple[int, int]]:
    del device
    total_steps = max(int(total_steps), 1)
    sample_steps = max(1, min(int(sample_steps), total_steps))
    boundaries = [
        int(round(total_steps * (1.0 - idx / sample_steps)))
        for idx in range(sample_steps + 1)
    ]
    boundaries[0] = total_steps
    boundaries[-1] = 0
    pairs: list[tuple[int, int]] = []
    previous_t = total_steps
    for boundary in boundaries[1:]:
        s_step = max(0, min(int(boundary), previous_t - 1))
        pairs.append((s_step, previous_t))
        previous_t = s_step
    return pairs


def categorical_posterior_step_probabilities(
    clean_logits: torch.Tensor,
    current_ids: torch.Tensor,
    marginal: torch.Tensor,
    *,
    alpha_s: torch.Tensor,
    alpha_t: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Compute sum_x0 q(x_s | x_t, x0) p_theta(x0 | x_t) for marginal corruption."""

    if clean_logits.shape[:-1] != current_ids.shape:
        raise ValueError(
            "clean_logits/current_ids shape mismatch: "
            f"{tuple(clean_logits.shape)} vs {tuple(current_ids.shape)}"
        )
    device = clean_logits.device
    dtype = clean_logits.dtype
    num_classes = clean_logits.size(-1)
    batch_size = clean_logits.size(0)
    flat_logits = clean_logits.reshape(batch_size, -1, num_classes)
    flat_current = current_ids.reshape(batch_size, -1).long().clamp(0, num_classes - 1)
    marginal = marginal.to(device=device, dtype=dtype).clamp_min(1e-12)
    marginal = marginal / marginal.sum().clamp_min(1e-12)

    scaled_logits = flat_logits / max(float(temperature), 1e-6)
    clean_prob = torch.softmax(scaled_logits, dim=-1).clamp_min(1e-12)
    current_one_hot = F.one_hot(flat_current, num_classes=num_classes).to(dtype=dtype)
    current_marginal = marginal.index_select(0, flat_current.reshape(-1)).view_as(flat_current).to(dtype)
    current_marginal = current_marginal.unsqueeze(-1)

    alpha_s = alpha_s.to(device=device, dtype=dtype).view(batch_size, 1, 1).clamp(0.0, 1.0)
    alpha_t = alpha_t.to(device=device, dtype=dtype).view(batch_size, 1, 1).clamp(0.0, 1.0)
    alpha_t_given_s = (alpha_t / alpha_s.clamp_min(1e-8)).clamp(0.0, 1.0)

    q_t_given_s = (1.0 - alpha_t_given_s) * current_marginal + alpha_t_given_s * current_one_hot
    q_t_given_0 = (1.0 - alpha_t) * current_marginal + alpha_t * current_one_hot

    ratio = clean_prob / q_t_given_0.clamp_min(1e-12)
    ratio_sum = ratio.sum(dim=-1, keepdim=True)
    q_s_given_0_marginal = (1.0 - alpha_s) * marginal.view(1, 1, num_classes) * ratio_sum
    q_s_given_0_clean = alpha_s * ratio
    unnormalized = q_t_given_s * (q_s_given_0_marginal + q_s_given_0_clean)
    probabilities = unnormalized / unnormalized.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return probabilities.reshape(*current_ids.shape, num_classes)


def _select_ids_from_probs(
    probabilities: torch.Tensor,
    *,
    sample_mode: str,
) -> torch.Tensor:
    if sample_mode == "argmax":
        return probabilities.argmax(dim=-1)
    flat = probabilities.reshape(-1, probabilities.size(-1)).clamp_min(1e-12)
    sampled = torch.distributions.Categorical(probs=flat).sample()
    return sampled.view(probabilities.shape[:-1])


def _posterior_step_sample_mode(sample_mode: str, *, is_final: bool) -> str:
    if sample_mode == "sample_argmax_final":
        return "argmax" if is_final else "sample"
    return sample_mode


def output_with_sampled_categories(
    output: ProcedureGraphDiffusionOutput,
    *,
    op_ids: torch.Tensor,
    material_ids: torch.Tensor,
    condition_ids: torch.Tensor,
    quantity_gate_ids: torch.Tensor,
    unit_ids: torch.Tensor,
    numeric_candidate_ids: torch.Tensor,
    action_dim: int,
    material_dim: int,
    condition_dim: int,
    unit_dim: int,
    numeric_candidate_dim: int,
    op_logits: torch.Tensor | None = None,
) -> ProcedureGraphDiffusionOutput:
    """Return sampled graph categories plus calibrated numeric decode logits.

    Material/condition categories are the actual terminal diffusion sample.
    Quantity gates, units, and record-local numeric candidates retain the final
    denoiser logits so confidence thresholds and joint candidate rescoring
    remain effective during deterministic decoding.  Replacing these logits
    with synthetic +/-20 one-hot values made every practical gate threshold
    and reuse penalty a no-op.
    """

    slot_output = output.slot_output
    return ProcedureGraphDiffusionOutput(
        slot_output=GraphDecoderOutput(
            op_logits=op_logits if op_logits is not None else _ids_to_logits(op_ids, action_dim),
            material_logits=_ids_to_logits(material_ids, material_dim),
            condition_logits=_ids_to_logits(condition_ids, condition_dim),
            quantity_gate_logits=slot_output.quantity_gate_logits,
            unit_logits=slot_output.unit_logits,
            quantity_values=slot_output.quantity_values,
            condition_values=slot_output.condition_values,
            numeric_candidate_logits=(
                slot_output.numeric_candidate_logits
                if slot_output.numeric_candidate_logits is not None
                else None
            ),
        ),
        structure_logits=output.structure_logits,
        skeleton_op_logits=output.skeleton_op_logits,
        skeleton_structure_logits=output.skeleton_structure_logits,
    )


def _ids_to_logits(ids: torch.Tensor, num_classes: int) -> torch.Tensor:
    logits = torch.full(
        (*ids.shape, num_classes),
        -20.0,
        dtype=torch.float32,
        device=ids.device,
    )
    logits.scatter_(-1, ids.long().clamp(0, num_classes - 1).unsqueeze(-1), 20.0)
    return logits


def _select_ids(
    logits: torch.Tensor,
    *,
    sample_mode: str,
    temperature: float,
) -> torch.Tensor:
    if sample_mode == "argmax":
        return logits.argmax(dim=-1)
    scaled = logits / max(float(temperature), 1e-6)
    probs = torch.softmax(scaled, dim=-1)
    return torch.distributions.Categorical(probs=probs).sample()


def _bincount_distribution(
    values: torch.Tensor,
    size: int,
    *,
    zero_ids: list[int] | None = None,
    smoothing: float = 1e-3,
) -> torch.Tensor:
    counts = torch.full((size,), smoothing, dtype=torch.float32, device=values.device)
    if values.numel() > 0:
        counts += torch.bincount(values.reshape(-1), minlength=size).float()
    for zero_id in zero_ids or []:
        counts[int(zero_id)] = 0.0
    if counts.sum() <= 0:
        counts.fill_(1.0)
        for zero_id in zero_ids or []:
            counts[int(zero_id)] = 0.0
    return counts / counts.sum().clamp_min(1e-12)
