"""Direct graph encoder-decoder backend for ReactGDiff."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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

BRANCH_OPS = {"FILTER", "PHASESEPARATION", "COLLECTLAYER", "EXTRACT", "PARTITION"}
WORKUP_OPS = {
    "CONCENTRATE",
    "FILTER",
    "WASH",
    "DRYSOLUTION",
    "DRYSOLID",
    "EXTRACT",
    "PARTITION",
    "PHASESEPARATION",
    "COLLECTLAYER",
    "RECRYSTALLIZE",
    "TRITURATE",
}
STRUCTURE_TARGET_DIM = 6


@dataclass(slots=True)
class GraphDecoderOutput:
    op_logits: torch.Tensor
    material_logits: torch.Tensor
    condition_logits: torch.Tensor
    quantity_gate_logits: torch.Tensor
    unit_logits: torch.Tensor
    quantity_values: torch.Tensor
    condition_values: torch.Tensor
    numeric_candidate_logits: torch.Tensor | None = None


class GraphTargetEncoder(nn.Module):
    """Encode supervised process-graph slots into a latent graph vector."""

    def __init__(
        self,
        *,
        action_dim: int,
        material_dim: int,
        condition_dim: int,
        unit_dim: int,
        max_material_slots: int,
        hidden_dim: int,
        latent_dim: int,
    ) -> None:
        super().__init__()
        embed_dim = max(16, hidden_dim // 4)
        self.max_material_slots = max_material_slots
        self.operation_embedding = nn.Embedding(action_dim, embed_dim, padding_idx=0)
        self.material_embedding = nn.Embedding(material_dim, embed_dim)
        self.condition_embedding = nn.Embedding(condition_dim, embed_dim)
        self.unit_embedding = nn.Embedding(unit_dim, embed_dim)
        self.quantity_gate_embedding = nn.Embedding(2, embed_dim)
        self.numeric_projection = nn.Linear(max_material_slots + 2, embed_dim)
        self.gru = nn.GRU(embed_dim * 6, hidden_dim, batch_first=True)
        self.to_latent = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(
        self,
        op_ids: torch.Tensor,
        material_ids: torch.Tensor,
        condition_ids: torch.Tensor,
        quantity_gate_ids: torch.Tensor,
        unit_ids: torch.Tensor,
        quantity_values: torch.Tensor,
        condition_values: torch.Tensor,
    ) -> torch.Tensor:
        material_features = self.material_embedding(material_ids).mean(dim=2)
        unit_features = self.unit_embedding(unit_ids).mean(dim=2)
        quantity_gate_features = self.quantity_gate_embedding(quantity_gate_ids).mean(dim=2)
        numeric_features = self.numeric_projection(
            torch.cat((quantity_values, condition_values), dim=-1)
        )
        step_features = torch.cat(
            (
                self.operation_embedding(op_ids),
                material_features,
                self.condition_embedding(condition_ids),
                unit_features,
                quantity_gate_features,
                numeric_features,
            ),
            dim=-1,
        )
        _, hidden = self.gru(step_features)
        return self.to_latent(hidden[-1])


class GraphSlotDecoder(nn.Module):
    """Decode a latent graph vector into operation nodes and typed slots."""

    def __init__(
        self,
        *,
        condition_dim: int,
        latent_dim: int,
        hidden_dim: int,
        max_steps: int,
        max_material_slots: int,
        action_dim: int,
        material_dim: int,
        condition_slot_dim: int,
        unit_dim: int,
        graph_backbone: str = "dit",
        dit_depth: int = 4,
        dit_heads: int = 8,
    ) -> None:
        super().__init__()
        if graph_backbone not in {"dit", "mlp"}:
            raise ValueError(f"Unsupported graph_backbone: {graph_backbone}")
        self.max_steps = max_steps
        self.max_material_slots = max_material_slots
        self.graph_backbone = graph_backbone
        self.dit_depth = dit_depth
        self.dit_heads = dit_heads
        self.position_embedding = nn.Embedding(max_steps, hidden_dim)
        if graph_backbone == "mlp":
            self.context = nn.Sequential(
                nn.Linear(condition_dim + latent_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
            )
            self.step_mixer = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
            )
        else:
            self.conditioning = nn.Sequential(
                nn.Linear(condition_dim + latent_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.token_seed = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.dit_blocks = nn.ModuleList(
                DiTBlock(hidden_dim, dit_heads) for _ in range(dit_depth)
            )
            self.final_norm = nn.LayerNorm(hidden_dim)
        self.operation_head = nn.Linear(hidden_dim, action_dim)
        self.material_head = nn.Linear(hidden_dim, max_material_slots * material_dim)
        self.condition_head = nn.Linear(hidden_dim, condition_slot_dim)
        self.quantity_gate_head = nn.Linear(hidden_dim, max_material_slots * 2)
        self.unit_head = nn.Linear(hidden_dim, max_material_slots * unit_dim)
        self.quantity_value_head = nn.Linear(hidden_dim, max_material_slots)
        self.condition_value_head = nn.Linear(hidden_dim, 2)
        self.material_dim = material_dim
        self.unit_dim = unit_dim

    def forward(self, condition: torch.Tensor, latent: torch.Tensor) -> GraphDecoderOutput:
        batch_size = condition.size(0)
        positions = torch.arange(self.max_steps, device=condition.device)
        position = self.position_embedding(positions).unsqueeze(0).expand(batch_size, -1, -1)
        if self.graph_backbone == "mlp":
            shared = self.context(torch.cat((condition, latent), dim=-1))
            shared_steps = shared.unsqueeze(1).expand(-1, self.max_steps, -1)
            hidden = self.step_mixer(torch.cat((shared_steps, position), dim=-1))
        else:
            conditioning = self.conditioning(torch.cat((condition, latent), dim=-1))
            hidden = self.token_seed(position + conditioning.unsqueeze(1))
            for block in self.dit_blocks:
                hidden = block(hidden, conditioning)
            hidden = self.final_norm(hidden)
        batch_size = hidden.size(0)
        return GraphDecoderOutput(
            op_logits=self.operation_head(hidden),
            material_logits=self.material_head(hidden).view(
                batch_size,
                self.max_steps,
                self.max_material_slots,
                self.material_dim,
            ),
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
        )


class DirectGraphEncoderDecoder(nn.Module):
    """Conditioned graph generator with a supervised graph encoder."""

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
        hidden_dim: int = 256,
        latent_dim: int = 128,
        graph_backbone: str = "dit",
        dit_depth: int = 4,
        dit_heads: int = 8,
        structure_target_dim: int = STRUCTURE_TARGET_DIM,
    ) -> None:
        super().__init__()
        if graph_backbone not in {"dit", "mlp"}:
            raise ValueError(f"Unsupported graph_backbone: {graph_backbone}")
        self.condition_dim = condition_dim
        self.action_dim = action_dim
        self.material_dim = material_dim
        self.condition_slot_dim = condition_slot_dim
        self.unit_dim = unit_dim
        self.quantity_dim = unit_dim
        self.max_steps = max_steps
        self.max_material_slots = max_material_slots
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.graph_backbone = graph_backbone
        self.dit_depth = dit_depth
        self.dit_heads = dit_heads
        self.structure_target_dim = structure_target_dim
        self.encoder = GraphTargetEncoder(
            action_dim=action_dim,
            material_dim=material_dim,
            condition_dim=condition_slot_dim,
            unit_dim=unit_dim,
            max_material_slots=max_material_slots,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
        )
        self.condition_prior = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = GraphSlotDecoder(
            condition_dim=condition_dim,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            max_steps=max_steps,
            max_material_slots=max_material_slots,
            action_dim=action_dim,
            material_dim=material_dim,
            condition_slot_dim=condition_slot_dim,
            unit_dim=unit_dim,
            graph_backbone=graph_backbone,
            dit_depth=dit_depth,
            dit_heads=dit_heads,
        )
        self.structure_head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, structure_target_dim),
        )

    def encode_graph(
        self,
        op_ids: torch.Tensor,
        material_ids: torch.Tensor,
        condition_ids: torch.Tensor,
        quantity_gate_ids: torch.Tensor,
        unit_ids: torch.Tensor,
        quantity_values: torch.Tensor,
        condition_values: torch.Tensor,
    ) -> torch.Tensor:
        return self.encoder(
            op_ids,
            material_ids,
            condition_ids,
            quantity_gate_ids,
            unit_ids,
            quantity_values,
            condition_values,
        )

    def prior(self, condition: torch.Tensor) -> torch.Tensor:
        return self.condition_prior(condition)

    def decode(self, condition: torch.Tensor, latent: torch.Tensor) -> GraphDecoderOutput:
        return self.decoder(condition, latent)

    def predict_structure(self, latent: torch.Tensor) -> torch.Tensor:
        return self.structure_head(latent)

    def forward(
        self,
        condition: torch.Tensor,
        op_ids: torch.Tensor,
        material_ids: torch.Tensor,
        condition_ids: torch.Tensor,
        quantity_gate_ids: torch.Tensor,
        unit_ids: torch.Tensor,
        quantity_values: torch.Tensor,
        condition_values: torch.Tensor,
    ) -> tuple[GraphDecoderOutput, GraphDecoderOutput, torch.Tensor, torch.Tensor]:
        graph_latent = self.encode_graph(
            op_ids,
            material_ids,
            condition_ids,
            quantity_gate_ids,
            unit_ids,
            quantity_values,
            condition_values,
        )
        prior_latent = self.prior(condition)
        return (
            self.decode(condition, prior_latent),
            self.decode(condition, graph_latent),
            prior_latent,
            graph_latent,
        )

    def config(self) -> dict[str, Any]:
        return {
            "condition_dim": self.condition_dim,
            "action_dim": self.action_dim,
            "material_dim": self.material_dim,
            "condition_slot_dim": self.condition_slot_dim,
            "unit_dim": self.unit_dim,
            "max_steps": self.max_steps,
            "max_material_slots": self.max_material_slots,
            "hidden_dim": self.hidden_dim,
            "latent_dim": self.latent_dim,
            "graph_backbone": self.graph_backbone,
            "dit_depth": self.dit_depth,
            "dit_heads": self.dit_heads,
            "structure_target_dim": self.structure_target_dim,
        }


def train_direct_graph_encoder_decoder(
    records: list[dict[str, Any]],
    *,
    condition_vectors: list[list[float]],
    codec: GraphTargetCodec,
    condition_dim: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    hidden_dim: int,
    latent_dim: int,
    graph_backbone: str,
    dit_depth: int,
    dit_heads: int,
    prior_alignment_weight: float,
    teacher_decoder_weight: float,
    latent_norm_weight: float,
    material_loss_weight: float,
    condition_loss_weight: float,
    quantity_gate_loss_weight: float,
    material_none_weight: float,
    material_present_weight: float,
    condition_none_weight: float,
    condition_present_weight: float,
    quantity_negative_weight: float,
    quantity_positive_weight: float,
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
    seed: int,
    device: str | torch.device,
    log_every: int = 1,
) -> tuple[DirectGraphEncoderDecoder, list[dict[str, float]]]:
    if not records:
        raise ValueError("records must not be empty")
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
    if condition_value_loss_weight <= 0:
        condition_values = torch.zeros_like(condition_values)
        condition_value_masks = torch.zeros_like(condition_value_masks)
    if numeric_value_clip > 0:
        clip_value = float(numeric_value_clip)
        quantity_values = quantity_values.clamp(min=-clip_value, max=clip_value)
        condition_values = condition_values.clamp(min=-clip_value, max=clip_value)
    structure_targets = build_structure_targets(
        op_ids,
        material_ids,
        condition_ids,
        quantity_value_masks,
        slot_mask,
        codec=codec,
    )
    _ensure_finite_training_tensors(
        condition=condition,
        quantity_values=quantity_values,
        condition_values=condition_values,
        structure_targets=structure_targets,
    )
    dataset = TensorDataset(
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
        structure_targets,
    )
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
    model = DirectGraphEncoderDecoder(
        condition_dim=condition_dim,
        action_dim=codec.action_dim,
        material_dim=codec.material_dim,
        condition_slot_dim=codec.condition_dim,
        unit_dim=codec.unit_dim,
        max_steps=codec.max_steps,
        max_material_slots=codec.max_material_slots,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        graph_backbone=graph_backbone,
        dit_depth=dit_depth,
        dit_heads=dit_heads,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    history: list[dict[str, float]] = []
    log_every = max(int(log_every), 0)

    for epoch in range(1, epochs + 1):
        model.train()
        totals = {
            "loss": 0.0,
            "prior_graph_loss": 0.0,
            "teacher_graph_loss": 0.0,
            "structure_loss": 0.0,
            "latent_alignment_loss": 0.0,
            "latent_norm_loss": 0.0,
            "grad_norm": 0.0,
            "prior_latent_norm": 0.0,
            "graph_latent_norm": 0.0,
        }
        total_items = 0
        for batch in loader:
            (
                batch_condition,
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
                batch_structure_target,
            ) = batch
            prior_output, teacher_output, prior_latent, graph_latent = model(
                batch_condition,
                batch_op,
                batch_material,
                batch_condition_id,
                batch_quantity_gate,
                batch_unit,
                batch_quantity_value,
                batch_condition_value,
            )
            prior_graph_loss = graph_slot_loss(
                prior_output,
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
                quantity_gate_loss_weight=quantity_gate_loss_weight,
                material_none_weight=material_none_weight,
                material_present_weight=material_present_weight,
                condition_none_weight=condition_none_weight,
                condition_present_weight=condition_present_weight,
                quantity_negative_weight=quantity_negative_weight,
                quantity_positive_weight=quantity_positive_weight,
                condition_value_loss_weight=condition_value_loss_weight,
                op_class_weights=op_class_weights,
            )
            teacher_graph_loss = graph_slot_loss(
                teacher_output,
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
                quantity_gate_loss_weight=quantity_gate_loss_weight,
                material_none_weight=material_none_weight,
                material_present_weight=material_present_weight,
                condition_none_weight=condition_none_weight,
                condition_present_weight=condition_present_weight,
                quantity_negative_weight=quantity_negative_weight,
                quantity_positive_weight=quantity_positive_weight,
                condition_value_loss_weight=condition_value_loss_weight,
                op_class_weights=op_class_weights,
            )
            prior_structure_loss = F.mse_loss(
                torch.sigmoid(model.predict_structure(prior_latent)),
                batch_structure_target,
            )
            teacher_structure_loss = F.mse_loss(
                torch.sigmoid(model.predict_structure(graph_latent)),
                batch_structure_target,
            )
            structure_loss = prior_structure_loss + teacher_decoder_weight * teacher_structure_loss
            latent_alignment_loss = (
                1.0
                - F.cosine_similarity(prior_latent, graph_latent.detach(), dim=-1).mean()
            )
            latent_norm_loss = prior_latent.pow(2).mean() + graph_latent.pow(2).mean()
            loss = (
                prior_graph_loss
                + teacher_decoder_weight * teacher_graph_loss
                + structure_loss_weight * structure_loss
                + prior_alignment_weight * latent_alignment_loss
                + latent_norm_weight * latent_norm_loss
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    "Non-finite training loss before backward "
                    f"(epoch={epoch}, prior={float(prior_graph_loss.detach())}, "
                    f"teacher={float(teacher_graph_loss.detach())}, "
                    f"struct={float(structure_loss.detach())}, "
                    f"align={float(latent_alignment_loss.detach())}, "
                    f"norm={float(latent_norm_loss.detach())})."
                )
            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=float(gradient_clip_norm),
                error_if_nonfinite=False,
            ) if gradient_clip_norm > 0 else torch.tensor(0.0, device=batch_condition.device)
            if not torch.isfinite(grad_norm):
                raise FloatingPointError(f"Non-finite gradient norm at epoch {epoch}.")
            optimizer.step()

            batch_size_now = batch_condition.size(0)
            totals["loss"] += float(loss.detach()) * batch_size_now
            totals["prior_graph_loss"] += float(prior_graph_loss.detach()) * batch_size_now
            totals["teacher_graph_loss"] += float(teacher_graph_loss.detach()) * batch_size_now
            totals["structure_loss"] += float(structure_loss.detach()) * batch_size_now
            totals["latent_alignment_loss"] += float(latent_alignment_loss.detach()) * batch_size_now
            totals["latent_norm_loss"] += float(latent_norm_loss.detach()) * batch_size_now
            totals["grad_norm"] += float(grad_norm.detach()) * batch_size_now
            totals["prior_latent_norm"] += float(prior_latent.detach().norm(dim=-1).mean()) * batch_size_now
            totals["graph_latent_norm"] += float(graph_latent.detach().norm(dim=-1).mean()) * batch_size_now
            total_items += batch_size_now
        epoch_metrics = {
            "epoch": float(epoch),
            **{key: value / max(total_items, 1) for key, value in totals.items()},
        }
        history.append(epoch_metrics)
        if log_every and (epoch == 1 or epoch == epochs or epoch % log_every == 0):
            print(
                "[epoch "
                f"{epoch:03d}/{epochs:03d}] "
                f"loss={epoch_metrics['loss']:.4f} "
                f"prior={epoch_metrics['prior_graph_loss']:.4f} "
                f"teacher={epoch_metrics['teacher_graph_loss']:.4f} "
                f"struct={epoch_metrics['structure_loss']:.4f} "
                f"align={epoch_metrics['latent_alignment_loss']:.4f} "
                f"norm={epoch_metrics['latent_norm_loss']:.4f} "
                f"grad={epoch_metrics['grad_norm']:.2f} "
                f"|z_p|={epoch_metrics['prior_latent_norm']:.2f} "
                f"|z_g|={epoch_metrics['graph_latent_norm']:.2f}",
                flush=True,
            )
    return model, history


def graph_slot_loss(
    output: GraphDecoderOutput,
    op_ids: torch.Tensor,
    material_ids: torch.Tensor,
    condition_ids: torch.Tensor,
    quantity_gate_ids: torch.Tensor,
    unit_ids: torch.Tensor,
    quantity_values: torch.Tensor,
    quantity_value_masks: torch.Tensor,
    condition_values: torch.Tensor,
    condition_value_masks: torch.Tensor,
    slot_mask: torch.Tensor,
    *,
    pad_id: int,
    material_loss_weight: float = 0.8,
    condition_loss_weight: float = 0.9,
    quantity_gate_loss_weight: float = 0.7,
    unit_loss_weight: float = 0.7,
    quantity_value_loss_weight: float = 0.5,
    condition_value_loss_weight: float = 0.0,
    material_none_weight: float = 0.35,
    material_present_weight: float = 1.6,
    condition_none_weight: float = 0.45,
    condition_present_weight: float = 2.5,
    quantity_negative_weight: float = 0.9,
    quantity_positive_weight: float = 1.0,
    op_class_weights: torch.Tensor | None = None,
    ignore_pad_operations: bool = True,
    operation_loss_weight: float = 1.0,
    numeric_candidate_ids: torch.Tensor | None = None,
    numeric_candidate_loss_weight: float = 0.0,
) -> torch.Tensor:
    flat_op_logits = output.op_logits.reshape(-1, output.op_logits.size(-1))
    flat_op_ids = op_ids.reshape(-1)
    if ignore_pad_operations:
        op_loss = F.cross_entropy(
            flat_op_logits,
            flat_op_ids,
            ignore_index=pad_id,
            weight=op_class_weights,
        )
    else:
        op_loss = F.cross_entropy(flat_op_logits, flat_op_ids, weight=op_class_weights)
    active_steps = slot_mask.bool()
    active_material_slots = active_steps.unsqueeze(-1).expand_as(material_ids)
    quantity_present = quantity_value_masks.bool()
    condition_present = condition_value_masks.bool()

    material_loss = _weighted_none_present_cross_entropy(
        output.material_logits,
        material_ids,
        active_material_slots,
        none_id=0,
        none_weight=material_none_weight,
        present_weight=material_present_weight,
    )
    condition_loss = _weighted_none_present_cross_entropy(
        output.condition_logits,
        condition_ids,
        active_steps,
        none_id=0,
        none_weight=condition_none_weight,
        present_weight=condition_present_weight,
    )
    quantity_gate_loss = _weighted_quantity_gate_loss(
        output.quantity_gate_logits,
        quantity_gate_ids,
        active_material_slots,
        negative_weight=quantity_negative_weight,
        positive_weight=quantity_positive_weight,
    )
    unit_loss = _masked_cross_entropy(output.unit_logits, unit_ids, quantity_present)
    quantity_value_loss = _masked_mse(output.quantity_values, quantity_values, quantity_present)
    condition_value_loss = _masked_mse(output.condition_values, condition_values, condition_present)
    numeric_candidate_loss = (
        _masked_cross_entropy(
            output.numeric_candidate_logits,
            numeric_candidate_ids,
            quantity_present,
        )
        if output.numeric_candidate_logits is not None and numeric_candidate_ids is not None
        else op_loss.detach() * 0.0
    )
    weighted_components = {
        "operation": operation_loss_weight * op_loss,
        "material": material_loss_weight * material_loss,
        "condition": condition_loss_weight * condition_loss,
        "quantity_gate": quantity_gate_loss_weight * quantity_gate_loss,
        "unit": unit_loss_weight * unit_loss,
        "quantity_value": quantity_value_loss_weight * quantity_value_loss,
        "condition_value": condition_value_loss_weight * condition_value_loss,
        "numeric_candidate": numeric_candidate_loss_weight * numeric_candidate_loss,
    }
    loss = sum(weighted_components.values())
    if not torch.isfinite(loss):
        component_text = ", ".join(
            f"{name}={float(value.detach())}"
            for name, value in weighted_components.items()
        )
        raise FloatingPointError(f"Non-finite graph slot loss: {component_text}")
    return loss


def build_structure_targets(
    op_ids: torch.Tensor,
    material_ids: torch.Tensor,
    condition_ids: torch.Tensor,
    quantity_value_masks: torch.Tensor,
    slot_mask: torch.Tensor,
    *,
    codec: GraphTargetCodec,
) -> torch.Tensor:
    """Compact graph-shape target used to keep the prior from collapsing to templates."""

    active_steps = slot_mask.float()
    active_step_mask = active_steps.bool()
    active_material_slots = active_step_mask.unsqueeze(-1).expand_as(material_ids)
    max_action_steps = max(codec.max_steps - 1, 1)
    max_material_slots = max(max_action_steps * codec.max_material_slots, 1)

    branch_lookup = torch.zeros(codec.action_dim, dtype=torch.float32, device=op_ids.device)
    workup_lookup = torch.zeros(codec.action_dim, dtype=torch.float32, device=op_ids.device)
    for action_id, action in enumerate(codec.action_vocab):
        if action in BRANCH_OPS:
            branch_lookup[action_id] = 1.0
        if action in WORKUP_OPS:
            workup_lookup[action_id] = 1.0

    step_count = active_steps.sum(dim=-1) / max_action_steps
    material_count = ((material_ids != 0) & active_material_slots).float().sum(dim=(1, 2))
    material_count = material_count / max_material_slots
    condition_count = ((condition_ids != 0) & active_step_mask).float().sum(dim=-1)
    condition_count = condition_count / max_action_steps
    quantity_count = (
        (quantity_value_masks > 0).float()
        * active_step_mask.unsqueeze(-1).float()
    ).sum(dim=(1, 2))
    quantity_count = quantity_count / max_material_slots
    branch_count = (branch_lookup[op_ids] * active_steps).sum(dim=-1) / max_action_steps
    workup_count = (workup_lookup[op_ids] * active_steps).sum(dim=-1) / max_action_steps

    return torch.stack(
        (
            step_count,
            material_count,
            condition_count,
            quantity_count,
            branch_count,
            workup_count,
        ),
        dim=-1,
    ).clamp_(0.0, 1.0)


def build_balanced_sample_weights(
    records: list[dict[str, Any]],
    *,
    max_weight: float,
    skeleton_weight_max: float,
) -> torch.Tensor:
    if not records:
        return torch.empty(0, dtype=torch.double)
    skeletons = [_record_skeleton(record) for record in records]
    skeleton_counts = Counter(skeletons)
    most_common_count = max(skeleton_counts.values(), default=1)
    weight_cap = max(float(max_weight), 1.0)
    skeleton_cap = max(float(skeleton_weight_max), 1.0)

    weights: list[float] = []
    for record, skeleton in zip(records, skeletons, strict=True):
        weight = 1.0
        buckets = record.get("_buckets") or {}
        features = record.get("_features") or {}
        scale = str(buckets.get("scale") or "")
        special = buckets.get("special") or {}

        if scale == "medium":
            weight += 0.25
        elif scale == "large":
            weight += 1.0
        if _special_enabled(special, "branch_workup") or (
            _feature_count(features, "branch_op_count") > 0
            and _feature_count(features, "workup_op_count") >= 3
        ):
            weight += 1.0
        if _special_enabled(special, "multi_reference") or _feature_count(
            features, "unique_material_ref_count"
        ) >= 5:
            weight += 0.8
        if _special_enabled(special, "condition_heavy") or _feature_count(
            features, "condition_ref_count"
        ) >= 4:
            weight += 0.7
        if _special_enabled(special, "numeric_heavy") or _feature_count(
            features, "quantity_count"
        ) >= 5:
            weight += 0.6
        if _special_enabled(special, "hard_numeric_condition"):
            weight += 0.3
        if _special_enabled(special, "complex_overall") or _feature_count(
            features, "complexity_score"
        ) >= 20:
            weight += 0.2

        skeleton_count = max(skeleton_counts[skeleton], 1)
        rarity = min((most_common_count / skeleton_count) ** 0.5, skeleton_cap)
        weights.append(min(weight * rarity, weight_cap))
    return torch.tensor(weights, dtype=torch.double)


def build_operation_class_weights(
    records: list[dict[str, Any]],
    *,
    codec: GraphTargetCodec,
    device: str | torch.device,
    weighting: str,
    alpha: float,
    max_weight: float,
) -> torch.Tensor | None:
    if weighting == "none" or alpha <= 0:
        return None
    if weighting != "balanced":
        raise ValueError(f"Unsupported operation_weighting: {weighting}")

    action_to_id = {action: idx for idx, action in enumerate(codec.action_vocab)}
    counts: Counter[int] = Counter()
    for record in records:
        steps = parse_action_sequence(str(record.get("actions", "")))[: codec.max_steps - 1]
        for step in steps:
            counts[action_to_id.get(step.operation_type, codec.eos_id)] += 1
        counts[codec.eos_id] += 1
    if not counts:
        return None

    weight_cap = max(float(max_weight), 1.0)
    most_common_count = max(counts.values(), default=1)
    weights = torch.ones(codec.action_dim, dtype=torch.float32, device=device)
    for action_id, count in counts.items():
        if action_id == codec.pad_id:
            continue
        weights[action_id] = min((most_common_count / max(count, 1)) ** alpha, weight_cap)

    counted_ids = torch.tensor(
        [action_id for action_id in counts if action_id != codec.pad_id],
        dtype=torch.long,
        device=device,
    )
    if counted_ids.numel() > 0:
        weights[counted_ids] = weights[counted_ids] / weights[counted_ids].mean().clamp_min(1e-6)
        weights[counted_ids] = weights[counted_ids].clamp(max=weight_cap)
    weights[codec.pad_id] = 1.0
    return weights


def _record_skeleton(record: dict[str, Any]) -> str:
    steps = parse_action_sequence(str(record.get("actions", "")))
    if not steps:
        return "<EMPTY>"
    return " ; ".join(step.operation_type for step in steps)


def _special_enabled(special: Any, name: str) -> bool:
    return isinstance(special, dict) and bool(special.get(name))


def _feature_count(features: Any, name: str) -> float:
    if not isinstance(features, dict):
        return 0.0
    try:
        return float(features.get(name) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _ensure_finite_training_tensors(**named_tensors: torch.Tensor) -> None:
    for name, tensor in named_tensors.items():
        if torch.isfinite(tensor).all():
            continue
        finite = tensor[torch.isfinite(tensor)]
        if finite.numel() == 0:
            detail = "all values are non-finite"
        else:
            detail = f"finite_min={float(finite.min())}, finite_max={float(finite.max())}"
        raise ValueError(f"Non-finite values found in training tensor {name}: {detail}")


def _masked_cross_entropy(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    flat_logits = logits.reshape(-1, logits.size(-1))
    flat_targets = targets.reshape(-1)
    flat_mask = mask.reshape(-1).bool()
    if not flat_mask.any():
        return flat_logits.sum() * 0.0
    return F.cross_entropy(flat_logits[flat_mask], flat_targets[flat_mask])


def _weighted_quantity_gate_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    *,
    negative_weight: float,
    positive_weight: float,
) -> torch.Tensor:
    flat_logits = logits.reshape(-1, logits.size(-1))
    flat_targets = targets.reshape(-1)
    flat_mask = mask.reshape(-1).bool()
    if not flat_mask.any():
        return flat_logits.sum() * 0.0
    weights = torch.tensor(
        [negative_weight, positive_weight],
        dtype=flat_logits.dtype,
        device=flat_logits.device,
    )
    return F.cross_entropy(flat_logits[flat_mask], flat_targets[flat_mask], weight=weights)


def _weighted_none_present_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    *,
    none_id: int,
    none_weight: float,
    present_weight: float,
) -> torch.Tensor:
    flat_logits = logits.reshape(-1, logits.size(-1))
    flat_targets = targets.reshape(-1)
    flat_mask = mask.reshape(-1).bool()
    if not flat_mask.any():
        return flat_logits.sum() * 0.0
    weights = torch.ones(logits.size(-1), dtype=flat_logits.dtype, device=flat_logits.device)
    weights *= present_weight
    weights[none_id] = none_weight
    return F.cross_entropy(flat_logits[flat_mask], flat_targets[flat_mask], weight=weights)


def _masked_mse(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.bool()
    if not mask.any():
        return prediction.sum() * 0.0
    return F.mse_loss(prediction[mask], target[mask])


@torch.no_grad()
def predict_direct_graph_records(
    model: DirectGraphEncoderDecoder,
    codec: GraphTargetCodec,
    records: list[dict[str, Any]],
    *,
    condition_vectors: list[list[float]],
    argument_filler: ArgumentTextFiller | None = None,
    argument_text_codec: ArgumentTextCodec | None = None,
    argument_filler_target: str | None = None,
    argument_condition_on_quantity_units: bool | None = None,
    include_generated_graph: bool = False,
    quantity_gate_threshold: float = 0.65,
    condition_probability_threshold: float = 0.35,
    use_structure_length: bool = False,
    min_structure_steps: int = 2,
    device: str | torch.device,
) -> list[dict[str, Any]]:
    model = model.to(device)
    model.eval()
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
    for record, condition_vector in zip(records, condition_vectors, strict=True):
        condition = torch.tensor([condition_vector], dtype=torch.float32, device=device)
        latent = model.prior(condition)
        output = model.decode(condition, latent)
        forced_step_count = None
        if use_structure_length:
            structure = torch.sigmoid(model.predict_structure(latent)).squeeze(0)
            predicted_steps = int(round(float(structure[0]) * max(codec.max_steps - 1, 1)))
            forced_step_count = max(int(min_structure_steps), min(predicted_steps, codec.max_steps))
        slots = codec.decode_logits(
            output.op_logits.squeeze(0).cpu(),
            output.material_logits.squeeze(0).cpu(),
            output.condition_logits.squeeze(0).cpu(),
            output.quantity_gate_logits.squeeze(0).cpu(),
            output.unit_logits.squeeze(0).cpu(),
            output.quantity_values.squeeze(0).cpu(),
            output.condition_values.squeeze(0).cpu(),
            quantity_gate_threshold=quantity_gate_threshold,
            condition_probability_threshold=condition_probability_threshold,
            forced_step_count=forced_step_count,
        )
        if argument_filler is not None and argument_text_codec is not None:
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
                    condition,
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
            "input_vector": condition_vector,
            "reference_actions": reference,
            "predicted_actions": prediction,
            "decoded_slots": slots,
            "text_gap": gap,
            "levenshtein_similarity": 1.0 - gap,
            "decoder_backend": "direct_graph_encoder_decoder",
        }
        if include_generated_graph:
            row["generated_graph"] = graph
        rows.append(row)
    return rows


def save_graph_checkpoint(
    path: str | Path,
    *,
    model: DirectGraphEncoderDecoder,
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
        "checkpoint_type": "direct_graph_encoder_decoder",
        "model_state": model.cpu().state_dict(),
        "model_config": model.config(),
        "codec": codec.to_dict(),
        "condition_featurizer": condition_featurizer,
        "history": history,
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


def load_graph_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> tuple[DirectGraphEncoderDecoder, GraphTargetCodec, dict[str, Any], list[dict[str, float]]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    model_config = dict(payload["model_config"])
    if "unit_dim" not in model_config and "quantity_dim" in model_config:
        model_config["unit_dim"] = model_config.pop("quantity_dim")
    else:
        model_config.pop("quantity_dim", None)
    model_config.setdefault("max_material_slots", 1)
    model_config.setdefault("graph_backbone", "mlp")
    model_config.setdefault("dit_depth", 0)
    model_config.setdefault("dit_heads", 1)
    model = DirectGraphEncoderDecoder(**model_config).to(device)
    model.load_state_dict(payload["model_state"], strict=False)
    codec = GraphTargetCodec.from_dict(payload["codec"])
    return model, codec, payload["condition_featurizer"], payload.get("history", [])


def load_argument_filler_from_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> tuple[ArgumentTextFiller | None, ArgumentTextCodec | None, list[dict[str, float]]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    if "argument_filler_state" not in payload:
        return None, None, []
    text_codec = ArgumentTextCodec.from_dict(payload["argument_text_codec"])
    filler = ArgumentTextFiller(**payload["argument_filler_config"]).to(device)
    filler.load_state_dict(payload["argument_filler_state"], strict=False)
    filler.argument_filler_target = str(payload.get("argument_filler_target", "all"))
    filler.condition_on_quantity_units = bool(payload.get("argument_condition_on_quantity_units", True))
    return filler, text_codec, payload.get("argument_history", [])
