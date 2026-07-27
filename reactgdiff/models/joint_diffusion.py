"""Small-scale joint diffusion procedure generator.

This module implements a compact experimental loop for the current repository:

``4D reaction input -> joint denoising latents -> graph-memory decode -> text``.

It is a deliberately lightweight surrogate for the full ReactGDiff method. The
discrete chain predicts an operation-skeleton proxy, the continuous chain
predicts typed numeric/reference proxies, and decoding selects the nearest
training graph before deterministic graph-to-sequence compilation.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from reactgdiff.compile.graph_to_sequence import decompile_graph_to_sequence
from reactgdiff.data.action_parser import KNOWN_OPENEXP_ACTIONS
from reactgdiff.data.graph_builder import build_process_graph
from reactgdiff.data.numeric_evidence import numeric_condition_field
from reactgdiff.eval.lev import text_gap
from reactgdiff.models.continuous_attribute_diffusion import ContinuousAttributeDenoiser
from reactgdiff.models.coupling_module import (
    DiscreteContinuousCoupling,
    sinusoidal_timestep_embedding,
)
from reactgdiff.models.dit import DiTBlock
from reactgdiff.models.discrete_graph_diffusion import DiscreteGraphDenoiser
from reactgdiff.utils.io import read_jsonl

DEFAULT_ATTRIBUTE_KEYS = (
    "action_count",
    "material_ref_count",
    "unique_material_ref_count",
    "duration_ref_count",
    "temperature_ref_count",
    "condition_ref_count",
    "quantity_count",
    "quantity_component_count",
)

MINIMAL_RECORD_FIELDS = (
    "index",
    "REACTANT",
    "PRODUCT",
    "CATALYST",
    "SOLVENT",
    "actions",
    "source",
    "extracted_molecules",
    "extracted_duration",
    "extracted_temperature",
    "molecules",
    "score",
)


@dataclass(slots=True)
class DiffusionSchedule:
    """Linear-beta DDPM schedule for the latent MVP experiment."""

    betas: torch.Tensor

    @classmethod
    def linear(
        cls,
        *,
        num_steps: int,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        device: str | torch.device = "cpu",
    ) -> "DiffusionSchedule":
        if num_steps <= 0:
            raise ValueError("num_steps must be positive")
        betas = torch.linspace(beta_start, beta_end, num_steps, device=device)
        return cls(betas=betas)

    @property
    def num_steps(self) -> int:
        return int(self.betas.numel())

    @property
    def alpha_bars(self) -> torch.Tensor:
        return torch.cumprod(1.0 - self.betas, dim=0)

    def to(self, device: str | torch.device) -> "DiffusionSchedule":
        return DiffusionSchedule(betas=self.betas.to(device))

    def q_sample(
        self,
        clean: torch.Tensor,
        timesteps: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        alpha_bar = self.alpha_bars[timesteps].unsqueeze(-1)
        return alpha_bar.sqrt() * clean + (1.0 - alpha_bar).sqrt() * noise


@dataclass(slots=True)
class ReactGDiffFeaturizer:
    """Feature transforms for four reaction input fields and target proxies."""

    action_vocab: list[str]
    attribute_keys: list[str]
    attribute_mean: list[float]
    attribute_std: list[float]
    condition_encoding: str = "scalar_hash"
    field_dim: int = 1
    ngram_min: int = 2
    ngram_max: int = 5
    include_numeric_evidence: bool = False
    numeric_evidence_include_source: bool = False
    numeric_evidence_quantity_only: bool = False

    @classmethod
    def fit(
        cls,
        records: Iterable[dict[str, Any]],
        *,
        action_vocab: Iterable[str] | None = None,
        attribute_keys: Iterable[str] = DEFAULT_ATTRIBUTE_KEYS,
        condition_encoding: str = "field_hash",
        field_dim: int = 128,
        ngram_min: int = 2,
        ngram_max: int = 5,
        include_numeric_evidence: bool | None = None,
        numeric_evidence_include_source: bool = False,
        numeric_evidence_quantity_only: bool = True,
    ) -> "ReactGDiffFeaturizer":
        if condition_encoding not in {"scalar_hash", "field_hash", "reactxt_hash"}:
            raise ValueError(f"Unsupported condition_encoding: {condition_encoding}")
        if condition_encoding in {"field_hash", "reactxt_hash"} and field_dim < 8:
            raise ValueError("field_dim must be at least 8 for hash-based encodings")
        if include_numeric_evidence is None:
            include_numeric_evidence = condition_encoding == "reactxt_hash"
        records = list(records)
        keys = list(attribute_keys)
        raw = torch.tensor(
            [[float(_feature_value(record, key)) for key in keys] for record in records],
            dtype=torch.float32,
        )
        if raw.numel() == 0:
            mean = torch.zeros(len(keys), dtype=torch.float32)
            std = torch.ones(len(keys), dtype=torch.float32)
        else:
            mean = raw.mean(dim=0)
            std = raw.std(dim=0, unbiased=False).clamp_min(1e-6)
        vocab = sorted(action_vocab or KNOWN_OPENEXP_ACTIONS)
        return cls(
            action_vocab=vocab,
            attribute_keys=keys,
            attribute_mean=mean.tolist(),
            attribute_std=std.tolist(),
            condition_encoding=condition_encoding,
            field_dim=field_dim if condition_encoding in {"field_hash", "reactxt_hash"} else 1,
            ngram_min=ngram_min,
            ngram_max=ngram_max,
            include_numeric_evidence=bool(include_numeric_evidence),
            numeric_evidence_include_source=bool(numeric_evidence_include_source),
            numeric_evidence_quantity_only=bool(numeric_evidence_quantity_only),
        )

    @property
    def condition_dim(self) -> int:
        if self.condition_encoding == "reactxt_hash":
            field_count = 6 + int(self.include_numeric_evidence)
            return field_count * self.field_dim
        return 4 * self.field_dim

    @property
    def structure_dim(self) -> int:
        return len(self.action_vocab)

    @property
    def attribute_dim(self) -> int:
        return len(self.attribute_keys)

    def condition_vector(self, record: dict[str, Any]) -> list[float]:
        """Return an encoding of the four reaction-participant input fields.

        ``scalar_hash`` keeps the original four-scalar baseline. ``field_hash``
        keeps the same four input fields but encodes each field as a hashed
        character n-gram vector over the field's molecule strings. It does not
        read action text or extracted procedure conditions.

        ``reactxt_hash`` mirrors the ReactXT action prompt at the feature level:
        each molecule field includes role labels, placeholders, and SMILES, and
        two extra fields encode the provided temperature/duration lookup tables.
        It still does not read the target action sequence.
        """

        if self.condition_encoding == "scalar_hash":
            fields = (
                record.get("REACTANT") or [],
                record.get("PRODUCT") or [],
                record.get("CATALYST") or [],
                record.get("SOLVENT") or [],
            )
            return [_group_scalar(values) for values in fields]

        fields = (
            _reactxt_prompt_fields(
                record,
                include_numeric_evidence=self.include_numeric_evidence,
                numeric_evidence_include_source=self.numeric_evidence_include_source,
                numeric_evidence_quantity_only=self.numeric_evidence_quantity_only,
            )
            if self.condition_encoding == "reactxt_hash"
            else (
                record.get("REACTANT") or [],
                record.get("PRODUCT") or [],
                record.get("CATALYST") or [],
                record.get("SOLVENT") or [],
            )
        )
        vector: list[float] = []
        for field_idx, values in enumerate(fields):
            vector.extend(
                _field_hash_vector(
                    values,
                    dim=self.field_dim,
                    field_idx=field_idx,
                    ngram_min=self.ngram_min,
                    ngram_max=self.ngram_max,
                )
            )
        return vector

    def structure_vector(self, record: dict[str, Any]) -> list[float]:
        histogram = (record.get("_features") or {}).get("action_histogram") or {}
        total = max(float(sum(histogram.values())), 1.0)
        return [float(histogram.get(action, 0.0)) / total for action in self.action_vocab]

    def attribute_vector(self, record: dict[str, Any]) -> list[float]:
        raw = torch.tensor(
            [float(_feature_value(record, key)) for key in self.attribute_keys],
            dtype=torch.float32,
        )
        mean = torch.tensor(self.attribute_mean, dtype=torch.float32)
        std = torch.tensor(self.attribute_std, dtype=torch.float32)
        return ((raw - mean) / std).tolist()

    def encode_records(
        self,
        records: list[dict[str, Any]],
        *,
        device: str | torch.device = "cpu",
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        conditions = torch.tensor([self.condition_vector(record) for record in records])
        structures = torch.tensor([self.structure_vector(record) for record in records])
        attributes = torch.tensor([self.attribute_vector(record) for record in records])
        return (
            conditions.float().to(device),
            structures.float().to(device),
            attributes.float().to(device),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_vocab": self.action_vocab,
            "attribute_keys": self.attribute_keys,
            "attribute_mean": self.attribute_mean,
            "attribute_std": self.attribute_std,
            "condition_encoding": self.condition_encoding,
            "field_dim": self.field_dim,
            "ngram_min": self.ngram_min,
            "ngram_max": self.ngram_max,
            "include_numeric_evidence": self.include_numeric_evidence,
            "numeric_evidence_include_source": self.numeric_evidence_include_source,
            "numeric_evidence_quantity_only": self.numeric_evidence_quantity_only,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ReactGDiffFeaturizer":
        return cls(
            action_vocab=list(payload["action_vocab"]),
            attribute_keys=list(payload["attribute_keys"]),
            attribute_mean=list(payload["attribute_mean"]),
            attribute_std=list(payload["attribute_std"]),
            condition_encoding=str(payload.get("condition_encoding", "scalar_hash")),
            field_dim=int(payload.get("field_dim", 1)),
            ngram_min=int(payload.get("ngram_min", 2)),
            ngram_max=int(payload.get("ngram_max", 5)),
            include_numeric_evidence=bool(payload.get("include_numeric_evidence", False)),
            numeric_evidence_include_source=bool(
                payload.get("numeric_evidence_include_source", True)
            ),
            # Preserve the feature semantics of checkpoints created before the
            # quantity-only evidence field was introduced.
            numeric_evidence_quantity_only=bool(
                payload.get("numeric_evidence_quantity_only", False)
            ),
        )


class JointDiffusionProcedureModel(nn.Module):
    """Coupled denoiser over operation-skeleton and attribute proxy latents."""

    def __init__(
        self,
        *,
        condition_dim: int = 4,
        structure_dim: int,
        attribute_dim: int,
        hidden_dim: int = 128,
        time_dim: int = 32,
        context_dim: int = 64,
        base_model: str = "dit",
        dit_depth: int = 4,
        dit_heads: int = 4,
    ) -> None:
        super().__init__()
        if base_model not in {"dit", "mlp"}:
            raise ValueError(f"Unsupported base_model: {base_model}")
        self.condition_dim = condition_dim
        self.structure_dim = structure_dim
        self.attribute_dim = attribute_dim
        self.hidden_dim = hidden_dim
        self.time_dim = time_dim
        self.context_dim = context_dim
        self.base_model = base_model
        self.dit_depth = dit_depth
        self.dit_heads = dit_heads
        if base_model == "mlp":
            self.coupling = DiscreteContinuousCoupling(
                structure_dim=structure_dim,
                attribute_dim=attribute_dim,
                condition_dim=condition_dim,
                time_dim=time_dim,
                context_dim=context_dim,
                hidden_dim=hidden_dim,
            )
            self.discrete_denoiser = DiscreteGraphDenoiser(
                structure_dim=structure_dim,
                condition_dim=condition_dim,
                time_dim=time_dim,
                context_dim=context_dim,
                hidden_dim=hidden_dim,
            )
            self.attribute_denoiser = ContinuousAttributeDenoiser(
                attribute_dim=attribute_dim,
                condition_dim=condition_dim,
                time_dim=time_dim,
                context_dim=context_dim,
                hidden_dim=hidden_dim,
            )
        else:
            self.dit_denoiser = JointDiTDenoiser(
                structure_dim=structure_dim,
                attribute_dim=attribute_dim,
                condition_dim=condition_dim,
                time_dim=time_dim,
                hidden_dim=hidden_dim,
                dit_depth=dit_depth,
                dit_heads=dit_heads,
            )

    def forward(
        self,
        noisy_structure: torch.Tensor,
        noisy_attributes: torch.Tensor,
        condition: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        timestep_embedding = sinusoidal_timestep_embedding(timesteps, self.time_dim)
        if self.base_model == "dit":
            return self.dit_denoiser(
                noisy_structure,
                noisy_attributes,
                condition,
                timestep_embedding,
            )
        structure_context, attribute_context = self.coupling(
            noisy_structure,
            noisy_attributes,
            condition,
            timestep_embedding,
        )
        clean_structure = self.discrete_denoiser(
            noisy_structure,
            condition,
            timestep_embedding,
            structure_context,
        )
        clean_attributes = self.attribute_denoiser(
            noisy_attributes,
            condition,
            timestep_embedding,
            attribute_context,
        )
        return clean_structure, clean_attributes

    def config(self) -> dict[str, Any]:
        return {
            "condition_dim": self.condition_dim,
            "structure_dim": self.structure_dim,
            "attribute_dim": self.attribute_dim,
            "hidden_dim": self.hidden_dim,
            "time_dim": self.time_dim,
            "context_dim": self.context_dim,
            "base_model": self.base_model,
            "dit_depth": self.dit_depth,
            "dit_heads": self.dit_heads,
        }


class JointDiTDenoiser(nn.Module):
    """DiT denoiser for the memory-baseline joint diffusion path."""

    def __init__(
        self,
        *,
        structure_dim: int,
        attribute_dim: int,
        condition_dim: int,
        time_dim: int,
        hidden_dim: int,
        dit_depth: int,
        dit_heads: int,
    ) -> None:
        super().__init__()
        self.structure_in = nn.Linear(structure_dim, hidden_dim)
        self.attribute_in = nn.Linear(attribute_dim, hidden_dim)
        self.type_embedding = nn.Parameter(torch.zeros(2, hidden_dim))
        self.conditioning = nn.Sequential(
            nn.Linear(condition_dim + time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.blocks = nn.ModuleList(DiTBlock(hidden_dim, dit_heads) for _ in range(dit_depth))
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.structure_out = nn.Linear(hidden_dim, structure_dim)
        self.attribute_out = nn.Linear(hidden_dim, attribute_dim)

    def forward(
        self,
        noisy_structure: torch.Tensor,
        noisy_attributes: torch.Tensor,
        condition: torch.Tensor,
        timestep_embedding: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        structure_token = self.structure_in(noisy_structure) + self.type_embedding[0]
        attribute_token = self.attribute_in(noisy_attributes) + self.type_embedding[1]
        tokens = torch.stack((structure_token, attribute_token), dim=1)
        conditioning = self.conditioning(torch.cat((condition, timestep_embedding), dim=-1))
        for block in self.blocks:
            tokens = block(tokens, conditioning)
        tokens = self.final_norm(tokens)
        clean_structure = torch.sigmoid(self.structure_out(tokens[:, 0]))
        clean_attributes = self.attribute_out(tokens[:, 1])
        return clean_structure, clean_attributes


def load_split_records(path: str | Path, *, limit: int | None = None) -> list[dict[str, Any]]:
    return list(read_jsonl(path, limit=limit))


def build_candidate_memory(
    records: list[dict[str, Any]],
    featurizer: ReactGDiffFeaturizer,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for record in records:
        minimal_record = {key: record.get(key) for key in MINIMAL_RECORD_FIELDS if key in record}
        graph = build_process_graph(record)
        decoded = decompile_graph_to_sequence(graph, mode="exact")
        candidates.append(
            {
                "index": record.get("index"),
                "record": minimal_record,
                "decoded_actions": decoded,
                "condition": featurizer.condition_vector(record),
                "structure": featurizer.structure_vector(record),
                "attributes": featurizer.attribute_vector(record),
            }
        )
    return candidates


def train_joint_diffusion(
    records: list[dict[str, Any]],
    *,
    featurizer: ReactGDiffFeaturizer,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    diffusion_steps: int,
    hidden_dim: int,
    time_dim: int,
    context_dim: int,
    base_model: str,
    dit_depth: int,
    dit_heads: int,
    seed: int,
    device: str | torch.device,
    log_every: int = 1,
) -> tuple[JointDiffusionProcedureModel, DiffusionSchedule, list[dict[str, float]]]:
    if not records:
        raise ValueError("records must not be empty")
    torch.manual_seed(seed)
    conditions, structures, attributes = featurizer.encode_records(records, device=device)
    dataset = TensorDataset(conditions, structures, attributes)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    model = JointDiffusionProcedureModel(
        condition_dim=featurizer.condition_dim,
        structure_dim=featurizer.structure_dim,
        attribute_dim=featurizer.attribute_dim,
        hidden_dim=hidden_dim,
        time_dim=time_dim,
        context_dim=context_dim,
        base_model=base_model,
        dit_depth=dit_depth,
        dit_heads=dit_heads,
    ).to(device)
    schedule = DiffusionSchedule.linear(num_steps=diffusion_steps, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    history: list[dict[str, float]] = []
    log_every = max(int(log_every), 0)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_structure = 0.0
        total_attribute = 0.0
        total_items = 0
        for condition, clean_structure, clean_attributes in loader:
            batch_size_now = condition.size(0)
            timesteps = torch.randint(
                low=0,
                high=schedule.num_steps,
                size=(batch_size_now,),
                device=device,
            )
            structure_noise = torch.randn(clean_structure.shape, device=device)
            attribute_noise = torch.randn(clean_attributes.shape, device=device)
            noisy_structure = schedule.q_sample(clean_structure, timesteps, structure_noise)
            noisy_attributes = schedule.q_sample(clean_attributes, timesteps, attribute_noise)
            pred_structure, pred_attributes = model(
                noisy_structure,
                noisy_attributes,
                condition,
                timesteps,
            )
            structure_loss = F.mse_loss(pred_structure, clean_structure)
            attribute_loss = F.mse_loss(pred_attributes, clean_attributes)
            loss = structure_loss + attribute_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += float(loss.detach()) * batch_size_now
            total_structure += float(structure_loss.detach()) * batch_size_now
            total_attribute += float(attribute_loss.detach()) * batch_size_now
            total_items += batch_size_now
        epoch_metrics = {
            "epoch": float(epoch),
            "loss": total_loss / max(total_items, 1),
            "structure_loss": total_structure / max(total_items, 1),
            "attribute_loss": total_attribute / max(total_items, 1),
        }
        history.append(epoch_metrics)
        if log_every and (epoch == 1 or epoch == epochs or epoch % log_every == 0):
            print(
                "[epoch "
                f"{epoch:03d}/{epochs:03d}] "
                f"loss={epoch_metrics['loss']:.4f} "
                f"structure={epoch_metrics['structure_loss']:.4f} "
                f"attribute={epoch_metrics['attribute_loss']:.4f}",
                flush=True,
            )
    return model, schedule, history


@torch.no_grad()
def sample_latents(
    model: JointDiffusionProcedureModel,
    schedule: DiffusionSchedule,
    condition: torch.Tensor,
    *,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate one pair of clean latent proxies with deterministic DDIM steps."""

    model.eval()
    device = condition.device
    generator = torch.Generator(device=device).manual_seed(seed)
    structure = torch.randn(
        (1, model.structure_dim),
        generator=generator,
        device=device,
    )
    attributes = torch.randn(
        (1, model.attribute_dim),
        generator=generator,
        device=device,
    )
    for step in reversed(range(schedule.num_steps)):
        timesteps = torch.full((1,), step, dtype=torch.long, device=device)
        clean_structure, clean_attributes = model(structure, attributes, condition, timesteps)
        structure = clean_structure
        attributes = clean_attributes
    return structure.squeeze(0).cpu(), attributes.squeeze(0).cpu()


def decode_from_memory(
    structure: torch.Tensor,
    attributes: torch.Tensor,
    candidates: list[dict[str, Any]],
    *,
    structure_weight: float = 1.0,
    attribute_weight: float = 1.0,
) -> tuple[dict[str, Any], float]:
    if not candidates:
        raise ValueError("candidates must not be empty")
    best_candidate = candidates[0]
    best_distance = math.inf
    for candidate in candidates:
        candidate_structure = torch.tensor(candidate["structure"], dtype=torch.float32)
        candidate_attributes = torch.tensor(candidate["attributes"], dtype=torch.float32)
        distance = (
            structure_weight * F.mse_loss(structure, candidate_structure).item()
            + attribute_weight * F.mse_loss(attributes, candidate_attributes).item()
        )
        if distance < best_distance:
            best_candidate = candidate
            best_distance = distance
    return best_candidate, best_distance


def predict_records(
    model: JointDiffusionProcedureModel,
    schedule: DiffusionSchedule,
    featurizer: ReactGDiffFeaturizer,
    records: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    *,
    seed: int,
    device: str | torch.device,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    schedule = schedule.to(device)
    model = model.to(device)
    for offset, record in enumerate(records):
        condition = torch.tensor(
            [featurizer.condition_vector(record)],
            dtype=torch.float32,
            device=device,
        )
        structure, attributes = sample_latents(
            model,
            schedule,
            condition,
            seed=seed + offset,
        )
        candidate, distance = decode_from_memory(structure, attributes, candidates)
        prediction = candidate["decoded_actions"]
        reference = str(record.get("actions", ""))
        gap = text_gap(prediction, reference)
        rows.append(
            {
                "index": record.get("index"),
                "input_vector": featurizer.condition_vector(record),
                "reference_actions": reference,
                "predicted_actions": prediction,
                "text_gap": gap,
                "levenshtein_similarity": 1.0 - gap,
                "decoded_from_train_index": candidate.get("index"),
                "decode_distance": distance,
            }
        )
    return rows


def save_checkpoint(
    path: str | Path,
    *,
    model: JointDiffusionProcedureModel,
    schedule: DiffusionSchedule,
    featurizer: ReactGDiffFeaturizer,
    candidates: list[dict[str, Any]],
    history: list[dict[str, float]],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": model.cpu().state_dict(),
        "model_config": model.config(),
        "diffusion_steps": schedule.num_steps,
        "betas": schedule.betas.cpu(),
        "featurizer": featurizer.to_dict(),
        "candidates": candidates,
        "history": history,
    }
    torch.save(payload, path)


def load_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> tuple[
    JointDiffusionProcedureModel,
    DiffusionSchedule,
    ReactGDiffFeaturizer,
    list[dict[str, Any]],
    list[dict[str, float]],
]:
    payload = torch.load(path, map_location=device, weights_only=False)
    model_config = dict(payload["model_config"])
    model_config.setdefault("base_model", "mlp")
    model_config.setdefault("dit_depth", 0)
    model_config.setdefault("dit_heads", 1)
    model = JointDiffusionProcedureModel(**model_config).to(device)
    model.load_state_dict(payload["model_state"])
    schedule = DiffusionSchedule(betas=payload["betas"].to(device))
    featurizer = ReactGDiffFeaturizer.from_dict(payload["featurizer"])
    return model, schedule, featurizer, payload["candidates"], payload.get("history", [])


def _feature_value(record: dict[str, Any], key: str) -> float:
    features = record.get("_features") or {}
    if key in features:
        value = features[key]
        if isinstance(value, (int, float)):
            return float(value)
    return 0.0


def _group_scalar(values: Iterable[Any]) -> float:
    values = sorted(str(value) for value in values if value)
    if not values:
        return 0.0
    joined = ".".join(values)
    digest = hashlib.sha1(joined.encode("utf-8")).hexdigest()
    identity = (int(digest[:16], 16) / float(0xFFFFFFFFFFFFFFFF)) * 2.0 - 1.0
    count_term = min(len(values), 8) / 8.0
    return max(min(0.85 * identity + 0.15 * count_term, 1.0), -1.0)


def _reactxt_prompt_fields(
    record: dict[str, Any],
    *,
    include_numeric_evidence: bool = False,
    numeric_evidence_include_source: bool = False,
    numeric_evidence_quantity_only: bool = False,
) -> tuple[list[str], ...]:
    """Return ReactXT-style prompt fields without using target action text."""

    extracted = record.get("extracted_molecules") or {}

    def molecule_field(role: str) -> list[str]:
        values: list[str] = []
        for smiles in record.get(role) or []:
            smiles_text = str(smiles)
            placeholder = extracted.get(smiles_text)
            prefix = f"{role}: "
            if placeholder:
                prefix += f"{placeholder}: "
            values.append(f"{prefix}[START_SMILES]{smiles_text}[END_SMILES]")
        return values

    def mapping_field(title: str, mapping_name: str) -> list[str]:
        value_to_ref = record.get(mapping_name) or {}
        values = [
            f"{title}: {ref}: {value}"
            for value, ref in sorted(
                value_to_ref.items(),
                key=lambda item: _placeholder_sort_key(str(item[1])),
            )
        ]
        return values

    fields: tuple[list[str], ...] = (
        molecule_field("REACTANT"),
        molecule_field("PRODUCT"),
        molecule_field("CATALYST"),
        molecule_field("SOLVENT"),
        mapping_field("Temperatures", "extracted_temperature"),
        mapping_field("Durations", "extracted_duration"),
    )
    if include_numeric_evidence:
        fields = (
            *fields,
            numeric_condition_field(
                record,
                include_source=numeric_evidence_include_source,
                quantity_only=numeric_evidence_quantity_only,
            ),
        )
    return fields


def _placeholder_sort_key(value: str) -> tuple[str, int, str]:
    match = re_placeholder_index(value)
    if match is None:
        return (value[:1], 10**9, value)
    return (value[:1], match, value)


def re_placeholder_index(value: str) -> int | None:
    if len(value) < 3:
        return None
    if value[0] not in "$@#" or value[-1] != value[0]:
        return None
    try:
        return int(value[1:-1])
    except ValueError:
        return None


def _field_hash_vector(
    values: Iterable[Any],
    *,
    dim: int,
    field_idx: int,
    ngram_min: int,
    ngram_max: int,
) -> list[float]:
    vector = [0.0] * dim
    strings = sorted(str(value).strip() for value in values if str(value).strip())
    if not strings:
        vector[0] = 1.0
        return vector

    vector[0] = min(len(strings), 16) / 16.0
    lengths = [len(value) for value in strings]
    vector[1] = min(sum(lengths) / len(lengths), 256) / 256.0
    vector[2] = min(max(lengths), 256) / 256.0
    vector[3] = min(len(set(strings)), 16) / 16.0

    for value in strings:
        normalized = f"<{value}>"
        _accumulate_hash_feature(
            vector,
            token=f"field{field_idx}:whole:{normalized}",
            dim=dim,
            weight=1.0,
        )
        for ngram_size in range(ngram_min, ngram_max + 1):
            if len(normalized) < ngram_size:
                continue
            weight = 1.0 / max(ngram_size, 1)
            for start in range(0, len(normalized) - ngram_size + 1):
                token = f"field{field_idx}:ng{ngram_size}:{normalized[start:start + ngram_size]}"
                _accumulate_hash_feature(vector, token=token, dim=dim, weight=weight)

    norm = math.sqrt(sum(value * value for value in vector[4:]))
    if norm > 0:
        for idx in range(4, dim):
            vector[idx] /= norm
    return vector


def _accumulate_hash_feature(
    vector: list[float],
    *,
    token: str,
    dim: int,
    weight: float,
) -> None:
    if dim <= 4:
        return
    digest = hashlib.sha1(token.encode("utf-8")).hexdigest()
    bucket = 4 + (int(digest[:8], 16) % (dim - 4))
    sign = -1.0 if int(digest[8:10], 16) % 2 else 1.0
    vector[bucket] += sign * weight
