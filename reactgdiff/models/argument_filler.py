"""Autoregressive argument text filler for graph-decoded OpenExp steps."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from reactgdiff.data.action_parser import parse_action_sequence
from reactgdiff.models.graph_codec import GraphTargetCodec, NONE_TOKEN, combine_condition_token

PAD_TOKEN = "<PAD>"
BOS_TOKEN = "<BOS>"
EOS_TOKEN = "<EOS>"
UNK_TOKEN = "<UNK>"


@dataclass(slots=True)
class ArgumentTextCodec:
    vocab: list[str]
    max_length: int

    @classmethod
    def fit(
        cls,
        records: list[dict[str, Any]],
        *,
        max_length: int = 160,
        max_vocab_size: int = 192,
    ) -> "ArgumentTextCodec":
        counts: Counter[str] = Counter()
        for record in records:
            for step in parse_action_sequence(str(record.get("actions", ""))):
                counts.update(step.arguments[: max(0, max_length - 2)])
        vocab = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN]
        for char, _ in counts.most_common(max(0, max_vocab_size - len(vocab))):
            if char not in vocab:
                vocab.append(char)
        return cls(vocab=vocab, max_length=max_length)

    @property
    def pad_id(self) -> int:
        return 0

    @property
    def bos_id(self) -> int:
        return 1

    @property
    def eos_id(self) -> int:
        return 2

    @property
    def unk_id(self) -> int:
        return 3

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def encode(self, text: str) -> list[int]:
        char_to_id = {char: idx for idx, char in enumerate(self.vocab)}
        ids = [self.bos_id]
        for char in str(text)[: max(0, self.max_length - 2)]:
            ids.append(char_to_id.get(char, self.unk_id))
        ids.append(self.eos_id)
        ids.extend([self.pad_id] * max(0, self.max_length - len(ids)))
        return ids[: self.max_length]

    def decode(self, ids: list[int] | torch.Tensor) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.detach().cpu().tolist()
        chars: list[str] = []
        for idx in ids:
            idx = int(idx)
            if idx in (self.pad_id, self.bos_id):
                continue
            if idx == self.eos_id:
                break
            if 0 <= idx < len(self.vocab):
                token = self.vocab[idx]
                chars.append("" if token == UNK_TOKEN else token)
        return "".join(chars).strip()

    def to_dict(self) -> dict[str, Any]:
        return {"vocab": self.vocab, "max_length": self.max_length}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ArgumentTextCodec":
        return cls(vocab=list(payload["vocab"]), max_length=int(payload["max_length"]))


class ArgumentStepDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        records: list[dict[str, Any]],
        condition_vectors: list[list[float]],
        *,
        graph_codec: GraphTargetCodec,
        text_codec: ArgumentTextCodec,
        max_examples: int | None = None,
        condition_on_quantities: bool = True,
        condition_on_quantity_units: bool = True,
        target: str = "all",
    ) -> None:
        if target not in {"all", "numeric"}:
            raise ValueError(f"Unsupported argument filler target: {target}")
        self.condition_vectors = condition_vectors
        self.text_codec = text_codec
        self.target = target
        self.examples: list[dict[str, Any]] = []
        for record_idx, record in enumerate(records):
            encoded = graph_codec.encode_record(record)
            steps = parse_action_sequence(str(record.get("actions", "")))[: graph_codec.max_steps - 1]
            for step_idx, step in enumerate(steps):
                if target == "numeric" and not step.quantities:
                    continue
                self.examples.append(
                    {
                        "record_idx": record_idx,
                        "step_idx": step_idx,
                        "op_id": encoded["op_ids"][step_idx],
                        "material_ids": encoded["material_ids"][step_idx],
                        "condition_id": encoded["condition_ids"][step_idx],
                        "quantity_gate_ids": (
                            encoded["quantity_gate_ids"][step_idx]
                            if condition_on_quantities
                            else [0] * graph_codec.max_material_slots
                        ),
                        "unit_ids": (
                            encoded["unit_ids"][step_idx]
                            if condition_on_quantities and condition_on_quantity_units
                            else [0] * graph_codec.max_material_slots
                        ),
                        "target": step.arguments,
                    }
                )
                if max_examples is not None and len(self.examples) >= max_examples:
                    return

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        example = self.examples[idx]
        return {
            "condition": torch.tensor(
                self.condition_vectors[int(example["record_idx"])],
                dtype=torch.float32,
            ),
            "step_id": torch.tensor(int(example["step_idx"]), dtype=torch.long),
            "op_id": torch.tensor(int(example["op_id"]), dtype=torch.long),
            "material_ids": torch.tensor(example["material_ids"], dtype=torch.long),
            "condition_id": torch.tensor(int(example["condition_id"]), dtype=torch.long),
            "quantity_gate_ids": torch.tensor(example["quantity_gate_ids"], dtype=torch.long),
            "unit_ids": torch.tensor(example["unit_ids"], dtype=torch.long),
            "tokens": torch.tensor(self.text_codec.encode(str(example["target"])), dtype=torch.long),
        }


class ArgumentTextFiller(nn.Module):
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
        vocab_size: int,
        max_length: int,
        hidden_dim: int = 256,
        layers: int = 4,
        heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.condition_dim = condition_dim
        self.action_dim = action_dim
        self.material_dim = material_dim
        self.condition_slot_dim = condition_slot_dim
        self.unit_dim = unit_dim
        self.max_steps = max_steps
        self.max_material_slots = max_material_slots
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.hidden_dim = hidden_dim
        self.layers = layers
        self.heads = heads
        self.dropout = dropout
        self.condition_projection = nn.Linear(condition_dim, hidden_dim)
        self.operation_embedding = nn.Embedding(action_dim, hidden_dim)
        self.step_embedding = nn.Embedding(max_steps, hidden_dim)
        self.material_embedding = nn.Embedding(material_dim, hidden_dim)
        self.condition_embedding = nn.Embedding(condition_slot_dim, hidden_dim)
        self.quantity_gate_embedding = nn.Embedding(2, hidden_dim)
        self.unit_embedding = nn.Embedding(unit_dim, hidden_dim)
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.position_embedding = nn.Embedding(max_length, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.norm = nn.LayerNorm(hidden_dim)
        self.output_head = nn.Linear(hidden_dim, vocab_size)

    def config(self) -> dict[str, Any]:
        return {
            "condition_dim": self.condition_dim,
            "action_dim": self.action_dim,
            "material_dim": self.material_dim,
            "condition_slot_dim": self.condition_slot_dim,
            "unit_dim": self.unit_dim,
            "max_steps": self.max_steps,
            "max_material_slots": self.max_material_slots,
            "vocab_size": self.vocab_size,
            "max_length": self.max_length,
            "hidden_dim": self.hidden_dim,
            "layers": self.layers,
            "heads": self.heads,
            "dropout": self.dropout,
        }

    def context(
        self,
        condition: torch.Tensor,
        step_id: torch.Tensor,
        op_id: torch.Tensor,
        material_ids: torch.Tensor,
        condition_id: torch.Tensor,
        quantity_gate_ids: torch.Tensor,
        unit_ids: torch.Tensor,
    ) -> torch.Tensor:
        return (
            self.condition_projection(condition)
            + self.operation_embedding(op_id)
            + self.step_embedding(step_id.clamp(max=self.max_steps - 1))
            + self.material_embedding(material_ids).mean(dim=1)
            + self.condition_embedding(condition_id)
            + self.quantity_gate_embedding(quantity_gate_ids).mean(dim=1)
            + self.unit_embedding(unit_ids).mean(dim=1)
        )

    def forward(
        self,
        condition: torch.Tensor,
        step_id: torch.Tensor,
        op_id: torch.Tensor,
        material_ids: torch.Tensor,
        condition_id: torch.Tensor,
        quantity_gate_ids: torch.Tensor,
        unit_ids: torch.Tensor,
        input_tokens: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, length = input_tokens.shape
        positions = torch.arange(length, device=input_tokens.device).unsqueeze(0)
        hidden = self.token_embedding(input_tokens) + self.position_embedding(positions)
        hidden = hidden + self.context(
            condition,
            step_id,
            op_id,
            material_ids,
            condition_id,
            quantity_gate_ids,
            unit_ids,
        ).unsqueeze(1)
        causal_mask = torch.triu(
            torch.ones(length, length, device=input_tokens.device, dtype=torch.bool),
            diagonal=1,
        )
        hidden = self.transformer(hidden, mask=causal_mask)
        return self.output_head(self.norm(hidden))

    @torch.no_grad()
    def generate(
        self,
        text_codec: ArgumentTextCodec,
        condition: torch.Tensor,
        step_id: torch.Tensor,
        op_id: torch.Tensor,
        material_ids: torch.Tensor,
        condition_id: torch.Tensor,
        quantity_gate_ids: torch.Tensor,
        unit_ids: torch.Tensor,
    ) -> str:
        self.eval()
        tokens = torch.full(
            (1, text_codec.max_length),
            text_codec.pad_id,
            dtype=torch.long,
            device=condition.device,
        )
        tokens[0, 0] = text_codec.bos_id
        for pos in range(1, text_codec.max_length):
            logits = self(
                condition,
                step_id,
                op_id,
                material_ids,
                condition_id,
                quantity_gate_ids,
                unit_ids,
                tokens[:, :pos],
            )
            next_id = int(logits[0, -1].argmax(dim=-1))
            tokens[0, pos] = next_id
            if next_id == text_codec.eos_id:
                break
        return text_codec.decode(tokens[0])


def train_argument_text_filler(
    records: list[dict[str, Any]],
    *,
    condition_vectors: list[list[float]],
    graph_codec: GraphTargetCodec,
    text_codec: ArgumentTextCodec,
    condition_dim: int,
    hidden_dim: int,
    layers: int,
    heads: int,
    dropout: float,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    gradient_clip_norm: float,
    max_examples: int | None,
    seed: int,
    device: str | torch.device,
    log_every: int = 1,
    condition_on_quantities: bool = True,
    condition_on_quantity_units: bool = True,
    target: str = "all",
) -> tuple[ArgumentTextFiller, list[dict[str, float]]]:
    torch.manual_seed(seed)
    dataset = ArgumentStepDataset(
        records,
        condition_vectors,
        graph_codec=graph_codec,
        text_codec=text_codec,
        max_examples=max_examples,
        condition_on_quantities=condition_on_quantities,
        condition_on_quantity_units=condition_on_quantity_units,
        target=target,
    )
    if not dataset:
        raise ValueError(f"No argument filler examples were built for target={target!r}.")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)
    model = ArgumentTextFiller(
        condition_dim=condition_dim,
        action_dim=graph_codec.action_dim,
        material_dim=graph_codec.material_dim,
        condition_slot_dim=graph_codec.condition_dim,
        unit_dim=graph_codec.unit_dim,
        max_steps=graph_codec.max_steps,
        max_material_slots=graph_codec.max_material_slots,
        vocab_size=text_codec.vocab_size,
        max_length=text_codec.max_length,
        hidden_dim=hidden_dim,
        layers=layers,
        heads=heads,
        dropout=dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_tokens = 0
        total_grad = 0.0
        total_items = 0
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            input_tokens = batch["tokens"][:, :-1]
            target_tokens = batch["tokens"][:, 1:]
            logits = model(
                batch["condition"],
                batch["step_id"],
                batch["op_id"],
                batch["material_ids"],
                batch["condition_id"],
                batch["quantity_gate_ids"],
                batch["unit_ids"],
                input_tokens,
            )
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                target_tokens.reshape(-1),
                ignore_index=text_codec.pad_id,
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite argument filler loss at epoch {epoch}.")
            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=float(gradient_clip_norm),
                error_if_nonfinite=False,
            ) if gradient_clip_norm > 0 else torch.tensor(0.0, device=device)
            if not torch.isfinite(grad_norm):
                raise FloatingPointError(f"Non-finite argument filler gradient at epoch {epoch}.")
            optimizer.step()
            token_count = int((target_tokens != text_codec.pad_id).sum())
            item_count = int(target_tokens.size(0))
            total_loss += float(loss.detach()) * token_count
            total_tokens += token_count
            total_grad += float(grad_norm.detach()) * item_count
            total_items += item_count
        metrics = {
            "epoch": float(epoch),
            "loss": total_loss / max(total_tokens, 1),
            "grad_norm": total_grad / max(total_items, 1),
            "examples": float(len(dataset)),
        }
        history.append(metrics)
        if log_every and (epoch == 1 or epoch == epochs or epoch % log_every == 0):
            print(
                f"[argument epoch {epoch:03d}/{epochs:03d}] "
                f"loss={metrics['loss']:.4f} grad={metrics['grad_norm']:.2f} "
                f"examples={len(dataset)}",
                flush=True,
            )
    return model, history


def should_fill_argument_slot(slot: dict[str, Any], *, target: str = "all") -> bool:
    if target == "all":
        return True
    if target == "numeric":
        return has_numeric_slot(slot)
    raise ValueError(f"Unsupported argument filler target: {target}")


def has_numeric_slot(slot: dict[str, Any]) -> bool:
    if slot.get("quantity_slots"):
        return True
    quantity = str(slot.get("quantity") or NONE_TOKEN)
    return quantity != NONE_TOKEN


def slot_features_from_decoded_slot(
    slot: dict[str, Any],
    *,
    graph_codec: GraphTargetCodec,
    device: str | torch.device,
    condition_on_quantity_units: bool = True,
) -> dict[str, torch.Tensor]:
    op_id = graph_codec._id(graph_codec.action_vocab, str(slot.get("operation_type") or ""), graph_codec.eos_id)
    material_refs = list(slot.get("material_refs") or [])
    material_ids = [
        graph_codec._id(graph_codec.material_vocab, str(ref), 0)
        for ref in material_refs[: graph_codec.max_material_slots]
    ]
    material_ids.extend([0] * max(0, graph_codec.max_material_slots - len(material_ids)))
    condition = str(slot.get("condition") or NONE_TOKEN)
    condition_id = graph_codec._id(graph_codec.condition_vocab, condition, 0)
    quantity_slots = list(slot.get("quantity_slots") or [])
    quantity_gate_ids = [0] * graph_codec.max_material_slots
    unit_ids = [0] * graph_codec.max_material_slots
    for quantity in quantity_slots:
        slot_id = int(quantity.get("slot_id", 0))
        if not 0 <= slot_id < graph_codec.max_material_slots:
            continue
        quantity_gate_ids[slot_id] = 1
        if condition_on_quantity_units:
            unit_ids[slot_id] = graph_codec._id(
                graph_codec.unit_vocab,
                str(quantity.get("unit") or NONE_TOKEN),
                0,
            )
    return {
        "op_id": torch.tensor([op_id], dtype=torch.long, device=device),
        "material_ids": torch.tensor([material_ids], dtype=torch.long, device=device),
        "condition_id": torch.tensor([condition_id], dtype=torch.long, device=device),
        "quantity_gate_ids": torch.tensor([quantity_gate_ids], dtype=torch.long, device=device),
        "unit_ids": torch.tensor([unit_ids], dtype=torch.long, device=device),
    }
