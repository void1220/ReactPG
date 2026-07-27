"""Shared MolT5 encoder inputs and checkpoint loading for graph diffusion."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from reactgdiff.data.procedure_prompt import build_encoder_prompt
from reactgdiff.models.graph_codec import GraphTargetCodec, NONE_TOKEN


def load_skeleton_text_encoder(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device,
    trainable: bool,
    local_files_only: bool = True,
) -> tuple[nn.Module, Any, dict[str, Any]]:
    try:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("transformers is required for shared MolT5 conditioning") from exc

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload.get("checkpoint_type") != "seq2seq_skeleton":
        raise ValueError(f"Not a seq2seq skeleton checkpoint: {checkpoint_path}")
    model_name = str(payload["model_name"])
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        local_files_only=local_files_only,
    )
    action_tokens = payload.get("action_tokens") or {}
    if action_tokens:
        tokenizer.add_special_tokens(
            {"additional_special_tokens": list(action_tokens.values())}
        )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        local_files_only=local_files_only,
    )
    if len(tokenizer) != int(base_model.get_input_embeddings().num_embeddings):
        base_model.resize_token_embeddings(len(tokenizer))
    base_state = {
        name[len("base_model.") :]: tensor
        for name, tensor in payload["model_state"].items()
        if name.startswith("base_model.")
    }
    base_model.load_state_dict(base_state, strict=False)
    encoder = base_model.get_encoder().to(device)
    for parameter in encoder.parameters():
        parameter.requires_grad = bool(trainable)
    if not trainable:
        encoder.eval()
    hidden_size = _infer_encoder_hidden_size(base_model)
    metadata = {
        "checkpoint": str(checkpoint_path),
        "model_name": model_name,
        "hidden_size": hidden_size,
        "trainable": bool(trainable),
        "prompt_style": str((payload.get("config") or {}).get("prompt_style", "compact")),
        "include_numeric_evidence": bool(
            (payload.get("config") or {}).get("include_numeric_evidence", True)
        ),
        "action_tokens": action_tokens,
        "tokenizer_length": len(tokenizer),
        "local_files_only": bool(local_files_only),
    }
    del base_model
    return encoder, tokenizer, metadata


def build_shared_text_inputs(
    records: list[dict[str, Any]],
    *,
    codec: GraphTargetCodec,
    tokenizer: Any,
    prompt_style: str,
    include_numeric_evidence: bool,
    numeric_evidence_include_source: bool,
    max_length: int,
) -> dict[str, torch.Tensor]:
    """Tokenize reaction context plus record-local candidates.

    T5 sentinel tokens mark material and numeric candidates. Their contextual
    encoder states become dynamic pointer keys in graph diffusion.
    """

    candidate_class_count = codec.material_dim + codec.numeric_candidate_dim
    marker_ids = _candidate_marker_ids(tokenizer, candidate_class_count)
    pad_id = int(tokenizer.pad_token_id)
    eos_id = int(
        tokenizer.eos_token_id
        if tokenizer.eos_token_id is not None
        else tokenizer.sep_token_id
    )
    max_length = max(int(max_length), 64)
    encoded_rows: list[list[int]] = []
    material_positions: list[list[int]] = []
    numeric_positions: list[list[int]] = []

    for record in records:
        prompt = build_encoder_prompt(
            record,
            prompt_style=prompt_style,
            include_numeric_evidence=include_numeric_evidence,
            numeric_evidence_include_source=numeric_evidence_include_source,
        )
        prompt_ids = list(
            tokenizer(
                prompt,
                add_special_tokens=False,
                truncation=False,
            )["input_ids"]
        )
        entries = _candidate_entries(record, codec=codec)
        if len(entries) + 2 > max_length:
            raise ValueError(
                f"max_length={max_length} cannot hold {len(entries)} candidate markers"
            )
        minimum_prompt_tokens = min(128, max(8, max_length // 3))
        minimum_prompt_tokens = min(
            minimum_prompt_tokens,
            max_length - len(entries) - 1,
        )
        candidate_content_budget = max(
            max_length - minimum_prompt_tokens - len(entries) - 1,
            0,
        )
        content_tokens_per_entry = min(
            10,
            candidate_content_budget // max(len(entries), 1),
        )
        extra_content_entries = (
            candidate_content_budget
            - content_tokens_per_entry * len(entries)
        )
        chunks: list[tuple[str, int, list[int]]] = []
        for entry_index, (kind, class_id, text) in enumerate(entries):
            marker_index = class_id if kind == "material" else codec.material_dim + class_id
            content_limit = content_tokens_per_entry + int(
                entry_index < extra_content_entries
                and content_tokens_per_entry < 10
            )
            content_ids = list(
                tokenizer(
                    text,
                    add_special_tokens=False,
                    truncation=True,
                    max_length=max(content_limit, 1),
                )["input_ids"]
            )[:content_limit]
            chunks.append((kind, class_id, [marker_ids[marker_index], *content_ids]))

        candidate_token_count = sum(len(chunk) for _, _, chunk in chunks)
        prompt_budget = max_length - candidate_token_count - 1
        input_ids = prompt_ids[:prompt_budget]
        input_ids.append(eos_id)
        row_material_positions = [-1] * codec.material_dim
        row_numeric_positions = [-1] * codec.numeric_candidate_dim
        for kind, class_id, chunk in chunks:
            marker_position = len(input_ids)
            input_ids.extend(chunk)
            if kind == "material":
                row_material_positions[class_id] = marker_position
            else:
                row_numeric_positions[class_id] = marker_position
        if len(input_ids) > max_length:
            raise AssertionError("Candidate input packing exceeded max_length")
        encoded_rows.append(input_ids[:max_length])
        material_positions.append(row_material_positions)
        numeric_positions.append(row_numeric_positions)

    padded_length = min(
        max((len(row) for row in encoded_rows), default=1),
        max_length,
    )
    input_tensor = torch.full(
        (len(encoded_rows), padded_length),
        pad_id,
        dtype=torch.long,
    )
    attention_mask = torch.zeros_like(input_tensor)
    for row_idx, row in enumerate(encoded_rows):
        row = row[:padded_length]
        input_tensor[row_idx, : len(row)] = torch.tensor(row, dtype=torch.long)
        attention_mask[row_idx, : len(row)] = 1
    return {
        "input_ids": input_tensor,
        "attention_mask": attention_mask,
        "material_candidate_positions": torch.tensor(material_positions, dtype=torch.long),
        "numeric_candidate_positions": torch.tensor(numeric_positions, dtype=torch.long),
    }


def slice_shared_text_inputs(
    inputs: dict[str, torch.Tensor],
    start: int,
    end: int,
    *,
    device: str | torch.device,
) -> dict[str, torch.Tensor]:
    return {
        key: value[start:end].to(device)
        for key, value in inputs.items()
    }


def _candidate_entries(
    record: dict[str, Any],
    *,
    codec: GraphTargetCodec,
) -> list[tuple[str, int, str]]:
    entries: list[tuple[str, int, str]] = [
        ("material", 0, "MATERIAL NONE"),
    ]
    extracted = record.get("extracted_molecules") or {}
    placeholder_to_smiles = {
        str(placeholder): str(smiles)
        for smiles, placeholder in extracted.items()
    }
    role_by_smiles: dict[str, str] = {}
    for role in ("REACTANT", "PRODUCT", "CATALYST", "SOLVENT"):
        for smiles in record.get(role) or []:
            role_by_smiles[str(smiles)] = role
    for class_id, placeholder in enumerate(codec.material_vocab[1:], start=1):
        smiles = placeholder_to_smiles.get(placeholder)
        if smiles is None and placeholder != "$-1$":
            continue
        if smiles is None:
            products = record.get("PRODUCT") or []
            smiles = str(products[0]) if products else "PRODUCT"
        role = role_by_smiles.get(smiles, "PRODUCT" if placeholder == "$-1$" else "MATERIAL")
        entries.append(
            ("material", class_id, f"MATERIAL {placeholder} {role} {smiles}")
        )

    entries.extend(
        [
            ("numeric", 0, "NUMERIC NONE"),
            ("numeric", 1, "NUMERIC MISSING"),
        ]
    )
    for candidate in codec.numeric_candidates_from_record(record):
        class_id = codec.numeric_candidate_class_id(candidate.candidate_id)
        if class_id <= 1 or class_id >= codec.numeric_candidate_dim:
            continue
        entries.append(
            (
                "numeric",
                class_id,
                " ".join(
                    [
                        "NUMERIC",
                        candidate.candidate_id,
                        candidate.numeric_type,
                        str(candidate.normalized_value),
                        str(candidate.normalized_unit or NONE_TOKEN),
                        candidate.source,
                    ]
                ),
            )
        )
    return entries


def _candidate_marker_ids(tokenizer: Any, count: int) -> list[int]:
    ids: list[int] = []
    unk_id = tokenizer.unk_token_id
    for index in range(count):
        token = f"<extra_id_{index}>"
        token_id = int(tokenizer.convert_tokens_to_ids(token))
        if unk_id is not None and token_id == int(unk_id):
            raise ValueError(
                f"Tokenizer lacks {token}; shared candidate pointers require "
                f"at least {count} distinct sentinel tokens."
            )
        if token_id in ids:
            raise ValueError(f"Candidate sentinel token collision at {token}")
        ids.append(token_id)
    return ids


def _infer_encoder_hidden_size(model: nn.Module) -> int:
    config = getattr(model, "config", None)
    for name in ("d_model", "hidden_size", "encoder_embed_dim"):
        value = getattr(config, name, None)
        if value is not None:
            return int(value)
    raise ValueError("Could not infer shared text encoder hidden size")
