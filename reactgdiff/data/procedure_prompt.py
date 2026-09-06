"""Shared reaction prompts for skeleton prediction and graph conditioning."""

from __future__ import annotations

from typing import Any

from reactgdiff.data.numeric_evidence import numeric_condition_field


def build_encoder_prompt(
    record: dict[str, Any],
    *,
    prompt_style: str,
    include_numeric_evidence: bool,
    numeric_evidence_include_source: bool = False,
) -> str:
    if prompt_style == "compact":
        return build_compact_encoder_prompt(record)
    if prompt_style == "reactxt":
        return build_reactxt_encoder_prompt(
            record,
            include_numeric_evidence=include_numeric_evidence,
            numeric_evidence_include_source=numeric_evidence_include_source,
        )
    raise ValueError(f"Unsupported prompt_style: {prompt_style}")


def build_compact_encoder_prompt(record: dict[str, Any]) -> str:
    fields = _reactxt_prompt_fields(record, include_numeric_evidence=False)
    parts = ["TASK: Predict operation skeleton."]
    for label, values in zip(
        ("REACTANT", "PRODUCT", "CATALYST", "SOLVENT"),
        fields[:4],
        strict=False,
    ):
        cleaned = [_compact_molecule_value(label, value) for value in values]
        value_text = " | ".join(value for value in cleaned if value)
        if value_text:
            parts.append(f"{label}: {value_text}")
    temperature_text = _compact_mapping_values(record, "extracted_temperature")
    if temperature_text:
        parts.append(f"TEMPERATURE: {temperature_text}")
    duration_text = _compact_mapping_values(record, "extracted_duration")
    if duration_text:
        parts.append(f"DURATION: {duration_text}")
    return "\n".join(parts)


def build_reactxt_encoder_prompt(
    record: dict[str, Any],
    *,
    include_numeric_evidence: bool,
    numeric_evidence_include_source: bool = False,
) -> str:
    labels = (
        "REACTANT",
        "PRODUCT",
        "CATALYST",
        "SOLVENT",
        "TEMPERATURE",
        "DURATION",
        "NUMERIC_EVIDENCE",
    )
    fields = list(
        _reactxt_prompt_fields(
            record,
            include_numeric_evidence=False,
        )
    )
    if include_numeric_evidence:
        fields.append(
            numeric_condition_field(
                record,
                include_source=numeric_evidence_include_source,
            )
        )
    parts = ["TASK: Predict the experimental operation skeleton."]
    for label, values in zip(labels, fields, strict=False):
        value_text = " | ".join(str(value) for value in values if str(value).strip())
        parts.append(f"{label}: {value_text if value_text else '<EMPTY>'}")
    return "\n".join(parts)


def _compact_molecule_value(label: str, value: Any) -> str:
    text = str(value).strip()
    prefix = f"{label}: "
    if text.startswith(prefix):
        text = text[len(prefix) :]
    text = text.replace("[START_SMILES]", "")
    text = text.replace("[END_SMILES]", "")
    return " ".join(text.split())


def _compact_mapping_values(record: dict[str, Any], mapping_name: str) -> str:
    value_to_ref = record.get(mapping_name) or {}
    if not isinstance(value_to_ref, dict):
        return ""
    return " | ".join(
        f"{ref}:{value}"
        for value, ref in sorted(
            value_to_ref.items(),
            key=lambda item: _placeholder_sort_key(str(item[1])),
        )
        if str(ref).strip() or str(value).strip()
    )


def _reactxt_prompt_fields(
    record: dict[str, Any],
    *,
    include_numeric_evidence: bool = False,
    numeric_evidence_include_source: bool = False,
) -> tuple[list[str], ...]:
    """Build ReactXT input fields without importing the model package.

    Keeping this data-only implementation local prevents the skeleton prompt
    path from recursively importing graph-diffusion model modules.
    """

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
        return [
            f"{title}: {ref}: {value}"
            for value, ref in sorted(
                value_to_ref.items(),
                key=lambda item: _placeholder_sort_key(str(item[1])),
            )
        ]

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
            ),
        )
    return fields


def _placeholder_sort_key(value: str) -> tuple[str, int, str]:
    index = _placeholder_index(value)
    if index is None:
        return (value[:1], 10**9, value)
    return (value[:1], index, value)


def _placeholder_index(value: str) -> int | None:
    if len(value) < 3:
        return None
    if value[0] not in "$@#" or value[-1] != value[0]:
        return None
    try:
        return int(value[1:-1])
    except ValueError:
        return None
