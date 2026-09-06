"""Numeric evidence extraction and grounding helpers.

The graph completion task should bind numeric slots to evidence that is already
available in the input record instead of freely generating precise values.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable


NONE_CANDIDATE_ID = "<NUMERIC_SLOT_MISSING>"
QUANTITY_CANDIDATE_TYPES = frozenset({"amount", "concentration", "yield"})

UNIT_PATTERN = (
    r"ug|µg|mg|g|kg|ul|µl|ml|l|liter|liters|litre|litres|"
    r"umol|µmol|mmol|mol|molar|normal|m|n|"
    r"equiv|equivalent|eq|percent|%|"
    r"h|hr|hrs|hour|hours|min|minute|minutes|s|sec|second|seconds|"
    r"day|days|c|°c|℃|degc|degree|degrees|times|x"
)

NUMBER_UNIT_RE = re.compile(
    rf"(?P<value>[-+]?\d+(?:[.,]\d+)?)\s*(?:°\s*)?(?P<unit>{UNIT_PATTERN})(?=\b|[^A-Za-z0-9_])",
    re.IGNORECASE,
)


@dataclass(slots=True)
class NumericCandidate:
    candidate_id: str
    raw_text: str
    value: float | None
    unit: str | None
    normalized_value: float | None
    normalized_unit: str | None
    numeric_type: str
    source: str
    confidence: float
    linked_material_id: str | None = None
    linked_op_hint: int | None = None
    placeholder: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "raw_text": self.raw_text,
            "value": self.value,
            "unit": self.unit,
            "normalized_value": self.normalized_value,
            "normalized_unit": self.normalized_unit,
            "numeric_type": self.numeric_type,
            "source": self.source,
            "confidence": self.confidence,
            "linked_material_id": self.linked_material_id,
            "linked_op_hint": self.linked_op_hint,
            "placeholder": self.placeholder,
        }


def numeric_candidates_from_record(
    record: dict[str, Any],
    *,
    include_source: bool = True,
    include_actions: bool = False,
    include_condition_maps: bool = True,
    allowed_numeric_types: Iterable[str] | None = None,
) -> list[NumericCandidate]:
    """Build a de-duplicated pool of numeric evidence candidates."""

    candidates: list[NumericCandidate] = []
    if include_condition_maps:
        for raw, placeholder in sorted(
            (record.get("extracted_duration") or {}).items(),
            key=lambda item: str(item[1]),
        ):
            candidate = parse_numeric_candidate(
                str(raw),
                source="input_duration",
                confidence=1.0,
                placeholder=str(placeholder),
            )
            if candidate is not None:
                candidates.append(candidate)
        for raw, placeholder in sorted(
            (record.get("extracted_temperature") or {}).items(),
            key=lambda item: str(item[1]),
        ):
            candidate = parse_numeric_candidate(
                str(raw),
                source="input_temperature",
                confidence=1.0,
                placeholder=str(placeholder),
            )
            if candidate is not None:
                candidates.append(candidate)

    if include_source:
        for raw in iter_numeric_spans(str(record.get("source") or "")):
            candidate = parse_numeric_candidate(raw, source="input_source", confidence=0.85)
            if candidate is not None:
                candidates.append(candidate)

    if include_actions:
        from reactgdiff.data.action_parser import parse_action_sequence

        for step in parse_action_sequence(str(record.get("actions") or "")):
            for raw in step.quantities:
                candidate = parse_numeric_candidate(
                    raw,
                    source="target_action",
                    confidence=0.6,
                    linked_op_hint=step.step_id,
                )
                if candidate is not None:
                    candidates.append(candidate)

    if allowed_numeric_types is not None:
        allowed = frozenset(str(value) for value in allowed_numeric_types)
        candidates = [
            candidate
            for candidate in candidates
            if candidate.numeric_type in allowed
        ]
    return _deduplicate_candidates(candidates)


def numeric_condition_field(
    record: dict[str, Any],
    *,
    include_source: bool = False,
    quantity_only: bool = False,
) -> list[str]:
    """Render all input-side numeric evidence as hashable prompt strings."""

    candidates = numeric_candidates_from_record(
        record,
        include_source=include_source,
        include_actions=False,
        include_condition_maps=not quantity_only,
        allowed_numeric_types=QUANTITY_CANDIDATE_TYPES if quantity_only else None,
    )
    values: list[str] = []
    for candidate in candidates:
        placeholder = f" placeholder={candidate.placeholder}" if candidate.placeholder else ""
        values.append(
            "NumericEvidence: "
            f"{candidate.candidate_id}: type={candidate.numeric_type} "
            f"value={candidate.normalized_value} unit={candidate.normalized_unit} "
            f"source={candidate.source}{placeholder} text={candidate.raw_text}"
        )
    return values


def parse_numeric_candidate(
    raw_text: str,
    *,
    source: str,
    confidence: float,
    placeholder: str | None = None,
    linked_material_id: str | None = None,
    linked_op_hint: int | None = None,
) -> NumericCandidate | None:
    parsed = parse_numeric_value_unit(raw_text)
    if parsed is None:
        return None
    value, unit = parsed
    normalized_unit = normalize_unit(unit)
    numeric_type = infer_numeric_type(raw_text, normalized_unit)
    return NumericCandidate(
        candidate_id="",
        raw_text=raw_text.strip(),
        value=value,
        unit=unit,
        normalized_value=value,
        normalized_unit=normalized_unit,
        numeric_type=numeric_type,
        source=source,
        confidence=float(confidence),
        linked_material_id=linked_material_id,
        linked_op_hint=linked_op_hint,
        placeholder=placeholder,
    )


def parse_numeric_value_unit(text: str) -> tuple[float, str] | None:
    match = NUMBER_UNIT_RE.search(str(text))
    if match is None:
        return None
    raw_value = match.group("value").replace(",", ".")
    try:
        value = float(raw_value)
    except ValueError:
        return None
    return value, normalize_unit(match.group("unit"))


def iter_numeric_spans(text: str) -> Iterable[str]:
    for match in NUMBER_UNIT_RE.finditer(text):
        yield match.group(0).strip()


def normalize_unit(unit: str) -> str:
    normalized = str(unit).strip().lower()
    aliases = {
        "µg": "ug",
        "µl": "ul",
        "µmol": "umol",
        "liter": "l",
        "liters": "l",
        "litre": "l",
        "litres": "l",
        "equivalent": "eq",
        "equiv": "eq",
        "percent": "%",
        "m": "molar",
        "n": "normal",
        "hr": "h",
        "hrs": "h",
        "hour": "h",
        "hours": "h",
        "minute": "min",
        "minutes": "min",
        "sec": "s",
        "second": "s",
        "seconds": "s",
        "days": "day",
        "c": "degc",
        "°c": "degc",
        "℃": "degc",
        "degree": "degc",
        "degrees": "degc",
        "x": "times",
    }
    return aliases.get(normalized, normalized)


def infer_numeric_type(text: str, unit: str | None) -> str:
    raw = str(text).lower()
    normalized_unit = normalize_unit(unit or "")
    if "yield" in raw:
        return "yield"
    if normalized_unit in {"degc"}:
        return "temperature"
    if normalized_unit in {"h", "min", "s", "day"}:
        return "duration"
    if normalized_unit in {"molar", "normal"}:
        return "concentration"
    if normalized_unit in {"times"}:
        return "repetition"
    if normalized_unit in {
        "ug",
        "mg",
        "g",
        "kg",
        "ul",
        "ml",
        "l",
        "umol",
        "mmol",
        "mol",
        "eq",
        "%",
    }:
        return "amount"
    return "number"


def best_numeric_candidate(
    candidates: list[NumericCandidate],
    *,
    value: float | None = None,
    unit: str | None = None,
    numeric_type: str | None = None,
    raw_text: str | None = None,
    used_candidate_ids: set[str] | None = None,
) -> NumericCandidate | None:
    if not candidates:
        return None
    used = used_candidate_ids or set()
    unit = normalize_unit(unit or "") if unit else None
    scored: list[tuple[float, int, NumericCandidate]] = []
    raw_normalized = _normalize_raw(raw_text or "")
    for idx, candidate in enumerate(candidates):
        if candidate.candidate_id in used:
            continue
        score = float(candidate.confidence)
        if raw_normalized and _normalize_raw(candidate.raw_text) == raw_normalized:
            score += 5.0
        if unit and candidate.normalized_unit == unit:
            score += 2.0
        if numeric_type and candidate.numeric_type == numeric_type:
            score += 1.5
        if value is not None and candidate.normalized_value is not None:
            if abs(float(candidate.normalized_value) - float(value)) <= max(abs(float(value)) * 1e-3, 1e-3):
                score += 3.0
        scored.append((score, -idx, candidate))
    if not scored:
        return None
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return scored[0][2]


def format_candidate_quantity(candidate: NumericCandidate) -> str:
    if candidate.raw_text:
        return candidate.raw_text
    if candidate.normalized_value is None or candidate.normalized_unit is None:
        return NONE_CANDIDATE_ID
    value = candidate.normalized_value
    if abs(value - round(value)) < 1e-6:
        number = str(int(round(value)))
    else:
        number = f"{value:.4g}"
    return f"{number} {candidate.normalized_unit}"


def _deduplicate_candidates(candidates: list[NumericCandidate]) -> list[NumericCandidate]:
    deduped: list[NumericCandidate] = []
    seen: set[tuple[str, str | None, float | None, str]] = set()
    for candidate in candidates:
        value_key = (
            round(candidate.normalized_value, 6)
            if candidate.normalized_value is not None
            else None
        )
        key = (
            candidate.numeric_type,
            candidate.normalized_unit,
            value_key,
            candidate.placeholder or "",
        )
        if key in seen:
            continue
        seen.add(key)
        candidate.candidate_id = f"NUM_{len(deduped)}"
        deduped.append(candidate)
    return deduped


def _normalize_raw(text: str) -> str:
    return " ".join(str(text).lower().replace(",", ".").split())
