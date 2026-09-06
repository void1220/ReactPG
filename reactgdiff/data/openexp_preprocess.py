"""Cleaning, feature extraction, and bucketing for OpenExp."""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from reactgdiff.data.action_parser import (
    KNOWN_OPENEXP_ACTIONS,
    MATERIAL_REF_RE,
    DURATION_REF_RE,
    TEMPERATURE_REF_RE,
    ActionStep,
    parse_action_sequence,
)

VOLUME_UNITS = {"ul", "µl", "ml", "l"}
MASS_UNITS = {"ug", "µg", "mg", "g", "kg"}
AMOUNT_UNITS = {"umol", "µmol", "mmol", "mol"}
CONCENTRATION_UNITS = {"molar", "normal", "m", "n"}
EQUIVALENT_UNITS = {"equiv", "equivalent", "eq"}
PERCENT_UNITS = {"percent", "%"}
UNIT_RE = re.compile(
    r"(?<![A-Za-z])(?P<unit>ug|µg|mg|g|kg|ul|µl|ml|l|umol|µmol|mmol|mol|"
    r"molar|normal|equiv|equivalent|eq|percent|%|M|N)\b",
    re.IGNORECASE,
)
NUMERIC_RE = re.compile(r"[-+]?\d+(?:[.;]\d+)?")

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
CONDITION_OPS = {
    "STIR",
    "WAIT",
    "REFLUX",
    "MICROWAVE",
    "SONICATE",
    "SETTEMPERATURE",
    "DRYSOLID",
    "DEGAS",
    "ADD",
    "QUENCH",
    "PH",
}


@dataclass(slots=True)
class PreparedRecord:
    """OpenExp row plus derived metadata."""

    record: dict[str, Any]
    features: dict[str, Any]
    quality: dict[str, Any]
    buckets: dict[str, Any]
    split: str

    def to_dict(self) -> dict[str, Any]:
        output = dict(self.record)
        output["_features"] = self.features
        output["_quality"] = self.quality
        output["_buckets"] = self.buckets
        output["_split"] = self.split
        return output


@dataclass
class SummaryBuilder:
    """Accumulate a compact preparation report."""

    total_records: int = 0
    kept_records: int = 0
    rejected_records: int = 0
    split_counts: Counter[str] = field(default_factory=Counter)
    scale_counts: Counter[str] = field(default_factory=Counter)
    special_bucket_counts: Counter[str] = field(default_factory=Counter)
    reject_reasons: Counter[str] = field(default_factory=Counter)
    action_counts: Counter[str] = field(default_factory=Counter)
    action_count_histogram: Counter[int] = field(default_factory=Counter)
    examples: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    def add_prepared(self, prepared: PreparedRecord) -> None:
        self.total_records += 1
        if prepared.quality["is_clean"]:
            self.kept_records += 1
            self.split_counts[prepared.split] += 1
            scale = prepared.buckets["scale"]
            self.scale_counts[scale] += 1
            for name, enabled in prepared.buckets["special"].items():
                if enabled:
                    self.special_bucket_counts[name] += 1
                    self._add_example(name, prepared)
            self.action_count_histogram[prepared.features["action_count"]] += 1
            self.action_counts.update(prepared.features["action_histogram"])
            self._add_example(scale, prepared)
        else:
            self.rejected_records += 1
            self.reject_reasons.update(prepared.quality["reasons"])
            self._add_example("rejected", prepared)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_records": self.total_records,
            "kept_records": self.kept_records,
            "rejected_records": self.rejected_records,
            "split_counts": dict(self.split_counts),
            "scale_counts": dict(self.scale_counts),
            "special_bucket_counts": dict(self.special_bucket_counts),
            "reject_reasons": dict(self.reject_reasons),
            "top_actions": self.action_counts.most_common(),
            "action_count_histogram": dict(sorted(self.action_count_histogram.items())),
            "examples": self.examples,
            "criteria": preparation_criteria(),
        }

    def _add_example(self, bucket_name: str, prepared: PreparedRecord) -> None:
        examples = self.examples.setdefault(bucket_name, [])
        if len(examples) >= 5:
            return
        examples.append(
            {
                "index": prepared.record.get("index"),
                "split": prepared.split,
                "actions": prepared.record.get("actions"),
                "features": {
                    key: prepared.features[key]
                    for key in (
                        "action_count",
                        "material_entity_count",
                        "quantity_count",
                        "quantity_component_count",
                        "duration_ref_count",
                        "temperature_ref_count",
                        "branch_op_count",
                        "complexity_score",
                    )
                },
                "quality": prepared.quality,
                "buckets": prepared.buckets,
            }
        )


def prepare_openexp_record(record: dict[str, Any]) -> PreparedRecord:
    steps = parse_action_sequence(str(record.get("actions", "")))
    features = extract_openexp_features(record, steps)
    quality = assess_quality(record, steps, features)
    split = assign_split(record)
    buckets = assign_buckets(features, quality)
    return PreparedRecord(
        record=record,
        features=features,
        quality=quality,
        buckets=buckets,
        split=split,
    )


def extract_openexp_features(record: dict[str, Any], steps: list[ActionStep]) -> dict[str, Any]:
    action_histogram = Counter(step.operation_type for step in steps)
    material_placeholders = set((record.get("extracted_molecules") or {}).values())
    duration_placeholders = set((record.get("extracted_duration") or {}).values())
    temperature_placeholders = set((record.get("extracted_temperature") or {}).values())
    material_refs = [ref for step in steps for ref in step.material_refs]
    duration_refs = [ref for step in steps for ref in step.duration_refs]
    temperature_refs = [ref for step in steps for ref in step.temperature_refs]
    quantities = [quantity for step in steps for quantity in step.quantities]
    quantity_components = [component for quantity in quantities for component in split_quantity_components(quantity)]
    unit_counts = Counter()
    quantity_type_counts = Counter()
    for component in quantity_components:
        for unit in extract_units(component):
            unit_counts[unit] += 1
            quantity_type_counts[unit_type(unit)] += 1

    timed_temperature_steps = sum(
        1 for step in steps if step.duration_refs and step.temperature_refs
    )
    literal_material_step_count = sum(1 for step in steps if has_literal_material_arg(step))
    branch_op_count = sum(action_histogram[op] for op in BRANCH_OPS)
    workup_op_count = sum(action_histogram[op] for op in WORKUP_OPS)
    condition_op_count = sum(action_histogram[op] for op in CONDITION_OPS)
    condition_ref_count = len(duration_refs) + len(temperature_refs)
    material_entity_count = len(material_placeholders)
    participant_count = (
        len(record.get("REACTANT") or [])
        + len(record.get("PRODUCT") or [])
        + len(record.get("CATALYST") or [])
        + len(record.get("SOLVENT") or [])
    )
    complexity_score = (
        len(steps)
        + 0.6 * material_entity_count
        + 0.8 * condition_ref_count
        + 0.45 * len(quantity_components)
        + 0.9 * branch_op_count
        + 0.4 * literal_material_step_count
    )

    return {
        "action_count": len(steps),
        "unique_action_count": len(action_histogram),
        "action_histogram": dict(action_histogram),
        "unknown_action_count": sum(
            count for op, count in action_histogram.items() if op not in KNOWN_OPENEXP_ACTIONS
        ),
        "material_entity_count": material_entity_count,
        "material_ref_count": len(material_refs),
        "unique_material_ref_count": len(set(material_refs)),
        "duration_ref_count": len(duration_refs),
        "temperature_ref_count": len(temperature_refs),
        "condition_ref_count": condition_ref_count,
        "duration_symbol_count": len(duration_placeholders),
        "temperature_symbol_count": len(temperature_placeholders),
        "quantity_count": len(quantities),
        "quantity_component_count": len(quantity_components),
        "quantity_type_counts": dict(quantity_type_counts),
        "unit_counts": dict(unit_counts),
        "timed_temperature_step_count": timed_temperature_steps,
        "literal_material_step_count": literal_material_step_count,
        "branch_op_count": branch_op_count,
        "workup_op_count": workup_op_count,
        "condition_op_count": condition_op_count,
        "participant_count": participant_count,
        "reactant_count": len(record.get("REACTANT") or []),
        "product_count": len(record.get("PRODUCT") or []),
        "catalyst_count": len(record.get("CATALYST") or []),
        "solvent_count": len(record.get("SOLVENT") or []),
        "source_char_count": len(str(record.get("source", ""))),
        "action_char_count": len(str(record.get("actions", ""))),
        "complexity_score": round(complexity_score, 3),
    }


def assess_quality(
    record: dict[str, Any],
    steps: list[ActionStep],
    features: dict[str, Any],
) -> dict[str, Any]:
    reasons: list[str] = []
    required_fields = (
        "index",
        "REACTANT",
        "PRODUCT",
        "actions",
        "source",
        "extracted_molecules",
        "molecules",
        "score",
    )
    for field_name in required_fields:
        value = record.get(field_name)
        if value in (None, "", [], {}):
            reasons.append(f"missing_{field_name}")

    if not steps:
        reasons.append("empty_action_sequence")
    if features["unknown_action_count"]:
        reasons.append("unknown_action")
    if "YIELD" not in features["action_histogram"]:
        reasons.append("missing_yield")
    if len(record.get("PRODUCT") or []) != 1:
        reasons.append("product_count_not_one")

    material_symbols = set((record.get("extracted_molecules") or {}).values())
    duration_symbols = set((record.get("extracted_duration") or {}).values())
    temperature_symbols = set((record.get("extracted_temperature") or {}).values())
    material_refs = {ref for step in steps for ref in step.material_refs}
    duration_refs = {ref for step in steps for ref in step.duration_refs}
    temperature_refs = {ref for step in steps for ref in step.temperature_refs}
    if missing := sorted(material_refs - material_symbols):
        reasons.append("unresolved_material_ref")
    else:
        missing = []
    if missing_duration := sorted(duration_refs - duration_symbols):
        reasons.append("unresolved_duration_ref")
    else:
        missing_duration = []
    if missing_temperature := sorted(temperature_refs - temperature_symbols):
        reasons.append("unresolved_temperature_ref")
    else:
        missing_temperature = []

    return {
        "is_clean": not reasons,
        "reasons": reasons,
        "missing_material_refs": missing,
        "missing_duration_refs": missing_duration,
        "missing_temperature_refs": missing_temperature,
    }


def assign_buckets(features: dict[str, Any], quality: dict[str, Any]) -> dict[str, Any]:
    action_count = features["action_count"]
    if action_count <= 8:
        scale = "small"
    elif action_count <= 13:
        scale = "medium"
    else:
        scale = "large"

    numeric_heavy = (
        features["quantity_component_count"] >= 14
        or features["quantity_count"] >= 10
        or len(features["quantity_type_counts"]) >= 4
    )
    condition_heavy = (
        features["condition_ref_count"] >= 5
        or features["timed_temperature_step_count"] >= 3
        or features["duration_ref_count"] >= 3
        or features["temperature_ref_count"] >= 3
    )
    multi_reference = (
        features["unique_material_ref_count"] >= 7
        or features["material_ref_count"] >= 10
        or features["material_entity_count"] >= 8
    )
    branch_workup = features["branch_op_count"] >= 4 or (
        features["workup_op_count"] >= 8 and features["branch_op_count"] >= 2
    )
    hard_numeric_condition = numeric_heavy or condition_heavy
    complex_overall = (
        hard_numeric_condition
        or multi_reference
        or branch_workup
        or features["complexity_score"] >= 31
    )

    return {
        "scale": scale,
        "special": {
            "numeric_heavy": bool(numeric_heavy),
            "condition_heavy": bool(condition_heavy),
            "hard_numeric_condition": bool(hard_numeric_condition),
            "multi_reference": bool(multi_reference),
            "branch_workup": bool(branch_workup),
            "complex_overall": bool(complex_overall),
        },
        "eligible_for_main": bool(quality["is_clean"]),
    }


def assign_split(record: dict[str, Any]) -> str:
    key = reaction_key(record)
    bucket = int(hashlib.sha1(key.encode("utf-8")).hexdigest()[:8], 16) % 1000
    if bucket < 800:
        return "train"
    if bucket < 900:
        return "val"
    return "test"


def reaction_key(record: dict[str, Any]) -> str:
    reactants = ".".join(sorted(record.get("REACTANT") or []))
    products = ".".join(sorted(record.get("PRODUCT") or []))
    catalysts = ".".join(sorted(record.get("CATALYST") or []))
    solvents = ".".join(sorted(record.get("SOLVENT") or []))
    return "|".join((reactants, products, catalysts, solvents))


def split_quantity_components(quantity: str) -> list[str]:
    parts = []
    for raw_part in re.split(r",", quantity):
        part = raw_part.strip()
        if part:
            parts.append(part)
    return parts or [quantity.strip()]


def extract_units(component: str) -> list[str]:
    units: list[str] = []
    for match in UNIT_RE.finditer(component):
        units.append(match.group("unit").lower())
    return units


def unit_type(unit: str) -> str:
    normalized = unit.lower()
    if normalized in VOLUME_UNITS:
        return "volume"
    if normalized in MASS_UNITS:
        return "mass"
    if normalized in AMOUNT_UNITS:
        return "amount"
    if normalized in CONCENTRATION_UNITS:
        return "concentration"
    if normalized in EQUIVALENT_UNITS:
        return "equivalent"
    if normalized in PERCENT_UNITS:
        return "percent"
    return "other"


def has_literal_material_arg(step: ActionStep) -> bool:
    if step.material_refs:
        return False
    text = f" {step.raw_text} "
    return any(token in text for token in (" with ", " from ", " over ")) or step.operation_type in {
        "ADD",
        "QUENCH",
        "PH",
    }


def preparation_criteria() -> dict[str, Any]:
    return {
        "main_clean_requirements": [
            "required OpenExp fields are present",
            "action sequence parses into at least one step",
            "all operation types are in the OpenExp action vocabulary",
            "all $material, @duration, and #temperature references resolve to extracted symbols",
            "a YIELD action is present",
            "exactly one PRODUCT is present",
        ],
        "splitting": {
            "method": "deterministic SHA1 hash of sorted reaction participants",
            "train": "hash bucket < 800 / 1000",
            "val": "800 <= hash bucket < 900 / 1000",
            "test": "hash bucket >= 900 / 1000",
        },
        "scale_buckets": {
            "small": "action_count <= 8",
            "medium": "9 <= action_count <= 13",
            "large": "action_count >= 14",
        },
        "special_buckets": {
            "numeric_heavy": "quantity_component_count >= 14 OR quantity_count >= 10 OR at least 4 quantity types",
            "condition_heavy": "condition_ref_count >= 5 OR at least 3 timed-temperature steps OR duration_ref_count >= 3 OR temperature_ref_count >= 3",
            "hard_numeric_condition": "numeric_heavy OR condition_heavy",
            "multi_reference": "unique_material_ref_count >= 7 OR material_ref_count >= 10 OR material_entity_count >= 8",
            "branch_workup": "branch_op_count >= 4 OR workup_op_count >= 8 with at least two branch ops",
            "complex_overall": "any hard/special bucket OR complexity_score >= 31",
        },
    }
