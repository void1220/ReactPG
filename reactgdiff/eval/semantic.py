"""Template-insensitive semantic metrics for OpenExp action strings.

The metrics operate only on prediction/reference text, so the same evaluator
can score ReactXT outputs and graph-decoder outputs without access to hidden
model slots. Surface argument wording is discarded, while operation order,
placeholder bindings, condition references, normalized units, and normalized
numeric values remain observable.
"""

from __future__ import annotations

from collections import Counter
from typing import Iterable

from reactgdiff.data.action_parser import ActionStep, parse_action_sequence
from reactgdiff.data.numeric_evidence import (
    infer_numeric_type,
    normalize_unit,
    parse_numeric_value_unit,
)
from reactgdiff.eval.lev import edit_distance


def corpus_semantic_metrics(pairs: Iterable[tuple[str, str]]) -> dict[str, float]:
    """Score action semantics after deterministic OpenExp parsing.

    ``semantic_score`` is a bounded composite intended for checkpoint selection:
    operation order 40%, material bindings 20%, duration/temperature references
    15%, numeric type/unit 15%, and normalized numeric value/unit 10%.
    Component metrics are also returned and should be reported alongside it.
    """

    pairs = list(pairs)
    metric_names = (
        "semantic_operation_similarity",
        "semantic_operation_exact_rate",
        "semantic_material_f1",
        "semantic_condition_f1",
        "semantic_numeric_type_unit_f1",
        "semantic_numeric_value_unit_f1",
        "semantic_procedure_exact_rate",
        "semantic_score",
        "semantic_score_75_rate",
        "canonical_mean_levenshtein_similarity",
        "canonical_levenshtein_90_rate",
        "canonical_levenshtein_75_rate",
        "canonical_levenshtein_50_rate",
    )
    if not pairs:
        return {"semantic_count": 0.0, **{name: 0.0 for name in metric_names}}

    totals = {name: 0.0 for name in metric_names}
    for prediction, reference in pairs:
        pred_steps = parse_action_sequence(prediction)
        ref_steps = parse_action_sequence(reference)
        pred_ops = [step.operation_type for step in pred_steps]
        ref_ops = [step.operation_type for step in ref_steps]
        operation_similarity = 1.0 - edit_distance(pred_ops, ref_ops) / max(
            len(pred_ops), len(ref_ops), 1
        )
        operation_exact = float(pred_ops == ref_ops)
        material_f1 = _counter_f1(
            _material_items(pred_steps),
            _material_items(ref_steps),
        )
        condition_f1 = _counter_f1(
            _condition_items(pred_steps),
            _condition_items(ref_steps),
        )
        numeric_type_unit_f1 = _counter_f1(
            _numeric_items(pred_steps, include_value=False),
            _numeric_items(ref_steps, include_value=False),
        )
        numeric_value_unit_f1 = _counter_f1(
            _numeric_items(pred_steps, include_value=True),
            _numeric_items(ref_steps, include_value=True),
        )

        pred_canonical = canonical_action_tokens(prediction)
        ref_canonical = canonical_action_tokens(reference)
        canonical_similarity = 1.0 - edit_distance(pred_canonical, ref_canonical) / max(
            len(pred_canonical), len(ref_canonical), 1
        )
        procedure_exact = float(pred_canonical == ref_canonical)
        semantic_score = min(
            max(
                0.40 * operation_similarity
                + 0.20 * material_f1
                + 0.15 * condition_f1
                + 0.15 * numeric_type_unit_f1
                + 0.10 * numeric_value_unit_f1,
                0.0,
            ),
            1.0,
        )

        totals["semantic_operation_similarity"] += operation_similarity
        totals["semantic_operation_exact_rate"] += operation_exact
        totals["semantic_material_f1"] += material_f1
        totals["semantic_condition_f1"] += condition_f1
        totals["semantic_numeric_type_unit_f1"] += numeric_type_unit_f1
        totals["semantic_numeric_value_unit_f1"] += numeric_value_unit_f1
        totals["semantic_procedure_exact_rate"] += procedure_exact
        totals["semantic_score"] += semantic_score
        totals["semantic_score_75_rate"] += float(semantic_score >= 0.75)
        totals["canonical_mean_levenshtein_similarity"] += canonical_similarity
        totals["canonical_levenshtein_90_rate"] += float(canonical_similarity >= 0.90)
        totals["canonical_levenshtein_75_rate"] += float(canonical_similarity >= 0.75)
        totals["canonical_levenshtein_50_rate"] += float(canonical_similarity >= 0.50)

    count = len(pairs)
    return {
        "semantic_count": float(count),
        **{name: value / count for name, value in totals.items()},
    }


def canonical_action_signature(text: str, *, include_numeric_values: bool = True) -> str:
    """Return a stable signature independent of natural-language templates."""

    return " ; ".join(
        _canonical_step(step, include_numeric_values=include_numeric_values)
        for step in parse_action_sequence(text)
    )


def canonical_action_tokens(
    text: str,
    *,
    include_numeric_values: bool = True,
) -> list[str]:
    """Return field tokens for efficient, interpretable canonical edit distance."""

    tokens: list[str] = []
    for step in parse_action_sequence(text):
        tokens.append(f"OP:{step.operation_type}")
        tokens.extend(f"MAT:{ref}" for ref in step.material_refs)
        tokens.extend(f"DUR:{ref}" for ref in step.duration_refs)
        tokens.extend(f"TEMP:{ref}" for ref in step.temperature_refs)
        tokens.extend(
            f"NUM:{item[1]}"
            for item in _step_numeric_items(
                step,
                include_value=include_numeric_values,
            )
        )
        tokens.append("<STEP>")
    return tokens


def _canonical_step(step: ActionStep, *, include_numeric_values: bool) -> str:
    materials = ",".join(step.material_refs) or "-"
    durations = ",".join(step.duration_refs) or "-"
    temperatures = ",".join(step.temperature_refs) or "-"
    quantities = ",".join(
        item[1]
        for item in _step_numeric_items(step, include_value=include_numeric_values)
    ) or "-"
    return (
        f"OP={step.operation_type}|MAT={materials}|DUR={durations}|"
        f"TEMP={temperatures}|NUM={quantities}"
    )


def _material_items(steps: list[ActionStep]) -> Counter[tuple[str, str]]:
    return Counter(
        (step.operation_type, material_ref)
        for step in steps
        for material_ref in step.material_refs
    )


def _condition_items(steps: list[ActionStep]) -> Counter[tuple[str, str, str]]:
    items: Counter[tuple[str, str, str]] = Counter()
    for step in steps:
        items.update((step.operation_type, "duration", ref) for ref in step.duration_refs)
        items.update((step.operation_type, "temperature", ref) for ref in step.temperature_refs)
    return items


def _numeric_items(
    steps: list[ActionStep],
    *,
    include_value: bool,
) -> Counter[tuple[str, str]]:
    return Counter(
        item
        for step in steps
        for item in _step_numeric_items(step, include_value=include_value)
    )


def _step_numeric_items(
    step: ActionStep,
    *,
    include_value: bool,
) -> list[tuple[str, str]]:
    items: list[tuple[str, str]] = []
    for raw_quantity in step.quantities:
        parsed = parse_numeric_value_unit(raw_quantity)
        if parsed is None:
            continue
        value, unit = parsed
        normalized_unit = normalize_unit(unit)
        numeric_type = infer_numeric_type(raw_quantity, normalized_unit)
        signature = f"{numeric_type}:{normalized_unit}"
        if include_value:
            signature += f":{_normalized_number(value)}"
        items.append((step.operation_type, signature))
    return items


def _normalized_number(value: float) -> str:
    if abs(value - round(value)) <= 1e-8:
        return str(int(round(value)))
    return f"{value:.8g}"


def _counter_f1(prediction: Counter, reference: Counter) -> float:
    pred_count = sum(prediction.values())
    ref_count = sum(reference.values())
    if pred_count == 0 and ref_count == 0:
        return 1.0
    if pred_count == 0 or ref_count == 0:
        return 0.0
    true_positive = sum((prediction & reference).values())
    precision = true_positive / pred_count
    recall = true_positive / ref_count
    return 0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall)
