"""Discrete graph-slot metrics for decoded procedure graphs."""

from __future__ import annotations

from typing import Any, Iterable

from reactgdiff.eval.lev import edit_distance
from reactgdiff.models.graph_codec import GraphTargetCodec, NONE_TOKEN


def discrete_slot_metrics(
    prediction_rows: Iterable[dict[str, Any]],
    reference_records: Iterable[dict[str, Any]],
    *,
    codec: GraphTargetCodec,
) -> dict[str, float]:
    """Compare decoded graph slots with target graph slots.

    These metrics intentionally evaluate the discrete graph skeleton: operation
    sequence, material references, duration/temperature condition slots, and
    numeric-slot presence. They do not compare rendered argument text.
    """

    rows = list(prediction_rows)
    records = list(reference_records)
    if len(rows) != len(records):
        raise ValueError(f"prediction/reference count mismatch: {len(rows)} != {len(records)}")

    totals = {
        "operation_sequence_similarity": 0.0,
        "operation_slot_accuracy": 0.0,
        "material_slot_accuracy": 0.0,
        "material_pointer_accuracy": 0.0,
        "condition_slot_accuracy": 0.0,
        "numeric_slot_accuracy": 0.0,
        "unit_slot_accuracy": 0.0,
        "numeric_candidate_step_accuracy": 0.0,
        "discrete_slot_score": 0.0,
        "length_accuracy": 0.0,
        "absolute_length_error": 0.0,
        "predicted_steps": 0.0,
        "reference_steps": 0.0,
        "operation_sequence_exact_rate": 0.0,
        "discrete_graph_exact_rate": 0.0,
        "grounded_discrete_graph_exact_rate": 0.0,
        "unsupported_numeric_rate": 0.0,
    }
    unsupported_numeric_count = 0
    predicted_numeric_count = 0
    reference_numeric_candidate_count = 0
    numeric_candidate_match_count = 0
    reference_evidence_candidate_count = 0
    evidence_candidate_match_count = 0

    for row, record in zip(rows, records, strict=True):
        pred_slots = list(row.get("decoded_slots") or [])
        ref_slots = codec.target_slots_from_record(record)
        pred_ops = [_operation(slot) for slot in pred_slots]
        ref_ops = [_operation(slot) for slot in ref_slots]
        max_len = max(len(pred_slots), len(ref_slots), 1)

        op_similarity = 1.0 - edit_distance(pred_ops, ref_ops) / max(len(pred_ops), len(ref_ops), 1)
        length_error = abs(len(pred_slots) - len(ref_slots))
        length_accuracy = 1.0 - length_error / max_len

        op_matches = 0
        material_matches = 0
        condition_matches = 0
        numeric_matches = 0
        unit_matches = 0
        numeric_candidate_matches = 0
        graph_exact = len(pred_slots) == len(ref_slots)
        grounded_graph_exact = graph_exact

        for slot_idx in range(max_len):
            pred = pred_slots[slot_idx] if slot_idx < len(pred_slots) else None
            ref = ref_slots[slot_idx] if slot_idx < len(ref_slots) else None
            if pred is None or ref is None:
                graph_exact = False
                grounded_graph_exact = False
                continue

            op_match = _operation(pred) == _operation(ref)
            material_match = _material_signature(pred, codec=codec) == _material_signature(ref, codec=codec)
            condition_match = _condition_signature(pred) == _condition_signature(ref)
            numeric_match = _numeric_signature(pred, codec=codec) == _numeric_signature(ref, codec=codec)
            unit_match = _unit_signature(pred, codec=codec) == _unit_signature(ref, codec=codec)
            candidate_match = _numeric_candidate_signature(pred, codec=codec) == _numeric_candidate_signature(
                ref,
                codec=codec,
            )

            op_matches += int(op_match)
            material_matches += int(material_match)
            condition_matches += int(condition_match)
            numeric_matches += int(numeric_match)
            unit_matches += int(unit_match)
            numeric_candidate_matches += int(candidate_match)
            graph_exact = graph_exact and op_match and material_match and condition_match and numeric_match
            grounded_graph_exact = (
                grounded_graph_exact
                and op_match
                and material_match
                and condition_match
                and numeric_match
                and unit_match
                and candidate_match
            )
            pred_candidate_signature = _numeric_candidate_signature(pred, codec=codec)
            ref_candidate_signature = _numeric_candidate_signature(ref, codec=codec)
            for pred_candidate, ref_candidate in zip(
                pred_candidate_signature,
                ref_candidate_signature,
                strict=True,
            ):
                if ref_candidate == NONE_TOKEN:
                    continue
                reference_numeric_candidate_count += 1
                numeric_candidate_match_count += int(pred_candidate == ref_candidate)
                if ref_candidate.startswith("NUM_"):
                    reference_evidence_candidate_count += 1
                    evidence_candidate_match_count += int(pred_candidate == ref_candidate)
            for quantity in pred.get("quantity_slots") or []:
                predicted_numeric_count += 1
                source = str(quantity.get("source") or "")
                candidate_id = str(quantity.get("candidate_id") or "")
                if source in {"model_continuous", "missing_evidence"} or not candidate_id:
                    unsupported_numeric_count += 1

        op_accuracy = op_matches / max_len
        material_accuracy = material_matches / max_len
        condition_accuracy = condition_matches / max_len
        numeric_accuracy = numeric_matches / max_len
        unit_accuracy = unit_matches / max_len
        numeric_candidate_accuracy = numeric_candidate_matches / max_len
        discrete_slot_score = (
            op_accuracy
            + material_accuracy
            + condition_accuracy
            + numeric_accuracy
            + length_accuracy
        ) / 5.0

        totals["operation_sequence_similarity"] += op_similarity
        totals["operation_slot_accuracy"] += op_accuracy
        totals["material_slot_accuracy"] += material_accuracy
        totals["material_pointer_accuracy"] += material_accuracy
        totals["condition_slot_accuracy"] += condition_accuracy
        totals["numeric_slot_accuracy"] += numeric_accuracy
        totals["unit_slot_accuracy"] += unit_accuracy
        totals["numeric_candidate_step_accuracy"] += numeric_candidate_accuracy
        totals["discrete_slot_score"] += discrete_slot_score
        totals["length_accuracy"] += length_accuracy
        totals["absolute_length_error"] += float(length_error)
        totals["predicted_steps"] += float(len(pred_slots))
        totals["reference_steps"] += float(len(ref_slots))
        totals["operation_sequence_exact_rate"] += float(pred_ops == ref_ops)
        totals["discrete_graph_exact_rate"] += float(graph_exact)
        totals["grounded_discrete_graph_exact_rate"] += float(grounded_graph_exact)

    total = len(rows)
    if total == 0:
        return {
            "discrete_eval_count": 0.0,
            **{key: 0.0 for key in totals},
            "numeric_candidate_pointer_accuracy": 0.0,
            "numeric_evidence_candidate_pointer_accuracy": 0.0,
            "numeric_candidate_target_count": 0.0,
            "numeric_evidence_candidate_target_count": 0.0,
        }

    metrics = {
        "discrete_eval_count": float(total),
        **{
            key: (
                unsupported_numeric_count / max(predicted_numeric_count, 1)
                if key == "unsupported_numeric_rate"
                else value / total
            )
            for key, value in totals.items()
        },
    }
    metrics.update(
        {
            "numeric_candidate_pointer_accuracy": (
                numeric_candidate_match_count / max(reference_numeric_candidate_count, 1)
            ),
            "numeric_evidence_candidate_pointer_accuracy": (
                evidence_candidate_match_count / max(reference_evidence_candidate_count, 1)
            ),
            "numeric_candidate_target_count": float(reference_numeric_candidate_count),
            "numeric_evidence_candidate_target_count": float(
                reference_evidence_candidate_count
            ),
        }
    )
    return metrics


def _operation(slot: dict[str, Any]) -> str:
    return str(slot.get("operation_type") or "")


def _material_signature(slot: dict[str, Any], *, codec: GraphTargetCodec) -> tuple[str, ...]:
    refs = list(slot.get("material_refs") or [])
    if not refs and slot.get("material_ref") not in (None, NONE_TOKEN):
        refs = [str(slot["material_ref"])]
    refs = [str(ref) for ref in refs if str(ref) != NONE_TOKEN]
    refs = refs[: codec.max_material_slots]
    refs.extend([NONE_TOKEN] * max(0, codec.max_material_slots - len(refs)))
    return tuple(refs)


def _condition_signature(slot: dict[str, Any]) -> tuple[str, str]:
    return (
        str(slot.get("duration_ref") or ""),
        str(slot.get("temperature_ref") or ""),
    )


def _numeric_signature(slot: dict[str, Any], *, codec: GraphTargetCodec) -> tuple[int, ...]:
    signature = [0] * codec.max_material_slots
    for quantity in slot.get("quantity_slots") or []:
        slot_id = int(quantity.get("slot_id", 0))
        if 0 <= slot_id < codec.max_material_slots:
            signature[slot_id] = 1
    return tuple(signature)


def _unit_signature(slot: dict[str, Any], *, codec: GraphTargetCodec) -> tuple[str, ...]:
    signature = [NONE_TOKEN] * codec.max_material_slots
    for quantity in slot.get("quantity_slots") or []:
        slot_id = int(quantity.get("slot_id", 0))
        if 0 <= slot_id < codec.max_material_slots:
            signature[slot_id] = str(quantity.get("unit") or NONE_TOKEN)
    return tuple(signature)


def _numeric_candidate_signature(slot: dict[str, Any], *, codec: GraphTargetCodec) -> tuple[str, ...]:
    signature = [NONE_TOKEN] * codec.max_material_slots
    for quantity in slot.get("quantity_slots") or []:
        slot_id = int(quantity.get("slot_id", 0))
        if 0 <= slot_id < codec.max_material_slots:
            signature[slot_id] = str(quantity.get("candidate_id") or NONE_TOKEN)
    return tuple(signature)
