"""Integrity metrics for deterministic graph-slot rendering."""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable


def deterministic_render_metrics(
    prediction_rows: Iterable[dict[str, Any]],
) -> dict[str, float]:
    """Verify that every decoded occurrence is rendered exactly once.

    The renderer trace uses occurrence IDs, so legitimate repetitions of the
    same material placeholder or numeric candidate are counted independently
    instead of being collapsed by value-based de-duplication.
    """

    rows = list(prediction_rows)
    totals = Counter()
    exact_rows = 0
    duplicate_candidate_occurrences = 0
    rendered_duplicate_candidate_occurrences = 0
    duplicate_material_occurrences = 0
    rendered_duplicate_material_occurrences = 0

    for row in rows:
        trace = list(row.get("deterministic_render_trace") or [])
        row_exact = len(trace) == len(row.get("decoded_slots") or [])
        row_source_transparent = row_exact
        row_candidates: list[str] = []
        row_materials: list[str] = []

        for step in trace:
            for field, prefix in (
                ("material_occurrences", "material"),
                ("quantity_occurrences", "quantity"),
                ("condition_occurrences", "condition"),
            ):
                occurrences = list(step.get(field) or [])
                totals[f"{prefix}_decoded"] += len(occurrences)
                rendered = sum(
                    int(occurrence.get("render_count") == 1)
                    for occurrence in occurrences
                )
                totals[f"{prefix}_rendered_once"] += rendered
                row_exact = row_exact and rendered == len(occurrences)

            injected = list(step.get("injected_material_occurrences") or [])
            totals["material_injected"] += len(injected)
            row_exact = row_exact and not injected
            structurally_removed = list(
                step.get("structurally_removed_material_occurrences") or []
            )
            structurally_added = list(
                step.get("structurally_added_material_occurrences") or []
            )
            totals["material_structurally_removed"] += len(structurally_removed)
            totals["material_structurally_added"] += len(structurally_added)
            row_source_transparent = (
                row_source_transparent
                and not structurally_removed
                and not structurally_added
            )

            for occurrence in step.get("quantity_occurrences") or []:
                candidate_id = str(occurrence.get("candidate_id") or "")
                if candidate_id:
                    row_candidates.append(candidate_id)
            for occurrence in step.get("material_occurrences") or []:
                placeholder = str(occurrence.get("placeholder") or "")
                if placeholder:
                    row_materials.append(placeholder)

        candidate_counts = Counter(row_candidates)
        material_counts = Counter(row_materials)
        for candidate_id, count in candidate_counts.items():
            if count <= 1:
                continue
            duplicates = count - 1
            duplicate_candidate_occurrences += duplicates
            rendered_duplicate_candidate_occurrences += max(
                sum(
                    int(occurrence.get("render_count") == 1)
                    for step in trace
                    for occurrence in step.get("quantity_occurrences") or []
                    if str(occurrence.get("candidate_id") or "") == candidate_id
                )
                - 1,
                0,
            )
        for placeholder, count in material_counts.items():
            if count <= 1:
                continue
            duplicates = count - 1
            duplicate_material_occurrences += duplicates
            rendered_duplicate_material_occurrences += max(
                sum(
                    int(occurrence.get("render_count") == 1)
                    for step in trace
                    for occurrence in step.get("material_occurrences") or []
                    if str(occurrence.get("placeholder") or "") == placeholder
                )
                - 1,
                0,
            )

        exact_rows += int(row_exact)
        totals["source_transparent_rows"] += int(row_source_transparent)

    count = len(rows)
    return {
        "deterministic_render_count": float(count),
        "render_all_occurrences_once_rate": exact_rows / max(count, 1),
        "render_material_occurrence_count": float(totals["material_decoded"]),
        "rendered_material_occurrence_count": float(totals["material_rendered_once"]),
        "render_material_occurrence_coverage": (
            totals["material_rendered_once"] / max(totals["material_decoded"], 1)
        ),
        "render_quantity_occurrence_count": float(totals["quantity_decoded"]),
        "rendered_quantity_occurrence_count": float(totals["quantity_rendered_once"]),
        "render_quantity_occurrence_coverage": (
            totals["quantity_rendered_once"] / max(totals["quantity_decoded"], 1)
        ),
        "render_condition_occurrence_count": float(totals["condition_decoded"]),
        "rendered_condition_occurrence_count": float(totals["condition_rendered_once"]),
        "render_condition_occurrence_coverage": (
            totals["condition_rendered_once"] / max(totals["condition_decoded"], 1)
        ),
        "render_injected_material_occurrence_count": float(totals["material_injected"]),
        "render_structurally_removed_material_occurrence_count": float(
            totals["material_structurally_removed"]
        ),
        "render_structurally_added_material_occurrence_count": float(
            totals["material_structurally_added"]
        ),
        "render_source_occurrence_transparency_rate": (
            totals["source_transparent_rows"] / max(count, 1)
        ),
        "render_duplicate_candidate_occurrence_count": float(
            duplicate_candidate_occurrences
        ),
        "render_duplicate_candidate_preservation_rate": (
            rendered_duplicate_candidate_occurrences
            / max(duplicate_candidate_occurrences, 1)
        ),
        "render_duplicate_material_occurrence_count": float(
            duplicate_material_occurrences
        ),
        "render_duplicate_material_preservation_rate": (
            rendered_duplicate_material_occurrences
            / max(duplicate_material_occurrences, 1)
        ),
    }
