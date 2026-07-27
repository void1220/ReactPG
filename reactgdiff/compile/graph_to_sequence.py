"""Deterministic RGDL graph-to-sequence compilers."""

from __future__ import annotations

from typing import Any


def decompile_graph_to_sequence(graph: dict[str, Any], *, mode: str = "exact") -> str:
    """Compile an RGDL graph back to an OpenExp-style action sequence."""

    if mode == "exact":
        return exact_decompile(graph)
    if mode == "canonical":
        return canonical_decompile(graph)
    raise ValueError(f"Unsupported decompile mode: {mode}")


def exact_decompile(graph: dict[str, Any]) -> str:
    """Recover the original action sequence from graph surface metadata."""

    operations = sorted_operations(graph)
    segments = [
        str(operation.get("attrs", {}).get("raw_text", "")).strip().rstrip(".")
        for operation in operations
    ]
    return join_operation_segments(segments, operations)


def canonical_decompile(graph: dict[str, Any]) -> str:
    """Regenerate action syntax from operation nodes and ordered slot nodes."""

    nodes_by_id = {node["id"]: node for node in graph.get("nodes", [])}
    slots_by_operation = operation_slots(graph, nodes_by_id)
    operations = sorted_operations(graph)
    segments = []
    for operation in operations:
        attrs = operation.get("attrs", {})
        operation_type = attrs.get("operation_type") or operation.get("label", "")
        slot_text = "".join(slot.get("attrs", {}).get("text", "") for slot in slots_by_operation.get(operation["id"], []))
        segment = f"{operation_type} {slot_text}".strip()
        segments.append(segment)
    return join_operation_segments(segments, operations)


def sorted_operations(graph: dict[str, Any]) -> list[dict[str, Any]]:
    return sorted(
        (node for node in graph.get("nodes", []) if node.get("type") == "operation"),
        key=lambda node: int(node.get("attrs", {}).get("step_id", 0)),
    )


def join_operation_segments(
    segments: list[str],
    operations: list[dict[str, Any]],
) -> str:
    segments = [segment for segment in segments if segment]
    if not segments:
        return ""
    rendered = segments[0]
    for idx, segment in enumerate(segments[1:], start=1):
        previous_operation = operations[idx - 1]
        separator = previous_operation.get("attrs", {}).get("separator_after") or " ; "
        rendered += separator + segment
    return rendered + "."


def operation_slots(
    graph: dict[str, Any],
    nodes_by_id: dict[str, dict[str, Any]] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Collect slot nodes connected to each operation by ordered ``has_slot`` edges."""

    if nodes_by_id is None:
        nodes_by_id = {node["id"]: node for node in graph.get("nodes", [])}
    slots: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    for edge in graph.get("edges", []):
        if edge.get("type") != "has_slot":
            continue
        slot = nodes_by_id.get(edge.get("target"))
        if not slot:
            continue
        order = int(edge.get("attrs", {}).get("order", slot.get("attrs", {}).get("order", 0)))
        slots.setdefault(edge["source"], []).append((order, slot))
    return {
        operation_id: [slot for _, slot in sorted(items, key=lambda item: item[0])]
        for operation_id, items in slots.items()
    }


def normalized_sequence(sequence: str) -> str:
    """Normalize superficial spacing for exact round-trip checks."""

    return " ".join(sequence.strip().split()).rstrip(".") + "."


def exact_match(graph: dict[str, Any]) -> bool:
    original = graph.get("surface", {}).get("original_actions", "")
    recovered = exact_decompile(graph)
    return normalized_sequence(str(original)) == normalized_sequence(recovered)


def canonical_match(graph: dict[str, Any]) -> bool:
    original = graph.get("surface", {}).get("original_actions", "")
    recovered = canonical_decompile(graph)
    return normalized_sequence(str(original)) == normalized_sequence(recovered)
