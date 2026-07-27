"""Traceability helpers for RGDL compiled graphs."""

from __future__ import annotations

from typing import Any


def operation_trace(graph: dict[str, Any]) -> list[dict[str, Any]]:
    """Return operation-level alignment between graph nodes and surface steps."""

    operation_nodes = {
        node["id"]: node
        for node in graph.get("nodes", [])
        if node.get("type") == "operation"
    }
    by_step = {
        int(node.get("attrs", {}).get("step_id", -1)): node_id
        for node_id, node in operation_nodes.items()
    }
    mentions_by_op: dict[str, list[str]] = {}
    conditions_by_op: dict[str, list[str]] = {}
    outputs_by_op: dict[str, list[str]] = {}
    for edge in graph.get("edges", []):
        if edge.get("type") == "mentions":
            mentions_by_op.setdefault(edge["source"], []).append(edge["target"])
        elif edge.get("type") == "has_condition":
            conditions_by_op.setdefault(edge["source"], []).append(edge["target"])
        elif edge.get("type") == "output_from":
            outputs_by_op.setdefault(edge["source"], []).append(edge["target"])

    rows = []
    for step in sorted(
        graph.get("surface", {}).get("steps", []),
        key=lambda item: int(item.get("step_id", 0)),
    ):
        step_id = int(step.get("step_id", 0))
        operation_id = by_step.get(step_id)
        rows.append(
            {
                "step_id": step_id,
                "compiled_action": step.get("raw_text", ""),
                "source_operation_node": operation_id,
                "material_mention_nodes": mentions_by_op.get(operation_id, []),
                "condition_nodes": conditions_by_op.get(operation_id, []),
                "output_nodes": outputs_by_op.get(operation_id, []),
            }
        )
    return rows
