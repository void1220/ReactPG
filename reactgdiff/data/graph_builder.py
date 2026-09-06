"""Build lightweight heterogeneous process graphs from OpenExp records."""

from __future__ import annotations

import re
from typing import Any

from reactgdiff.data.action_parser import ActionStep, parse_action_sequence
from reactgdiff.data.graph_schema import ProcessGraph

TERMINAL_OUTPUT_OPERATIONS = {"YIELD", "OBTAIN", "ISOLATE", "COLLECT"}


def build_process_graph(record: dict[str, Any]) -> dict[str, Any]:
    """Convert one OpenExp record into a JSON-serializable process graph."""

    graph = ProcessGraph(
        graph_id=f"openexp_{record.get('index', 'unknown')}",
        metadata={
            "dataset": "OpenExp",
            "index": record.get("index"),
            "actions": record.get("actions", ""),
            "source": record.get("source", ""),
            "score": record.get("score"),
            "reactants": record.get("REACTANT", []),
            "products": record.get("PRODUCT", []),
            "catalysts": record.get("CATALYST", []),
            "solvents": record.get("SOLVENT", []),
        },
    )

    placeholder_to_smiles = _placeholder_to_smiles(record)
    smiles_to_names = _smiles_to_names(record)
    placeholder_to_condition = _placeholder_to_condition(record)
    material_node_ids = _add_material_nodes(graph, record, placeholder_to_smiles, smiles_to_names)
    condition_node_ids = _add_condition_nodes(graph, placeholder_to_condition)

    steps = parse_action_sequence(str(record.get("actions", "")))
    previous_operation_id: str | None = None
    previous_state_id: str | None = None

    for step in steps:
        operation_id = f"op_{step.step_id:03d}"
        graph.add_node(
            operation_id,
            "operation",
            step.operation_type,
            {
                "step_id": step.step_id,
                "raw_text": step.raw_text,
                "arguments": step.arguments,
                "quantities": step.quantities,
            },
        )

        if previous_operation_id is not None:
            graph.add_edge(previous_operation_id, operation_id, "precede")
        if previous_state_id is not None and _uses_previous_state(step):
            graph.add_edge(
                previous_state_id,
                operation_id,
                "input_to",
                {"implicit": True, "source": "previous_state"},
            )

        for ref in step.material_refs:
            material_id = material_node_ids.get(ref)
            if material_id is None:
                material_id = _material_node_id(ref)
                material_node_ids[ref] = material_id
                graph.add_node(
                    material_id,
                    "material",
                    ref,
                    {"placeholder": ref, "role": "unresolved"},
                )
            if _is_terminal_output(step, ref):
                graph.add_edge(operation_id, material_id, "output_from", {"placeholder": ref})
            else:
                graph.add_edge(material_id, operation_id, "input_to", {"placeholder": ref})
                graph.add_edge(operation_id, material_id, "refer_to", {"placeholder": ref})

        for ref in [*step.duration_refs, *step.temperature_refs]:
            condition_id = condition_node_ids.get(ref)
            if condition_id is None:
                condition_id = _condition_node_id(ref)
                condition_node_ids[ref] = condition_id
                graph.add_node(
                    condition_id,
                    "condition",
                    ref,
                    {"placeholder": ref, "condition_type": "unknown"},
                )
            graph.add_edge(operation_id, condition_id, "has_condition", {"placeholder": ref})

        state_id = f"state_{step.step_id:03d}"
        graph.add_node(
            state_id,
            "state",
            "final_state" if step.step_id == len(steps) - 1 else f"state_{step.step_id}",
            {"created_by": operation_id},
        )
        graph.add_edge(operation_id, state_id, "output_from")

        previous_operation_id = operation_id
        previous_state_id = state_id

    graph.metadata["parsed_actions"] = [step.to_dict() for step in steps]
    return graph.to_dict()


def _placeholder_to_smiles(record: dict[str, Any]) -> dict[str, str]:
    extracted = record.get("extracted_molecules") or {}
    return {placeholder: smiles for smiles, placeholder in extracted.items()}


def _smiles_to_names(record: dict[str, Any]) -> dict[str, list[str]]:
    names: dict[str, list[str]] = {}
    for name, smiles in (record.get("molecules") or {}).items():
        names.setdefault(smiles, []).append(name)
    return names


def _placeholder_to_condition(record: dict[str, Any]) -> dict[str, dict[str, str]]:
    conditions: dict[str, dict[str, str]] = {}
    for value, placeholder in (record.get("extracted_duration") or {}).items():
        conditions[placeholder] = {"condition_type": "duration", "value": value}
    for value, placeholder in (record.get("extracted_temperature") or {}).items():
        conditions[placeholder] = {"condition_type": "temperature", "value": value}
    return conditions


def _add_material_nodes(
    graph: ProcessGraph,
    record: dict[str, Any],
    placeholder_to_smiles: dict[str, str],
    smiles_to_names: dict[str, list[str]],
) -> dict[str, str]:
    material_node_ids: dict[str, str] = {}
    for placeholder, smiles in sorted(
        placeholder_to_smiles.items(),
        key=lambda item: _placeholder_sort_key(item[0]),
    ):
        node_id = _material_node_id(placeholder)
        names = smiles_to_names.get(smiles, [])
        role = _material_role(smiles, record)
        label = names[0] if names else placeholder
        graph.add_node(
            node_id,
            "material",
            label,
            {
                "placeholder": placeholder,
                "smiles": smiles,
                "names": names,
                "role": role,
            },
        )
        material_node_ids[placeholder] = node_id
    return material_node_ids


def _add_condition_nodes(
    graph: ProcessGraph,
    placeholder_to_condition: dict[str, dict[str, str]],
) -> dict[str, str]:
    condition_node_ids: dict[str, str] = {}
    for placeholder, attrs in sorted(
        placeholder_to_condition.items(),
        key=lambda item: _placeholder_sort_key(item[0]),
    ):
        node_id = _condition_node_id(placeholder)
        label = attrs.get("value", placeholder)
        graph.add_node(node_id, "condition", label, {"placeholder": placeholder, **attrs})
        condition_node_ids[placeholder] = node_id
    return condition_node_ids


def _material_role(smiles: str, record: dict[str, Any]) -> str:
    if smiles in set(record.get("PRODUCT") or []):
        return "product"
    if smiles in set(record.get("REACTANT") or []):
        return "reactant"
    if smiles in set(record.get("CATALYST") or []):
        return "catalyst"
    if smiles in set(record.get("SOLVENT") or []):
        return "solvent"
    return "mentioned"


def _material_node_id(placeholder: str) -> str:
    return f"mat_{_safe_placeholder(placeholder)}"


def _condition_node_id(placeholder: str) -> str:
    return f"cond_{_safe_placeholder(placeholder)}"


def _safe_placeholder(placeholder: str) -> str:
    inner = placeholder.strip("$@")
    return re.sub(r"[^0-9A-Za-z]+", "_", inner.replace("-", "neg")).strip("_")


def _placeholder_sort_key(placeholder: str) -> tuple[int, int | str]:
    match = re.search(r"-?\d+", placeholder)
    if not match:
        return (1, placeholder)
    return (0, int(match.group(0)))


def _uses_previous_state(step: ActionStep) -> bool:
    return step.operation_type not in {"ADD"}


def _is_terminal_output(step: ActionStep, ref: str) -> bool:
    return step.operation_type in TERMINAL_OUTPUT_OPERATIONS or ref.startswith("$-")
