"""Compile OpenExp action sequences into RGDL graphs."""

from __future__ import annotations

import re
from typing import Any

from reactgdiff.data.action_parser import (
    DURATION_REF_RE,
    MATERIAL_REF_RE,
    PAREN_RE,
    TEMPERATURE_REF_RE,
    ActionStep,
    is_quantity_text,
    parse_action_sequence,
    split_action_sequence_with_separators,
)
from reactgdiff.data.openexp_preprocess import (
    extract_units,
    split_quantity_components,
    unit_type,
)

SCHEMA_VERSION = "rgdl-0.1"

STATE_OUTPUT_OPS = {
    "ADD",
    "MAKESOLUTION",
    "STIR",
    "SETTEMPERATURE",
    "WAIT",
    "REFLUX",
    "MICROWAVE",
    "SONICATE",
    "CONCENTRATE",
    "FILTER",
    "WASH",
    "DRYSOLUTION",
    "DRYSOLID",
    "EXTRACT",
    "PARTITION",
    "PHASESEPARATION",
    "COLLECTLAYER",
    "PH",
    "QUENCH",
    "RECRYSTALLIZE",
    "TRITURATE",
    "DEGAS",
}
BRANCH_OPS = {"FILTER", "PHASESEPARATION", "COLLECTLAYER", "EXTRACT", "PARTITION"}
TERMINAL_OPS = {"YIELD"}
INPUT_MATERIAL_FIELDS = (
    ("REACTANT", "reactant"),
    ("CATALYST", "catalyst"),
    ("SOLVENT", "solvent"),
    ("PRODUCT", "product"),
)
CONNECTOR_ONLY_LITERALS = {
    "and",
    "at",
    "for",
    "from",
    "in",
    "into",
    "of",
    "over",
    "to",
    "under",
    "with",
}


class GraphBuilder:
    """Small helper for stable node and edge construction."""

    def __init__(self) -> None:
        self.nodes: list[dict[str, Any]] = []
        self.edges: list[dict[str, Any]] = []
        self._node_ids: set[str] = set()

    def add_node(
        self,
        node_id: str,
        node_type: str,
        label: str,
        attrs: dict[str, Any] | None = None,
    ) -> str:
        if node_id in self._node_ids:
            return node_id
        self._node_ids.add(node_id)
        self.nodes.append(
            {
                "id": node_id,
                "type": node_type,
                "label": label,
                "attrs": dict(attrs or {}),
            }
        )
        return node_id

    def add_edge(
        self,
        source: str,
        target: str,
        edge_type: str,
        attrs: dict[str, Any] | None = None,
    ) -> str:
        edge_id = f"e_{len(self.edges):05d}"
        self.edges.append(
            {
                "id": edge_id,
                "source": source,
                "target": target,
                "type": edge_type,
                "attrs": dict(attrs or {}),
            }
        )
        return edge_id


def compile_openexp_record_to_rgdl(
    record: dict[str, Any],
    *,
    include_slots: bool = True,
    include_surface_attrs: bool = True,
) -> dict[str, Any]:
    """Compile one processed or raw OpenExp row into an RGDL graph."""

    actions = str(record.get("actions", ""))
    steps = parse_action_sequence(actions)
    separators = separators_for_steps(actions)
    graph = GraphBuilder()
    graph.add_node(
        "reaction",
        "reaction",
        f"OpenExp {record.get('index')}",
        {
            "dataset": "OpenExp",
            "index": record.get("index"),
            "score": record.get("score"),
            "split": record.get("_split"),
            "buckets": record.get("_buckets", {}),
        },
    )

    symbol_table = build_symbol_table(record, steps)
    compact_target = not include_slots
    add_symbol_nodes(graph, symbol_table, include_material_entities=not compact_target)
    if not compact_target:
        add_reaction_participant_edges(graph, symbol_table)
    add_operation_subgraph(
        graph,
        record,
        steps,
        symbol_table,
        separators,
        include_slots=include_slots,
        include_surface_attrs=include_surface_attrs,
        compact_target=compact_target,
    )
    compiler_modes = ["semantic"]
    if include_slots:
        compiler_modes.append("canonical")
    if include_surface_attrs:
        compiler_modes.append("exact")

    return {
        "schema_version": SCHEMA_VERSION,
        "graph_id": f"openexp_{record.get('index')}",
        "source_dataset": "OpenExp",
        "reaction": {
            "index": record.get("index"),
            "reactants": record.get("REACTANT", []),
            "products": record.get("PRODUCT", []),
            "catalysts": record.get("CATALYST", []),
            "solvents": record.get("SOLVENT", []),
            "source": record.get("source", ""),
            "score": record.get("score"),
            "split": record.get("_split"),
            "features": record.get("_features", {}),
            "buckets": record.get("_buckets", {}),
            "input_materials": symbol_table["input_materials"],
        },
        "symbol_table": public_symbol_table(symbol_table),
        "nodes": graph.nodes,
        "edges": graph.edges,
        "surface": {
            "original_actions": record.get("actions", ""),
            "steps": [step.to_dict() for step in steps],
        },
        "constraints": {
            "operation_count": len(steps),
            "requires_topological_order": True,
            "requires_resolved_symbols": True,
            "compiler_modes": compiler_modes,
            "target_profile": "surface_round_trip" if include_slots else "semantic_no_slot",
            "includes_slots": include_slots,
            "includes_surface_attrs": include_surface_attrs,
            "includes_material_entities": not compact_target,
            "state_policy": "distinct_only" if compact_target else "all_steps",
        },
    }


def build_symbol_table(record: dict[str, Any], steps: list[ActionStep]) -> dict[str, Any]:
    extracted_molecules = record.get("extracted_molecules") or {}
    molecules = record.get("molecules") or {}
    names_by_smiles: dict[str, list[str]] = {}
    smiles_by_normalized_name: dict[str, str] = {}
    for name, smiles in molecules.items():
        names_by_smiles.setdefault(smiles, []).append(name)
        smiles_by_normalized_name[normalize_material_name(name)] = smiles

    materials: dict[str, dict[str, Any]] = {}
    materials_by_smiles: dict[str, dict[str, Any]] = {}
    input_materials: dict[str, list[dict[str, Any]]] = {
        field_name: [] for field_name, _ in INPUT_MATERIAL_FIELDS
    }
    input_category_members: dict[str, set[str]] = {
        field_name: set() for field_name, _ in INPUT_MATERIAL_FIELDS
    }
    vocab_index = 0
    for field_name, role in INPUT_MATERIAL_FIELDS:
        for field_index, smiles in enumerate(record.get(field_name) or []):
            entry, vocab_index = ensure_input_material(
                smiles,
                field_name,
                role,
                field_index,
                vocab_index,
                extracted_molecules,
                names_by_smiles,
                materials,
                materials_by_smiles,
                source="reaction_input",
            )
            add_input_material_view(input_materials, input_category_members, field_name, entry)

    for step in steps:
        for literal in extract_literal_material_mentions(step):
            smiles = smiles_by_normalized_name.get(normalize_material_name(literal))
            if not smiles:
                continue
            field_name, role = inferred_literal_input_field(step, smiles, record)
            field_index = len(input_materials[field_name])
            entry, vocab_index = ensure_input_material(
                smiles,
                field_name,
                role,
                field_index,
                vocab_index,
                extracted_molecules,
                names_by_smiles,
                materials,
                materials_by_smiles,
                source="action_literal",
            )
            add_input_material_view(input_materials, input_category_members, field_name, entry)

    literal_lookup: dict[str, dict[str, Any]] = {}
    for entry in materials_by_smiles.values():
        for name in entry["names"]:
            literal_lookup[normalize_material_name(name)] = entry

    durations = {
        placeholder: {
            "symbol": placeholder,
            "raw": raw,
            "kind": "duration",
            "node_id": condition_id(placeholder),
        }
        for raw, placeholder in sorted(
            (record.get("extracted_duration") or {}).items(),
            key=lambda item: _symbol_sort_key(item[1]),
        )
    }
    temperatures = {
        placeholder: {
            "symbol": placeholder,
            "raw": raw,
            "kind": "temperature",
            "node_id": condition_id(placeholder),
        }
        for raw, placeholder in sorted(
            (record.get("extracted_temperature") or {}).items(),
            key=lambda item: _symbol_sort_key(item[1]),
        )
    }
    return {
        "materials": materials,
        "materials_by_smiles": materials_by_smiles,
        "literal_lookup": literal_lookup,
        "input_materials": input_materials,
        "durations": durations,
        "temperatures": temperatures,
    }


def ensure_input_material(
    smiles: str,
    field_name: str,
    role: str,
    field_index: int,
    vocab_index: int,
    extracted_molecules: dict[str, str],
    names_by_smiles: dict[str, list[str]],
    materials: dict[str, dict[str, Any]],
    materials_by_smiles: dict[str, dict[str, Any]],
    *,
    source: str,
) -> tuple[dict[str, Any], int]:
    entry = materials_by_smiles.get(smiles)
    if entry is not None:
        if role not in entry["roles"]:
            entry["roles"].append(role)
        if source not in entry["sources"]:
            entry["sources"].append(source)
        return entry, vocab_index

    symbol = extracted_molecules.get(smiles) or f"{role}_{field_index}"
    material_id = f"m{vocab_index}"
    node_id = f"mat_{vocab_index:03d}"
    entry = {
        "material_id": material_id,
        "symbol": symbol,
        "smiles": smiles,
        "role": role,
        "roles": [role],
        "input_category": field_name,
        "names": names_by_smiles.get(smiles, []),
        "node_id": node_id,
        "vocab_index": vocab_index,
        "sources": [source],
    }
    materials[symbol] = entry
    materials_by_smiles[smiles] = entry
    return entry, vocab_index + 1


def add_input_material_view(
    input_materials: dict[str, list[dict[str, Any]]],
    input_category_members: dict[str, set[str]],
    field_name: str,
    entry: dict[str, Any],
) -> None:
    if entry["material_id"] in input_category_members[field_name]:
        return
    input_category_members[field_name].add(entry["material_id"])
    input_materials[field_name].append(material_input_view(entry))


def inferred_literal_input_field(
    step: ActionStep,
    smiles: str,
    record: dict[str, Any],
) -> tuple[str, str]:
    for field_name, role in INPUT_MATERIAL_FIELDS:
        if smiles in set(record.get(field_name) or []):
            return field_name, role
    if step.operation_type in {"WASH", "EXTRACT", "PARTITION", "RECRYSTALLIZE", "TRITURATE"}:
        return "SOLVENT", "solvent"
    if step.operation_type in {"DRYSOLUTION", "DRYSOLID"}:
        return "REACTANT", "reactant"
    return "REACTANT", "reactant"


def material_input_view(entry: dict[str, Any]) -> dict[str, Any]:
    return {
        "material_id": entry["material_id"],
        "material_label": material_index_label(entry),
        "vocab_index": entry["vocab_index"],
        "symbol": entry["symbol"],
        "role": entry["role"],
        "roles": entry["roles"],
        "input_category": entry["input_category"],
        "smiles": entry["smiles"],
        "names": entry["names"],
        "sources": entry["sources"],
    }


def material_index_label(entry: dict[str, Any]) -> str:
    return entry["material_id"]


def public_symbol_table(symbol_table: dict[str, Any]) -> dict[str, Any]:
    return {
        "materials": {
            symbol: {
                "material_id": entry["material_id"],
                "symbol": entry["symbol"],
                "smiles": entry["smiles"],
                "role": entry["role"],
                "roles": entry["roles"],
                "input_category": entry["input_category"],
                "names": entry["names"],
                "node_id": entry["node_id"],
                "vocab_index": entry["vocab_index"],
                "sources": entry["sources"],
            }
            for symbol, entry in symbol_table["materials"].items()
        },
        "durations": symbol_table["durations"],
        "temperatures": symbol_table["temperatures"],
    }


def resolve_literal_material(
    literal: str,
    symbol_table: dict[str, Any],
) -> dict[str, Any] | None:
    return symbol_table["literal_lookup"].get(normalize_material_name(literal))


def normalize_material_name(name: str) -> str:
    normalized = re.sub(r"\s+", " ", name.strip().lower())
    return normalized.strip(" ,;.")


def add_symbol_nodes(
    graph: GraphBuilder,
    symbol_table: dict[str, Any],
    *,
    include_material_entities: bool,
) -> None:
    if include_material_entities:
        for symbol, entry in symbol_table["materials"].items():
            graph.add_node(
                entry["node_id"],
                "material_entity",
                material_index_label(entry),
                {
                    "material_id": entry["material_id"],
                    "material_index": entry["vocab_index"],
                    "role": entry["role"],
                    "roles": entry["roles"],
                    "input_category": entry["input_category"],
                },
            )
    for collection in ("durations", "temperatures"):
        for symbol, entry in symbol_table[collection].items():
            graph.add_node(
                entry["node_id"],
                "condition",
                entry["raw"],
                {
                    "symbol": symbol,
                    "kind": entry["kind"],
                    "raw": entry["raw"],
                },
            )


def add_reaction_participant_edges(graph: GraphBuilder, symbol_table: dict[str, Any]) -> None:
    for entry in symbol_table["materials"].values():
        graph.add_edge(
            "reaction",
            entry["node_id"],
            "has_participant",
            {
                "material_id": entry["material_id"],
                "material_index": entry["vocab_index"],
                "role": entry["role"],
                "roles": entry["roles"],
            },
        )


def add_operation_subgraph(
    graph: GraphBuilder,
    record: dict[str, Any],
    steps: list[ActionStep],
    symbol_table: dict[str, Any],
    separators: list[str],
    *,
    include_slots: bool,
    include_surface_attrs: bool,
    compact_target: bool,
) -> None:
    previous_operation_id: str | None = None
    previous_state_id: str | None = None
    if not compact_target:
        previous_state_id = "state_initial"
        graph.add_node(
            previous_state_id,
            "state",
            "initial_state",
            {"state_kind": "initial", "step_id": None},
        )
        graph.add_edge("reaction", previous_state_id, "initializes")

    for step in steps:
        operation_id = operation_node_id(step.step_id)
        output_state_kind = state_kind(step)
        operation_attrs: dict[str, Any] = {
            "step_id": step.step_id,
            "operation_type": step.operation_type,
        }
        if compact_target and is_distinct_state_kind(output_state_kind):
            operation_attrs["output_state"] = output_state_kind
        retained = retained_branch(step)
        if retained:
            operation_attrs["retained"] = retained
        if include_surface_attrs:
            operation_attrs.update(
                {
                    "raw_text": step.raw_text,
                    "separator_after": separators[step.step_id]
                    if step.step_id < len(separators)
                    else "",
                }
            )
        graph.add_node(
            operation_id,
            "operation",
            step.operation_type,
            operation_attrs,
        )
        if previous_operation_id is not None:
            graph.add_edge(previous_operation_id, operation_id, "next", {"order": step.step_id})
        if (
            previous_state_id
            and step.operation_type not in {"ADD", "MAKESOLUTION"}
            and step.operation_type not in TERMINAL_OPS
        ):
            graph.add_edge(
                previous_state_id,
                operation_id,
                "input_to",
                {"implicit": True, "role": "current_state"},
            )

        quantity_ids_by_raw = None
        if include_slots:
            add_slots(graph, operation_id, step, symbol_table)
        else:
            quantity_ids_by_raw = add_operation_quantities(graph, operation_id, step)
        add_mentions(graph, operation_id, step, symbol_table, quantity_ids_by_raw)
        add_conditions(graph, operation_id, step, symbol_table)

        if step.operation_type in STATE_OUTPUT_OPS and (
            not compact_target or is_distinct_state_kind(output_state_kind)
        ):
            state_id = state_node_id(step.step_id)
            graph.add_node(
                state_id,
                "state",
                state_label(step),
                {
                    "step_id": step.step_id,
                    "state_kind": output_state_kind,
                    "created_by": operation_id,
                    "retained": retained_branch(step),
                },
            )
            graph.add_edge(operation_id, state_id, "output_from")
            if previous_state_id and not compact_target:
                graph.add_edge(state_id, previous_state_id, "derived_from")
            previous_state_id = state_id

        if previous_state_id and step.operation_type in TERMINAL_OPS:
            graph.add_edge(previous_state_id, operation_id, "input_to", {"role": "terminal_state"})

        previous_operation_id = operation_id


def separators_for_steps(actions: str) -> list[str]:
    separators: list[str] = []
    previous_nonempty: int | None = None
    for segment, separator in split_action_sequence_with_separators(actions):
        text = segment.strip().rstrip(".").strip()
        if text:
            separators.append(separator)
            previous_nonempty = len(separators) - 1
        else:
            if previous_nonempty is not None:
                separators[previous_nonempty] += separator
    return separators


def add_slots(
    graph: GraphBuilder,
    operation_id: str,
    step: ActionStep,
    symbol_table: dict[str, Any],
) -> None:
    for slot in semantic_slots_from_step(step):
        slot_id = f"slot_{step.step_id:03d}_{slot['order']:03d}"
        graph.add_node(
            slot_id,
            "slot",
            slot_label(slot),
            {
                "step_id": step.step_id,
                "operation_id": operation_id,
                **slot,
            },
        )
        graph.add_edge(operation_id, slot_id, "has_slot", {"order": slot["order"]})
        if slot["kind"] == "material_ref":
            target = symbol_table["materials"].get(slot["symbol"], {}).get("node_id")
            if target:
                graph.add_edge(slot_id, target, "realizes", {"kind": "material_ref"})
        elif slot["kind"] == "duration_ref":
            target = symbol_table["durations"].get(slot["symbol"], {}).get("node_id")
            if target:
                graph.add_edge(slot_id, target, "realizes", {"kind": "duration_ref"})
        elif slot["kind"] == "temperature_ref":
            target = symbol_table["temperatures"].get(slot["symbol"], {}).get("node_id")
            if target:
                graph.add_edge(slot_id, target, "realizes", {"kind": "temperature_ref"})
        elif slot["kind"] == "quantity":
            quantity_id = add_quantity_node(graph, step, slot["raw"], f"slot_{slot['order']:03d}")
            graph.add_edge(slot_id, quantity_id, "realizes", {"kind": "quantity"})
            graph.add_edge(operation_id, quantity_id, "has_quantity", {"slot_id": slot_id})


def add_operation_quantities(
    graph: GraphBuilder,
    operation_id: str,
    step: ActionStep,
) -> dict[str, list[str]]:
    quantity_ids_by_raw: dict[str, list[str]] = {}
    for q_idx, quantity in enumerate(step.quantities):
        quantity_id = add_quantity_node(graph, step, quantity, f"op_{q_idx:02d}")
        quantity_ids_by_raw.setdefault(quantity, []).append(quantity_id)
        graph.add_edge(operation_id, quantity_id, "has_quantity", {"order": q_idx})
    return quantity_ids_by_raw


def semantic_slots_from_step(step: ActionStep) -> list[dict[str, Any]]:
    slots: list[dict[str, Any]] = []
    text = step.arguments
    pos = 0
    while pos < len(text):
        match = _next_semantic_token(text, pos)
        if match is None:
            _append_literal_slot(slots, text[pos:])
            break
        start, end, kind, raw = match
        if start > pos:
            _append_literal_slot(slots, text[pos:start])
        if kind == "quantity":
            slot_text = f"({raw})"
            slot = {
                "kind": kind,
                "text": slot_text,
                "raw": raw,
                "components": parse_quantity_components(raw),
            }
        elif kind == "material_ref":
            slot = {"kind": kind, "text": raw, "raw": raw, "symbol": raw}
        elif kind == "duration_ref":
            slot = {"kind": kind, "text": raw, "raw": raw, "symbol": raw}
        elif kind == "temperature_ref":
            slot = {"kind": kind, "text": raw, "raw": raw, "symbol": raw}
        else:
            slot = {"kind": kind, "text": raw, "raw": raw}
        slot["order"] = len(slots)
        slots.append(slot)
        pos = end
    return slots


def _next_semantic_token(text: str, pos: int) -> tuple[int, int, str, str] | None:
    candidates: list[tuple[int, int, str, str]] = []
    for kind, pattern in (
        ("material_ref", MATERIAL_REF_RE),
        ("duration_ref", DURATION_REF_RE),
        ("temperature_ref", TEMPERATURE_REF_RE),
    ):
        match = pattern.search(text, pos)
        if match:
            candidates.append((match.start(), match.end(), kind, match.group(0)))
    for match in PAREN_RE.finditer(text, pos):
        raw = match.group(1)
        if is_quantity_text(raw):
            candidates.append((match.start(), match.end(), "quantity", raw))
            break
    if not candidates:
        return None
    return min(candidates, key=lambda item: (item[0], item[1]))


def _append_literal_slot(slots: list[dict[str, Any]], text: str) -> None:
    if not text:
        return
    slots.append(
        {
            "order": len(slots),
            "kind": "literal_text",
            "text": text,
            "raw": text,
            "semantic_role": infer_literal_role(text),
        }
    )


def infer_literal_role(text: str) -> str:
    lowered = text.strip().lower()
    if not lowered:
        return "spacing"
    if lowered in {"with", "and", "from", "over", "under", "for", "at", "to"}:
        return "connector"
    if lowered.startswith(("with ", "from ", "over ", "under ", "for ", "at ", "to ")):
        return "connector_phrase"
    if "keep filtrate" in lowered or "keep precipitate" in lowered:
        return "retention_policy"
    if "pH" in text or "ph " in lowered:
        return "condition_text"
    return "literal_text"


def slot_label(slot: dict[str, Any]) -> str:
    if slot["kind"] in {"material_ref", "duration_ref", "temperature_ref"}:
        return slot["symbol"]
    if slot["kind"] == "quantity":
        return slot["raw"]
    return slot["text"].strip() or "space"


def add_mentions(
    graph: GraphBuilder,
    operation_id: str,
    step: ActionStep,
    symbol_table: dict[str, Any],
    quantity_ids_by_raw: dict[str, list[str]] | None = None,
) -> None:
    mention_specs: list[dict[str, Any]] = []
    for idx, symbol in enumerate(step.material_refs):
        if symbol not in symbol_table["materials"]:
            continue
        target = symbol_table["materials"][symbol]
        mention_specs.append(
            {
                "kind": "symbol",
                "symbol": symbol,
                "raw": symbol,
                "span_index": idx,
                "target": target,
                "resolution": "placeholder",
            }
        )
    for idx, literal in enumerate(extract_literal_material_mentions(step)):
        target = resolve_literal_material(literal, symbol_table)
        if target is None:
            continue
        mention_specs.append(
            {
                "kind": "literal",
                "symbol": target["symbol"],
                "raw": literal,
                "span_index": len(mention_specs) + idx,
                "target": target,
                "resolution": "input_name",
            }
        )

    for idx, spec in enumerate(mention_specs):
        mention_id = f"men_{step.step_id:03d}_{idx:02d}"
        target = spec["target"]
        graph.add_node(
            mention_id,
            "material_mention",
            material_index_label(target),
            {
                "step_id": step.step_id,
                "operation_id": operation_id,
                "mention_kind": spec["kind"],
                "material_id": target["material_id"],
                "material_index": target["vocab_index"],
                "role": material_mention_role(step, spec["symbol"]),
                "input_category": target["input_category"],
                "resolution": spec["resolution"],
            },
        )
        graph.add_edge(operation_id, mention_id, "mentions", {"order": idx})
        for q_idx, quantity in enumerate(step.quantities):
            if quantity in spec["raw"] or len(mention_specs) == 1:
                if quantity_ids_by_raw is None:
                    quantity_id = add_quantity_node(graph, step, quantity, f"{idx:02d}_{q_idx:02d}")
                    graph.add_edge(mention_id, quantity_id, "has_quantity")
                else:
                    for quantity_id in quantity_ids_by_raw.get(quantity, []):
                        graph.add_edge(mention_id, quantity_id, "has_quantity")


def add_conditions(
    graph: GraphBuilder,
    operation_id: str,
    step: ActionStep,
    symbol_table: dict[str, Any],
) -> None:
    for symbol in step.duration_refs:
        target = symbol_table["durations"].get(symbol, {}).get("node_id") or condition_id(symbol)
        graph.add_edge(operation_id, target, "has_condition", {"kind": "duration", "symbol": symbol})
    for symbol in step.temperature_refs:
        target = symbol_table["temperatures"].get(symbol, {}).get("node_id") or condition_id(symbol)
        graph.add_edge(operation_id, target, "has_condition", {"kind": "temperature", "symbol": symbol})
    atmosphere = extract_atmosphere(step.raw_text)
    if atmosphere:
        node_id = f"cond_atm_{step.step_id:03d}"
        graph.add_node(
            node_id,
            "condition",
            atmosphere,
            {"kind": "atmosphere", "raw": atmosphere, "step_id": step.step_id},
        )
        graph.add_edge(operation_id, node_id, "has_condition", {"kind": "atmosphere"})
    target_ph = extract_ph_target(step.raw_text)
    if target_ph:
        node_id = f"cond_ph_{step.step_id:03d}"
        graph.add_node(
            node_id,
            "condition",
            target_ph,
            {"kind": "pH_target", "raw": target_ph, "step_id": step.step_id},
        )
        graph.add_edge(operation_id, node_id, "has_condition", {"kind": "pH_target"})


def add_quantity_node(
    graph: GraphBuilder,
    step: ActionStep,
    quantity: str,
    suffix: str,
) -> str:
    quantity_id = f"qty_{step.step_id:03d}_{suffix}"
    graph.add_node(
        quantity_id,
        "quantity",
        quantity,
        {
            "step_id": step.step_id,
            "raw": quantity,
            "components": parse_quantity_components(quantity),
        },
    )
    return quantity_id


def parse_quantity_components(quantity: str) -> list[dict[str, Any]]:
    components = []
    for component in split_quantity_components(quantity):
        units = extract_units(component)
        components.append(
            {
                "raw": component,
                "value": extract_first_number(component),
                "units": units,
                "types": [unit_type(unit) for unit in units],
            }
        )
    return components


def extract_literal_material_mentions(step: ActionStep) -> list[str]:
    text = remove_refs_and_quantities(step.arguments)
    if not text:
        return []
    if step.operation_type == "MAKESOLUTION":
        return split_after_keyword(text, "with", delimiter="and")
    if step.operation_type in {"WASH", "EXTRACT", "QUENCH", "TRITURATE", "DEGAS"}:
        return split_after_keyword(text, "with", delimiter="and")
    if step.operation_type == "RECRYSTALLIZE":
        return split_after_keyword(text, "from", delimiter="and")
    if step.operation_type == "DRYSOLUTION":
        return split_after_keyword(text, "over", delimiter="and")
    if step.operation_type == "PH":
        return split_after_keyword(text, "with", stop_tokens=(" to pH ",), delimiter="and")
    if step.operation_type == "PARTITION":
        return split_after_keyword(text, "with", delimiter="and")
    if step.operation_type == "ADD" and not step.material_refs:
        return split_literal_material_phrase(text)
    return []


def split_after_keyword(
    text: str,
    keyword: str,
    *,
    delimiter: str,
    stop_tokens: tuple[str, ...] = (),
) -> list[str]:
    lowered = text.lower()
    prefix = f"{keyword.lower()} "
    if prefix in lowered:
        start = lowered.index(prefix) + len(prefix)
        text = text[start:]
    for token in stop_tokens:
        idx = text.lower().find(token)
        if idx >= 0:
            text = text[:idx]
    return clean_literal_parts(re.split(rf"\b{re.escape(delimiter)}\b", text, flags=re.IGNORECASE))


def split_literal_material_phrase(text: str) -> list[str]:
    return clean_literal_parts(re.split(r"\b(?:with|and)\b", text, flags=re.IGNORECASE))


def clean_literal_parts(parts: list[str]) -> list[str]:
    literals = []
    for part in parts:
        literal = clean_literal(part)
        if literal and literal.lower() not in CONNECTOR_ONLY_LITERALS:
            literals.append(literal)
    return literals


def remove_refs_and_quantities(text: str) -> str:
    text = MATERIAL_REF_RE.sub("", text)
    text = DURATION_REF_RE.sub("", text)
    text = TEMPERATURE_REF_RE.sub("", text)
    text = re.sub(r"\([^()]*\)", "", text)
    text = re.sub(r"\b(?:for|at|over|under)\s*$", "", text, flags=re.IGNORECASE)
    return clean_literal(text)


def clean_literal(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip(" ;,")
    text = re.sub(r"\s+\d+\s*x$", "", text, flags=re.IGNORECASE)
    return text.strip()


def material_mention_role(step: ActionStep, symbol: str | None) -> str:
    if step.operation_type == "YIELD" or (symbol and symbol.startswith("$-")):
        return "product"
    if step.operation_type in {"WASH", "EXTRACT", "PARTITION", "QUENCH", "PH", "DRYSOLUTION"}:
        return "agent"
    return "input"


def material_role(smiles: str, record: dict[str, Any]) -> str:
    if smiles in set(record.get("PRODUCT") or []):
        return "product"
    if smiles in set(record.get("REACTANT") or []):
        return "reactant"
    if smiles in set(record.get("CATALYST") or []):
        return "catalyst"
    if smiles in set(record.get("SOLVENT") or []):
        return "solvent"
    return "mentioned"


def state_kind(step: ActionStep) -> str:
    if step.operation_type == "FILTER":
        return "filtered_branch"
    if step.operation_type == "COLLECTLAYER":
        return f"{retained_branch(step) or 'collected'}_layer"
    if step.operation_type in BRANCH_OPS:
        return "branched_state"
    if step.operation_type == "CONCENTRATE":
        return "concentrated_state"
    if step.operation_type == "MAKESOLUTION":
        return "solution"
    if step.operation_type == "DRYSOLID":
        return "solid"
    return "process_state"


def is_distinct_state_kind(kind: str) -> bool:
    return kind not in {"process_state", "initial"}


def state_label(step: ActionStep) -> str:
    kind = state_kind(step)
    return f"{kind}_{step.step_id}"


def retained_branch(step: ActionStep) -> str | None:
    lowered = step.raw_text.lower()
    if "keep filtrate" in lowered:
        return "filtrate"
    if "keep precipitate" in lowered:
        return "precipitate"
    if "organic" in lowered:
        return "organic"
    if "aqueous" in lowered:
        return "aqueous"
    return None


def extract_atmosphere(text: str) -> str | None:
    match = re.search(r"\bunder\s+([A-Za-z0-9 ]+)$", text, flags=re.IGNORECASE)
    if match:
        return f"under {match.group(1).strip()}"
    return None


def extract_ph_target(text: str) -> str | None:
    match = re.search(r"\bto\s+pH\s+([A-Za-z0-9_.-]+)", text, flags=re.IGNORECASE)
    if match:
        return f"pH {match.group(1)}"
    return None


def extract_first_number(text: str) -> float | None:
    match = re.search(r"[-+]?\d+(?:[.;]\d+)?", text)
    if not match:
        return None
    return float(match.group(0).replace(";", "."))


def operation_node_id(step_id: int) -> str:
    return f"op_{step_id:03d}"


def state_node_id(step_id: int) -> str:
    return f"state_{step_id:03d}"


def material_entity_id(symbol: str) -> str:
    return f"mat_{safe_symbol(symbol)}"


def condition_id(symbol: str) -> str:
    return f"cond_{safe_symbol(symbol)}"


def safe_symbol(symbol: str) -> str:
    inner = symbol.strip("$@#").replace("-", "neg")
    return re.sub(r"[^0-9A-Za-z]+", "_", inner).strip("_")


def _symbol_sort_key(symbol: str) -> tuple[int, int | str]:
    match = re.search(r"-?\d+", symbol)
    if not match:
        return (1, symbol)
    return (0, int(match.group(0)))
