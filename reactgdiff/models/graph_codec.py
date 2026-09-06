"""Detailed graph target codec for direct ReactGDiff graph decoding."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable

import torch

from reactgdiff.compile.graph_to_sequence import decompile_graph_to_sequence
from reactgdiff.data.action_parser import KNOWN_OPENEXP_ACTIONS, ActionStep, parse_action_sequence
from reactgdiff.data.graph_schema import ProcessGraph
from reactgdiff.data.numeric_evidence import (
    NONE_CANDIDATE_ID,
    QUANTITY_CANDIDATE_TYPES,
    best_numeric_candidate,
    format_candidate_quantity,
    infer_numeric_type,
    normalize_unit as normalize_numeric_unit,
    numeric_candidates_from_record,
    parse_numeric_value_unit,
)

PAD_TOKEN = "<PAD>"
EOS_TOKEN = "<EOS>"
NONE_TOKEN = "<NONE>"

LEGACY_CONDITION_TOKENS = (
    NONE_TOKEN,
    "DURATION",
    "TEMPERATURE",
    "DURATION_TEMPERATURE",
)
NUMERIC_CANDIDATE_TYPE_VOCAB = (
    "<NONE>",
    "amount",
    "concentration",
    "yield",
    "duration",
    "temperature",
    "repetition",
    "number",
)
NUMERIC_CANDIDATE_SOURCE_VOCAB = (
    "<NONE>",
    "input_source",
    "input_duration",
    "input_temperature",
    "target_action",
    "missing_evidence",
)

NUMERIC_RE = re.compile(r"[-+]?\d+(?:[.,]\d+)?")
DISPLAY_NUMBER_RE = re.compile(
    r"(?<![A-Za-z0-9_.])[-+]?(?:\d+(?:[.,]\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
)
MATERIAL_PLACEHOLDER_RE = re.compile(r"^\$-?\d+\$$")
DURATION_PLACEHOLDER_RE = re.compile(r"^@\d+@$")
TEMPERATURE_PLACEHOLDER_RE = re.compile(r"^#\d+#$")
UNIT_RE = re.compile(
    r"(?<![A-Za-z])(?P<unit>ug|mg|g|kg|ul|ml|l|umol|mmol|mol|molar|normal|"
    r"equiv|equivalent|eq|percent|%|m|n|h|hr|hrs|hour|hours|min|minutes|"
    r"c|degc|degree|degrees)(?=\b|[^A-Za-z0-9_])",
    re.IGNORECASE,
)


@dataclass(slots=True)
class GraphTargetCodec:
    """Encode OpenExp process graphs into detailed fixed-width graph slots.

    Variable graph arity is represented by fixed upper bounds plus masks:
    every operation step has up to ``max_material_slots`` material slots and
    matching numeric slots. The discrete part decides whether a slot exists and
    which unit it uses; the continuous part predicts normalized values only for
    slots marked present in the target.
    """

    action_vocab: list[str]
    material_vocab: list[str]
    condition_vocab: list[str]
    unit_vocab: list[str]
    max_steps: int
    max_material_slots: int
    quantity_mean: float
    quantity_std: float
    duration_mean: float
    duration_std: float
    temperature_mean: float
    temperature_std: float
    max_numeric_candidates: int = 64
    numeric_candidate_include_source: bool = False
    numeric_candidate_quantity_only: bool = False

    @classmethod
    def fit(
        cls,
        records: Iterable[dict[str, Any]],
        *,
        max_steps: int = 32,
        max_material_refs: int = 16,
        max_material_slots: int = 4,
        max_quantity_vocab: int = 256,
        max_numeric_candidates: int = 64,
        numeric_candidate_include_source: bool = False,
        numeric_candidate_quantity_only: bool = True,
    ) -> "GraphTargetCodec":
        unit_counts: Counter[str] = Counter()
        quantity_values: list[float] = []
        duration_values: list[float] = []
        temperature_values: list[float] = []

        for record in records:
            for step in parse_action_sequence(str(record.get("actions", ""))):
                for quantity in step.quantities:
                    parsed = parse_numeric_unit(quantity)
                    if parsed is None:
                        continue
                    value, unit = parsed
                    unit_counts[unit] += 1
                    quantity_values.append(value)
            for raw in (record.get("extracted_duration") or {}).keys():
                parsed = parse_numeric_unit(raw)
                if parsed is not None:
                    duration_values.append(parsed[0])
            for raw in (record.get("extracted_temperature") or {}).keys():
                parsed = parse_numeric_unit(raw)
                if parsed is not None:
                    temperature_values.append(parsed[0])

        action_vocab = [PAD_TOKEN, EOS_TOKEN, *sorted(KNOWN_OPENEXP_ACTIONS)]
        material_vocab = [
            NONE_TOKEN,
            "$-1$",
            *[f"${idx}$" for idx in range(1, max_material_refs + 1)],
        ]
        unit_vocab = [
            NONE_TOKEN,
            *[
                unit
                for unit, _ in unit_counts.most_common(max(0, max_quantity_vocab - 1))
            ],
        ]
        condition_refs = [f"@{idx}@" for idx in range(1, max_material_refs + 1)]
        temperature_refs = [f"#{idx}#" for idx in range(1, max_material_refs + 1)]
        condition_vocab = [
            NONE_TOKEN,
            *condition_refs,
            *temperature_refs,
            *[
                combine_condition_token(duration_ref, temperature_ref)
                for duration_ref in condition_refs
                for temperature_ref in temperature_refs
            ],
        ]
        quantity_mean, quantity_std = robust_mean_std(quantity_values)
        duration_mean, duration_std = robust_mean_std(duration_values)
        temperature_mean, temperature_std = robust_mean_std(temperature_values)
        return cls(
            action_vocab=action_vocab,
            material_vocab=material_vocab,
            condition_vocab=condition_vocab,
            unit_vocab=unit_vocab,
            max_steps=max_steps,
            max_material_slots=max_material_slots,
            quantity_mean=quantity_mean,
            quantity_std=quantity_std,
            duration_mean=duration_mean,
            duration_std=duration_std,
            temperature_mean=temperature_mean,
            temperature_std=temperature_std,
            max_numeric_candidates=max(1, int(max_numeric_candidates)),
            numeric_candidate_include_source=bool(numeric_candidate_include_source),
            numeric_candidate_quantity_only=bool(numeric_candidate_quantity_only),
        )

    @property
    def pad_id(self) -> int:
        return 0

    @property
    def eos_id(self) -> int:
        return 1

    @property
    def action_dim(self) -> int:
        return len(self.action_vocab)

    @property
    def material_dim(self) -> int:
        return len(self.material_vocab)

    @property
    def condition_dim(self) -> int:
        return len(self.condition_vocab)

    @property
    def unit_dim(self) -> int:
        return len(self.unit_vocab)

    @property
    def quantity_dim(self) -> int:
        return self.unit_dim

    @property
    def numeric_candidate_dim(self) -> int:
        """NONE, MISSING, and one class for each record-local candidate."""

        return self.max_numeric_candidates + 2

    @property
    def numeric_candidate_type_dim(self) -> int:
        return len(NUMERIC_CANDIDATE_TYPE_VOCAB)

    @property
    def numeric_candidate_source_dim(self) -> int:
        return len(NUMERIC_CANDIDATE_SOURCE_VOCAB)

    def numeric_candidate_features_from_record(
        self,
        record: dict[str, Any],
    ) -> dict[str, list[Any]]:
        """Encode a compact record-local candidate table for pointer scoring."""

        size = self.numeric_candidate_dim
        values = [0.0] * size
        confidences = [0.0] * size
        unit_ids = [0] * size
        type_ids = [0] * size
        source_ids = [0] * size
        mask = [False] * size
        # NONE and MISSING are learned special candidates.
        mask[0] = True
        mask[1] = True
        for candidate_idx, candidate in enumerate(
            self.numeric_candidates_from_record(record)
        ):
            class_id = candidate_idx + 2
            if class_id >= size:
                break
            raw_value = float(candidate.normalized_value or 0.0)
            values[class_id] = max(
                min(math.copysign(math.log1p(abs(raw_value)), raw_value) / 8.0, 1.0),
                -1.0,
            )
            confidences[class_id] = float(candidate.confidence)
            unit_ids[class_id] = self._id(
                self.unit_vocab,
                str(candidate.normalized_unit or NONE_TOKEN),
                0,
            )
            type_ids[class_id] = self._id(
                list(NUMERIC_CANDIDATE_TYPE_VOCAB),
                str(candidate.numeric_type),
                0,
            )
            source_ids[class_id] = self._id(
                list(NUMERIC_CANDIDATE_SOURCE_VOCAB),
                str(candidate.source),
                0,
            )
            mask[class_id] = True
        return {
            "values": values,
            "confidences": confidences,
            "unit_ids": unit_ids,
            "type_ids": type_ids,
            "source_ids": source_ids,
            "mask": mask,
        }

    def encode_numeric_candidate_features(
        self,
        records: list[dict[str, Any]],
        *,
        device: str | torch.device,
    ) -> tuple[torch.Tensor, ...]:
        rows = [self.numeric_candidate_features_from_record(record) for record in records]
        return (
            torch.tensor([row["values"] for row in rows], dtype=torch.float32, device=device),
            torch.tensor(
                [row["confidences"] for row in rows],
                dtype=torch.float32,
                device=device,
            ),
            torch.tensor([row["unit_ids"] for row in rows], dtype=torch.int16, device=device),
            torch.tensor([row["type_ids"] for row in rows], dtype=torch.int8, device=device),
            torch.tensor([row["source_ids"] for row in rows], dtype=torch.int8, device=device),
            torch.tensor([row["mask"] for row in rows], dtype=torch.bool, device=device),
        )

    def encode_records(
        self,
        records: list[dict[str, Any]],
        condition_vectors: list[list[float]],
        *,
        device: str | torch.device,
    ) -> tuple[torch.Tensor, ...]:
        encoded_rows = [self.encode_record(record) for record in records]

        def collect(name: str, dtype: torch.dtype) -> torch.Tensor:
            return torch.tensor([row[name] for row in encoded_rows], dtype=dtype, device=device)

        return (
            torch.tensor(condition_vectors, dtype=torch.float32, device=device),
            collect("op_ids", torch.long),
            collect("material_ids", torch.long),
            collect("condition_ids", torch.long),
            collect("quantity_gate_ids", torch.long),
            collect("unit_ids", torch.long),
            collect("quantity_values", torch.float32),
            collect("quantity_value_masks", torch.float32),
            collect("condition_values", torch.float32),
            collect("condition_value_masks", torch.float32),
            collect("slot_mask", torch.float32),
        )

    def encode_record(self, record: dict[str, Any]) -> dict[str, Any]:
        steps = parse_action_sequence(str(record.get("actions", "")))[: self.max_steps - 1]
        op_ids = [self.pad_id] * self.max_steps
        material_ids = [[0] * self.max_material_slots for _ in range(self.max_steps)]
        condition_ids = [0] * self.max_steps
        quantity_gate_ids = [[0] * self.max_material_slots for _ in range(self.max_steps)]
        unit_ids = [[0] * self.max_material_slots for _ in range(self.max_steps)]
        numeric_candidate_ids = [[0] * self.max_material_slots for _ in range(self.max_steps)]
        quantity_values = [[0.0] * self.max_material_slots for _ in range(self.max_steps)]
        quantity_value_masks = [[0.0] * self.max_material_slots for _ in range(self.max_steps)]
        condition_values = [[0.0, 0.0] for _ in range(self.max_steps)]
        condition_value_masks = [[0.0, 0.0] for _ in range(self.max_steps)]
        slot_mask = [0.0] * self.max_steps
        surface_arguments = [""] * self.max_steps
        surface_raw_texts = [""] * self.max_steps
        numeric_candidates = self.numeric_candidates_from_record(record)

        for step_idx, step in enumerate(steps):
            op_ids[step_idx] = self._id(self.action_vocab, step.operation_type, self.eos_id)
            condition_ids[step_idx] = self._condition_id(step)
            slot_mask[step_idx] = 1.0
            surface_arguments[step_idx] = step.arguments
            surface_raw_texts[step_idx] = step.raw_text

            for material_slot, material_ref in enumerate(step.material_refs[: self.max_material_slots]):
                material_ids[step_idx][material_slot] = self._id(self.material_vocab, material_ref, 0)

            for quantity_slot, quantity in enumerate(step.quantities[: self.max_material_slots]):
                parsed = parse_numeric_unit(quantity)
                if parsed is None:
                    continue
                value, unit = parsed
                quantity_gate_ids[step_idx][quantity_slot] = 1
                unit_ids[step_idx][quantity_slot] = self._id(self.unit_vocab, unit, 0)
                candidate = matching_numeric_candidate(
                    numeric_candidates,
                    value=value,
                    unit=unit,
                    numeric_type=infer_numeric_type(quantity, unit),
                    raw_text=quantity,
                )
                numeric_candidate_ids[step_idx][quantity_slot] = self.numeric_candidate_class_id(
                    candidate.candidate_id if candidate is not None else NONE_CANDIDATE_ID
                )
                quantity_values[step_idx][quantity_slot] = self.normalize_quantity(value)
                quantity_value_masks[step_idx][quantity_slot] = 1.0

            duration_value = self._first_condition_value(record, step, "duration")
            if duration_value is not None:
                condition_values[step_idx][0] = self.normalize_duration(duration_value)
                condition_value_masks[step_idx][0] = 1.0
            temperature_value = self._first_condition_value(record, step, "temperature")
            if temperature_value is not None:
                condition_values[step_idx][1] = self.normalize_temperature(temperature_value)
                condition_value_masks[step_idx][1] = 1.0

        eos_idx = min(len(steps), self.max_steps - 1)
        op_ids[eos_idx] = self.eos_id
        return {
            "op_ids": op_ids,
            "material_ids": material_ids,
            "condition_ids": condition_ids,
            "quantity_gate_ids": quantity_gate_ids,
            "unit_ids": unit_ids,
            "numeric_candidate_ids": numeric_candidate_ids,
            "quantity_values": quantity_values,
            "quantity_value_masks": quantity_value_masks,
            "condition_values": condition_values,
            "condition_value_masks": condition_value_masks,
            "slot_mask": slot_mask,
            "surface_arguments": surface_arguments,
            "surface_raw_texts": surface_raw_texts,
        }

    def target_slots_from_record(self, record: dict[str, Any]) -> list[dict[str, Any]]:
        """Return target slots with exact surface argument text preserved.

        The supervised graph slots intentionally keep both structured fields and
        surface arguments. The structured fields are what the graph generator
        learns; ``argument_text`` is the target for the text filler and makes the
        graph exactly decompilable back to the OpenExp action string.
        """

        slots: list[dict[str, Any]] = []
        numeric_candidates = self.numeric_candidates_from_record(record)
        steps = parse_action_sequence(str(record.get("actions", "")))[: self.max_steps - 1]
        from reactgdiff.data.action_parser import quantity_material_bindings
        for step_idx, step in enumerate(steps):
            bindings = quantity_material_bindings(step.arguments)
            quantity_slots: list[dict[str, Any]] = []
            for quantity_slot, quantity in enumerate(step.quantities[: self.max_material_slots]):
                parsed = parse_numeric_unit(quantity)
                if parsed is None:
                    continue
                value, unit = parsed
                quantity_slots.append(
                    {
                        "slot_id": quantity_slot,
                        "material_ref": bindings[quantity_slot] if quantity_slot < len(bindings) else "",
                        "value": value,
                        "unit": unit,
                        "unit_type": infer_numeric_type(quantity, unit),
                        "numeric_type": infer_numeric_type(quantity, unit),
                        "text": quantity,
                    }
                )
                candidate = matching_numeric_candidate(
                    numeric_candidates,
                    value=value,
                    unit=unit,
                    numeric_type=infer_numeric_type(quantity, unit),
                    raw_text=quantity,
                )
                if candidate is not None:
                    quantity_slots[-1].update(
                        {
                            "candidate_id": candidate.candidate_id,
                            "evidence_text": candidate.raw_text,
                            "source": candidate.source,
                            "confidence": candidate.confidence,
                        }
                    )
                else:
                    quantity_slots[-1].update(
                        {
                            "candidate_id": NONE_CANDIDATE_ID,
                            "source": "missing_evidence",
                            "confidence": 0.0,
                        }
                    )
            duration_ref = step.duration_refs[0] if step.duration_refs else ""
            temperature_ref = step.temperature_refs[0] if step.temperature_refs else ""
            condition = combine_condition_token(duration_ref, temperature_ref)
            first_material = step.material_refs[0] if step.material_refs else NONE_TOKEN
            first_quantity = quantity_slots[0]["text"] if quantity_slots else NONE_TOKEN
            slots.append(
                {
                    "step_id": step_idx,
                    "operation_type": step.operation_type,
                    "material_refs": list(step.material_refs[: self.max_material_slots]),
                    "material_ref": first_material,
                    "condition": condition,
                    "duration_ref": duration_ref,
                    "temperature_ref": temperature_ref,
                    "condition_values": {},
                    "quantity_slots": quantity_slots,
                    "quantity": first_quantity,
                    "argument_text": step.arguments,
                    "raw_text": step.raw_text,
                }
            )
        return slots

    def build_target_graph(self, record: dict[str, Any]) -> dict[str, Any]:
        """Build a graph from target slots that exact-decompiles to the target text."""

        return self.build_generated_graph(record, self.target_slots_from_record(record))

    def decode_logits(
        self,
        op_logits: torch.Tensor,
        material_logits: torch.Tensor,
        condition_logits: torch.Tensor,
        quantity_gate_logits: torch.Tensor,
        unit_logits: torch.Tensor,
        quantity_values: torch.Tensor,
        condition_values: torch.Tensor,
        numeric_candidate_logits: torch.Tensor | None = None,
        *,
        quantity_gate_threshold: float = 0.65,
        condition_probability_threshold: float = 0.35,
        forced_step_count: int | None = None,
        decode_quantities: bool = True,
        decode_quantity_values: bool = True,
        numeric_candidates: list[Any] | None = None,
        numeric_candidate_reuse_penalty: float = 0.0,
        numeric_candidate_unit_weight: float = 0.0,
        drop_unsupported_numeric_slots: bool = False,
    ) -> list[dict[str, Any]]:
        op_ids = self._decode_operation_ids(op_logits, forced_step_count=forced_step_count)
        material_ids = material_logits.argmax(dim=-1).tolist()
        condition_ids = self._decode_condition_ids(
            condition_logits,
            probability_threshold=condition_probability_threshold,
        )
        if decode_quantities:
            quantity_gate_probs = torch.softmax(quantity_gate_logits, dim=-1)[..., 1]
            unit_ids = unit_logits.argmax(dim=-1).tolist()
            numeric_candidate_ids = (
                numeric_candidate_logits.argmax(dim=-1).tolist()
                if numeric_candidate_logits is not None
                else None
            )
        else:
            quantity_gate_probs = None
            unit_ids = None
            numeric_candidate_ids = None
        slots: list[dict[str, Any]] = []
        used_numeric_candidates: Counter[str] = Counter()

        for step_idx, op_id in enumerate(op_ids):
            if op_id in (self.pad_id, self.eos_id):
                break
            operation = self.action_vocab[op_id]
            material_refs = [
                self.material_vocab[material_id]
                for material_id in material_ids[step_idx]
                if material_id and self.material_vocab[material_id] != NONE_TOKEN
            ]
            quantity_slots: list[dict[str, Any]] = []
            if decode_quantities and quantity_gate_probs is not None and unit_ids is not None:
                for quantity_slot in range(self.max_material_slots):
                    gate_is_open = (
                        float(quantity_gate_probs[step_idx][quantity_slot]) >= quantity_gate_threshold
                    )
                    unit = self.unit_vocab[unit_ids[step_idx][quantity_slot]]
                    if not gate_is_open or unit == NONE_TOKEN:
                        continue
                    numeric_type = infer_numeric_type("", unit)
                    candidate_id: str | None = None
                    if numeric_candidate_logits is not None and numeric_candidates is not None:
                        candidate_id = self._select_numeric_candidate(
                            numeric_candidate_logits[step_idx][quantity_slot],
                            unit_logits[step_idx][quantity_slot],
                            numeric_candidates,
                            used_numeric_candidates,
                            reuse_penalty=numeric_candidate_reuse_penalty,
                            unit_weight=numeric_candidate_unit_weight,
                        )
                    elif numeric_candidate_ids is not None:
                        candidate_id = self.numeric_candidate_token(
                            numeric_candidate_ids[step_idx][quantity_slot]
                        )
                    if (
                        drop_unsupported_numeric_slots
                        and (
                            not candidate_id
                            or candidate_id in {NONE_TOKEN, NONE_CANDIDATE_ID}
                        )
                    ):
                        continue
                    if candidate_id and candidate_id.startswith("NUM_"):
                        used_numeric_candidates[candidate_id] += 1
                    if decode_quantity_values:
                        normalized_value = float(quantity_values[step_idx][quantity_slot])
                        value = self.denormalize_quantity(normalized_value)
                        quantity_slots.append(
                            {
                                "slot_id": quantity_slot,
                                "value": value,
                                "unit": unit,
                                "unit_type": numeric_type,
                                "numeric_type": numeric_type,
                                "text": format_numeric_quantity(value, unit),
                                "source": "model_continuous",
                                "candidate_id": candidate_id,
                            }
                        )
                    else:
                        quantity_slots.append(
                            {
                                "slot_id": quantity_slot,
                                "value": None,
                                "unit": unit,
                                "unit_type": numeric_type,
                                "numeric_type": numeric_type,
                                "text": NONE_CANDIDATE_ID,
                                "source": "needs_grounding",
                                "candidate_id": candidate_id,
                            }
                        )
            condition = self.condition_vocab[condition_ids[step_idx]]
            duration_ref, temperature_ref = split_condition_token(condition)
            decoded_condition_values = (
                {
                    "duration": self.denormalize_duration(float(condition_values[step_idx][0])),
                    "temperature": self.denormalize_temperature(float(condition_values[step_idx][1])),
                }
                if self.uses_legacy_condition_classes
                else {}
            )
            first_material = material_refs[0] if material_refs else NONE_TOKEN
            first_quantity = quantity_slots[0]["text"] if quantity_slots else NONE_TOKEN
            slots.append(
                {
                    "step_id": step_idx,
                    "operation_type": operation,
                    "material_refs": material_refs,
                    "material_ref": first_material,
                    "condition": condition,
                    "duration_ref": duration_ref,
                    "temperature_ref": temperature_ref,
                    "condition_values": decoded_condition_values,
                    "quantity_slots": quantity_slots,
                    "quantity": first_quantity,
                }
            )
            if operation == "YIELD":
                break

        if not slots:
            slots.append(
                {
                    "step_id": 0,
                    "operation_type": "ADD",
                    "material_refs": ["$1$"],
                    "material_ref": "$1$",
                    "condition": NONE_TOKEN,
                    "duration_ref": "",
                    "temperature_ref": "",
                    "condition_values": {},
                    "quantity_slots": [],
                    "quantity": NONE_TOKEN,
                }
            )
        if not any(slot["operation_type"] == "YIELD" for slot in slots):
            terminal_slot = {
                "step_id": min(len(slots), self.max_steps - 1),
                "operation_type": "YIELD",
                "material_refs": ["$-1$"],
                "material_ref": "$-1$",
                "condition": NONE_TOKEN,
                "duration_ref": "",
                "temperature_ref": "",
                "condition_values": {},
                "quantity_slots": [],
                "quantity": NONE_TOKEN,
            }
            if len(slots) >= self.max_steps:
                slots[-1] = terminal_slot
            else:
                slots.append(terminal_slot)
        return slots[: self.max_steps]

    def _select_numeric_candidate(
        self,
        candidate_logits: torch.Tensor,
        unit_logits: torch.Tensor,
        candidates: list[Any],
        used_candidates: Counter[str],
        *,
        reuse_penalty: float,
        unit_weight: float,
    ) -> str:
        """Jointly score candidate identity, unit agreement, and prior reuse."""

        scores = candidate_logits.detach().float().clone()
        scores[0] = -1.0e4
        valid_end = min(len(candidates) + 2, scores.numel())
        if valid_end < scores.numel():
            scores[valid_end:] = -1.0e4
        unit_log_probs = torch.log_softmax(unit_logits.detach().float(), dim=-1)
        for candidate_idx, candidate in enumerate(candidates[: self.max_numeric_candidates]):
            class_id = candidate_idx + 2
            candidate_id = f"NUM_{candidate_idx}"
            unit_id = self._id(
                self.unit_vocab,
                str(candidate.normalized_unit or NONE_TOKEN),
                0,
            )
            if unit_id > 0 and unit_weight:
                scores[class_id] += float(unit_weight) * unit_log_probs[unit_id]
            if reuse_penalty and used_candidates[candidate_id]:
                scores[class_id] -= float(reuse_penalty) * used_candidates[candidate_id]
        return self.numeric_candidate_token(int(scores.argmax().item()))

    def skeleton_ids_from_record(self, record: dict[str, Any]) -> list[int]:
        steps = parse_action_sequence(str(record.get("actions", "")))[: self.max_steps - 1]
        return self.operation_ids_from_sequence(step.operation_type for step in steps)

    def operation_ids_from_sequence(self, operations: Iterable[str]) -> list[int]:
        op_ids = [self.pad_id] * self.max_steps
        step_count = 0
        for step_idx, operation in enumerate(list(operations)[: self.max_steps - 1]):
            op_ids[step_idx] = self._id(self.action_vocab, str(operation).upper(), self.eos_id)
            step_count = step_idx + 1
        eos_idx = min(step_count, self.max_steps - 1)
        op_ids[eos_idx] = self.eos_id
        return op_ids

    def ground_numeric_slots(
        self,
        record: dict[str, Any],
        slots: list[dict[str, Any]],
        *,
        allow_missing: bool = True,
    ) -> list[dict[str, Any]]:
        candidates = self.numeric_candidates_from_record(record)
        candidates_by_id = {candidate.candidate_id: candidate for candidate in candidates}
        used_candidate_ids: set[str] = set()
        grounded: list[dict[str, Any]] = []
        for slot in slots:
            copied_slot = dict(slot)
            quantity_slots: list[dict[str, Any]] = []
            for quantity in copied_slot.get("quantity_slots") or []:
                copied_quantity = dict(quantity)
                existing_source = str(copied_quantity.get("source") or "")
                needs_grounding = existing_source in {"", "needs_grounding", "model_continuous"}
                if not needs_grounding:
                    quantity_slots.append(copied_quantity)
                    continue
                numeric_type = str(
                    copied_quantity.get("numeric_type")
                    or infer_numeric_type("", str(copied_quantity.get("unit") or ""))
                )
                predicted_candidate_id = str(copied_quantity.get("candidate_id") or "")
                if predicted_candidate_id:
                    candidate = candidates_by_id.get(predicted_candidate_id)
                else:
                    # Backward-compatible path for checkpoints trained before the
                    # record-local numeric-candidate diffusion variable existed.
                    candidate = best_numeric_candidate(
                        candidates,
                        unit=str(copied_quantity.get("unit") or "") or None,
                        numeric_type=numeric_type,
                        used_candidate_ids=used_candidate_ids,
                    )
                if candidate is None:
                    if not allow_missing:
                        continue
                    copied_quantity.update(
                        {
                            "candidate_id": NONE_CANDIDATE_ID,
                            "text": NONE_CANDIDATE_ID,
                            "value": None,
                            "unit": copied_quantity.get("unit") or NONE_TOKEN,
                            "numeric_type": numeric_type,
                            "unit_type": numeric_type,
                            "source": "missing_evidence",
                            "confidence": 0.0,
                            "numeric_grounding_status": "missing",
                        }
                    )
                else:
                    used_candidate_ids.add(candidate.candidate_id)
                    copied_quantity.update(
                        {
                            "candidate_id": candidate.candidate_id,
                            "text": format_candidate_quantity(candidate),
                            "value": candidate.normalized_value,
                            "unit": candidate.normalized_unit or copied_quantity.get("unit"),
                            "numeric_type": candidate.numeric_type,
                            "unit_type": candidate.numeric_type,
                            "source": candidate.source,
                            "confidence": candidate.confidence,
                            "evidence_text": candidate.raw_text,
                            "numeric_grounding_status": "grounded",
                        }
                    )
                quantity_slots.append(copied_quantity)
            copied_slot["quantity_slots"] = quantity_slots
            copied_slot["quantity"] = quantity_slots[0]["text"] if quantity_slots else NONE_TOKEN
            grounded.append(copied_slot)
        return grounded

    def _decode_operation_ids(
        self,
        op_logits: torch.Tensor,
        *,
        forced_step_count: int | None,
    ) -> list[int]:
        if forced_step_count is None:
            return op_logits.argmax(dim=-1).tolist()
        step_count = max(1, min(int(forced_step_count), self.max_steps))
        yield_id = self._id(self.action_vocab, "YIELD", self.eos_id)
        op_ids: list[int] = []
        for step_idx in range(step_count):
            if step_idx == step_count - 1:
                op_ids.append(yield_id)
                continue
            scores = op_logits[step_idx].clone()
            scores[self.pad_id] = -torch.inf
            scores[self.eos_id] = -torch.inf
            scores[yield_id] = -torch.inf
            op_ids.append(int(scores.argmax(dim=-1)))
        op_ids.extend([self.eos_id] * max(0, self.max_steps - len(op_ids)))
        return op_ids[: self.max_steps]

    @staticmethod
    def _decode_condition_ids(
        logits: torch.Tensor,
        *,
        probability_threshold: float,
    ) -> list[int]:
        probabilities = torch.softmax(logits, dim=-1)
        if probabilities.size(-1) <= 1:
            return [0] * probabilities.size(0)
        non_none_probs, non_none_offsets = probabilities[..., 1:].max(dim=-1)
        argmax_ids = probabilities.argmax(dim=-1)
        chosen = torch.where(
            non_none_probs >= probability_threshold,
            non_none_offsets + 1,
            argmax_ids,
        )
        chosen = torch.where(non_none_probs >= probability_threshold, chosen, torch.zeros_like(chosen))
        return chosen.tolist()

    def build_generated_graph(
        self,
        record: dict[str, Any],
        slots: list[dict[str, Any]],
    ) -> dict[str, Any]:
        graph = ProcessGraph(
            graph_id=f"generated_openexp_{record.get('index', 'unknown')}",
            metadata={
                "dataset": "OpenExp",
                "index": record.get("index"),
                "generator": "direct_graph_encoder_decoder",
                "reactants": record.get("REACTANT", []),
                "products": record.get("PRODUCT", []),
                "catalysts": record.get("CATALYST", []),
                "solvents": record.get("SOLVENT", []),
            },
        )
        material_node_ids = self._add_material_nodes(graph, record)
        duration_refs = self._condition_refs(record, "duration")
        temperature_refs = self._condition_refs(record, "temperature")
        condition_node_ids = self._add_condition_nodes(graph, duration_refs, temperature_refs)

        previous_operation_id: str | None = None
        previous_state_id: str | None = None
        rendered_segments: list[str] = []
        render_trace: list[dict[str, Any]] = []
        for step_idx, slot in enumerate(slots):
            operation = str(slot["operation_type"])
            material_refs = self._normalized_material_refs(slot, material_node_ids)
            for material_ref in material_refs:
                if material_ref in material_node_ids:
                    continue
                node_id = (
                    f"mat_unresolved_{material_ref.strip('$').replace('-', 'neg')}"
                )
                graph.add_node(
                    node_id,
                    "material",
                    material_ref,
                    {
                        "placeholder": material_ref,
                        "smiles": None,
                        "role": "unresolved_prediction",
                        "source": "decoded_slot",
                    },
                )
                material_node_ids[material_ref] = node_id
            duration_ref, temperature_ref = self._slot_condition_refs(
                slot,
                duration_refs=duration_refs,
                temperature_refs=temperature_refs,
            )
            for condition_type, ref in (
                ("duration", duration_ref),
                ("temperature", temperature_ref),
            ):
                if not ref or ref in condition_node_ids:
                    continue
                node_id = f"cond_unresolved_{condition_type}_{len(condition_node_ids):03d}"
                graph.add_node(
                    node_id,
                    "condition",
                    ref,
                    {
                        "placeholder": ref,
                        "condition_type": condition_type,
                        "source": "decoded_slot",
                        "unresolved": True,
                    },
                )
                condition_node_ids[ref] = node_id
            raw_text, step_render_trace = self._render_slot(
                operation,
                slot,
                step_idx=step_idx,
                material_refs=material_refs,
                duration_ref=duration_ref,
                temperature_ref=temperature_ref,
            )
            rendered_segments.append(raw_text)
            render_trace.append(step_render_trace)

            operation_id = f"op_{step_idx:03d}"
            graph.add_node(
                operation_id,
                "operation",
                operation,
                {
                    "step_id": step_idx,
                    "raw_text": raw_text,
                    "arguments": str(slot.get("argument_text") or raw_text[len(operation) :]).strip(),
                    "operation_type": operation,
                    "decoded_slots": slot,
                },
            )
            if previous_operation_id is not None:
                graph.add_edge(previous_operation_id, operation_id, "precede")
            if previous_state_id is not None and operation != "ADD":
                graph.add_edge(
                    previous_state_id,
                    operation_id,
                    "input_to",
                    {"implicit": True, "source": "previous_state"},
                )
            for material_idx, material_ref in enumerate(material_refs):
                material_id = material_node_ids.get(material_ref)
                if material_id is None:
                    continue
                edge_type = "output_from" if operation == "YIELD" else "input_to"
                occurrence_id = f"step_{step_idx}:material_{material_idx}"
                graph.add_edge(
                    operation_id if edge_type == "output_from" else material_id,
                    material_id if edge_type == "output_from" else operation_id,
                    edge_type,
                    {
                        "placeholder": material_ref,
                        "order": material_idx,
                        "occurrence_id": occurrence_id,
                    },
                )
                if operation != "YIELD":
                    graph.add_edge(
                        operation_id,
                        material_id,
                        "refer_to",
                        {
                            "placeholder": material_ref,
                            "order": material_idx,
                            "occurrence_id": occurrence_id,
                        },
                    )
            rendered_quantities = step_render_trace["quantity_occurrences"]
            for quantity_idx, quantity in enumerate(slot.get("quantity_slots") or []):
                slot_id = int(quantity.get("slot_id", quantity_idx))
                occurrence_id = f"step_{step_idx}:quantity_{quantity_idx}"
                rendered_quantity = rendered_quantities[quantity_idx]
                quantity_node_id = f"qty_{step_idx:03d}_{quantity_idx:02d}_{slot_id:02d}"
                graph.add_node(
                    quantity_node_id,
                    "condition",
                    rendered_quantity["text"],
                    {
                        "condition_type": "quantity",
                        "slot_id": slot_id,
                        "order": quantity_idx,
                        "occurrence_id": occurrence_id,
                        "rendered_text": rendered_quantity["text"],
                        "value": quantity.get("value"),
                        "unit": quantity.get("unit"),
                        "numeric_type": quantity.get("numeric_type"),
                        "candidate_id": quantity.get("candidate_id"),
                        "source": quantity.get("source"),
                        "confidence": quantity.get("confidence"),
                        "evidence_text": quantity.get("evidence_text"),
                    },
                )
                graph.add_edge(
                    operation_id,
                    quantity_node_id,
                    "has_condition",
                    {
                        "order": quantity_idx,
                        "occurrence_id": occurrence_id,
                        "binding": "operation",
                    },
                )
            for condition_idx, ref in enumerate((duration_ref, temperature_ref)):
                condition_id = condition_node_ids.get(ref)
                if condition_id:
                    graph.add_edge(
                        operation_id,
                        condition_id,
                        "has_condition",
                        {
                            "placeholder": ref,
                            "order": condition_idx,
                            "occurrence_id": (
                                f"step_{step_idx}:"
                                f"{'duration' if condition_idx == 0 else 'temperature'}_0"
                            ),
                        },
                    )

            state_id = f"state_{step_idx:03d}"
            graph.add_node(
                state_id,
                "state",
                "final_state" if operation == "YIELD" else f"state_{step_idx}",
                {"created_by": operation_id},
            )
            graph.add_edge(operation_id, state_id, "output_from")
            previous_operation_id = operation_id
            previous_state_id = state_id

        graph.metadata["decoded_slots"] = slots
        graph.metadata["decoded_actions"] = " ; ".join(rendered_segments).rstrip(".") + "."
        graph.metadata["deterministic_render_trace"] = render_trace
        graph.metadata["deterministic_renderer_version"] = "lossless_operation_group_v1"
        return graph.to_dict()

    def decompile_generated_graph(self, graph: dict[str, Any]) -> str:
        return decompile_graph_to_sequence(graph, mode="exact")

    def normalize_quantity(self, value: float) -> float:
        return (math.log1p(max(value, 0.0)) - self.quantity_mean) / self.quantity_std

    def denormalize_quantity(self, value: float) -> float:
        return max(math.expm1(value * self.quantity_std + self.quantity_mean), 0.0)

    def normalize_duration(self, value: float) -> float:
        return (math.log1p(max(value, 0.0)) - self.duration_mean) / self.duration_std

    def denormalize_duration(self, value: float) -> float:
        return max(math.expm1(value * self.duration_std + self.duration_mean), 0.0)

    def normalize_temperature(self, value: float) -> float:
        return (math.log1p(max(value, 0.0)) - self.temperature_mean) / self.temperature_std

    def denormalize_temperature(self, value: float) -> float:
        return max(math.expm1(value * self.temperature_std + self.temperature_mean), 0.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_vocab": self.action_vocab,
            "material_vocab": self.material_vocab,
            "condition_vocab": self.condition_vocab,
            "unit_vocab": self.unit_vocab,
            "max_steps": self.max_steps,
            "max_material_slots": self.max_material_slots,
            "quantity_mean": self.quantity_mean,
            "quantity_std": self.quantity_std,
            "duration_mean": self.duration_mean,
            "duration_std": self.duration_std,
            "temperature_mean": self.temperature_mean,
            "temperature_std": self.temperature_std,
            "max_numeric_candidates": self.max_numeric_candidates,
            "numeric_candidate_include_source": self.numeric_candidate_include_source,
            "numeric_candidate_quantity_only": self.numeric_candidate_quantity_only,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "GraphTargetCodec":
        if "unit_vocab" not in payload:
            payload = {
                **payload,
                "unit_vocab": payload.get("quantity_vocab", [NONE_TOKEN]),
                "condition_vocab": payload.get("condition_vocab", list(LEGACY_CONDITION_TOKENS)),
                "max_material_slots": 1,
                "quantity_mean": 0.0,
                "quantity_std": 1.0,
                "duration_mean": 0.0,
                "duration_std": 1.0,
                "temperature_mean": 0.0,
                "temperature_std": 1.0,
            }
        if "condition_vocab" not in payload:
            payload = {**payload, "condition_vocab": list(LEGACY_CONDITION_TOKENS)}
        return cls(
            action_vocab=list(payload["action_vocab"]),
            material_vocab=list(payload["material_vocab"]),
            condition_vocab=list(payload["condition_vocab"]),
            unit_vocab=list(payload["unit_vocab"]),
            max_steps=int(payload["max_steps"]),
            max_material_slots=int(payload["max_material_slots"]),
            quantity_mean=float(payload["quantity_mean"]),
            quantity_std=float(payload["quantity_std"]),
            duration_mean=float(payload["duration_mean"]),
            duration_std=float(payload["duration_std"]),
            temperature_mean=float(payload["temperature_mean"]),
            temperature_std=float(payload["temperature_std"]),
            max_numeric_candidates=int(payload.get("max_numeric_candidates", 64)),
            # Old checkpoints used source text implicitly during grounding.
            numeric_candidate_include_source=bool(
                payload.get("numeric_candidate_include_source", True)
            ),
            # Old checkpoints used one mixed condition/quantity candidate pool.
            numeric_candidate_quantity_only=bool(
                payload.get("numeric_candidate_quantity_only", False)
            ),
        )

    def numeric_candidates_from_record(self, record: dict[str, Any]):
        return numeric_candidates_from_record(
            record,
            include_source=self.numeric_candidate_include_source,
            include_actions=False,
            include_condition_maps=not self.numeric_candidate_quantity_only,
            allowed_numeric_types=(
                QUANTITY_CANDIDATE_TYPES
                if self.numeric_candidate_quantity_only
                else None
            ),
        )[: self.max_numeric_candidates]

    def numeric_candidate_class_id(self, candidate_id: str | None) -> int:
        if not candidate_id or candidate_id == NONE_TOKEN:
            return 0
        if candidate_id == NONE_CANDIDATE_ID:
            return 1
        match = re.fullmatch(r"NUM_(\d+)", str(candidate_id))
        if match is None:
            return 1
        candidate_index = int(match.group(1))
        return candidate_index + 2 if candidate_index < self.max_numeric_candidates else 1

    def numeric_candidate_token(self, class_id: int) -> str:
        class_id = int(class_id)
        if class_id <= 0:
            return NONE_TOKEN
        if class_id == 1:
            return NONE_CANDIDATE_ID
        candidate_index = class_id - 2
        if candidate_index >= self.max_numeric_candidates:
            return NONE_CANDIDATE_ID
        return f"NUM_{candidate_index}"

    @staticmethod
    def _id(vocab: list[str], token: str, fallback: int) -> int:
        try:
            return vocab.index(token)
        except ValueError:
            return fallback

    def _condition_id(self, step: ActionStep) -> int:
        if self.uses_legacy_condition_classes:
            return self._legacy_condition_id(step)
        duration_ref = step.duration_refs[0] if step.duration_refs else ""
        temperature_ref = step.temperature_refs[0] if step.temperature_refs else ""
        token = combine_condition_token(duration_ref, temperature_ref)
        return self._id(self.condition_vocab, token, 0)

    @property
    def uses_legacy_condition_classes(self) -> bool:
        return tuple(self.condition_vocab) == LEGACY_CONDITION_TOKENS

    @staticmethod
    def _slot_condition_refs(
        slot: dict[str, Any],
        *,
        duration_refs: list[str],
        temperature_refs: list[str],
    ) -> tuple[str, str]:
        duration_ref = str(slot.get("duration_ref") or "")
        temperature_ref = str(slot.get("temperature_ref") or "")
        if duration_ref or temperature_ref:
            return (
                duration_ref if DURATION_PLACEHOLDER_RE.fullmatch(duration_ref) else "",
                temperature_ref
                if TEMPERATURE_PLACEHOLDER_RE.fullmatch(temperature_ref)
                else "",
            )

        condition = str(slot.get("condition") or NONE_TOKEN)
        if condition == "DURATION":
            return (duration_refs[0] if duration_refs else "", "")
        if condition == "TEMPERATURE":
            return ("", temperature_refs[0] if temperature_refs else "")
        if condition == "DURATION_TEMPERATURE":
            return (
                duration_refs[0] if duration_refs else "",
                temperature_refs[0] if temperature_refs else "",
            )
        parsed_duration, parsed_temperature = split_condition_token(condition)
        return (
            parsed_duration
            if DURATION_PLACEHOLDER_RE.fullmatch(parsed_duration)
            else "",
            parsed_temperature
            if TEMPERATURE_PLACEHOLDER_RE.fullmatch(parsed_temperature)
            else "",
        )


    @staticmethod
    def _legacy_condition_id(step: ActionStep) -> int:
        has_duration = bool(step.duration_refs)
        has_temperature = bool(step.temperature_refs)
        if has_duration and has_temperature:
            return 3
        if has_temperature:
            return 2
        if has_duration:
            return 1
        return 0

    @staticmethod
    def _first_condition_value(
        record: dict[str, Any],
        step: ActionStep,
        condition_type: str,
    ) -> float | None:
        refs = step.duration_refs if condition_type == "duration" else step.temperature_refs
        if not refs:
            return None
        field = "extracted_duration" if condition_type == "duration" else "extracted_temperature"
        value_to_ref = record.get(field) or {}
        ref_to_value = {ref: value for value, ref in value_to_ref.items()}
        raw = ref_to_value.get(refs[0])
        if raw is None:
            return None
        parsed = parse_numeric_unit(str(raw))
        return parsed[0] if parsed is not None else None

    @staticmethod
    def _condition_refs(record: dict[str, Any], condition_type: str) -> list[str]:
        field = "extracted_duration" if condition_type == "duration" else "extracted_temperature"
        refs = sorted((record.get(field) or {}).values(), key=_placeholder_sort_key)
        if refs:
            return refs
        return ["@1@"] if condition_type == "duration" else ["#1#"]

    @staticmethod
    def _add_condition_nodes(
        graph: ProcessGraph,
        duration_refs: list[str],
        temperature_refs: list[str],
    ) -> dict[str, str]:
        node_ids: dict[str, str] = {}
        for ref in duration_refs:
            node_id = f"cond_duration_{len(node_ids):03d}"
            graph.add_node(node_id, "condition", ref, {"placeholder": ref, "condition_type": "duration"})
            node_ids[ref] = node_id
        for ref in temperature_refs:
            node_id = f"cond_temperature_{len(node_ids):03d}"
            graph.add_node(
                node_id,
                "condition",
                ref,
                {"placeholder": ref, "condition_type": "temperature"},
            )
            node_ids[ref] = node_id
        return node_ids

    @staticmethod
    def _add_material_nodes(graph: ProcessGraph, record: dict[str, Any]) -> dict[str, str]:
        extracted = record.get("extracted_molecules") or {}
        placeholder_to_smiles = {placeholder: smiles for smiles, placeholder in extracted.items()}
        if not placeholder_to_smiles:
            placeholder_to_smiles = {"$-1$": "PRODUCT"}
            idx = 1
            for field_name in ("REACTANT", "CATALYST", "SOLVENT"):
                for smiles in record.get(field_name) or []:
                    placeholder_to_smiles[f"${idx}$"] = str(smiles)
                    idx += 1

        node_ids: dict[str, str] = {}
        for placeholder in sorted(placeholder_to_smiles, key=_placeholder_sort_key):
            node_id = f"mat_{placeholder.strip('$').replace('-', 'neg')}"
            graph.add_node(
                node_id,
                "material",
                placeholder,
                {
                    "placeholder": placeholder,
                    "smiles": placeholder_to_smiles[placeholder],
                    "role": "product" if placeholder == "$-1$" else "participant",
                },
            )
            node_ids[placeholder] = node_id
        return node_ids

    @staticmethod
    def _normalized_material_refs(
        slot: dict[str, Any],
        material_node_ids: dict[str, str],
    ) -> list[str]:
        operation = str(slot["operation_type"])
        raw_refs = list(slot.get("material_refs") or [])
        if not raw_refs and slot.get("material_ref") not in (None, NONE_TOKEN):
            raw_refs = [str(slot["material_ref"])]
        if operation == "YIELD" and "$-1$" in material_node_ids:
            return ["$-1$"]

        material_refs = [
            str(ref)
            for ref in raw_refs
            if MATERIAL_PLACEHOLDER_RE.fullmatch(str(ref))
        ]
        return material_refs

    @staticmethod
    def _render_slot(
        operation: str,
        slot: dict[str, Any],
        *,
        step_idx: int,
        material_refs: list[str],
        duration_ref: str,
        temperature_ref: str,
    ) -> tuple[str, dict[str, Any]]:
        raw_text = str(slot.get("raw_text") or "").strip().rstrip(".")
        argument_text = str(slot.get("argument_text") or "").strip()
        if raw_text:
            rendered = raw_text
            render_mode = "preserved_raw_text"
        elif argument_text:
            rendered = f"{operation} {argument_text}".strip()
            render_mode = "preserved_argument_text"
        else:
            rendered = GraphTargetCodec._render_step(
                operation,
                material_refs=material_refs,
                quantity_slots=list(slot.get("quantity_slots") or []),
                duration_ref=duration_ref,
                temperature_ref=temperature_ref,
            )
            render_mode = "deterministic_template"

        quantity_occurrences = [
            {
                "occurrence_id": f"step_{step_idx}:quantity_{quantity_idx}",
                "slot_id": int(quantity.get("slot_id", quantity_idx)),
                "candidate_id": quantity.get("candidate_id"),
                "text": format_quantity_slot_text(quantity),
                "render_count": 1 if render_mode == "deterministic_template" else None,
                "binding": "operation",
            }
            for quantity_idx, quantity in enumerate(slot.get("quantity_slots") or [])
        ]
        material_occurrences = [
            {
                "occurrence_id": f"step_{step_idx}:material_{material_idx}",
                "placeholder": material_ref,
                "render_count": 1 if render_mode == "deterministic_template" else None,
            }
            for material_idx, material_ref in enumerate(material_refs)
        ]
        condition_occurrences = []
        if duration_ref:
            condition_occurrences.append(
                {
                    "occurrence_id": f"step_{step_idx}:duration_0",
                    "condition_type": "duration",
                    "placeholder": duration_ref,
                    "render_count": 1 if render_mode == "deterministic_template" else None,
                }
            )
        if temperature_ref:
            condition_occurrences.append(
                {
                    "occurrence_id": f"step_{step_idx}:temperature_0",
                    "condition_type": "temperature",
                    "placeholder": temperature_ref,
                    "render_count": 1 if render_mode == "deterministic_template" else None,
                }
            )
        source_material_refs = list(slot.get("material_refs") or [])
        if not source_material_refs and slot.get("material_ref") not in (
            None,
            NONE_TOKEN,
        ):
            source_material_refs = [str(slot["material_ref"])]
        source_material_refs = [
            str(ref)
            for ref in source_material_refs
            if MATERIAL_PLACEHOLDER_RE.fullmatch(str(ref))
        ]
        removed_materials = list(
            (Counter(source_material_refs) - Counter(material_refs)).elements()
        )
        added_materials = list(
            (Counter(material_refs) - Counter(source_material_refs)).elements()
        )
        return rendered, {
            "step_id": step_idx,
            "operation_type": operation,
            "render_mode": render_mode,
            "quantity_binding_mode": "operation_ordered_group",
            "material_occurrences": material_occurrences,
            "quantity_occurrences": quantity_occurrences,
            "condition_occurrences": condition_occurrences,
            "injected_material_occurrences": [],
            "structurally_removed_material_occurrences": [
                {
                    "placeholder": placeholder,
                    "reason": "yield_product_constraint",
                }
                for placeholder in removed_materials
            ],
            "structurally_added_material_occurrences": [
                {
                    "placeholder": placeholder,
                    "reason": "yield_product_constraint",
                }
                for placeholder in added_materials
            ],
        }

    @staticmethod
    def _render_step(
        operation: str,
        *,
        material_refs: list[str],
        quantity_slots: list[dict[str, Any]],
        duration_ref: str,
        temperature_ref: str,
    ) -> str:
        joined_materials = " and ".join(material_refs)
        quantities = [format_quantity_slot_text(quantity) for quantity in quantity_slots]
        quantity_suffix = f" ({', '.join(quantities)})" if quantities else ""

        if operation == "YIELD":
            product_ref = material_refs[0] if material_refs else "$-1$"
            base = f"YIELD {product_ref}"
        elif operation == "MAKESOLUTION":
            base = f"MAKESOLUTION{' with ' + joined_materials if joined_materials else ''}"
        elif operation == "ADD":
            base = f"ADD{' ' + joined_materials if joined_materials else ''}"
        elif operation in {"QUENCH", "PH"}:
            base = f"{operation}{' with ' + joined_materials if joined_materials else ''}"
        elif operation in {"WASH", "EXTRACT", "TRITURATE", "RECRYSTALLIZE"}:
            base = f"{operation}{' with ' + joined_materials if joined_materials else ''}"
        elif operation == "DRYSOLUTION":
            base = f"DRYSOLUTION{' over ' + joined_materials if joined_materials else ''}"
        elif operation == "FILTER":
            base = f"FILTER{' ' + joined_materials if joined_materials else ''} keep filtrate"
        elif operation == "COLLECTLAYER":
            base = f"COLLECTLAYER{' ' + joined_materials if joined_materials else ''} organic"
        elif operation in {"PHASESEPARATION", "CONCENTRATE", "DRYSOLID", "DEGAS"}:
            base = f"{operation}{' ' + joined_materials if joined_materials else ''}"
        elif operation == "SETTEMPERATURE":
            base = f"SETTEMPERATURE{' ' + joined_materials if joined_materials else ''}"
        else:
            base = f"{operation}{' ' + joined_materials if joined_materials else ''}"

        base = f"{base.strip()}{quantity_suffix}"
        if operation == "SETTEMPERATURE" and temperature_ref:
            base = f"{base} {temperature_ref}"
            temperature_ref = ""
        if operation == "ADD":
            if temperature_ref:
                base = f"{base} at {temperature_ref}"
            if duration_ref:
                base = f"{base} over {duration_ref}"
            return base.strip()
        if duration_ref:
            base = f"{base} for {duration_ref}"
        if temperature_ref:
            base = f"{base} at {temperature_ref}"
        return base.strip()

    @staticmethod
    def _slot_raw_text(
        operation: str,
        slot: dict[str, Any],
        *,
        material_refs: list[str],
        duration_ref: str,
        temperature_ref: str,
    ) -> str:
        rendered, _ = GraphTargetCodec._render_slot(
            operation,
            slot,
            step_idx=int(slot.get("step_id", 0)),
            material_refs=material_refs,
            duration_ref=duration_ref,
            temperature_ref=temperature_ref,
        )
        return rendered

def matching_numeric_candidate(
    candidates,
    *,
    value: float,
    unit: str,
    numeric_type: str,
    raw_text: str,
):
    """Return an input candidate only when it actually supports the target value."""

    normalized_unit = normalize_numeric_unit(unit)
    matching = [
        candidate
        for candidate in candidates
        if candidate.normalized_value is not None
        and abs(float(candidate.normalized_value) - float(value))
        <= max(abs(float(value)) * 1e-3, 1e-3)
        and candidate.normalized_unit == normalized_unit
    ]
    return best_numeric_candidate(
        matching,
        value=value,
        unit=normalized_unit,
        numeric_type=numeric_type,
        raw_text=raw_text,
    )


def parse_numeric_unit(text: str) -> tuple[float, str] | None:
    parsed = parse_numeric_value_unit(text)
    if parsed is not None:
        return parsed
    number_match = NUMERIC_RE.search(text)
    if number_match is None:
        return None
    raw_number = number_match.group(0).replace(",", ".")
    try:
        value = float(raw_number)
    except ValueError:
        return None
    unit_match = UNIT_RE.search(text)
    unit = unit_match.group("unit").lower() if unit_match else "number"
    unit = normalize_unit(unit)
    return value, unit


def normalize_unit(unit: str) -> str:
    return normalize_numeric_unit(unit)


def robust_mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 1.0
    transformed = [math.log1p(max(value, 0.0)) for value in values]
    mean = sum(transformed) / len(transformed)
    variance = sum((value - mean) ** 2 for value in transformed) / max(len(transformed), 1)
    return mean, max(math.sqrt(variance), 1e-3)


def format_numeric_quantity(value: float, unit: str) -> str:
    if unit == NONE_TOKEN:
        return NONE_TOKEN
    if value >= 100:
        number = f"{value:.0f}"
    elif value >= 10:
        number = f"{value:.1f}".rstrip("0").rstrip(".")
    elif value >= 1:
        number = f"{value:.2f}".rstrip("0").rstrip(".")
    else:
        number = f"{value:.3f}".rstrip("0").rstrip(".")
    return f"{number} {unit}"


def format_quantity_slot_text(quantity: dict[str, Any]) -> str:
    """Render one decoded quantity occurrence without collapsing duplicates.

    Candidate evidence keeps its original number spelling because that is the
    closest available surface form. Only the ambiguous one-letter
    concentration aliases are expanded to a stable parser-visible unit.
    Missing values stay explicit rather than disappearing from the
    deterministic output.
    """

    text = str(quantity.get("text") or "").strip()
    unit = normalize_numeric_unit(str(quantity.get("unit") or ""))
    if text and text != NONE_TOKEN:
        if unit not in {"molar", "normal"}:
            return text
        lowered = text.lower()
        if re.search(r"\b(?:molar|normal)\b", lowered):
            return text
        number_match = DISPLAY_NUMBER_RE.search(text)
        if number_match is not None:
            return f"{number_match.group(0)} {unit}"
        return text

    value = quantity.get("value")
    if value is not None and unit and unit != NONE_TOKEN:
        try:
            return format_numeric_quantity(float(value), unit)
        except (TypeError, ValueError):
            pass
    return NONE_CANDIDATE_ID


def render_material_texts(
    material_refs: list[str],
    quantity_slots: list[dict[str, Any]],
) -> list[str]:
    texts: list[str] = []
    for idx, material_ref in enumerate(material_refs):
        quantity_text = ""
        if idx < len(quantity_slots):
            text = str(quantity_slots[idx].get("text") or "")
            if text and text != NONE_TOKEN:
                quantity_text = f" ({text})"
        texts.append(f"{material_ref}{quantity_text}")
    if not texts:
        for quantity in quantity_slots:
            text = str(quantity.get("text") or "")
            if text and text != NONE_TOKEN:
                texts.append(f"unknown ({text})")
    return texts


def combine_condition_token(duration_ref: str, temperature_ref: str) -> str:
    if duration_ref and temperature_ref:
        return f"{duration_ref}|{temperature_ref}"
    if duration_ref:
        return duration_ref
    if temperature_ref:
        return temperature_ref
    return NONE_TOKEN


def split_condition_token(token: str) -> tuple[str, str]:
    if not token or token == NONE_TOKEN:
        return "", ""
    if token in LEGACY_CONDITION_TOKENS:
        return "", ""
    duration_ref = ""
    temperature_ref = ""
    for part in token.split("|"):
        if part.startswith("@") and part.endswith("@"):
            duration_ref = part
        elif part.startswith("#") and part.endswith("#"):
            temperature_ref = part
    return duration_ref, temperature_ref


def _placeholder_sort_key(placeholder: str) -> tuple[int, int | str]:
    if placeholder == "$-1$":
        return (0, -1)
    if len(placeholder) >= 3 and placeholder[0] in "$@#" and placeholder[-1] == placeholder[0]:
        order = {"$": 0, "@": 1, "#": 2}[placeholder[0]]
        try:
            return (order, int(placeholder[1:-1]))
        except ValueError:
            pass
    return (3, placeholder)
