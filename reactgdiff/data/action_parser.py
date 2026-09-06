"""Parser for OpenExp action strings.

This parser intentionally preserves the original action text while extracting
the references needed to build a lightweight process graph. It is not meant to
be a chemistry executor.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

MATERIAL_REF_RE = re.compile(r"\$-?\d+\$")
DURATION_REF_RE = re.compile(r"@\d+@")
TEMPERATURE_REF_RE = re.compile(r"#\d+#")
PAREN_RE = re.compile(r"\(([^()]*)\)")
QUANTITY_UNIT_RE = re.compile(
    r"(?<![A-Za-z])(?:ug|µg|mg|g|kg|ul|µl|ml|l|liter|liters|litre|litres|"
    r"umol|µmol|mmol|mol|molar|normal|equiv|equivalent|eq|percent|%)"
    r"(?=\b|[^A-Za-z0-9_])",
    re.IGNORECASE,
)

KNOWN_OPENEXP_ACTIONS = {
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
    "YIELD",
}


@dataclass(slots=True)
class ActionStep:
    """Parsed representation of one OpenExp action segment."""

    step_id: int
    operation_type: str
    raw_text: str
    arguments: str
    material_refs: list[str] = field(default_factory=list)
    duration_refs: list[str] = field(default_factory=list)
    temperature_refs: list[str] = field(default_factory=list)
    quantities: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "operation_type": self.operation_type,
            "raw_text": self.raw_text,
            "arguments": self.arguments,
            "material_refs": self.material_refs,
            "duration_refs": self.duration_refs,
            "temperature_refs": self.temperature_refs,
            "quantities": self.quantities,
        }


def parse_action_sequence(actions: str) -> list[ActionStep]:
    """Parse the semicolon-separated OpenExp action field."""

    steps: list[ActionStep] = []
    for raw_segment in split_action_sequence(actions):
        text = raw_segment.strip()
        if not text:
            continue
        text = text.rstrip(".").strip()
        if not text:
            continue

        if " " in text:
            operation_type, arguments = text.split(" ", 1)
        else:
            operation_type, arguments = text, ""

        steps.append(
            ActionStep(
                step_id=len(steps),
                operation_type=operation_type.upper(),
                raw_text=text,
                arguments=arguments,
                material_refs=MATERIAL_REF_RE.findall(text),
                duration_refs=DURATION_REF_RE.findall(text),
                temperature_refs=TEMPERATURE_REF_RE.findall(text),
                quantities=extract_quantity_texts(text),
            )
        )
    return steps


def split_action_sequence(actions: str) -> list[str]:
    """Split an action string on top-level semicolons.

    A few OpenExp rows contain semicolons inside quantities or bracketed
    chemical names. Treating those as action separators creates bogus operation
    types, so the splitter tracks simple bracket depth.
    """

    return [segment for segment, _ in split_action_sequence_with_separators(actions)]


def split_action_sequence_with_separators(actions: str) -> list[tuple[str, str]]:
    """Split an action string and preserve each segment's following separator."""

    segments: list[tuple[str, str]] = []
    start = 0
    paren_depth = 0
    bracket_depth = 0
    brace_depth = 0
    for idx, char in enumerate(actions):
        if char == "(":
            paren_depth += 1
        elif char == ")" and paren_depth:
            paren_depth -= 1
        elif char == "[":
            bracket_depth += 1
        elif char == "]" and bracket_depth:
            bracket_depth -= 1
        elif char == "{":
            brace_depth += 1
        elif char == "}" and brace_depth:
            brace_depth -= 1
        elif char == ";":
            if _next_segment_starts_known_action(actions, idx) or (
                paren_depth == 0
                and bracket_depth == 0
                and brace_depth == 0
            ) or not _open_groups_close_before_next_separator(
                actions,
                idx,
                paren_depth=paren_depth,
                bracket_depth=bracket_depth,
                brace_depth=brace_depth,
            ):
                raw_segment = actions[start:idx]
                segment = raw_segment.rstrip()
                separator_start = idx
                separator_end = idx + 1
                while separator_end < len(actions) and actions[separator_end].isspace():
                    separator_end += 1
                separator = raw_segment[len(segment) :] + actions[separator_start:separator_end]
                segments.append((segment, separator))
                start = separator_end
                paren_depth = 0
                bracket_depth = 0
                brace_depth = 0
                continue
    segments.append((actions[start:].rstrip(), ""))
    return segments


def _next_segment_starts_known_action(actions: str, separator_idx: int) -> bool:
    tail = actions[separator_idx + 1 :].lstrip()
    for action in KNOWN_OPENEXP_ACTIONS:
        if tail.upper() == action or tail.upper().startswith(f"{action} "):
            return True
    return False


def _open_groups_close_before_next_separator(
    actions: str,
    separator_idx: int,
    *,
    paren_depth: int,
    bracket_depth: int,
    brace_depth: int,
) -> bool:
    next_separator = actions.find(";", separator_idx + 1)
    if next_separator == -1:
        next_separator = len(actions)
    window = actions[separator_idx + 1 : next_separator]
    if paren_depth and ")" not in window:
        return False
    if bracket_depth and "]" not in window:
        return False
    if brace_depth and "}" not in window:
        return False
    return True


def extract_quantity_texts(text: str) -> list[str]:
    """Extract parenthetical quantity-like spans.

    Parentheses in chemical names are common, for example ``Pd(PPh3)4``. A span
    is treated as a quantity only when it contains a digit plus a known unit or
    yield/percent marker.
    """

    quantities: list[str] = []
    for match in PAREN_RE.finditer(text):
        raw = match.group(1).strip()
        quantities.extend(split_quantity_text(raw))
    return quantities


def split_quantity_text(raw: str) -> list[str]:
    """Split a parenthetical quantity span into individual numeric components."""

    if not is_quantity_text(raw):
        return []
    if "yield" in raw.lower():
        return [raw]
    parts = [part.strip() for part in re.split(r"\s*[,;]\s*", raw) if part.strip()]
    quantity_parts = [part for part in parts if is_quantity_text(part)]
    if len(quantity_parts) >= 2:
        return quantity_parts
    return [raw]


def is_quantity_text(raw: str) -> bool:
    normalized = raw.strip().lower()
    if not normalized:
        return False
    if "yield" in normalized and re.search(r"\d", normalized):
        return True
    return bool(re.search(r"\d", normalized) and QUANTITY_UNIT_RE.search(normalized))


def quantity_material_bindings(text: str) -> list[str]:
    """Conservative structural labels: only a material immediately before a quantity group.

    Multiple components in one group share the material. Ambiguous groups remain unbound.
    No numerical values are used to choose a material.
    """
    bindings = []
    for match in PAREN_RE.finditer(text):
        quantities = split_quantity_text(match.group(1).strip())
        adjacent = re.search(r"(\$-?\d+\$)\s*$", text[:match.start()])
        bindings.extend([adjacent.group(1) if adjacent else ''] * len(quantities))
    return bindings
