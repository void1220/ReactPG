from __future__ import annotations

import subprocess
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from reactgdiff.eval.rendering import deterministic_render_metrics
from reactgdiff.eval.semantic import corpus_semantic_metrics
from reactgdiff.models.graph_codec import GraphTargetCodec
from reactgdiff.models.graph_encoder_decoder import GraphDecoderOutput
from reactgdiff.models.procedure_graph_diffusion import (
    ProcedureGraphDiffusion,
    ProcedureGraphDiffusionOutput,
    output_with_sampled_categories,
)


ROOT = Path(__file__).resolve().parents[1]


def _record() -> dict:
    return {
        "index": 1,
        "REACTANT": ["CCO"],
        "PRODUCT": ["CC=O"],
        "CATALYST": [],
        "SOLVENT": [],
        "extracted_molecules": {"CCO": "$1$", "CC=O": "$-1$"},
        "extracted_duration": {},
        "extracted_temperature": {},
        "source": "The starting material (5 g) was added.",
        "actions": "ADD $1$ (5 g) ; YIELD $-1$.",
    }


def test_semantic_metrics_ignore_template_wording() -> None:
    prediction = "ADD $1$ ; STIR for @1@ at #1# ; YIELD $-1$."
    reference = "ADD slowly $1$ ; STIR the mixture for @1@ at #1# ; YIELD $-1$."
    metrics = corpus_semantic_metrics([(prediction, reference)])
    assert metrics["semantic_procedure_exact_rate"] == 1.0
    assert metrics["canonical_levenshtein_75_rate"] == 1.0


def test_yield_template_renders_all_decoded_quantities() -> None:
    record = _record()
    codec = GraphTargetCodec.fit(
        [record],
        max_steps=4,
        max_material_refs=2,
        max_material_slots=2,
    )
    slots = [
        {
            "step_id": 0,
            "operation_type": "YIELD",
            "material_refs": ["$-1$"],
            "quantity_slots": [
                {"slot_id": 0, "text": "5 g"},
                {"slot_id": 1, "text": "80 %"},
            ],
        }
    ]
    graph = codec.build_generated_graph(record, slots)
    assert codec.decompile_generated_graph(graph) == "YIELD $-1$ (5 g, 80 %)."


def test_yield_product_constraint_is_explicit_in_render_trace() -> None:
    record = _record()
    codec = GraphTargetCodec.fit(
        [record],
        max_steps=4,
        max_material_refs=2,
        max_material_slots=2,
    )
    graph = codec.build_generated_graph(
        record,
        [
            {
                "step_id": 0,
                "operation_type": "YIELD",
                "material_refs": ["$1$"],
                "quantity_slots": [],
            }
        ],
    )
    trace = graph["metadata"]["deterministic_render_trace"][0]
    assert codec.decompile_generated_graph(graph) == "YIELD $-1$."
    assert trace["structurally_removed_material_occurrences"] == [
        {"placeholder": "$1$", "reason": "yield_product_constraint"}
    ]
    assert trace["structurally_added_material_occurrences"] == [
        {"placeholder": "$-1$", "reason": "yield_product_constraint"}
    ]


def test_deterministic_renderer_preserves_repeated_occurrences_without_binding() -> None:
    record = {
        **_record(),
        "extracted_duration": {"2 hours": "@1@"},
        "extracted_temperature": {"20 C": "#1#"},
    }
    codec = GraphTargetCodec.fit(
        [record],
        max_steps=4,
        max_material_refs=2,
        max_material_slots=2,
    )
    slots = [
        {
            "step_id": 0,
            "operation_type": "ADD",
            "material_refs": ["$1$", "$1$"],
            "duration_ref": "@1@",
            "temperature_ref": "#1#",
            "quantity_slots": [
                {
                    "slot_id": 0,
                    "candidate_id": "NUM_0",
                    "text": "5 g",
                    "unit": "g",
                    "value": 5.0,
                },
                {
                    "slot_id": 1,
                    "candidate_id": "NUM_0",
                    "text": "5 g",
                    "unit": "g",
                    "value": 5.0,
                },
            ],
        }
    ]
    graph = codec.build_generated_graph(record, slots)
    assert (
        codec.decompile_generated_graph(graph)
        == "ADD $1$ and $1$ (5 g, 5 g) at #1# over @1@."
    )
    trace = graph["metadata"]["deterministic_render_trace"][0]
    assert [item["occurrence_id"] for item in trace["material_occurrences"]] == [
        "step_0:material_0",
        "step_0:material_1",
    ]
    assert [item["occurrence_id"] for item in trace["quantity_occurrences"]] == [
        "step_0:quantity_0",
        "step_0:quantity_1",
    ]
    assert all(item["render_count"] == 1 for item in trace["quantity_occurrences"])
    quantity_nodes = [
        node
        for node in graph["nodes"]
        if node.get("attrs", {}).get("condition_type") == "quantity"
    ]
    assert len(quantity_nodes) == 2
    assert len({node["id"] for node in quantity_nodes}) == 2

    metrics = deterministic_render_metrics(
        [
            {
                "decoded_slots": slots,
                "deterministic_render_trace": graph["metadata"][
                    "deterministic_render_trace"
                ],
            }
        ]
    )
    assert metrics["render_all_occurrences_once_rate"] == 1.0
    assert metrics["render_duplicate_candidate_preservation_rate"] == 1.0
    assert metrics["render_duplicate_material_preservation_rate"] == 1.0


def test_deterministic_renderer_does_not_inject_default_material() -> None:
    record = _record()
    codec = GraphTargetCodec.fit(
        [record],
        max_steps=4,
        max_material_refs=2,
        max_material_slots=2,
    )
    slots = [
        {
            "step_id": 0,
            "operation_type": "ADD",
            "material_refs": [],
            "quantity_slots": [
                {"slot_id": 0, "text": "1M", "unit": "molar", "value": 1.0},
                {"slot_id": 1, "text": "2N", "unit": "normal", "value": 2.0},
            ],
        }
    ]
    graph = codec.build_generated_graph(record, slots)
    assert codec.decompile_generated_graph(graph) == "ADD (1 molar, 2 normal)."
    assert graph["metadata"]["deterministic_render_trace"][0][
        "injected_material_occurrences"
    ] == []


def test_deterministic_renderer_exposes_unresolved_predicted_placeholders() -> None:
    record = _record()
    codec = GraphTargetCodec.fit(
        [record],
        max_steps=4,
        max_material_refs=4,
        max_material_slots=2,
    )
    graph = codec.build_generated_graph(
        record,
        [
            {
                "step_id": 0,
                "operation_type": "ADD",
                "material_refs": ["$7$"],
                "duration_ref": "@8@",
                "temperature_ref": "#9#",
                "quantity_slots": [],
            }
        ],
    )
    assert codec.decompile_generated_graph(graph) == "ADD $7$ at #9# over @8@."
    unresolved_nodes = [
        node
        for node in graph["nodes"]
        if node.get("attrs", {}).get("source") == "decoded_slot"
    ]
    assert {
        (node["type"], node["attrs"]["placeholder"])
        for node in unresolved_nodes
    } == {
        ("material", "$7$"),
        ("condition", "@8@"),
        ("condition", "#9#"),
    }


def test_set_temperature_renders_condition_once_without_extra_preposition() -> None:
    record = {
        **_record(),
        "extracted_temperature": {"20 C": "#1#"},
    }
    codec = GraphTargetCodec.fit(
        [record],
        max_steps=4,
        max_material_refs=2,
        max_material_slots=2,
    )
    graph = codec.build_generated_graph(
        record,
        [
            {
                "step_id": 0,
                "operation_type": "SETTEMPERATURE",
                "material_refs": [],
                "temperature_ref": "#1#",
                "quantity_slots": [],
            }
        ],
    )
    assert codec.decompile_generated_graph(graph) == "SETTEMPERATURE #1#."
    condition_trace = graph["metadata"]["deterministic_render_trace"][0][
        "condition_occurrences"
    ]
    assert len(condition_trace) == 1
    assert condition_trace[0]["render_count"] == 1


def test_skeleton_cli_import_has_no_prompt_model_cycle() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/train_skeleton_seq2seq.py", "--help"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_numeric_candidate_target_requires_input_evidence() -> None:
    record = _record()
    codec = GraphTargetCodec.fit(
        [record],
        max_steps=4,
        max_material_refs=2,
        max_material_slots=2,
        max_numeric_candidates=4,
        numeric_candidate_include_source=False,
    )
    assert codec.encode_record(record)["numeric_candidate_ids"][0][0] == 1

    codec.numeric_candidate_include_source = True
    assert codec.encode_record(record)["numeric_candidate_ids"][0][0] == 2


def test_quantity_candidate_pool_excludes_conditions_and_nmr_hydrogens() -> None:
    record = {
        **_record(),
        "source": (
            "1H NMR: -1H; the starting material (5 g) was stirred "
            "for 2 hours at 20 C."
        ),
        "extracted_duration": {"2 hours": "@1@"},
        "extracted_temperature": {"20 C": "#1#"},
    }
    codec = GraphTargetCodec.fit(
        [record],
        max_steps=4,
        max_material_refs=2,
        max_material_slots=2,
        max_numeric_candidates=8,
        numeric_candidate_include_source=True,
        numeric_candidate_quantity_only=True,
    )
    candidates = codec.numeric_candidates_from_record(record)
    assert [candidate.raw_text for candidate in candidates] == ["5 g"]
    features = codec.numeric_candidate_features_from_record(record)
    assert sum(features["mask"]) == 3  # NONE, MISSING, NUM_0
    assert features["unit_ids"][2] == codec.unit_vocab.index("g")


def test_graph_diffusion_emits_numeric_candidate_logits() -> None:
    model = ProcedureGraphDiffusion(
        condition_dim=8,
        action_dim=5,
        material_dim=4,
        condition_slot_dim=3,
        unit_dim=3,
        numeric_candidate_dim=6,
        max_steps=4,
        max_material_slots=2,
        hidden_dim=16,
        dit_depth=1,
        dit_heads=4,
        diffusion_steps=2,
    )
    output = model(
        torch.zeros(2, 8),
        torch.ones(2, dtype=torch.long),
        torch.zeros(2, 4, dtype=torch.long),
        torch.zeros(2, 4, 2, dtype=torch.long),
        torch.zeros(2, 4, dtype=torch.long),
        torch.zeros(2, 4, 2, dtype=torch.long),
        torch.zeros(2, 4, 2, dtype=torch.long),
        torch.zeros(2, 4, 2, dtype=torch.long),
    )
    assert output.slot_output.numeric_candidate_logits is not None
    assert output.slot_output.numeric_candidate_logits.shape == (2, 4, 2, 6)


def test_sampled_output_preserves_numeric_logits_for_decode_controls() -> None:
    raw_gate_logits = torch.tensor([[[[0.2, 0.8]]]])
    raw_unit_logits = torch.tensor([[[[0.1, 0.7, 0.2]]]])
    raw_candidate_logits = torch.tensor([[[[0.0, -1.0, 1.5, 0.5]]]])
    output = ProcedureGraphDiffusionOutput(
        slot_output=GraphDecoderOutput(
            op_logits=torch.zeros(1, 1, 3),
            material_logits=torch.zeros(1, 1, 1, 3),
            condition_logits=torch.zeros(1, 1, 2),
            quantity_gate_logits=raw_gate_logits,
            unit_logits=raw_unit_logits,
            quantity_values=torch.zeros(1, 1, 1),
            condition_values=torch.zeros(1, 1, 2),
            numeric_candidate_logits=raw_candidate_logits,
        ),
        structure_logits=torch.zeros(1, 6),
    )
    sampled = output_with_sampled_categories(
        output,
        op_ids=torch.zeros(1, 1, dtype=torch.long),
        material_ids=torch.zeros(1, 1, 1, dtype=torch.long),
        condition_ids=torch.zeros(1, 1, dtype=torch.long),
        quantity_gate_ids=torch.ones(1, 1, 1, dtype=torch.long),
        unit_ids=torch.ones(1, 1, 1, dtype=torch.long),
        numeric_candidate_ids=torch.full((1, 1, 1), 2, dtype=torch.long),
        action_dim=3,
        material_dim=3,
        condition_dim=2,
        unit_dim=3,
        numeric_candidate_dim=4,
    )
    assert torch.equal(sampled.slot_output.quantity_gate_logits, raw_gate_logits)
    assert torch.equal(sampled.slot_output.unit_logits, raw_unit_logits)
    assert torch.equal(sampled.slot_output.numeric_candidate_logits, raw_candidate_logits)


def test_hash_mode_uses_compact_dynamic_numeric_pointer() -> None:
    model = ProcedureGraphDiffusion(
        condition_dim=8,
        action_dim=5,
        material_dim=4,
        condition_slot_dim=3,
        unit_dim=3,
        numeric_candidate_dim=6,
        max_steps=4,
        max_material_slots=2,
        hidden_dim=16,
        dit_depth=1,
        dit_heads=4,
        diffusion_steps=2,
        numeric_candidate_feature_pointer=True,
        numeric_candidate_type_dim=8,
        numeric_candidate_source_dim=6,
    )
    context = model.prepare_numeric_candidate_context(
        values=torch.tensor([[0.0, 0.0, 0.2, 0.4, 0.0, 0.0]]),
        confidences=torch.tensor([[0.0, 0.0, 0.85, 0.85, 0.0, 0.0]]),
        unit_ids=torch.tensor([[0, 0, 1, 2, 0, 0]]),
        type_ids=torch.tensor([[0, 0, 1, 1, 0, 0]]),
        source_ids=torch.tensor([[0, 0, 1, 1, 0, 0]]),
        mask=torch.tensor([[1, 1, 1, 1, 0, 0]], dtype=torch.bool),
    )
    output = model(
        torch.zeros(1, 8),
        torch.ones(1, dtype=torch.long),
        torch.zeros(1, 4, dtype=torch.long),
        torch.zeros(1, 4, 2, dtype=torch.long),
        torch.zeros(1, 4, dtype=torch.long),
        torch.zeros(1, 4, 2, dtype=torch.long),
        torch.zeros(1, 4, 2, dtype=torch.long),
        torch.zeros(1, 4, 2, dtype=torch.long),
        numeric_candidate_context=context,
    )
    logits = output.slot_output.numeric_candidate_logits
    assert logits is not None
    assert logits.shape == (1, 4, 2, 6)
    assert torch.all(logits[..., 4:] <= -1.0e3)
    logits[..., :4].mean().backward()
    assert model.numeric_candidate_pointer_query.weight.grad is not None
    assert model.numeric_candidate_key_scalar_projection.weight.grad is not None


def test_joint_candidate_decode_penalizes_excessive_reuse() -> None:
    record = {
        **_record(),
        "source": "The material (5 g) was diluted with solvent (10 ml).",
    }
    codec = GraphTargetCodec.fit(
        [record],
        max_steps=4,
        max_material_refs=2,
        max_material_slots=2,
        max_numeric_candidates=4,
        numeric_candidate_include_source=True,
        numeric_candidate_quantity_only=True,
    )
    candidates = codec.numeric_candidates_from_record(record)
    logits = torch.tensor([-10.0, -10.0, 5.0, 4.0, -10.0, -10.0])
    unit_logits = torch.zeros(codec.unit_dim)
    selected = codec._select_numeric_candidate(
        logits,
        unit_logits,
        candidates,
        Counter({"NUM_0": 1}),
        reuse_penalty=2.0,
        unit_weight=0.0,
    )
    assert selected == "NUM_1"


class _DummyTextEncoder(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(32, hidden_size)
        self.call_count = 0

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_dict: bool,
    ) -> SimpleNamespace:
        del attention_mask, return_dict
        self.call_count += 1
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


def test_shared_encoder_cross_attention_and_candidate_pointer_masks() -> None:
    encoder = _DummyTextEncoder(hidden_size=12)
    model = ProcedureGraphDiffusion(
        condition_dim=8,
        action_dim=5,
        material_dim=4,
        condition_slot_dim=3,
        unit_dim=3,
        numeric_candidate_dim=6,
        max_steps=4,
        max_material_slots=2,
        hidden_dim=16,
        dit_depth=1,
        dit_heads=4,
        diffusion_steps=2,
        shared_text_encoder=encoder,
        shared_encoder_dim=12,
        shared_encoder_trainable=True,
    )
    context = model.prepare_shared_text_context(
        input_ids=torch.tensor([[1, 2, 3, 4, 5, 0], [6, 7, 8, 9, 0, 0]]),
        attention_mask=torch.tensor([[1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 0, 0]]),
        material_candidate_positions=torch.tensor([[0, 1, -1, 3], [0, -1, 2, 3]]),
        numeric_candidate_positions=torch.tensor(
            [[0, 1, 2, -1, -1, 4], [0, 1, -1, 3, -1, -1]]
        ),
    )
    inputs = (
        torch.zeros(2, 8),
        torch.ones(2, dtype=torch.long),
        torch.zeros(2, 4, dtype=torch.long),
        torch.zeros(2, 4, 2, dtype=torch.long),
        torch.zeros(2, 4, dtype=torch.long),
        torch.zeros(2, 4, 2, dtype=torch.long),
        torch.zeros(2, 4, 2, dtype=torch.long),
        torch.zeros(2, 4, 2, dtype=torch.long),
    )
    output = model(*inputs, shared_context=context)
    assert encoder.call_count == 1
    assert output.slot_output.material_logits.shape == (2, 4, 2, 4)
    assert output.slot_output.numeric_candidate_logits is not None
    assert output.slot_output.numeric_candidate_logits.shape == (2, 4, 2, 6)
    assert torch.all(output.slot_output.material_logits[0, :, :, 2] <= -1.0e3)
    assert torch.all(output.slot_output.numeric_candidate_logits[0, :, :, 3] <= -1.0e3)

    second_output = model(*inputs, shared_context=context)
    assert encoder.call_count == 1
    loss = (
        second_output.slot_output.material_logits[..., 0].mean()
        + second_output.slot_output.numeric_candidate_logits[..., 0].mean()
    )
    loss.backward()
    assert encoder.embedding.weight.grad is not None
    assert model.material_pointer_query.weight.grad is not None
