"""Re-render saved graph slots with the current deterministic templates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reactgdiff.eval.semantic import corpus_semantic_metrics
from reactgdiff.eval.rendering import deterministic_render_metrics
from reactgdiff.eval.slots import discrete_slot_metrics
from reactgdiff.eval.text import corpus_text_metrics
from reactgdiff.models.procedure_graph_diffusion import (
    load_procedure_graph_diffusion_checkpoint,
)
from reactgdiff.utils.io import read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metrics", required=True)
    args = parser.parse_args()

    records = list(read_jsonl(args.input))
    records_by_index = {str(record.get("index")): record for record in records}
    _, codec, _, _ = load_procedure_graph_diffusion_checkpoint(
        args.checkpoint,
        device="cpu",
    )
    rows = []
    ordered_records = []
    for saved in read_jsonl(args.predictions):
        key = str(saved.get("index"))
        record = records_by_index.get(key)
        if record is None:
            raise KeyError(f"Prediction index {key!r} is missing from {args.input}")
        slots = list(saved.get("decoded_slots") or [])
        graph = codec.build_generated_graph(record, slots)
        prediction = codec.decompile_generated_graph(graph)
        graph_metadata = graph.get("metadata") or {}
        rows.append(
            {
                "index": saved.get("index"),
                "reference_actions": str(record.get("actions", "")),
                "predicted_actions": prediction,
                "decoded_slots": slots,
                "decoder_backend": saved.get(
                    "decoder_backend",
                    "procedure_graph_diffusion",
                ),
                "skeleton_source": saved.get("skeleton_source"),
                "skeleton_operations": saved.get("skeleton_operations"),
                "template_recompiled": True,
                "deterministic_renderer_version": graph_metadata.get(
                    "deterministic_renderer_version"
                ),
                "deterministic_render_trace": graph_metadata.get(
                    "deterministic_render_trace",
                    [],
                ),
            }
        )
        ordered_records.append(record)

    pairs = [
        (row["predicted_actions"], row["reference_actions"])
        for row in rows
    ]
    metrics = corpus_text_metrics(pairs)
    metrics.update(corpus_semantic_metrics(pairs))
    metrics.update(discrete_slot_metrics(rows, ordered_records, codec=codec))
    metrics.update(deterministic_render_metrics(rows))
    metrics.update(
        {
            "checkpoint": args.checkpoint,
            "source_predictions": args.predictions,
            "predictions": args.output,
            "template_recompiled": True,
            "deterministic_renderer_version": "lossless_operation_group_v1",
            "quantity_binding_mode": "operation_ordered_group",
            "yield_template": "YIELD $-1$ (q1, q2, ...)",
        }
    )

    write_jsonl(args.output, rows)
    metrics_path = Path(args.metrics)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(
        f"Recompiled {len(rows)} predictions: BLEU-2={metrics['bleu_2']:.6f}, "
        f"ROUGE-1={metrics['rouge_1']:.6f}, "
        f"raw75={metrics['levenshtein_75_rate']:.6f}, "
        f"norm75={metrics['number_normalized_levenshtein_75_rate']:.6f}, "
        f"semantic={metrics['semantic_score']:.6f}.",
        flush=True,
    )
    print(f"Wrote {args.output} and {args.metrics}.")


if __name__ == "__main__":
    main()
