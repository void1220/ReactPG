"""Diagnose the deterministic ground-truth graph-to-action decode ceiling."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reactgdiff.eval.text import corpus_text_metrics
from reactgdiff.models.graph_codec import GraphTargetCodec
from reactgdiff.models.joint_diffusion import load_split_records
from reactgdiff.utils.io import write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", default="data/processed/openexp_sample")
    parser.add_argument("--split", default="test")
    parser.add_argument("--data", default=None, help="Explicit JSONL path; overrides --data_dir/--split.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output_dir", default="outputs/diagnostics/gt_graph_decode")
    parser.add_argument("--max-steps", type=int, default=32)
    parser.add_argument("--max-material-refs", type=int, default=16)
    parser.add_argument("--max-material-slots", type=int, default=4)
    args = parser.parse_args()

    data_path = Path(args.data) if args.data else Path(args.data_dir) / "splits" / f"{args.split}.jsonl"
    records = load_split_records(data_path, limit=args.limit)
    codec = GraphTargetCodec.fit(
        records,
        max_steps=args.max_steps,
        max_material_refs=args.max_material_refs,
        max_material_slots=args.max_material_slots,
    )

    rows = []
    for record in records:
        graph = codec.build_target_graph(record)
        prediction = codec.decompile_generated_graph(graph)
        reference = str(record.get("actions", ""))
        rows.append(
            {
                "index": record.get("index"),
                "reference_actions": reference,
                "predicted_actions": prediction,
                "decoded_slots": codec.target_slots_from_record(record),
                "decoder_backend": "gt_graph_decode",
            }
        )

    metrics = corpus_text_metrics((row["predicted_actions"], row["reference_actions"]) for row in rows)
    metrics.update(
        {
            "records": len(records),
            "data": str(data_path),
            "split": args.split,
            "diagnostic": "gt_graph_decode",
        }
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / f"{args.split}.gt_graph_decode.jsonl"
    metrics_path = output_dir / f"{args.split}.metrics.json"
    write_jsonl(predictions_path, rows)
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print(
        f"GT graph decode: records={len(records)} "
        f"LEV90={metrics['levenshtein_90_rate']:.4f} "
        f"LEV75={metrics['levenshtein_75_rate']:.4f} "
        f"num-norm-LEV75={metrics['number_normalized_levenshtein_75_rate']:.4f}"
    )
    print(f"Wrote predictions to {predictions_path}")
    print(f"Wrote metrics to {metrics_path}")


if __name__ == "__main__":
    main()
