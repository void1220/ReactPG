"""Analyze decoded graph-slot richness in prediction JSONL files."""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reactgdiff.eval.text import corpus_text_metrics
from reactgdiff.utils.io import read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", help="Prediction JSONL path.")
    args = parser.parse_args()

    rows = list(read_jsonl(args.predictions))
    if not rows:
        print("No prediction rows found.")
        return

    op_counts: collections.Counter[str] = collections.Counter()
    unique_predictions: collections.Counter[str] = collections.Counter()
    pred_lengths: list[int] = []
    ref_lengths: list[int] = []
    text_pairs: list[tuple[str, str]] = []
    material_slots = 0
    quantity_slots = 0
    condition_slots = 0
    total_steps = 0

    for row in rows:
        slots = row.get("decoded_slots") or []
        pred_lengths.append(len(slots))
        ref_lengths.append(str(row.get("reference_actions", "")).count(";") + 1)
        predicted_actions = str(row.get("predicted_actions", ""))
        reference_actions = str(row.get("reference_actions", ""))
        text_pairs.append((predicted_actions, reference_actions))
        unique_predictions[predicted_actions] += 1
        for slot in slots:
            total_steps += 1
            op_counts[str(slot.get("operation_type", ""))] += 1
            material_slots += bool(slot.get("material_refs") or slot.get("material_ref") not in (None, "<NONE>"))
            quantity_slots += bool(slot.get("quantity_slots") or slot.get("quantity") not in (None, "<NONE>"))
            condition_slots += str(slot.get("condition", "<NONE>")) != "<NONE>"

    print(f"rows={len(rows)}")
    print(f"unique_predictions={len(unique_predictions)}")
    print(
        "pred_len="
        f"{mean(pred_lengths):.3f}/{min(pred_lengths)}/{max(pred_lengths)} "
        "(mean/min/max)"
    )
    print(
        "ref_len="
        f"{mean(ref_lengths):.3f}/{min(ref_lengths)}/{max(ref_lengths)} "
        "(mean/min/max)"
    )
    text_metrics = corpus_text_metrics(text_pairs)
    print(
        "text_metrics="
        f"BLEU-2:{text_metrics['bleu_2']:.4f} "
        f"ROUGE-1:{text_metrics['rouge_1']:.4f} "
        f"LEV75:{text_metrics['levenshtein_75_rate']:.4f} "
        f"LEV50:{text_metrics['levenshtein_50_rate']:.4f}"
    )
    print(f"material_step_rate={material_slots / max(total_steps, 1):.4f}")
    print(f"quantity_step_rate={quantity_slots / max(total_steps, 1):.4f}")
    print(f"condition_step_rate={condition_slots / max(total_steps, 1):.4f}")
    print(f"top_ops={op_counts.most_common(15)}")
    print(f"top_predictions={unique_predictions.most_common(5)}")


def mean(values: list[float] | list[int]) -> float:
    return sum(values) / max(len(values), 1)


if __name__ == "__main__":
    main()
