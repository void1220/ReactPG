"""Evaluate predicted operation skeleton quality."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reactgdiff.data.action_parser import parse_action_sequence
from reactgdiff.eval.lev import edit_distance, levenshtein_similarity
from reactgdiff.utils.io import read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred", required=True, help="Prediction/skeleton JSONL path.")
    parser.add_argument("--gold", required=True, help="Gold processed JSONL path.")
    parser.add_argument("--output", default=None, help="Optional JSON metrics path.")
    args = parser.parse_args()

    pred_rows = list(read_jsonl(args.pred))
    gold_rows = list(read_jsonl(args.gold))
    if len(pred_rows) != len(gold_rows):
        gold_by_index = {str(row.get("index")): row for row in gold_rows if row.get("index") is not None}
        if all(str(row.get("index")) in gold_by_index for row in pred_rows):
            gold_rows = [gold_by_index[str(row.get("index"))] for row in pred_rows]
        else:
            raise ValueError(f"prediction/gold count mismatch: {len(pred_rows)} != {len(gold_rows)}")

    totals = Counter()
    topk_scores: list[float] = []
    for pred_row, gold_row in zip(pred_rows, gold_rows, strict=True):
        pred_ops = extract_operations(pred_row)
        gold_ops = [step.operation_type for step in parse_action_sequence(str(gold_row.get("actions", "")))]
        max_len = max(len(pred_ops), len(gold_ops), 1)
        aligned_matches = sum(
            pred == gold
            for pred, gold in zip(pred_ops, gold_ops, strict=False)
        )
        sequence_similarity = 1.0 - edit_distance(pred_ops, gold_ops) / max_len
        pred_counter = Counter(pred_ops)
        gold_counter = Counter(gold_ops)
        overlap = sum((pred_counter & gold_counter).values())
        precision = overlap / max(len(pred_ops), 1)
        recall = overlap / max(len(gold_ops), 1)
        coverage_f1 = 0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall)
        skeleton_lev = levenshtein_similarity(" ".join(pred_ops), " ".join(gold_ops))
        length_error = abs(len(pred_ops) - len(gold_ops))

        totals["operation_type_accuracy"] += aligned_matches / max_len
        totals["operation_sequence_edit_similarity"] += sequence_similarity
        totals["operation_coverage"] += recall
        totals["operation_coverage_f1"] += coverage_f1
        totals["skeleton_lev"] += skeleton_lev
        totals["skeleton_length_error"] += length_error
        totals["skeleton_exact_rate"] += float(pred_ops == gold_ops)
        oracle = topk_oracle_score(pred_row, gold_ops)
        if oracle is not None:
            topk_scores.append(oracle)

    count = len(pred_rows)
    metrics = {
        "count": float(count),
        **{key: value / max(count, 1) for key, value in totals.items()},
        "topk_skeleton_oracle_score": (
            sum(topk_scores) / len(topk_scores) if topk_scores else 0.0
        ),
        "topk_skeleton_oracle_count": float(len(topk_scores)),
        "pred": args.pred,
        "gold": args.gold,
    }

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(f"Wrote skeleton metrics to {args.output}")
    print(
        f"Skeleton eval: count={count} "
        f"op_acc={metrics['operation_type_accuracy']:.4f} "
        f"edit_sim={metrics['operation_sequence_edit_similarity']:.4f} "
        f"coverage={metrics['operation_coverage']:.4f} "
        f"len_err={metrics['skeleton_length_error']:.2f} "
        f"exact={metrics['skeleton_exact_rate']:.4f}"
    )


def extract_operations(row: dict[str, Any]) -> list[str]:
    for key in ("predicted_skeleton", "skeleton", "operations", "operation_types"):
        value = row.get(key)
        if isinstance(value, list):
            return [str(item).upper() for item in value]
        if isinstance(value, str):
            return [step.operation_type for step in parse_action_sequence(value)]
    slots = row.get("decoded_slots")
    if isinstance(slots, list):
        return [str(slot.get("operation_type") or "").upper() for slot in slots if isinstance(slot, dict)]
    for key in ("predicted_actions", "prediction", "decoded_actions", "output"):
        value = row.get(key)
        if isinstance(value, str):
            return [step.operation_type for step in parse_action_sequence(value)]
    return []


def topk_oracle_score(row: dict[str, Any], gold_ops: list[str]) -> float | None:
    candidates = row.get("topk_skeletons") or row.get("top_k_skeletons")
    if not isinstance(candidates, list) or not candidates:
        return None
    scores = []
    for candidate in candidates:
        if isinstance(candidate, dict):
            ops = extract_operations(candidate)
        elif isinstance(candidate, list):
            ops = [str(item).upper() for item in candidate]
        elif isinstance(candidate, str):
            ops = [step.operation_type for step in parse_action_sequence(candidate)]
        else:
            continue
        scores.append(1.0 - edit_distance(ops, gold_ops) / max(len(ops), len(gold_ops), 1))
    return max(scores) if scores else None


if __name__ == "__main__":
    main()
