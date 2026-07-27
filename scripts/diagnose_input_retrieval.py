"""Diagnose whether input encodings retrieve procedure skeletons that generalize."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reactgdiff.data.action_parser import parse_action_sequence
from reactgdiff.eval.lev import edit_distance
from reactgdiff.eval.text import corpus_text_metrics
from reactgdiff.models.joint_diffusion import ReactGDiffFeaturizer, load_split_records
from reactgdiff.utils.io import write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", default="data/processed/openexp_sample/splits/train.jsonl")
    parser.add_argument("--val", default="data/processed/openexp_sample/splits/val.jsonl")
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--val-limit", type=int, default=None)
    parser.add_argument("--mode", choices=("condition", "source", "molecule_jaccard"), default="condition")
    parser.add_argument("--condition-encoding", choices=("reactxt_hash", "field_hash", "scalar_hash"), default="reactxt_hash")
    parser.add_argument("--field-dim", type=int, default=64)
    parser.add_argument("--ngram-min", type=int, default=2)
    parser.add_argument("--ngram-max", type=int, default=5)
    parser.add_argument("--no-numeric-evidence-input", action="store_true")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output", default="outputs/diagnostics/input_retrieval.json")
    parser.add_argument("--predictions", default=None)
    args = parser.parse_args()

    train_records = load_split_records(args.train, limit=args.train_limit)
    val_records = load_split_records(args.val, limit=args.val_limit)
    if not train_records or not val_records:
        raise ValueError("train and val splits must be non-empty")

    include_numeric_evidence = not args.no_numeric_evidence_input
    featurizer = ReactGDiffFeaturizer.fit(
        train_records,
        condition_encoding=args.condition_encoding,
        field_dim=args.field_dim,
        ngram_min=args.ngram_min,
        ngram_max=args.ngram_max,
        include_numeric_evidence=include_numeric_evidence,
    )

    if args.mode == "condition":
        train_vectors = [featurizer.condition_vector(record) for record in train_records]
        val_vectors = [featurizer.condition_vector(record) for record in val_records]
        scorer = lambda val_idx, train_idx: cosine_similarity(val_vectors[val_idx], train_vectors[train_idx])
    elif args.mode == "source":
        train_vectors = [hashed_text_vector(str(record.get("source") or ""), args.field_dim * 4) for record in train_records]
        val_vectors = [hashed_text_vector(str(record.get("source") or ""), args.field_dim * 4) for record in val_records]
        scorer = lambda val_idx, train_idx: cosine_similarity(val_vectors[val_idx], train_vectors[train_idx])
    else:
        train_sets = [molecule_set(record) for record in train_records]
        val_sets = [molecule_set(record) for record in val_records]
        scorer = lambda val_idx, train_idx: jaccard(val_sets[val_idx], train_sets[train_idx])

    train_skeletons = [operation_sequence(record) for record in train_records]
    val_skeletons = [operation_sequence(record) for record in val_records]
    train_skeleton_set = {tuple(skeleton) for skeleton in train_skeletons}

    rows: list[dict[str, Any]] = []
    top_k = max(int(args.top_k), 1)
    top1_sims: list[float] = []
    oracle_sims: list[float] = []
    seen_skeletons = 0
    exact_top1 = 0
    length_errors: list[int] = []
    for val_idx, val_record in enumerate(val_records):
        scores = [(scorer(val_idx, train_idx), train_idx) for train_idx in range(len(train_records))]
        scores.sort(reverse=True, key=lambda item: item[0])
        top = scores[:top_k]
        best_score, best_idx = top[0]
        pred_record = train_records[best_idx]
        pred_ops = train_skeletons[best_idx]
        ref_ops = val_skeletons[val_idx]
        top1_sim = skeleton_similarity(pred_ops, ref_ops)
        oracle_sim = max(skeleton_similarity(train_skeletons[idx], ref_ops) for _, idx in top)
        top1_sims.append(top1_sim)
        oracle_sims.append(oracle_sim)
        seen_skeletons += int(tuple(ref_ops) in train_skeleton_set)
        exact_top1 += int(pred_ops == ref_ops)
        length_errors.append(abs(len(pred_ops) - len(ref_ops)))
        rows.append(
            {
                "index": val_record.get("index"),
                "nearest_index": pred_record.get("index"),
                "nearest_score": best_score,
                "reference_actions": str(val_record.get("actions") or ""),
                "predicted_actions": str(pred_record.get("actions") or ""),
                "reference_skeleton": ref_ops,
                "predicted_skeleton": pred_ops,
                "topk_skeletons": [train_skeletons[idx] for _, idx in top],
                "top1_skeleton_similarity": top1_sim,
                "topk_oracle_skeleton_similarity": oracle_sim,
            }
        )

    text_metrics = corpus_text_metrics((row["predicted_actions"], row["reference_actions"]) for row in rows)
    metrics = {
        **text_metrics,
        "mode": args.mode,
        "train_records": len(train_records),
        "val_records": len(val_records),
        "condition_encoding": args.condition_encoding,
        "field_dim": args.field_dim,
        "include_numeric_evidence": include_numeric_evidence,
        "top_k": top_k,
        "top1_skeleton_similarity": sum(top1_sims) / len(top1_sims),
        "topk_oracle_skeleton_similarity": sum(oracle_sims) / len(oracle_sims),
        "top1_skeleton_exact_rate": exact_top1 / len(rows),
        "val_skeleton_seen_in_train_rate": seen_skeletons / len(rows),
        "mean_skeleton_length_error": sum(length_errors) / len(length_errors),
        "train_unique_skeletons": len(train_skeleton_set),
        "val_unique_skeletons": len({tuple(skeleton) for skeleton in val_skeletons}),
        "train_top_skeletons": [
            {"skeleton": " ; ".join(skeleton), "count": count}
            for skeleton, count in Counter(tuple(s) for s in train_skeletons).most_common(10)
        ],
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    predictions_path = Path(args.predictions) if args.predictions else output_path.with_suffix(".jsonl")
    write_jsonl(predictions_path, rows)
    print(
        f"Input retrieval[{args.mode}]: records={len(rows)} "
        f"top1_skel={metrics['top1_skeleton_similarity']:.4f} "
        f"top{top_k}_oracle={metrics['topk_oracle_skeleton_similarity']:.4f} "
        f"seen={metrics['val_skeleton_seen_in_train_rate']:.4f} "
        f"LEV75={metrics['levenshtein_75_rate']:.4f}"
    )
    print(f"Wrote metrics to {output_path}")
    print(f"Wrote predictions to {predictions_path}")


def operation_sequence(record: dict[str, Any]) -> list[str]:
    return [step.operation_type for step in parse_action_sequence(str(record.get("actions") or ""))]


def skeleton_similarity(left: list[str], right: list[str]) -> float:
    return 1.0 - edit_distance(left, right) / max(len(left), len(right), 1)


def molecule_set(record: dict[str, Any]) -> set[str]:
    values: set[str] = set()
    for field in ("REACTANT", "PRODUCT", "CATALYST", "SOLVENT"):
        values.update(str(value) for value in record.get(field) or [] if value)
    values.update(str(value) for value in (record.get("extracted_molecules") or {}).keys() if value)
    return values


def jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    return len(left & right) / max(len(left | right), 1)


def hashed_text_vector(text: str, dim: int) -> list[float]:
    dim = max(int(dim), 16)
    vector = [0.0] * dim
    normalized = f"<{text}>"
    vector[0] = min(len(normalized), 2048) / 2048.0
    vector[1] = min(len(set(normalized)), 128) / 128.0
    for ngram_size in range(2, 6):
        if len(normalized) < ngram_size:
            continue
        weight = 1.0 / ngram_size
        for start in range(0, len(normalized) - ngram_size + 1):
            token = normalized[start : start + ngram_size]
            digest = hashlib.sha1(token.encode("utf-8")).hexdigest()
            bucket = 2 + (int(digest[:8], 16) % (dim - 2))
            sign = -1.0 if int(digest[8:10], 16) % 2 else 1.0
            vector[bucket] += sign * weight
    norm = math.sqrt(sum(value * value for value in vector[2:]))
    if norm > 0:
        for idx in range(2, dim):
            vector[idx] /= norm
    return vector


def cosine_similarity(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    return numerator / max(left_norm * right_norm, 1e-12)


if __name__ == "__main__":
    main()
