"""Evaluate decoded procedure predictions with text metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reactgdiff.eval.text import corpus_text_metrics
from reactgdiff.eval.semantic import corpus_semantic_metrics
from reactgdiff.utils.io import read_jsonl


PREDICTION_KEYS = ("predicted_actions", "prediction", "decoded_actions", "output")
REFERENCE_KEYS = ("reference_actions", "reference", "target_actions", "target")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions",
        default="outputs/predictions/reactgdiff_small_val.jsonl",
        help="Prediction JSONL path.",
    )
    parser.add_argument(
        "--output",
        default="outputs/metrics/reactgdiff_small_eval.json",
        help="Metric JSON path.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Deprecated; LEV75 and LEV50 are always reported.",
    )
    args = parser.parse_args()

    rows = list(read_jsonl(args.predictions))
    pairs = [(_first_present(row, PREDICTION_KEYS), _first_present(row, REFERENCE_KEYS)) for row in rows]
    metrics = corpus_text_metrics(pairs)
    metrics.update(corpus_semantic_metrics(pairs))
    metrics["predictions"] = args.predictions

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(
        f"Evaluated {int(metrics['count'])} predictions: "
        f"BLEU-2={metrics['bleu_2']:.4f}, "
        f"ROUGE-1={metrics['rouge_1']:.4f}, "
        f"90%LEV={metrics['levenshtein_90_rate']:.4f}, "
        f"75%LEV={metrics['levenshtein_75_rate']:.4f}, "
        f"50%LEV={metrics['levenshtein_50_rate']:.4f}, "
        f"num-norm-75%LEV={metrics['number_normalized_levenshtein_75_rate']:.4f}"
        f", semantic={metrics['semantic_score']:.4f}"
        f", canonical-75%LEV={metrics['canonical_levenshtein_75_rate']:.4f}"
    )
    print(f"Wrote metrics to {args.output}")


def _first_present(row: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = row.get(key)
        if value is not None:
            return str(value)
    raise KeyError(f"None of {keys!r} were found in prediction row")


if __name__ == "__main__":
    main()
