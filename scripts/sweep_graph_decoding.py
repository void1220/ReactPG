"""Scan graph-diffusion numeric decoding without resampling between settings.

The reverse-diffusion outputs are cached once on CPU.  Every gate-threshold and
candidate-reuse setting is then decoded from exactly those tensors, so metric
differences come only from deterministic decoding.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reactgdiff.eval.semantic import corpus_semantic_metrics
from reactgdiff.eval.slots import discrete_slot_metrics
from reactgdiff.eval.text import (
    _corpus_bleu_2,
    _rouge_1,
    _tokens,
    corpus_text_metrics,
    number_normalize_text,
)
from reactgdiff.eval.lev import normalize_text
from reactgdiff.models.joint_diffusion import ReactGDiffFeaturizer
from reactgdiff.models.procedure_graph_diffusion import (
    _forced_skeleton_ids,
    load_procedure_graph_diffusion_checkpoint,
)
from reactgdiff.utils.io import read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--skeleton-cache", required=True)
    parser.add_argument("--existing-predictions", default=None)
    parser.add_argument("--sweep-output", required=True)
    parser.add_argument("--best-predictions", required=True)
    parser.add_argument("--best-metrics", required=True)
    parser.add_argument(
        "--thresholds",
        default="0.50,0.60,0.65,0.70,0.75,0.80,0.825,0.85,0.875,0.90,0.925,0.95,0.975,0.99",
    )
    parser.add_argument(
        "--reuse-penalties",
        default="0,0.5,1,2,3,4,5,6,8,10,12",
    )
    parser.add_argument("--initial-reuse-penalty", type=float, default=1.0)
    parser.add_argument("--numeric-candidate-unit-weight", type=float, default=1.0)
    parser.add_argument("--condition-probability-threshold", type=float, default=0.05)
    parser.add_argument("--sample-steps", type=int, default=32)
    parser.add_argument(
        "--sample-mode",
        choices=("argmax", "sample", "sample_argmax_final"),
        default="sample_argmax_final",
    )
    parser.add_argument(
        "--sampler",
        choices=("posterior", "single_step", "iterative"),
        default="posterior",
    )
    parser.add_argument("--sample-temperature", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    thresholds = _parse_float_list(args.thresholds)
    reuse_penalties = _parse_float_list(args.reuse_penalties)
    records = list(read_jsonl(args.input, limit=args.limit))
    if not records:
        raise ValueError("Input split is empty")
    skeleton_cache = _load_indexed_rows(args.skeleton_cache)
    model, codec, featurizer_payload, _ = load_procedure_graph_diffusion_checkpoint(
        args.checkpoint,
        device=args.device,
    )
    if model.shared_text_encoder is not None:
        raise ValueError(
            "This fixed-output sweep currently targets the hash-conditioned checkpoint; "
            "shared-text checkpoints should use the standard prediction path."
        )
    featurizer = ReactGDiffFeaturizer.from_dict(featurizer_payload)

    print(
        f"Sampling one fixed reverse-diffusion trajectory for {len(records)} records "
        f"on {args.device}; batch={args.batch_size}, seed={args.seed}.",
        flush=True,
    )
    cached_batches = _sample_fixed_outputs(
        model=model,
        codec=codec,
        featurizer=featurizer,
        records=records,
        skeleton_cache=skeleton_cache,
        sample_steps=args.sample_steps,
        sample_mode=args.sample_mode,
        sample_temperature=args.sample_temperature,
        sampler=args.sampler,
        batch_size=args.batch_size,
        seed=args.seed,
        device=args.device,
    )
    reference_actions = [str(record.get("actions", "")) for record in records]
    candidate_pools = [codec.numeric_candidates_from_record(record) for record in records]
    existing_predictions = (
        _recompile_existing_predictions(
            args.existing_predictions,
            records=records,
            codec=codec,
        )
        if args.existing_predictions
        else None
    )

    evaluated: dict[tuple[float, float], dict[str, Any]] = {}

    def evaluate(threshold: float, reuse_penalty: float) -> dict[str, Any]:
        key = (_round_setting(threshold), _round_setting(reuse_penalty))
        if key in evaluated:
            return evaluated[key]
        pairs, _, baseline_mismatches = _decode_configuration(
            cached_batches=cached_batches,
            records=records,
            reference_actions=reference_actions,
            candidate_pools=candidate_pools,
            codec=codec,
            threshold=key[0],
            reuse_penalty=key[1],
            unit_weight=args.numeric_candidate_unit_weight,
            condition_probability_threshold=args.condition_probability_threshold,
            existing_predictions=existing_predictions if key == (0.75, 1.0) else None,
            keep_rows=False,
        )
        metrics = fast_corpus_text_metrics(pairs)
        metrics.update(corpus_semantic_metrics(pairs))
        metrics.update(
            {
                "quantity_gate_threshold": key[0],
                "numeric_candidate_reuse_penalty": key[1],
            }
        )
        if baseline_mismatches is not None:
            metrics["existing_prediction_mismatch_count"] = float(baseline_mismatches)
            metrics["existing_prediction_match_rate"] = (
                1.0 - baseline_mismatches / len(records)
            )
        evaluated[key] = metrics
        print(
            f"decoded threshold={key[0]:.3f} reuse={key[1]:.3f}: "
            f"semantic={metrics['semantic_score']:.6f}, "
            f"raw75={metrics['levenshtein_75_rate']:.6f}, "
            f"norm75={metrics['number_normalized_levenshtein_75_rate']:.6f}, "
            f"BLEU-2={metrics['bleu_2']:.6f}",
            flush=True,
        )
        return metrics

    print("Pass 1/2: scan quantity-gate thresholds at the current reuse penalty.", flush=True)
    threshold_scan = [evaluate(threshold, args.initial_reuse_penalty) for threshold in thresholds]
    pass1_best = _best_metrics(threshold_scan)

    print(
        "Pass 2/2: scan candidate-reuse penalties at the best gate threshold, "
        "then confirm the threshold once at the selected reuse penalty.",
        flush=True,
    )
    reuse_scan = [
        evaluate(pass1_best["quantity_gate_threshold"], reuse_penalty)
        for reuse_penalty in reuse_penalties
    ]
    pass2_best = _best_metrics(reuse_scan)
    threshold_confirmation = [
        evaluate(threshold, pass2_best["numeric_candidate_reuse_penalty"])
        for threshold in thresholds
    ]
    confirmed_threshold_best = _best_metrics(threshold_confirmation)
    reuse_confirmation = [
        evaluate(confirmed_threshold_best["quantity_gate_threshold"], reuse_penalty)
        for reuse_penalty in reuse_penalties
    ]
    final_coordinate_best = _best_metrics(reuse_confirmation)
    best = _best_metrics(evaluated.values())

    if (
        best["quantity_gate_threshold"]
        != final_coordinate_best["quantity_gate_threshold"]
        or best["numeric_candidate_reuse_penalty"]
        != final_coordinate_best["numeric_candidate_reuse_penalty"]
    ):
        print(
            "The best previously visited coordinate is better than the final coordinate; "
            "keeping the globally best visited setting.",
            flush=True,
        )

    best_key = (
        float(best["quantity_gate_threshold"]),
        float(best["numeric_candidate_reuse_penalty"]),
    )
    best_pairs, best_rows, _ = _decode_configuration(
        cached_batches=cached_batches,
        records=records,
        reference_actions=reference_actions,
        candidate_pools=candidate_pools,
        codec=codec,
        threshold=best_key[0],
        reuse_penalty=best_key[1],
        unit_weight=args.numeric_candidate_unit_weight,
        condition_probability_threshold=args.condition_probability_threshold,
        existing_predictions=None,
        keep_rows=True,
    )
    exact_metrics = corpus_text_metrics(best_pairs)
    exact_metrics.update(corpus_semantic_metrics(best_pairs))
    exact_metrics.update(discrete_slot_metrics(best_rows, records, codec=codec))
    exact_metrics.update(
        {
            "checkpoint": args.checkpoint,
            "predictions": args.best_predictions,
            "input": args.input,
            "skeleton_cache": args.skeleton_cache,
            "quantity_gate_threshold": best_key[0],
            "numeric_candidate_reuse_penalty": best_key[1],
            "numeric_candidate_unit_weight": args.numeric_candidate_unit_weight,
            "condition_probability_threshold": args.condition_probability_threshold,
            "drop_unsupported_numeric_slots": True,
            "graph_diffusion_sample_steps": args.sample_steps,
            "graph_diffusion_sample_mode": args.sample_mode,
            "graph_diffusion_sampler": args.sampler,
            "graph_diffusion_sample_temperature": args.sample_temperature,
            "sample_batch_size": args.batch_size,
            "seed": args.seed,
            "yield_template": "YIELD $-1$ (q1, q2, ...)",
        }
    )
    _assert_fast_metrics_match(best, exact_metrics)

    write_jsonl(args.best_predictions, best_rows)
    _write_json(args.best_metrics, exact_metrics)
    sweep_payload = {
        "checkpoint": args.checkpoint,
        "input": args.input,
        "count": len(records),
        "selection_metric": "semantic_score",
        "fixed_reverse_diffusion": True,
        "thresholds": thresholds,
        "reuse_penalties": reuse_penalties,
        "pass1_threshold_scan": threshold_scan,
        "pass1_best": pass1_best,
        "pass2_reuse_scan": reuse_scan,
        "pass2_best": pass2_best,
        "threshold_confirmation": threshold_confirmation,
        "reuse_confirmation": reuse_confirmation,
        "best_visited": best,
        "exact_best_metrics": exact_metrics,
        "metric_maxima": {
            metric: _best_metrics(evaluated.values(), metric=metric)
            for metric in (
                "semantic_score",
                "bleu_2",
                "rouge_1",
                "levenshtein_75_rate",
                "number_normalized_levenshtein_75_rate",
                "canonical_levenshtein_75_rate",
            )
        },
    }
    _write_json(args.sweep_output, sweep_payload)
    print(
        f"Best fixed-output decode: threshold={best_key[0]:.3f}, "
        f"reuse={best_key[1]:.3f}, semantic={exact_metrics['semantic_score']:.6f}, "
        f"BLEU-2={exact_metrics['bleu_2']:.6f}, "
        f"raw75={exact_metrics['levenshtein_75_rate']:.6f}, "
        f"norm75={exact_metrics['number_normalized_levenshtein_75_rate']:.6f}.",
        flush=True,
    )
    print(f"Wrote {args.best_predictions}, {args.best_metrics}, and {args.sweep_output}.")


@torch.no_grad()
def _sample_fixed_outputs(
    *,
    model,
    codec,
    featurizer,
    records: list[dict[str, Any]],
    skeleton_cache: dict[str, dict[str, Any]],
    sample_steps: int,
    sample_mode: str,
    sample_temperature: float,
    sampler: str,
    batch_size: int,
    seed: int,
    device: str,
) -> list[dict[str, torch.Tensor]]:
    torch.manual_seed(seed)
    model = model.to(device)
    model.eval()
    cached: list[dict[str, torch.Tensor]] = []
    for start in range(0, len(records), batch_size):
        batch_records = records[start : start + batch_size]
        condition = torch.tensor(
            [featurizer.condition_vector(record) for record in batch_records],
            dtype=torch.float32,
            device=device,
        )
        candidate_batch = (
            codec.encode_numeric_candidate_features(batch_records, device=device)
            if model.numeric_candidate_feature_pointer
            else ()
        )
        forced_op_ids = _forced_skeleton_ids(
            codec,
            batch_records,
            source="cache",
            cache=skeleton_cache,
            device=device,
        )
        output = model.sample_output(
            condition,
            forced_op_ids=forced_op_ids,
            sample_steps=sample_steps,
            sample_mode=sample_mode,
            temperature=sample_temperature,
            sampler=sampler,
            numeric_candidate_values=candidate_batch[0] if candidate_batch else None,
            numeric_candidate_confidences=candidate_batch[1] if candidate_batch else None,
            numeric_candidate_unit_ids=candidate_batch[2] if candidate_batch else None,
            numeric_candidate_type_ids=candidate_batch[3] if candidate_batch else None,
            numeric_candidate_source_ids=candidate_batch[4] if candidate_batch else None,
            numeric_candidate_mask=candidate_batch[5] if candidate_batch else None,
        )
        slot_output = output.slot_output
        cached.append(
            {
                "start": torch.tensor(start),
                "op": slot_output.op_logits.detach().cpu(),
                "material": slot_output.material_logits.detach().cpu(),
                "condition": slot_output.condition_logits.detach().cpu(),
                "quantity_gate": slot_output.quantity_gate_logits.detach().cpu(),
                "unit": slot_output.unit_logits.detach().cpu(),
                "quantity_values": slot_output.quantity_values.detach().cpu(),
                "condition_values": slot_output.condition_values.detach().cpu(),
                "numeric_candidate": (
                    slot_output.numeric_candidate_logits.detach().cpu()
                    if slot_output.numeric_candidate_logits is not None
                    else torch.empty(0)
                ),
            }
        )
        batch_number = len(cached)
        total_batches = math.ceil(len(records) / batch_size)
        if batch_number == 1 or batch_number == total_batches or batch_number % 10 == 0:
            print(
                f"cached reverse output batch {batch_number}/{total_batches} "
                f"({min(start + len(batch_records), len(records))}/{len(records)})",
                flush=True,
            )
    return cached


def _decode_configuration(
    *,
    cached_batches: list[dict[str, torch.Tensor]],
    records: list[dict[str, Any]],
    reference_actions: list[str],
    candidate_pools: list[list[Any]],
    codec,
    threshold: float,
    reuse_penalty: float,
    unit_weight: float,
    condition_probability_threshold: float,
    existing_predictions: dict[str, str] | None,
    keep_rows: bool,
) -> tuple[list[tuple[str, str]], list[dict[str, Any]], int | None]:
    pairs: list[tuple[str, str]] = []
    rows: list[dict[str, Any]] = []
    mismatch_count = 0 if existing_predictions is not None else None
    for batch in cached_batches:
        start = int(batch["start"])
        batch_count = batch["op"].size(0)
        numeric_candidate_logits = batch["numeric_candidate"]
        for offset in range(batch_count):
            record_idx = start + offset
            record = records[record_idx]
            slots = codec.decode_logits(
                batch["op"][offset],
                batch["material"][offset],
                batch["condition"][offset],
                batch["quantity_gate"][offset],
                batch["unit"][offset],
                batch["quantity_values"][offset],
                batch["condition_values"][offset],
                (
                    numeric_candidate_logits[offset]
                    if numeric_candidate_logits.numel()
                    else None
                ),
                quantity_gate_threshold=threshold,
                condition_probability_threshold=condition_probability_threshold,
                decode_quantities=True,
                decode_quantity_values=False,
                numeric_candidates=candidate_pools[record_idx],
                numeric_candidate_reuse_penalty=reuse_penalty,
                numeric_candidate_unit_weight=unit_weight,
                drop_unsupported_numeric_slots=True,
            )
            slots = codec.ground_numeric_slots(record, slots)
            graph = codec.build_generated_graph(record, slots)
            prediction = codec.decompile_generated_graph(graph)
            reference = reference_actions[record_idx]
            pairs.append((prediction, reference))
            if mismatch_count is not None:
                expected = existing_predictions.get(str(record.get("index")))
                mismatch_count += int(expected != prediction)
            if keep_rows:
                similarity = _string_similarity(prediction, reference)
                rows.append(
                    {
                        "index": record.get("index"),
                        "reference_actions": reference,
                        "predicted_actions": prediction,
                        "decoded_slots": slots,
                        "text_gap": 1.0 - similarity,
                        "levenshtein_similarity": similarity,
                        "decoder_backend": "procedure_graph_diffusion",
                        "skeleton_source": "cache",
                        "skeleton_operations": [
                            str(slot.get("operation_type") or "") for slot in slots
                        ],
                    }
                )
    return pairs, rows, mismatch_count


def fast_corpus_text_metrics(
    pairs: Iterable[tuple[str, str]],
) -> dict[str, float]:
    """Exact text metrics using a bit-parallel string edit-distance kernel."""

    pairs = list(pairs)
    if not pairs:
        return corpus_text_metrics([])
    pred_tokens = [_tokens(prediction) for prediction, _ in pairs]
    ref_tokens = [_tokens(reference) for _, reference in pairs]
    similarities = [
        _string_similarity(prediction, reference)
        for prediction, reference in pairs
    ]
    normalized_similarities = [
        _string_similarity(
            number_normalize_text(prediction),
            number_normalize_text(reference),
        )
        for prediction, reference in pairs
    ]
    rouge_scores = [
        _rouge_1(prediction, reference)
        for prediction, reference in zip(pred_tokens, ref_tokens, strict=True)
    ]
    exact_matches = sum(
        normalize_text(prediction) == normalize_text(reference)
        for prediction, reference in pairs
    )
    total = len(pairs)
    return {
        "count": float(total),
        "bleu_2": _corpus_bleu_2(pred_tokens, ref_tokens),
        "rouge_1": sum(score[2] for score in rouge_scores) / total,
        "rouge_1_precision": sum(score[0] for score in rouge_scores) / total,
        "rouge_1_recall": sum(score[1] for score in rouge_scores) / total,
        "levenshtein_90_rate": sum(value >= 0.90 for value in similarities) / total,
        "levenshtein_75_rate": sum(value >= 0.75 for value in similarities) / total,
        "levenshtein_50_rate": sum(value >= 0.50 for value in similarities) / total,
        "mean_number_normalized_levenshtein_similarity": (
            sum(normalized_similarities) / total
        ),
        "number_normalized_levenshtein_90_rate": (
            sum(value >= 0.90 for value in normalized_similarities) / total
        ),
        "number_normalized_levenshtein_75_rate": (
            sum(value >= 0.75 for value in normalized_similarities) / total
        ),
        "number_normalized_levenshtein_50_rate": (
            sum(value >= 0.50 for value in normalized_similarities) / total
        ),
        "exact_match_rate": exact_matches / total,
    }


def _string_similarity(left: str, right: str) -> float:
    normalized_left = normalize_text(left)
    normalized_right = normalize_text(right)
    distance = _myers_string_edit_distance(normalized_left, normalized_right)
    return 1.0 - distance / max(len(normalized_left), len(normalized_right), 1)


def _myers_string_edit_distance(left: str, right: str) -> int:
    """Exact Levenshtein distance using Myers' bit-vector algorithm."""

    if not left:
        return len(right)
    if not right:
        return len(left)
    if len(right) > len(left):
        left, right = right, left
    width = len(right)
    mask = (1 << width) - 1
    high_bit = 1 << (width - 1)
    char_masks: dict[str, int] = {}
    for idx, char in enumerate(right):
        char_masks[char] = char_masks.get(char, 0) | (1 << idx)
    positive = mask
    negative = 0
    score = width
    for char in left:
        equal = char_masks.get(char, 0)
        vertical = equal | negative
        horizontal = (((equal & positive) + positive) ^ positive) | equal
        positive_horizontal = negative | ~(horizontal | positive)
        negative_horizontal = positive & horizontal
        if positive_horizontal & high_bit:
            score += 1
        elif negative_horizontal & high_bit:
            score -= 1
        positive_horizontal = ((positive_horizontal << 1) | 1) & mask
        negative_horizontal = (negative_horizontal << 1) & mask
        positive = (negative_horizontal | ~(vertical | positive_horizontal)) & mask
        negative = positive_horizontal & vertical
    return score


def _best_metrics(
    candidates: Iterable[dict[str, Any]],
    *,
    metric: str = "semantic_score",
) -> dict[str, Any]:
    return max(
        candidates,
        key=lambda item: (
            float(item[metric]),
            float(item["semantic_score"]),
            -abs(float(item["quantity_gate_threshold"]) - 0.75),
            -float(item["numeric_candidate_reuse_penalty"]),
        ),
    )


def _recompile_existing_predictions(
    path: str,
    *,
    records: list[dict[str, Any]],
    codec,
) -> dict[str, str]:
    records_by_index = {str(record.get("index")): record for record in records}
    compiled: dict[str, str] = {}
    for row in read_jsonl(path):
        key = str(row.get("index"))
        record = records_by_index.get(key)
        if record is None:
            continue
        graph = codec.build_generated_graph(record, list(row.get("decoded_slots") or []))
        compiled[key] = codec.decompile_generated_graph(graph)
    return compiled


def _assert_fast_metrics_match(
    fast_metrics: dict[str, Any],
    exact_metrics: dict[str, Any],
) -> None:
    for key in (
        "bleu_2",
        "rouge_1",
        "levenshtein_90_rate",
        "levenshtein_75_rate",
        "levenshtein_50_rate",
        "mean_number_normalized_levenshtein_similarity",
        "number_normalized_levenshtein_75_rate",
        "exact_match_rate",
    ):
        if abs(float(fast_metrics[key]) - float(exact_metrics[key])) > 1e-12:
            raise AssertionError(
                f"Fast metric mismatch for {key}: "
                f"{fast_metrics[key]} != {exact_metrics[key]}"
            )


def _load_indexed_rows(path: str) -> dict[str, dict[str, Any]]:
    return {
        str(row["index"]): row
        for row in read_jsonl(path)
        if row.get("index") is not None
    }


def _parse_float_list(value: str) -> list[float]:
    parsed = sorted({_round_setting(float(item)) for item in value.split(",") if item.strip()})
    if not parsed:
        raise ValueError("At least one numeric setting is required")
    return parsed


def _round_setting(value: float) -> float:
    return round(float(value), 6)


def _write_json(path: str, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
