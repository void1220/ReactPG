"""Sample decoded procedures from a trained ReactGDiff checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reactgdiff.models.graph_encoder_decoder import (
    load_argument_filler_from_checkpoint,
    load_graph_checkpoint,
    predict_direct_graph_records,
)
from reactgdiff.models.joint_diffusion import (
    ReactGDiffFeaturizer,
    load_checkpoint,
    load_split_records,
    predict_records,
)
from reactgdiff.models.procedure_graph_diffusion import (
    load_procedure_graph_diffusion_checkpoint,
    predict_procedure_graph_diffusion_records,
)
from reactgdiff.utils.io import read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default="outputs/checkpoints/reactgdiff_graph.pt",
        help="Checkpoint produced by scripts/train_reactgdiff.py.",
    )
    parser.add_argument(
        "--input",
        default="data/processed/openexp_sample/splits/test.jsonl",
        help="Input JSONL split.",
    )
    parser.add_argument(
        "--output",
        default="outputs/predictions/reactgdiff_graph_test.jsonl",
        help="Prediction JSONL path.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--decoder-backend",
        choices=("auto", "direct_graph", "graph_diffusion", "memory"),
        default="auto",
    )
    parser.add_argument(
        "--save-generated-graph",
        action="store_true",
        help="Store full generated graph JSON in predictions. Off by default to keep files small.",
    )
    parser.add_argument(
        "--quantity-gate-threshold",
        type=float,
        default=None,
        help=(
            "Decode a numeric slot only when its open probability reaches this threshold. "
            "Defaults to 0.999 for graph_diffusion and 0.65 for other backends."
        ),
    )
    parser.add_argument(
        "--condition-probability-threshold",
        type=float,
        default=0.05,
        help=(
            "Decode a duration/temperature ref only when the best non-empty condition "
            "reaches this probability. Concrete ref vocabularies usually need a lower "
            "threshold than the legacy 4-class condition target."
        ),
    )
    parser.add_argument(
        "--use-structure-length",
        action="store_true",
        help="Use the structure head to set decoded procedure length and final YIELD position.",
    )
    parser.add_argument(
        "--min-structure-steps",
        type=int,
        default=2,
        help="Minimum decoded step count when --use-structure-length is enabled.",
    )
    parser.add_argument(
        "--graph-diffusion-sample-steps",
        type=int,
        default=None,
        help="Reverse denoising steps for graph_diffusion checkpoints.",
    )
    parser.add_argument(
        "--graph-diffusion-sample-mode",
        choices=("argmax", "sample", "sample_argmax_final"),
        default="sample",
        help=(
            "Categorical selection mode inside graph diffusion reverse sampling. "
            "DiGress samples from each posterior step; argmax is a deterministic ablation; "
            "sample_argmax_final samples intermediate steps and argmaxes only the final step."
        ),
    )
    parser.add_argument(
        "--graph-diffusion-sampler",
        choices=("posterior", "single_step", "iterative"),
        default="posterior",
        help=(
            "posterior runs the standard x0-prediction reverse chain; "
            "single_step predicts the clean graph once from marginal noise; "
            "iterative keeps the legacy experimental re-noise loop."
        ),
    )
    parser.add_argument(
        "--graph-diffusion-sample-temperature",
        type=float,
        default=1.0,
        help="Sampling temperature when --graph-diffusion-sample-mode=sample.",
    )
    parser.add_argument(
        "--sample-batch-size",
        type=int,
        default=64,
        help="Sampling batch size for graph_diffusion checkpoints.",
    )
    parser.add_argument(
        "--skeleton-source",
        choices=("predicted", "gold", "cache", "none"),
        default="predicted",
        help="Skeleton source for graph_diffusion sampling.",
    )
    parser.add_argument(
        "--skeleton-cache",
        default=None,
        help="Optional JSONL cache with predicted skeletons keyed by index.",
    )
    parser.add_argument(
        "--ground-numeric-slots",
        action="store_true",
        help="Bind decoded numeric slots to input numeric evidence instead of continuous values.",
    )
    parser.add_argument(
        "--numeric-candidate-reuse-penalty",
        type=float,
        default=16.0,
        help="Penalty applied each time a record-local numeric candidate is reused.",
    )
    parser.add_argument(
        "--numeric-candidate-unit-weight",
        type=float,
        default=1.0,
        help="Weight of candidate/unit agreement during final numeric decoding.",
    )
    parser.add_argument(
        "--drop-unsupported-numeric-slots",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Close quantity gates that select NONE or missing numeric evidence.",
    )
    parser.add_argument(
        "--no-decode-quantity-values",
        action="store_true",
        help="Decode numeric gates/units only; values are expected to come from grounding.",
    )
    parser.add_argument(
        "--use-argument-filler",
        action="store_true",
        help="Legacy/debug only: apply a saved final AR argument filler if present.",
    )
    args = parser.parse_args()

    records = load_split_records(args.input, limit=args.limit)
    backend = args.decoder_backend
    if backend == "auto":
        backend = checkpoint_backend(args.checkpoint)
    quantity_gate_threshold = args.quantity_gate_threshold
    if quantity_gate_threshold is None:
        quantity_gate_threshold = 0.999 if backend == "graph_diffusion" else 0.65

    if backend == "direct_graph":
        model, codec, condition_featurizer, _ = load_graph_checkpoint(
            args.checkpoint,
            device=args.device,
        )
        argument_filler, argument_text_codec, _ = load_argument_filler_from_checkpoint(
            args.checkpoint,
            device=args.device,
        )
        featurizer = ReactGDiffFeaturizer.from_dict(condition_featurizer)
        rows = predict_direct_graph_records(
            model,
            codec,
            records,
            condition_vectors=[featurizer.condition_vector(record) for record in records],
            argument_filler=argument_filler,
            argument_text_codec=argument_text_codec,
            include_generated_graph=args.save_generated_graph,
            quantity_gate_threshold=quantity_gate_threshold,
            condition_probability_threshold=args.condition_probability_threshold,
            use_structure_length=args.use_structure_length,
            min_structure_steps=args.min_structure_steps,
            device=args.device,
        )
    elif backend == "graph_diffusion":
        model, codec, condition_featurizer, _ = load_procedure_graph_diffusion_checkpoint(
            args.checkpoint,
            device=args.device,
        )
        if args.use_argument_filler:
            argument_filler, argument_text_codec, _ = load_argument_filler_from_checkpoint(
                args.checkpoint,
                device=args.device,
            )
        else:
            argument_filler, argument_text_codec = None, None
        featurizer = ReactGDiffFeaturizer.from_dict(condition_featurizer)
        skeleton_cache = load_skeleton_cache(args.skeleton_cache) if args.skeleton_cache else None
        rows = predict_procedure_graph_diffusion_records(
            model,
            codec,
            records,
            condition_vectors=[featurizer.condition_vector(record) for record in records],
            argument_filler=argument_filler,
            argument_text_codec=argument_text_codec,
            include_generated_graph=args.save_generated_graph,
            quantity_gate_threshold=quantity_gate_threshold,
            condition_probability_threshold=args.condition_probability_threshold,
            sample_steps=args.graph_diffusion_sample_steps,
            sample_mode=args.graph_diffusion_sample_mode,
            sample_temperature=args.graph_diffusion_sample_temperature,
            sample_batch_size=args.sample_batch_size,
            sampler=args.graph_diffusion_sampler,
            decode_quantity_values=not args.no_decode_quantity_values,
            ground_numeric_slots=args.ground_numeric_slots,
            numeric_candidate_reuse_penalty=args.numeric_candidate_reuse_penalty,
            numeric_candidate_unit_weight=args.numeric_candidate_unit_weight,
            drop_unsupported_numeric_slots=args.drop_unsupported_numeric_slots,
            skeleton_source=args.skeleton_source,
            skeleton_cache=skeleton_cache,
            use_structure_length=args.use_structure_length,
            min_structure_steps=args.min_structure_steps,
            seed=args.seed,
            device=args.device,
        )
    else:
        model, schedule, featurizer, candidates, _ = load_checkpoint(
            args.checkpoint,
            device=args.device,
        )
        rows = predict_records(
            model,
            schedule,
            featurizer,
            records,
            candidates,
            seed=args.seed,
            device=args.device,
        )
    count = write_jsonl(args.output, rows)
    print(f"Wrote {count} decoded predictions to {args.output}")


def checkpoint_backend(path: str | Path) -> str:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("checkpoint_type") == "direct_graph_encoder_decoder":
        return "direct_graph"
    if payload.get("checkpoint_type") == "procedure_graph_diffusion":
        return "graph_diffusion"
    return "memory"


def load_skeleton_cache(path: str | None) -> dict[str, dict]:
    if not path:
        return {}
    cache: dict[str, dict] = {}
    for row in read_jsonl(path):
        index = row.get("index")
        if index is None:
            continue
        cache[str(index)] = row
    return cache


if __name__ == "__main__":
    main()
