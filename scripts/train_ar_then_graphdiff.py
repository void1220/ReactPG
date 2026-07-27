"""Run AR skeleton training followed by grounded graph diffusion training."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]


def _add_option(command: list[str], flag: str, value: object | None) -> None:
    if value is None:
        return
    command.extend([flag, str(value)])


def _run(command: list[str], *, dry_run: bool) -> None:
    print("\n$ " + " ".join(command), flush=True)
    if dry_run:
        return
    subprocess.run(command, cwd=ROOT, check=True)


def _split_name(path: str) -> str:
    stem = Path(path).stem
    return stem or "eval"


def _resolve_input_splits(args: argparse.Namespace) -> None:
    data_root = Path(args.data_root) if args.data_root else Path("data/processed/openexp")
    if args.train is None:
        args.train = str(data_root / "splits" / "train.jsonl")
    if args.val is None:
        args.val = str(data_root / "splits" / "val.jsonl")
    if args.scale:
        cache_root = Path(args.split_cache_dir)
        dataset_name = data_root.name if args.data_root else "openexp"
        scale_root = cache_root / dataset_name / f"scale_{args.scale}"
        args.train = str(_filter_scale_split(Path(args.train), scale_root / "train.jsonl", args.scale, dry_run=args.dry_run))
        args.val = str(_filter_scale_split(Path(args.val), scale_root / "val.jsonl", args.scale, dry_run=args.dry_run))
    print("Dataset splits:", flush=True)
    print(f"  train: {args.train}", flush=True)
    print(f"  val:   {args.val}", flush=True)


def _filter_scale_split(source: Path, target: Path, scale: str, *, dry_run: bool) -> Path:
    if dry_run:
        print(f"Would filter {source} -> {target} where _buckets.scale == {scale}", flush=True)
        return target
    if not source.exists():
        raise FileNotFoundError(f"Missing split file: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    total = 0
    with source.open("r", encoding="utf-8") as src, target.open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            total += 1
            record = json.loads(line)
            if (record.get("_buckets") or {}).get("scale") != scale:
                continue
            dst.write(json.dumps(record, ensure_ascii=False) + "\n")
            kept += 1
    print(f"Filtered scale={scale}: {source} -> {target} ({kept}/{total})", flush=True)
    if kept == 0:
        raise ValueError(f"No records matched scale={scale} in {source}")
    return target


def _default_paths(args: argparse.Namespace) -> dict[str, str]:
    eval_name = args.eval_name or _split_name(args.val)
    run_name = args.run_name
    skeleton_suffix = "seq2seq_skeleton" if args.skeleton_backend == "seq2seq" else "ar_skeleton"
    return {
        "ar_checkpoint": args.ar_checkpoint
        or f"outputs/checkpoints/{run_name}_{skeleton_suffix}.pt",
        "ar_predictions": args.ar_predictions
        or f"outputs/skeleton/{run_name}_{skeleton_suffix}_{eval_name}.jsonl",
        "ar_metrics": args.ar_metrics
        or f"outputs/metrics/{run_name}_{skeleton_suffix}_{eval_name}.json",
        "graph_checkpoint": args.graph_checkpoint
        or f"outputs/checkpoints/{run_name}.pt",
        "graph_predictions": args.graph_predictions
        or f"outputs/predictions/{run_name}_{eval_name}.jsonl",
        "graph_metrics": args.graph_metrics
        or f"outputs/metrics/{run_name}_{eval_name}.json",
    }


def _mkdirs(paths: Iterable[str], *, dry_run: bool) -> None:
    if dry_run:
        return
    for path in paths:
        Path(path).parent.mkdir(parents=True, exist_ok=True)


def build_ar_command(args: argparse.Namespace, paths: dict[str, str]) -> list[str]:
    if args.skeleton_backend == "seq2seq":
        return build_seq2seq_skeleton_command(args, paths)
    command = [
        sys.executable,
        "scripts/train_skeleton_ar.py",
        "--train",
        args.train,
        "--val",
        args.val,
    ]
    _add_option(command, "--train-limit", args.train_limit)
    _add_option(command, "--val-limit", args.val_limit)
    command.extend(
        [
            "--epochs",
            str(args.ar_epochs),
            "--batch-size",
            str(args.ar_batch_size),
            "--learning-rate",
            str(args.ar_learning_rate),
            "--hidden-dim",
            str(args.ar_hidden_dim),
            "--layers",
            str(args.ar_layers),
            "--dropout",
            str(args.ar_dropout),
            "--max-steps",
            str(args.max_steps),
            "--max-material-refs",
            str(args.max_material_refs),
            "--max-material-slots",
            str(args.max_material_slots),
            "--condition-encoding",
            args.condition_encoding,
            "--field-dim",
            str(args.field_dim),
            "--ngram-min",
            str(args.ngram_min),
            "--ngram-max",
            str(args.ngram_max),
            "--gradient-clip-norm",
            str(args.gradient_clip_norm),
            "--seed",
            str(args.seed),
            "--device",
            args.device,
            "--checkpoint",
            paths["ar_checkpoint"],
            "--predictions",
            paths["ar_predictions"],
            "--metrics",
            paths["ar_metrics"],
            "--log-every",
            str(args.log_every),
        ]
    )
    if args.no_numeric_evidence_input:
        command.append("--no-numeric-evidence-input")
    return command


def build_seq2seq_skeleton_command(args: argparse.Namespace, paths: dict[str, str]) -> list[str]:
    command = [
        sys.executable,
        "scripts/train_skeleton_seq2seq.py",
        "--train",
        args.train,
        "--val",
        args.val,
    ]
    _add_option(command, "--train-limit", args.train_limit)
    _add_option(command, "--val-limit", args.val_limit)
    command.extend(
        [
            "--model-name",
            args.skeleton_model_name,
            "--epochs",
            str(args.seq2seq_epochs),
            "--batch-size",
            str(args.seq2seq_batch_size),
            "--eval-batch-size",
            str(args.seq2seq_eval_batch_size),
            "--learning-rate",
            str(args.seq2seq_learning_rate),
            "--weight-decay",
            str(args.seq2seq_weight_decay),
            "--warmup-ratio",
            str(args.seq2seq_warmup_ratio),
            "--gradient-accumulation-steps",
            str(args.seq2seq_gradient_accumulation_steps),
            "--max-input-length",
            str(args.seq2seq_max_input_length),
            "--max-target-length",
            str(args.seq2seq_max_target_length),
            "--prompt-style",
            args.seq2seq_prompt_style,
            "--target-format",
            args.seq2seq_target_format,
            "--participant-shuffle-prob",
            str(args.seq2seq_participant_shuffle_prob),
            "--numeric-format-augment-prob",
            str(args.seq2seq_numeric_format_augment_prob),
            "--field-mask-prob",
            str(args.seq2seq_field_mask_prob),
            "--max-steps",
            str(args.max_steps),
            "--max-material-refs",
            str(args.max_material_refs),
            "--max-material-slots",
            str(args.max_material_slots),
            "--action-loss-weighting",
            args.seq2seq_action_loss_weighting,
            "--action-weight-beta",
            str(args.seq2seq_action_weight_beta),
            "--action-weight-min",
            str(args.seq2seq_action_weight_min),
            "--action-weight-max",
            str(args.seq2seq_action_weight_max),
            "--length-loss-weight",
            str(args.seq2seq_length_loss_weight),
            "--length-prior-weight",
            str(args.seq2seq_length_prior_weight),
            "--repetition-penalty-weight",
            str(args.seq2seq_repetition_penalty_weight),
            "--best-eval-interval",
            str(args.seq2seq_best_eval_interval),
            "--beam-size",
            str(args.seq2seq_beam_size),
            "--num-return-sequences",
            str(args.seq2seq_num_return_sequences),
            "--generation-length-penalty",
            str(args.seq2seq_generation_length_penalty),
            "--gradient-clip-norm",
            str(args.gradient_clip_norm),
            "--seed",
            str(args.seed),
            "--device",
            args.device,
            "--checkpoint",
            paths["ar_checkpoint"],
            "--predictions",
            paths["ar_predictions"],
            "--metrics",
            paths["ar_metrics"],
            "--log-every",
            str(args.log_every),
        ]
    )
    if args.no_numeric_evidence_input:
        command.append("--no-numeric-evidence-input")
    if args.seq2seq_prompt_augmentation:
        command.append("--prompt-augmentation")
    if args.seq2seq_warmup_steps is not None:
        command.extend(["--warmup-steps", str(args.seq2seq_warmup_steps)])
    if args.seq2seq_no_restore_best:
        command.append("--no-restore-best")
    if args.skeleton_random_init:
        command.append("--random-init")
    if args.skeleton_local_files_only:
        command.append("--local-files-only")
    if args.seq2seq_freeze_encoder:
        command.append("--freeze-encoder")
    if args.seq2seq_fp16:
        command.append("--fp16")
    elif args.seq2seq_bf16:
        command.append("--bf16")
    return command


def build_graph_command(args: argparse.Namespace, paths: dict[str, str]) -> list[str]:
    command = [
        sys.executable,
        "scripts/train_reactgdiff.py",
        "--train",
        args.train,
        "--val",
        args.val,
        "--decoder-backend",
        "graph_diffusion",
    ]
    _add_option(command, "--train-limit", args.train_limit)
    _add_option(command, "--val-limit", args.val_limit)
    command.extend(
        [
            "--epochs",
            str(args.graph_epochs),
            "--batch-size",
            str(args.graph_batch_size),
            "--learning-rate",
            str(args.graph_learning_rate),
            "--diffusion-steps",
            str(args.diffusion_steps),
            "--graph-diffusion-sample-steps",
            str(args.graph_diffusion_sample_steps),
            "--graph-diffusion-noise-schedule",
            args.graph_diffusion_noise_schedule,
            "--graph-diffusion-sampler",
            args.graph_diffusion_sampler,
            "--graph-diffusion-sample-mode",
            args.graph_diffusion_sample_mode,
            "--graph-diffusion-sample-temperature",
            str(args.graph_diffusion_sample_temperature),
            "--graph-diffusion-quantity-mode",
            "grounded",
            "--graph-condition-encoder",
            args.graph_condition_encoder,
            "--graph-diffusion-skeleton-conditioning",
            "--graph-diffusion-skeleton-teacher-forcing",
            str(args.graph_diffusion_skeleton_teacher_forcing),
            "--graph-diffusion-skeleton-teacher-forcing-final",
            str(args.graph_diffusion_skeleton_teacher_forcing_final),
            "--graph-diffusion-skeleton-corruption-probability",
            str(args.graph_diffusion_skeleton_corruption_probability),
            "--graph-diffusion-skeleton-loss-weight",
            str(args.graph_diffusion_skeleton_loss_weight),
            "--graph-diffusion-slot-operation-loss-weight",
            str(args.graph_diffusion_slot_operation_loss_weight),
            "--eval-skeleton-source",
            "cache",
            "--skeleton-cache",
            paths["ar_predictions"],
            "--condition-encoding",
            args.condition_encoding,
            "--field-dim",
            str(args.field_dim),
            "--ngram-min",
            str(args.ngram_min),
            "--ngram-max",
            str(args.ngram_max),
            "--hidden-dim",
            str(args.graph_hidden_dim),
            "--dit-depth",
            str(args.dit_depth),
            "--dit-heads",
            str(args.dit_heads),
            "--max-steps",
            str(args.max_steps),
            "--max-material-refs",
            str(args.max_material_refs),
            "--max-material-slots",
            str(args.max_material_slots),
            "--max-quantity-vocab",
            str(args.max_quantity_vocab),
            "--max-numeric-candidates",
            str(args.max_numeric_candidates),
            "--sampling-strategy",
            args.sampling_strategy,
            "--sample-weight-max",
            str(args.sample_weight_max),
            "--skeleton-weight-max",
            str(args.skeleton_weight_max),
            "--operation-weighting",
            args.operation_weighting,
            "--operation-weight-alpha",
            str(args.operation_weight_alpha),
            "--operation-weight-max",
            str(args.operation_weight_max),
            "--material-loss-weight",
            str(args.material_loss_weight),
            "--condition-loss-weight",
            str(args.condition_loss_weight),
            "--quantity-gate-loss-weight",
            str(args.quantity_gate_loss_weight),
            "--numeric-candidate-loss-weight",
            str(args.numeric_candidate_loss_weight),
            "--numeric-candidate-reuse-penalty",
            str(args.numeric_candidate_reuse_penalty),
            "--numeric-candidate-unit-weight",
            str(args.numeric_candidate_unit_weight),
            "--condition-value-loss-weight",
            str(args.condition_value_loss_weight),
            "--quantity-value-loss-weight",
            "0.0",
            "--structure-loss-weight",
            str(args.structure_loss_weight),
            "--quantity-gate-threshold",
            str(args.quantity_gate_threshold),
            "--condition-probability-threshold",
            str(args.condition_probability_threshold),
            "--best-eval-interval",
            str(args.best_eval_interval),
            "--best-eval-limit",
            str(args.best_eval_limit),
            "--best-eval-metric",
            args.best_eval_metric,
            "--sample-batch-size",
            str(args.sample_batch_size),
            "--gradient-clip-norm",
            str(args.gradient_clip_norm),
            "--seed",
            str(args.seed),
            "--device",
            args.device,
            "--checkpoint",
            paths["graph_checkpoint"],
            "--predictions",
            paths["graph_predictions"],
            "--metrics",
            paths["graph_metrics"],
            "--log-every",
            str(args.log_every),
        ]
    )
    if args.graph_condition_encoder == "shared_molt5":
        command.extend(
            [
                "--shared-encoder-checkpoint",
                paths["ar_checkpoint"],
                "--shared-encoder-mode",
                args.shared_encoder_mode,
                "--shared-encoder-learning-rate",
                str(args.shared_encoder_learning_rate),
                "--shared-encoder-max-length",
                str(args.shared_encoder_max_length),
            ]
        )
        if args.skeleton_local_files_only:
            command.append("--shared-encoder-local-files-only")
    if args.no_numeric_evidence_input:
        command.append("--no-numeric-evidence-input")
    if args.numeric_candidate_include_source:
        command.append("--numeric-candidate-include-source")
    if not args.numeric_candidate_quantity_only:
        command.append("--no-numeric-candidate-quantity-only")
    if not args.numeric_candidate_feature_pointer:
        command.append("--no-numeric-candidate-feature-pointer")
    if not args.drop_unsupported_numeric_slots:
        command.append("--no-drop-unsupported-numeric-slots")
    if args.no_restore_best:
        command.append("--no-restore-best")
    if args.save_generated_graph:
        command.append("--save-generated-graph")
    return command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        default="data/processed/openexp",
        help="Dataset root containing splits/train.jsonl and splits/val.jsonl.",
    )
    parser.add_argument("--train", default=None)
    parser.add_argument("--val", default=None)
    parser.add_argument("--scale", choices=("small", "medium", "large"), default="small")
    parser.add_argument("--split-cache-dir", default="outputs/prepared_splits")
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--val-limit", type=int, default=None)
    parser.add_argument(
        "--run-name",
        default="openexp_small_shared_molt5_pointer_source_finetune",
    )
    parser.add_argument("--eval-name", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-ar", action="store_true")
    parser.add_argument("--skip-graph", action="store_true")
    parser.add_argument(
        "--skeleton-backend",
        choices=("gru", "seq2seq"),
        default="seq2seq",
        help="gru keeps the existing lightweight AR model; seq2seq uses BART/T5-style skeleton generation.",
    )

    parser.add_argument("--condition-encoding", choices=("reactxt_hash", "field_hash", "scalar_hash"), default="reactxt_hash")
    parser.add_argument("--field-dim", type=int, default=256)
    parser.add_argument("--ngram-min", type=int, default=2)
    parser.add_argument("--ngram-max", type=int, default=5)
    parser.add_argument("--no-numeric-evidence-input", action="store_true")
    parser.add_argument(
        "--numeric-candidate-include-source",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use source-augmented numeric candidates by default for the current "
            "upper-bound experiment."
        ),
    )
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=32)
    parser.add_argument("--max-material-refs", type=int, default=16)
    parser.add_argument("--max-material-slots", type=int, default=4)
    parser.add_argument("--max-quantity-vocab", type=int, default=256)
    parser.add_argument("--max-numeric-candidates", type=int, default=64)
    parser.add_argument(
        "--numeric-candidate-quantity-only",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--numeric-candidate-feature-pointer",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--numeric-candidate-reuse-penalty", type=float, default=16.0)
    parser.add_argument("--numeric-candidate-unit-weight", type=float, default=1.0)
    parser.add_argument(
        "--drop-unsupported-numeric-slots",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument("--ar-epochs", type=int, default=500)
    parser.add_argument("--ar-batch-size", type=int, default=64)
    parser.add_argument("--ar-learning-rate", type=float, default=3e-4)
    parser.add_argument("--ar-hidden-dim", type=int, default=128)
    parser.add_argument("--ar-layers", type=int, default=4)
    parser.add_argument("--ar-dropout", type=float, default=0.1)
    parser.add_argument("--ar-checkpoint", default=None)
    parser.add_argument("--ar-predictions", default=None)
    parser.add_argument("--ar-metrics", default=None)

    parser.add_argument("--skeleton-model-name", default="laituan245/molt5-base")
    parser.add_argument("--skeleton-random-init", action="store_true")
    parser.add_argument("--skeleton-local-files-only", action="store_true")
    parser.add_argument("--seq2seq-epochs", type=int, default=200)
    parser.add_argument("--seq2seq-batch-size", type=int, default=8)
    parser.add_argument("--seq2seq-eval-batch-size", type=int, default=8)
    parser.add_argument("--seq2seq-learning-rate", type=float, default=3e-5)
    parser.add_argument("--seq2seq-weight-decay", type=float, default=0.01)
    parser.add_argument("--seq2seq-warmup-ratio", type=float, default=0.05)
    parser.add_argument("--seq2seq-warmup-steps", type=int, default=None)
    parser.add_argument("--seq2seq-gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--seq2seq-max-input-length", type=int, default=384)
    parser.add_argument("--seq2seq-max-target-length", type=int, default=96)
    parser.add_argument("--seq2seq-prompt-style", choices=("compact", "reactxt"), default="compact")
    parser.add_argument(
        "--seq2seq-target-format",
        choices=("natural_text", "special_tokens"),
        default="natural_text",
    )
    parser.add_argument("--seq2seq-prompt-augmentation", action="store_true")
    parser.add_argument("--seq2seq-participant-shuffle-prob", type=float, default=0.0)
    parser.add_argument("--seq2seq-numeric-format-augment-prob", type=float, default=0.0)
    parser.add_argument("--seq2seq-field-mask-prob", type=float, default=0.0)
    parser.add_argument("--seq2seq-action-loss-weighting", choices=("none", "class_balanced"), default="none")
    parser.add_argument("--seq2seq-action-weight-beta", type=float, default=0.9999)
    parser.add_argument("--seq2seq-action-weight-min", type=float, default=0.5)
    parser.add_argument("--seq2seq-action-weight-max", type=float, default=4.0)
    parser.add_argument("--seq2seq-length-loss-weight", type=float, default=0.2)
    parser.add_argument("--seq2seq-length-prior-weight", type=float, default=0.0)
    parser.add_argument("--seq2seq-repetition-penalty-weight", type=float, default=0.1)
    parser.add_argument("--seq2seq-best-eval-interval", type=int, default=10)
    parser.add_argument("--seq2seq-no-restore-best", action="store_true")
    parser.add_argument("--seq2seq-beam-size", type=int, default=4)
    parser.add_argument("--seq2seq-num-return-sequences", type=int, default=4)
    parser.add_argument("--seq2seq-generation-length-penalty", type=float, default=1.0)
    parser.add_argument("--seq2seq-freeze-encoder", action="store_true")
    parser.add_argument(
        "--seq2seq-fp16",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--seq2seq-bf16",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument("--graph-epochs", type=int, default=800)
    parser.add_argument("--graph-batch-size", type=int, default=64)
    parser.add_argument("--graph-learning-rate", type=float, default=2e-4)
    parser.add_argument("--graph-hidden-dim", type=int, default=256)
    parser.add_argument(
        "--graph-condition-encoder",
        choices=("hash", "shared_molt5"),
        default="hash",
    )
    parser.add_argument(
        "--shared-encoder-mode",
        choices=("frozen", "finetune"),
        default="finetune",
    )
    parser.add_argument("--shared-encoder-learning-rate", type=float, default=3e-6)
    parser.add_argument("--shared-encoder-max-length", type=int, default=512)
    parser.add_argument("--dit-depth", type=int, default=4)
    parser.add_argument("--dit-heads", type=int, default=8)
    parser.add_argument("--diffusion-steps", type=int, default=32)
    parser.add_argument("--graph-diffusion-sample-steps", type=int, default=32)
    parser.add_argument("--graph-diffusion-noise-schedule", choices=("cosine", "linear"), default="cosine")
    parser.add_argument("--graph-diffusion-sampler", choices=("posterior", "single_step", "iterative"), default="posterior")
    parser.add_argument(
        "--graph-diffusion-sample-mode",
        choices=("argmax", "sample", "sample_argmax_final"),
        default="sample_argmax_final",
    )
    parser.add_argument("--graph-diffusion-sample-temperature", type=float, default=1.0)
    parser.add_argument("--graph-diffusion-skeleton-teacher-forcing", type=float, default=1.0)
    parser.add_argument("--graph-diffusion-skeleton-teacher-forcing-final", type=float, default=1.0)
    parser.add_argument("--graph-diffusion-skeleton-corruption-probability", type=float, default=0.0)
    parser.add_argument("--graph-diffusion-skeleton-loss-weight", type=float, default=0.0)
    parser.add_argument("--graph-diffusion-slot-operation-loss-weight", type=float, default=0.0)
    parser.add_argument("--sampling-strategy", choices=("balanced", "random"), default="balanced")
    parser.add_argument("--sample-weight-max", type=float, default=8.0)
    parser.add_argument("--skeleton-weight-max", type=float, default=3.0)
    parser.add_argument("--operation-weighting", choices=("balanced", "none"), default="balanced")
    parser.add_argument("--operation-weight-alpha", type=float, default=0.5)
    parser.add_argument("--operation-weight-max", type=float, default=4.0)
    parser.add_argument("--material-loss-weight", type=float, default=0.8)
    parser.add_argument("--condition-loss-weight", type=float, default=0.9)
    parser.add_argument("--quantity-gate-loss-weight", type=float, default=0.7)
    parser.add_argument("--numeric-candidate-loss-weight", type=float, default=0.9)
    parser.add_argument("--condition-value-loss-weight", type=float, default=0.0)
    parser.add_argument("--structure-loss-weight", type=float, default=0.2)
    parser.add_argument("--quantity-gate-threshold", type=float, default=0.999)
    parser.add_argument("--condition-probability-threshold", type=float, default=0.05)
    parser.add_argument("--best-eval-interval", type=int, default=10)
    parser.add_argument("--best-eval-limit", type=int, default=2048)
    parser.add_argument(
        "--best-eval-metric",
        choices=("semantic_score", "discrete_slot_score", "canonical_levenshtein_75_rate"),
        default="semantic_score",
    )
    parser.add_argument("--sample-batch-size", type=int, default=64)
    parser.add_argument("--no-restore-best", action="store_true")
    parser.add_argument(
        "--save-generated-graph",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--graph-checkpoint", default=None)
    parser.add_argument("--graph-predictions", default=None)
    parser.add_argument("--graph-metrics", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _resolve_input_splits(args)
    paths = _default_paths(args)
    _mkdirs(paths.values(), dry_run=args.dry_run)
    if not args.skip_ar:
        _run(build_ar_command(args, paths), dry_run=args.dry_run)
    if not args.skip_graph:
        if args.skip_ar and not Path(paths["ar_predictions"]).exists() and not args.dry_run:
            raise FileNotFoundError(
                "跳过 AR 训练时必须已有 skeleton cache: "
                f"{paths['ar_predictions']}"
            )
        _run(build_graph_command(args, paths), dry_run=args.dry_run)
    print("\n完成。输出路径：", flush=True)
    for key, value in paths.items():
        print(f"  {key}: {value}", flush=True)


if __name__ == "__main__":
    main()
