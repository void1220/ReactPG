"""Train the ReactGDiff small graph-generation experiments."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reactgdiff.eval.slots import discrete_slot_metrics
from reactgdiff.eval.semantic import corpus_semantic_metrics
from reactgdiff.eval.text import corpus_text_metrics
from reactgdiff.models.argument_filler import ArgumentTextCodec, train_argument_text_filler
from reactgdiff.models.graph_codec import GraphTargetCodec
from reactgdiff.models.graph_encoder_decoder import (
    predict_direct_graph_records,
    save_graph_checkpoint,
    train_direct_graph_encoder_decoder,
)
from reactgdiff.models.joint_diffusion import (
    ReactGDiffFeaturizer,
    build_candidate_memory,
    load_split_records,
    predict_records,
    save_checkpoint,
    train_joint_diffusion,
)
from reactgdiff.models.procedure_graph_diffusion import (
    predict_procedure_graph_diffusion_records,
    save_procedure_graph_diffusion_checkpoint,
    train_procedure_graph_diffusion,
)
from reactgdiff.utils.io import read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train",
        default="data/processed/openexp_sample/splits/train.jsonl",
        help="Training JSONL split.",
    )
    parser.add_argument(
        "--val",
        default="data/processed/openexp_sample/splits/val.jsonl",
        help="Validation JSONL split used for validation metrics.",
    )
    parser.add_argument("--train-limit", type=int, default=None, help="Optional train limit.")
    parser.add_argument("--val-limit", type=int, default=None, help="Optional validation limit.")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--diffusion-steps", type=int, default=16)
    parser.add_argument(
        "--graph-diffusion-noise-schedule",
        choices=("cosine", "linear"),
        default="cosine",
        help="Categorical corruption schedule for decoder_backend=graph_diffusion.",
    )
    parser.add_argument(
        "--graph-diffusion-sample-steps",
        type=int,
        default=None,
        help="Reverse denoising steps used for validation sampling. Defaults to --diffusion-steps.",
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
        "--graph-diffusion-pad-loss-weight",
        type=float,
        default=0.25,
        help=(
            "Operation loss weight for PAD slots in decoder_backend=graph_diffusion. "
            "Use 0 to recover the old PAD-ignored behavior."
        ),
    )
    parser.add_argument(
        "--graph-diffusion-quantity-mode",
        choices=("argument", "diffusion", "grounded"),
        default="argument",
        help=(
            "argument leaves numeric quantity slots to the argument text filler; "
            "diffusion predicts quantity gates/units/values inside graph diffusion; "
            "grounded predicts discrete numeric slots and binds values to input evidence."
        ),
    )
    parser.add_argument(
        "--graph-diffusion-timestep-sampling",
        choices=("uniform", "endpoint", "endpoint_mix"),
        default="uniform",
        help=(
            "Training timestep distribution for graph_diffusion. uniform samples "
            "t in [0, T] as in DiGress; endpoint_mix oversamples t=T "
            "for single-step experiments."
        ),
    )
    parser.add_argument(
        "--graph-diffusion-endpoint-probability",
        type=float,
        default=0.0,
        help="Probability of replacing a random training timestep with t=T under endpoint_mix.",
    )
    parser.add_argument(
        "--graph-diffusion-skeleton-conditioning",
        action="store_true",
        help=(
            "Predict the operation skeleton from the condition first, then condition "
            "slot diffusion on that skeleton."
        ),
    )
    parser.add_argument(
        "--graph-diffusion-skeleton-loss-weight",
        type=float,
        default=0.0,
        help="Auxiliary loss weight for condition-to-operation-skeleton prediction.",
    )
    parser.add_argument(
        "--graph-diffusion-skeleton-teacher-forcing",
        type=float,
        default=1.0,
        help=(
            "Probability of using the gold skeleton as slot-diffusion context during "
            "training when skeleton conditioning is enabled; the rest uses predicted skeletons."
        ),
    )
    parser.add_argument(
        "--graph-diffusion-skeleton-teacher-forcing-final",
        type=float,
        default=1.0,
        help="Final-epoch gold-skeleton probability; linearly annealed from the initial value.",
    )
    parser.add_argument(
        "--graph-diffusion-skeleton-corruption-probability",
        type=float,
        default=0.0,
        help="Per-token corruption applied to non-teacher skeleton context during training.",
    )
    parser.add_argument(
        "--graph-diffusion-slot-operation-loss-weight",
        type=float,
        default=1.0,
        help=(
            "Operation CE weight inside the slot diffusion loss. Use a small value "
            "or 0 when skeleton conditioning owns operation prediction."
        ),
    )
    parser.add_argument(
        "--eval-skeleton-source",
        choices=("predicted", "gold", "cache", "none"),
        default="predicted",
        help=(
            "Skeleton source for graph-diffusion validation. gold fixes the target "
            "operation skeleton; cache uses --skeleton-cache rows; predicted uses "
            "the model's skeleton head when available."
        ),
    )
    parser.add_argument(
        "--skeleton-cache",
        default=None,
        help="Optional JSONL cache with predicted skeletons keyed by index.",
    )
    parser.add_argument(
        "--sample-batch-size",
        type=int,
        default=64,
        help="Validation sampling batch size for decoder_backend=graph_diffusion.",
    )
    parser.add_argument(
        "--best-eval-interval",
        type=int,
        default=0,
        help=(
            "For decoder_backend=graph_diffusion, run lightweight validation every N epochs "
            "and restore the best model by mean text gap. Use 0 to disable."
        ),
    )
    parser.add_argument(
        "--best-eval-limit",
        type=int,
        default=512,
        help="Maximum validation records used for periodic best-model selection; 0 uses all validation records.",
    )
    parser.add_argument(
        "--best-eval-metric",
        choices=("semantic_score", "discrete_slot_score", "canonical_levenshtein_75_rate"),
        default="semantic_score",
        help="Validation metric used to restore the best graph-diffusion checkpoint.",
    )
    parser.add_argument(
        "--no-restore-best",
        action="store_true",
        help="Keep the final epoch weights instead of restoring the best periodic validation weights.",
    )
    parser.add_argument(
        "--condition-encoding",
        choices=("reactxt_hash", "field_hash", "scalar_hash"),
        default="reactxt_hash",
        help=(
            "reactxt_hash hashes a ReactXT-style prompt with placeholders, SMILES, "
            "and temperature/duration lookup tables; field_hash keeps only the four "
            "molecule fields."
        ),
    )
    parser.add_argument(
        "--graph-condition-encoder",
        choices=("hash", "shared_molt5"),
        default="hash",
        help=(
            "hash uses the legacy global n-gram vector; shared_molt5 loads the "
            "seq2seq skeleton encoder and adds token cross-attention/pointer heads."
        ),
    )
    parser.add_argument(
        "--shared-encoder-checkpoint",
        default=None,
        help="Seq2seq skeleton checkpoint whose MolT5 encoder is shared with graph diffusion.",
    )
    parser.add_argument(
        "--shared-encoder-mode",
        choices=("frozen", "finetune"),
        default="frozen",
        help="Freeze the learned skeleton encoder or continue updating it from graph losses.",
    )
    parser.add_argument("--shared-encoder-learning-rate", type=float, default=1e-5)
    parser.add_argument("--shared-encoder-max-length", type=int, default=512)
    parser.add_argument(
        "--shared-encoder-local-files-only",
        action="store_true",
        help="Resolve the base tokenizer/model referenced by the skeleton checkpoint locally.",
    )
    parser.add_argument(
        "--field-dim",
        type=int,
        default=128,
        help=(
            "Per-field hash dimension. condition_dim = 6 * field_dim for reactxt_hash "
            "and 4 * field_dim for field_hash."
        ),
    )
    parser.add_argument("--ngram-min", type=int, default=2)
    parser.add_argument("--ngram-max", type=int, default=5)
    parser.add_argument(
        "--no-numeric-evidence-input",
        action="store_true",
        help=(
            "Disable the extra NumericEvidence input field for reactxt_hash. "
            "Use only for old-baseline ablations."
        ),
    )
    parser.add_argument(
        "--numeric-candidate-include-source",
        action="store_true",
        help=(
            "Allow procedure source text to populate numeric candidates. Off by default "
            "for a fair reaction-to-procedure comparison with ReactXT."
        ),
    )
    parser.add_argument(
        "--numeric-candidate-quantity-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Keep amount/concentration/yield candidates for quantity slots and "
            "exclude duration, temperature, and NMR-like 1H spans."
        ),
    )
    parser.add_argument(
        "--numeric-candidate-feature-pointer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use compact per-candidate value/unit/type/source embeddings for hash-mode pointer scoring.",
    )
    parser.add_argument(
        "--numeric-candidate-reuse-penalty",
        type=float,
        default=16.0,
        help="Greedy decoding penalty applied each time a record-local numeric candidate is reused.",
    )
    parser.add_argument(
        "--numeric-candidate-unit-weight",
        type=float,
        default=1.0,
        help="Weight of candidate/unit log-probability agreement during final decoding.",
    )
    parser.add_argument(
        "--drop-unsupported-numeric-slots",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Close decoded quantity gates whose selected candidate is NONE or MISSING.",
    )
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument(
        "--graph-backbone",
        choices=("dit", "mlp"),
        default="dit",
        help="direct_graph backbone: dit uses transformer blocks; mlp keeps the earlier simple decoder.",
    )
    parser.add_argument(
        "--diffusion-base",
        choices=("dit", "mlp"),
        default="dit",
        help="memory backend denoising base model; dit replaces the earlier pair of small MLP denoisers.",
    )
    parser.add_argument("--dit-depth", type=int, default=4, help="Number of DiT blocks.")
    parser.add_argument("--dit-heads", type=int, default=8, help="Attention heads per DiT block.")
    parser.add_argument("--time-dim", type=int, default=32)
    parser.add_argument("--context-dim", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=32)
    parser.add_argument("--max-material-refs", type=int, default=16)
    parser.add_argument(
        "--max-material-slots",
        type=int,
        default=4,
        help="Maximum material/quantity slots per operation step.",
    )
    parser.add_argument(
        "--max-quantity-vocab",
        type=int,
        default=256,
        help="Maximum unit categories kept for numeric slots; legacy option name.",
    )
    parser.add_argument(
        "--max-numeric-candidates",
        type=int,
        default=64,
        help="Maximum record-local numeric evidence candidates classified by graph diffusion.",
    )
    parser.add_argument("--prior-alignment-weight", type=float, default=0.1)
    parser.add_argument("--teacher-decoder-weight", type=float, default=0.3)
    parser.add_argument(
        "--latent-norm-weight",
        type=float,
        default=1e-3,
        help="L2 penalty on prior/graph latent magnitudes for direct_graph training.",
    )
    parser.add_argument("--material-loss-weight", type=float, default=0.8)
    parser.add_argument("--numeric-candidate-loss-weight", type=float, default=0.9)
    parser.add_argument("--condition-loss-weight", type=float, default=0.9)
    parser.add_argument(
        "--condition-value-loss-weight",
        type=float,
        default=0.0,
        help=(
            "MSE weight for generated duration/temperature numeric values. Keep 0.0 "
            "for ReactXT-aligned main experiments where condition values are provided "
            "as input lookup tables."
        ),
    )
    parser.add_argument(
        "--structure-loss-weight",
        type=float,
        default=0.2,
        help="Auxiliary weight for global graph-shape targets such as length, conditions, and workup count.",
    )
    parser.add_argument(
        "--numeric-value-clip",
        type=float,
        default=8.0,
        help="Clamp normalized continuous quantity/condition targets before they enter the graph encoder.",
    )
    parser.add_argument(
        "--sampling-strategy",
        choices=("balanced", "random"),
        default="balanced",
        help="balanced oversamples complex bucket labels and rare action skeletons.",
    )
    parser.add_argument(
        "--sample-weight-max",
        type=float,
        default=8.0,
        help="Maximum per-record weight used by the balanced sampler.",
    )
    parser.add_argument(
        "--skeleton-weight-max",
        type=float,
        default=3.0,
        help="Maximum rare-skeleton multiplier inside the balanced sampler.",
    )
    parser.add_argument(
        "--operation-weighting",
        choices=("balanced", "none"),
        default="balanced",
        help="balanced applies inverse-frequency class weights to operation prediction.",
    )
    parser.add_argument(
        "--operation-weight-alpha",
        type=float,
        default=0.5,
        help="Inverse-frequency exponent for operation class weights.",
    )
    parser.add_argument(
        "--operation-weight-max",
        type=float,
        default=4.0,
        help="Maximum operation class weight after balancing.",
    )
    parser.add_argument(
        "--gradient-clip-norm",
        type=float,
        default=1.0,
        help="Clip global gradient norm after backward. Use 0 to disable.",
    )
    parser.add_argument("--quantity-gate-loss-weight", type=float, default=0.7)
    parser.add_argument(
        "--quantity-value-loss-weight",
        type=float,
        default=0.25,
        help="MSE weight for generated normalized quantity values in graph_diffusion.",
    )
    parser.add_argument("--material-none-weight", type=float, default=0.35)
    parser.add_argument("--material-present-weight", type=float, default=1.6)
    parser.add_argument("--condition-none-weight", type=float, default=0.45)
    parser.add_argument("--condition-present-weight", type=float, default=2.5)
    parser.add_argument("--quantity-negative-weight", type=float, default=0.9)
    parser.add_argument("--quantity-positive-weight", type=float, default=1.0)
    parser.add_argument(
        "--quantity-gate-threshold",
        type=float,
        default=None,
        help=(
            "Decode a numeric slot only when its open probability reaches this threshold. "
            "Defaults to 0.999 for graph_diffusion and 0.75 for other backends."
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
        help="During validation decode, use the structure head to set procedure length and final YIELD position.",
    )
    parser.add_argument(
        "--min-structure-steps",
        type=int,
        default=2,
        help="Minimum decoded step count when --use-structure-length is enabled.",
    )
    parser.add_argument(
        "--argument-filler-backend",
        choices=("none", "transformer"),
        default="none",
        help="Optional autoregressive text filler for per-step argument strings.",
    )
    parser.add_argument(
        "--argument-filler-target",
        choices=("all", "numeric"),
        default="all",
        help=(
            "Argument filler target surface. all rewrites every decoded step; "
            "numeric trains/applies only on numeric-slot steps and lets the filler "
            "autoregressively generate concrete amount/unit text."
        ),
    )
    parser.add_argument("--argument-filler-epochs", type=int, default=150)
    parser.add_argument("--argument-filler-batch-size", type=int, default=256)
    parser.add_argument("--argument-filler-learning-rate", type=float, default=3e-4)
    parser.add_argument("--argument-filler-hidden-dim", type=int, default=256)
    parser.add_argument("--argument-filler-layers", type=int, default=4)
    parser.add_argument("--argument-filler-heads", type=int, default=8)
    parser.add_argument("--argument-filler-dropout", type=float, default=0.1)
    parser.add_argument("--argument-max-length", type=int, default=160)
    parser.add_argument("--argument-vocab-size", type=int, default=192)
    parser.add_argument(
        "--argument-max-examples",
        type=int,
        default=None,
        help="Optional cap on step-level examples for quick argument-filler experiments.",
    )
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", default="cpu", help="cpu, cuda, or cuda:N.")
    parser.add_argument(
        "--log-every",
        type=int,
        default=1,
        help="Print training loss every N epochs. Use 0 to disable epoch logs.",
    )
    parser.add_argument(
        "--decoder-backend",
        choices=("direct_graph", "graph_diffusion", "memory"),
        default="direct_graph",
        help=(
            "direct_graph is the one-shot slot decoder; graph_diffusion is the "
            "DiGress-style categorical procedure-graph denoiser; memory keeps "
            "the earlier nearest-memory baseline."
        ),
    )
    parser.add_argument(
        "--save-generated-graph",
        action="store_true",
        help="Store full generated graph JSON in predictions. Off by default to keep files small.",
    )
    parser.add_argument(
        "--checkpoint",
        default="outputs/checkpoints/reactgdiff_graph.pt",
        help="Checkpoint path.",
    )
    parser.add_argument(
        "--predictions",
        default="outputs/predictions/reactgdiff_graph_val.jsonl",
        help="Validation prediction JSONL path.",
    )
    parser.add_argument(
        "--metrics",
        default="outputs/metrics/reactgdiff_graph_val.json",
        help="Validation metrics JSON path.",
    )
    args = parser.parse_args()
    if args.quantity_gate_threshold is None:
        args.quantity_gate_threshold = (
            0.999 if args.decoder_backend == "graph_diffusion" else 0.75
        )
    argument_condition_on_quantity_units = args.argument_filler_target == "all"
    include_numeric_evidence = not args.no_numeric_evidence_input
    skeleton_cache = load_skeleton_cache(args.skeleton_cache) if args.skeleton_cache else None
    if args.decoder_backend == "graph_diffusion" and args.argument_filler_backend != "none":
        print(
            "已禁用 graph_diffusion 主流程中的最后 AR argument filler；"
            "当前训练只包含骨架预测和图扩散槽位补全。",
            flush=True,
        )
        args.argument_filler_backend = "none"

    print(
        "准备阶段：开始读取数据 "
        f"train={args.train} val={args.val} "
        f"train_limit={args.train_limit} val_limit={args.val_limit}",
        flush=True,
    )
    train_records = load_split_records(args.train, limit=args.train_limit)
    val_records = load_split_records(args.val, limit=args.val_limit)
    print(
        "准备阶段：数据读取完成 "
        f"训练集={len(train_records)} 验证集={len(val_records)} "
        f"后端={args.decoder_backend} 设备={args.device} seed={args.seed}",
        flush=True,
    )
    print(
        "准备阶段：开始构建条件 featurizer "
        f"encoding={args.condition_encoding} field_dim={args.field_dim} "
        f"numeric_evidence={'启用' if include_numeric_evidence else '关闭'} "
        f"source_numeric={'启用' if args.numeric_candidate_include_source else '关闭'}",
        flush=True,
    )
    featurizer = ReactGDiffFeaturizer.fit(
        train_records,
        condition_encoding=args.condition_encoding,
        field_dim=args.field_dim,
        ngram_min=args.ngram_min,
        ngram_max=args.ngram_max,
        include_numeric_evidence=include_numeric_evidence,
        numeric_evidence_include_source=args.numeric_candidate_include_source,
        numeric_evidence_quantity_only=args.numeric_candidate_quantity_only,
    )
    print(
        "准备阶段：条件 featurizer 完成 "
        f"condition_dim={featurizer.condition_dim}",
        flush=True,
    )
    input_description = (
        "ReactXT 输入字段：REACTANT / PRODUCT / CATALYST / SOLVENT / 温度 / 时长"
        if args.condition_encoding == "reactxt_hash"
        else "输入字段：REACTANT / PRODUCT / CATALYST / SOLVENT"
    )
    print(
        f"{input_description} "
        f"(encoding={args.condition_encoding}, condition_dim={featurizer.condition_dim}, "
            f"field_dim={featurizer.field_dim}, "
            f"numeric_evidence={'启用' if include_numeric_evidence else '关闭'}, "
            f"source_numeric={'启用' if args.numeric_candidate_include_source else '关闭'})",
        flush=True,
    )
    if args.decoder_backend == "direct_graph":
        print("准备阶段：开始构建 graph codec", flush=True)
        codec = GraphTargetCodec.fit(
            train_records,
            max_steps=args.max_steps,
            max_material_refs=args.max_material_refs,
            max_material_slots=args.max_material_slots,
            max_quantity_vocab=args.max_quantity_vocab,
            max_numeric_candidates=args.max_numeric_candidates,
            numeric_candidate_include_source=args.numeric_candidate_include_source,
            numeric_candidate_quantity_only=args.numeric_candidate_quantity_only,
        )
        print(
            "Graph decoder dims: "
            f"actions={codec.action_dim} materials={codec.material_dim} "
            f"conditions={codec.condition_dim} units={codec.unit_dim} "
            f"slots_per_step={codec.max_material_slots} "
            f"max_steps={codec.max_steps} hidden={args.hidden_dim} latent={args.latent_dim} "
            f"backbone={args.graph_backbone} dit_depth={args.dit_depth} dit_heads={args.dit_heads}",
            flush=True,
        )
        print(
            "Slot calibration: "
            f"material_none/present={args.material_none_weight}/{args.material_present_weight} "
            f"condition_none/present={args.condition_none_weight}/{args.condition_present_weight} "
            f"condition_value_weight={args.condition_value_loss_weight} "
            f"quantity_neg/pos={args.quantity_negative_weight}/{args.quantity_positive_weight} "
            f"quantity_threshold={args.quantity_gate_threshold} "
            f"condition_threshold={args.condition_probability_threshold}",
            flush=True,
        )
        print(
            "Anti-collapse training: "
            f"sampling={args.sampling_strategy} sample_cap={args.sample_weight_max} "
            f"skeleton_cap={args.skeleton_weight_max} "
            f"operation_weighting={args.operation_weighting} "
            f"operation_alpha={args.operation_weight_alpha} "
            f"operation_cap={args.operation_weight_max} "
            f"structure_weight={args.structure_loss_weight} "
            f"numeric_clip={args.numeric_value_clip} "
            f"grad_clip={args.gradient_clip_norm}",
            flush=True,
        )
        print("准备阶段：开始张量化 direct_graph 条件向量", flush=True)
        train_conditions = [featurizer.condition_vector(record) for record in train_records]
        print("准备阶段完成，开始 direct_graph 训练。", flush=True)
        model, history = train_direct_graph_encoder_decoder(
            train_records,
            condition_vectors=train_conditions,
            codec=codec,
            condition_dim=featurizer.condition_dim,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            hidden_dim=args.hidden_dim,
            latent_dim=args.latent_dim,
            graph_backbone=args.graph_backbone,
            dit_depth=args.dit_depth,
            dit_heads=args.dit_heads,
            prior_alignment_weight=args.prior_alignment_weight,
            teacher_decoder_weight=args.teacher_decoder_weight,
            latent_norm_weight=args.latent_norm_weight,
            material_loss_weight=args.material_loss_weight,
            condition_loss_weight=args.condition_loss_weight,
            quantity_gate_loss_weight=args.quantity_gate_loss_weight,
            material_none_weight=args.material_none_weight,
            material_present_weight=args.material_present_weight,
            condition_none_weight=args.condition_none_weight,
            condition_present_weight=args.condition_present_weight,
            quantity_negative_weight=args.quantity_negative_weight,
            quantity_positive_weight=args.quantity_positive_weight,
            condition_value_loss_weight=args.condition_value_loss_weight,
            numeric_value_clip=args.numeric_value_clip,
            structure_loss_weight=args.structure_loss_weight,
            sampling_strategy=args.sampling_strategy,
            sample_weight_max=args.sample_weight_max,
            skeleton_weight_max=args.skeleton_weight_max,
            operation_weighting=args.operation_weighting,
            operation_weight_alpha=args.operation_weight_alpha,
            operation_weight_max=args.operation_weight_max,
            gradient_clip_norm=args.gradient_clip_norm,
            seed=args.seed,
            device=args.device,
            log_every=args.log_every,
        )
        argument_filler = None
        argument_text_codec = None
        argument_history = []
        if args.argument_filler_backend == "transformer":
            print(
                "开始训练 argument text filler："
                f"target={args.argument_filler_target} "
                f"condition_on_quantity_units={argument_condition_on_quantity_units}",
                flush=True,
            )
            argument_text_codec = ArgumentTextCodec.fit(
                train_records,
                max_length=args.argument_max_length,
                max_vocab_size=args.argument_vocab_size,
            )
            argument_filler, argument_history = train_argument_text_filler(
                train_records,
                condition_vectors=train_conditions,
                graph_codec=codec,
                text_codec=argument_text_codec,
                condition_dim=featurizer.condition_dim,
                hidden_dim=args.argument_filler_hidden_dim,
                layers=args.argument_filler_layers,
                heads=args.argument_filler_heads,
                dropout=args.argument_filler_dropout,
                epochs=args.argument_filler_epochs,
                batch_size=args.argument_filler_batch_size,
                learning_rate=args.argument_filler_learning_rate,
                gradient_clip_norm=args.gradient_clip_norm,
                max_examples=args.argument_max_examples,
                seed=args.seed,
                device=args.device,
                log_every=args.log_every,
                condition_on_quantities=True,
                condition_on_quantity_units=argument_condition_on_quantity_units,
                target=args.argument_filler_target,
            )
        print(f"准备保存 checkpoint 并运行最终验证解码：{args.checkpoint}", flush=True)
        save_graph_checkpoint(
            args.checkpoint,
            model=model,
            codec=codec,
            condition_featurizer=featurizer.to_dict(),
            history=history,
            argument_filler=argument_filler,
            argument_text_codec=argument_text_codec,
            argument_history=argument_history,
            argument_filler_target=args.argument_filler_target,
            argument_condition_on_quantity_units=argument_condition_on_quantity_units,
        )
        val_conditions = [featurizer.condition_vector(record) for record in val_records]
        prediction_rows = predict_direct_graph_records(
            model,
            codec,
            val_records,
            condition_vectors=val_conditions,
            argument_filler=argument_filler,
            argument_text_codec=argument_text_codec,
            argument_filler_target=args.argument_filler_target,
            argument_condition_on_quantity_units=argument_condition_on_quantity_units,
            include_generated_graph=args.save_generated_graph,
            quantity_gate_threshold=args.quantity_gate_threshold,
            condition_probability_threshold=args.condition_probability_threshold,
            use_structure_length=args.use_structure_length,
            min_structure_steps=args.min_structure_steps,
            device=args.device,
        )
    elif args.decoder_backend == "graph_diffusion":
        if (
            args.graph_condition_encoder == "shared_molt5"
            and not args.shared_encoder_checkpoint
        ):
            raise ValueError(
                "--graph-condition-encoder shared_molt5 requires "
                "--shared-encoder-checkpoint"
            )
        print("准备阶段：开始构建 graph diffusion codec", flush=True)
        codec = GraphTargetCodec.fit(
            train_records,
            max_steps=args.max_steps,
            max_material_refs=args.max_material_refs,
            max_material_slots=args.max_material_slots,
            max_quantity_vocab=args.max_quantity_vocab,
            max_numeric_candidates=args.max_numeric_candidates,
            numeric_candidate_include_source=args.numeric_candidate_include_source,
            numeric_candidate_quantity_only=args.numeric_candidate_quantity_only,
        )
        print(
            "离散图扩散模型维度："
            f"操作类别={codec.action_dim} 材料类别={codec.material_dim} "
            f"条件类别={codec.condition_dim} 单位类别={codec.unit_dim} "
            f"数值候选类别={codec.numeric_candidate_dim} "
            f"每步材料槽={codec.max_material_slots} 最大步数={codec.max_steps} "
            f"hidden={args.hidden_dim} DiT层数={args.dit_depth} heads={args.dit_heads} "
            f"扩散步数={args.diffusion_steps} 噪声日程={args.graph_diffusion_noise_schedule}",
            flush=True,
        )
        print(
            "训练设置："
            f"epochs={args.epochs} batch={args.batch_size} lr={args.learning_rate:g} "
            f"采样={args.sampling_strategy} 样本权重上限={args.sample_weight_max} "
            f"骨架权重上限={args.skeleton_weight_max} "
            f"操作权重={args.operation_weighting}(alpha={args.operation_weight_alpha}, cap={args.operation_weight_max}) "
            f"结构loss权重={args.structure_loss_weight} PAD权重={args.graph_diffusion_pad_loss_weight} "
            f"骨架条件={'启用' if args.graph_diffusion_skeleton_conditioning else '关闭'} "
            f"骨架loss={args.graph_diffusion_skeleton_loss_weight} "
            f"骨架teacher={args.graph_diffusion_skeleton_teacher_forcing} "
            f"slot操作loss={args.graph_diffusion_slot_operation_loss_weight} "
            f"quantity模式={args.graph_diffusion_quantity_mode} 梯度裁剪={args.gradient_clip_norm}",
            flush=True,
        )
        print(
            "采样/验证设置："
            f"反向采样器={args.graph_diffusion_sampler} "
            f"采样模式={args.graph_diffusion_sample_mode} "
            f"采样步数={args.graph_diffusion_sample_steps or args.diffusion_steps} "
            f"temperature={args.graph_diffusion_sample_temperature} "
            f"验证batch={args.sample_batch_size} "
            f"结构长度={'启用' if args.use_structure_length else '关闭'} "
            f"最小步数={args.min_structure_steps}",
            flush=True,
        )
        graph_diffusion_numeric_slot_context = (
            args.argument_filler_backend == "transformer"
            and args.argument_filler_target == "numeric"
        )
        graph_diffusion_diffuse_quantities = (
            args.graph_diffusion_quantity_mode in {"diffusion", "grounded"}
            or graph_diffusion_numeric_slot_context
        )
        graph_diffusion_ground_numeric = args.graph_diffusion_quantity_mode == "grounded"
        graph_diffusion_decode_quantity_values = args.graph_diffusion_quantity_mode == "diffusion"
        effective_quantity_value_loss_weight = (
            args.quantity_value_loss_weight if args.graph_diffusion_quantity_mode == "diffusion" else 0.0
        )
        if graph_diffusion_numeric_slot_context:
            print(
                "数值类槽位设置：graph diffusion 训练 quantity gate/unit 作为 filler 上下文；"
                "具体数值文本由 argument filler 自回归生成，graph quantity value loss=0。",
                flush=True,
            )
        graph_diffusion_best_metric_key = args.best_eval_metric
        print("准备阶段：开始张量化 graph diffusion 条件向量", flush=True)
        train_conditions = [featurizer.condition_vector(record) for record in train_records]
        print("准备阶段完成，开始 graph diffusion 训练。", flush=True)
        graph_diffusion_validation_callback = None
        if args.best_eval_interval > 0:
            if args.best_eval_limit == 0 or args.best_eval_limit >= len(val_records):
                best_eval_records = val_records
            else:
                validation_rng = random.Random(args.seed)
                best_eval_indices = sorted(
                    validation_rng.sample(
                        range(len(val_records)),
                        max(int(args.best_eval_limit), 1),
                    )
                )
                best_eval_records = [val_records[index] for index in best_eval_indices]
            best_eval_conditions = [
                featurizer.condition_vector(record) for record in best_eval_records
            ]
            print(
                "最佳模型选择："
                f"每 {args.best_eval_interval} epoch 验证一次，"
                f"验证样本={len(best_eval_records)}（固定随机子集），"
                f"指标={graph_diffusion_best_metric_key} 越高越好，"
                f"训练结束后{'恢复最佳权重' if not args.no_restore_best else '保留最后一轮权重'}。",
                flush=True,
            )

            def graph_diffusion_validation_callback(current_model, epoch):
                print(
                    f"[验证 {epoch:03d}] 开始生成验证样本："
                    f"records={len(best_eval_records)} batch={args.sample_batch_size} "
                    f"sampler={args.graph_diffusion_sampler}/{args.graph_diffusion_sample_mode}",
                    flush=True,
                )
                rows = predict_procedure_graph_diffusion_records(
                    current_model,
                    codec,
                    best_eval_records,
                    condition_vectors=best_eval_conditions,
                    include_generated_graph=False,
                    quantity_gate_threshold=args.quantity_gate_threshold,
                    condition_probability_threshold=args.condition_probability_threshold,
                    sample_steps=args.graph_diffusion_sample_steps,
                    sample_mode=args.graph_diffusion_sample_mode,
                    sample_temperature=args.graph_diffusion_sample_temperature,
                    sample_batch_size=args.sample_batch_size,
                    sampler=args.graph_diffusion_sampler,
                    use_structure_length=args.use_structure_length,
                    min_structure_steps=args.min_structure_steps,
                    decode_quantities=graph_diffusion_diffuse_quantities,
                    decode_quantity_values=graph_diffusion_decode_quantity_values,
                    ground_numeric_slots=graph_diffusion_ground_numeric,
                    numeric_candidate_reuse_penalty=args.numeric_candidate_reuse_penalty,
                    numeric_candidate_unit_weight=args.numeric_candidate_unit_weight,
                    drop_unsupported_numeric_slots=args.drop_unsupported_numeric_slots,
                    skeleton_source=args.eval_skeleton_source,
                    skeleton_cache=skeleton_cache,
                    seed=args.seed,
                    device=args.device,
                )
                metrics = discrete_slot_metrics(rows, best_eval_records, codec=codec)
                pairs = [
                    (row["predicted_actions"], row["reference_actions"])
                    for row in rows
                ]
                metrics.update(corpus_text_metrics(pairs))
                metrics.update(corpus_semantic_metrics(pairs))
                print(
                    f"[验证 {epoch:03d}] 完成："
                    f"records={len(best_eval_records)} "
                    f"离散槽分数={metrics['discrete_slot_score']:.4f} "
                    f"语义分数={metrics['semantic_score']:.4f} "
                    f"canonical-LEV75={metrics['canonical_levenshtein_75_rate']:.4f} "
                    f"操作序列相似度={metrics['operation_sequence_similarity']:.4f} "
                    f"长度误差={metrics['absolute_length_error']:.2f}",
                    flush=True,
                )
                return metrics
        else:
            print("最佳模型选择：已关闭周期验证，训练结束后保存最后一轮权重。", flush=True)

        model, history = train_procedure_graph_diffusion(
            train_records,
            condition_vectors=train_conditions,
            codec=codec,
            condition_dim=featurizer.condition_dim,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            hidden_dim=args.hidden_dim,
            dit_depth=args.dit_depth,
            dit_heads=args.dit_heads,
            diffusion_steps=args.diffusion_steps,
            noise_schedule=args.graph_diffusion_noise_schedule,
            material_loss_weight=args.material_loss_weight,
            condition_loss_weight=args.condition_loss_weight,
            quantity_gate_loss_weight=args.quantity_gate_loss_weight,
            material_none_weight=args.material_none_weight,
            material_present_weight=args.material_present_weight,
            condition_none_weight=args.condition_none_weight,
            condition_present_weight=args.condition_present_weight,
            quantity_negative_weight=args.quantity_negative_weight,
            quantity_positive_weight=args.quantity_positive_weight,
            quantity_value_loss_weight=effective_quantity_value_loss_weight,
            numeric_candidate_loss_weight=args.numeric_candidate_loss_weight,
            condition_value_loss_weight=args.condition_value_loss_weight,
            numeric_value_clip=args.numeric_value_clip,
            structure_loss_weight=args.structure_loss_weight,
            sampling_strategy=args.sampling_strategy,
            sample_weight_max=args.sample_weight_max,
            skeleton_weight_max=args.skeleton_weight_max,
            operation_weighting=args.operation_weighting,
            operation_weight_alpha=args.operation_weight_alpha,
            operation_weight_max=args.operation_weight_max,
            gradient_clip_norm=args.gradient_clip_norm,
            pad_loss_weight=args.graph_diffusion_pad_loss_weight,
            diffuse_quantities=graph_diffusion_diffuse_quantities,
            skeleton_conditioning=args.graph_diffusion_skeleton_conditioning,
            skeleton_loss_weight=args.graph_diffusion_skeleton_loss_weight,
            skeleton_teacher_forcing_probability=args.graph_diffusion_skeleton_teacher_forcing,
            skeleton_teacher_forcing_final_probability=(
                args.graph_diffusion_skeleton_teacher_forcing_final
            ),
            skeleton_corruption_probability=(
                args.graph_diffusion_skeleton_corruption_probability
            ),
            slot_operation_loss_weight=args.graph_diffusion_slot_operation_loss_weight,
            timestep_sampling=args.graph_diffusion_timestep_sampling,
            endpoint_probability=args.graph_diffusion_endpoint_probability,
            shared_encoder_checkpoint=(
                args.shared_encoder_checkpoint
                if args.graph_condition_encoder == "shared_molt5"
                else None
            ),
            shared_encoder_mode=args.shared_encoder_mode,
            shared_encoder_learning_rate=args.shared_encoder_learning_rate,
            shared_encoder_max_length=args.shared_encoder_max_length,
            shared_encoder_prompt_style="checkpoint",
            shared_encoder_include_numeric_evidence=include_numeric_evidence,
            shared_encoder_numeric_evidence_include_source=(
                args.numeric_candidate_include_source
            ),
            shared_encoder_local_files_only=args.shared_encoder_local_files_only,
            numeric_candidate_feature_pointer=args.numeric_candidate_feature_pointer,
            seed=args.seed,
            device=args.device,
            log_every=args.log_every,
            validation_callback=graph_diffusion_validation_callback,
            validation_interval=args.best_eval_interval,
            restore_best=not args.no_restore_best,
            best_metric_key=graph_diffusion_best_metric_key,
            best_metric_mode="max",
        )
        argument_filler = None
        argument_text_codec = None
        argument_history = []
        if args.argument_filler_backend == "transformer":
            print(
                "开始训练 argument text filler："
                f"epochs={args.argument_filler_epochs} batch={args.argument_filler_batch_size} "
                f"hidden={args.argument_filler_hidden_dim} layers={args.argument_filler_layers} "
                f"target={args.argument_filler_target} "
                f"condition_on_quantities={graph_diffusion_diffuse_quantities} "
                f"condition_on_quantity_units={argument_condition_on_quantity_units}",
                flush=True,
            )
            argument_text_codec = ArgumentTextCodec.fit(
                train_records,
                max_length=args.argument_max_length,
                max_vocab_size=args.argument_vocab_size,
            )
            argument_filler, argument_history = train_argument_text_filler(
                train_records,
                condition_vectors=train_conditions,
                graph_codec=codec,
                text_codec=argument_text_codec,
                condition_dim=featurizer.condition_dim,
                hidden_dim=args.argument_filler_hidden_dim,
                layers=args.argument_filler_layers,
                heads=args.argument_filler_heads,
                dropout=args.argument_filler_dropout,
                epochs=args.argument_filler_epochs,
                batch_size=args.argument_filler_batch_size,
                learning_rate=args.argument_filler_learning_rate,
                gradient_clip_norm=args.gradient_clip_norm,
                max_examples=args.argument_max_examples,
                seed=args.seed,
                device=args.device,
                log_every=args.log_every,
                condition_on_quantities=graph_diffusion_diffuse_quantities,
                condition_on_quantity_units=argument_condition_on_quantity_units,
                target=args.argument_filler_target,
            )
        restored_best_epoch = history[-1].get("restored_best_epoch") if history else None
        restored_best_score = (
            history[-1].get(f"restored_best_{graph_diffusion_best_metric_key}") if history else None
        )
        if restored_best_epoch is not None:
            print(
                f"准备保存 checkpoint：将保存已恢复的最佳模型 "
                f"epoch={int(restored_best_epoch):03d} "
                f"{graph_diffusion_best_metric_key}={float(restored_best_score):.4f} "
                f"-> {args.checkpoint}",
                flush=True,
            )
        else:
            print(f"准备保存 checkpoint：保存当前模型权重 -> {args.checkpoint}", flush=True)
        save_procedure_graph_diffusion_checkpoint(
            args.checkpoint,
            model=model,
            codec=codec,
            condition_featurizer=featurizer.to_dict(),
            history=history,
            argument_filler=argument_filler,
            argument_text_codec=argument_text_codec,
            argument_history=argument_history,
            argument_filler_target=args.argument_filler_target,
            argument_condition_on_quantity_units=argument_condition_on_quantity_units,
        )
        print("checkpoint 保存完成，开始最终验证集采样。", flush=True)
        val_conditions = [featurizer.condition_vector(record) for record in val_records]
        prediction_rows = predict_procedure_graph_diffusion_records(
            model,
            codec,
            val_records,
            condition_vectors=val_conditions,
            argument_filler=argument_filler,
            argument_text_codec=argument_text_codec,
            argument_filler_target=args.argument_filler_target,
            argument_condition_on_quantity_units=argument_condition_on_quantity_units,
            include_generated_graph=args.save_generated_graph,
            quantity_gate_threshold=args.quantity_gate_threshold,
            condition_probability_threshold=args.condition_probability_threshold,
            sample_steps=args.graph_diffusion_sample_steps,
            sample_mode=args.graph_diffusion_sample_mode,
            sample_temperature=args.graph_diffusion_sample_temperature,
            sample_batch_size=args.sample_batch_size,
            sampler=args.graph_diffusion_sampler,
            use_structure_length=args.use_structure_length,
            min_structure_steps=args.min_structure_steps,
            decode_quantities=graph_diffusion_diffuse_quantities,
            decode_quantity_values=graph_diffusion_decode_quantity_values,
            ground_numeric_slots=graph_diffusion_ground_numeric,
            numeric_candidate_reuse_penalty=args.numeric_candidate_reuse_penalty,
            numeric_candidate_unit_weight=args.numeric_candidate_unit_weight,
            drop_unsupported_numeric_slots=args.drop_unsupported_numeric_slots,
            skeleton_source=args.eval_skeleton_source,
            skeleton_cache=skeleton_cache,
            seed=args.seed,
            device=args.device,
        )
    else:
        candidates = build_candidate_memory(train_records, featurizer)
        print(
            "Memory baseline dims: "
            f"structure={featurizer.structure_dim} attributes={featurizer.attribute_dim} "
            f"diffusion_steps={args.diffusion_steps} hidden={args.hidden_dim} "
            f"base={args.diffusion_base} dit_depth={args.dit_depth} dit_heads={args.dit_heads}",
            flush=True,
        )
        model, schedule, history = train_joint_diffusion(
            train_records,
            featurizer=featurizer,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            diffusion_steps=args.diffusion_steps,
            hidden_dim=args.hidden_dim,
            time_dim=args.time_dim,
            context_dim=args.context_dim,
            base_model=args.diffusion_base,
            dit_depth=args.dit_depth,
            dit_heads=args.dit_heads,
            seed=args.seed,
            device=args.device,
            log_every=args.log_every,
        )
        print(f"准备保存 checkpoint 并运行最终验证解码：{args.checkpoint}", flush=True)
        save_checkpoint(
            args.checkpoint,
            model=model,
            schedule=schedule,
            featurizer=featurizer,
            candidates=candidates,
            history=history,
        )
        prediction_rows = predict_records(
            model,
            schedule,
            featurizer,
            val_records,
            candidates,
            seed=args.seed,
            device=args.device,
        )

    write_jsonl(args.predictions, prediction_rows)
    print(
        f"最终验证采样完成：生成 {len(prediction_rows)} 条预测，"
        f"已写入 {args.predictions}",
        flush=True,
    )
    metrics = corpus_text_metrics(
        (row["predicted_actions"], row["reference_actions"]) for row in prediction_rows
    )
    metrics.update(
        corpus_semantic_metrics(
            (row["predicted_actions"], row["reference_actions"])
            for row in prediction_rows
        )
    )
    if args.decoder_backend in {"direct_graph", "graph_diffusion"}:
        metrics.update(discrete_slot_metrics(prediction_rows, val_records, codec=codec))
    metrics.update(
        {
            "train_records": len(train_records),
            "val_records": len(val_records),
            "epochs": args.epochs,
            "condition_dim": featurizer.condition_dim,
            "condition_encoding": args.condition_encoding,
            "include_numeric_evidence": include_numeric_evidence,
            "numeric_evidence_include_source": args.numeric_candidate_include_source,
            "numeric_evidence_quantity_only": args.numeric_candidate_quantity_only,
            "field_dim": featurizer.field_dim,
            "ngram_min": featurizer.ngram_min,
            "ngram_max": featurizer.ngram_max,
            "decoder_backend": args.decoder_backend,
            "final_train_loss": history[-1]["loss"] if history else None,
            "checkpoint": args.checkpoint,
            "predictions": args.predictions,
        }
    )
    if args.decoder_backend == "direct_graph":
        metrics.update(
            {
                "max_steps": args.max_steps,
                "latent_dim": args.latent_dim,
                "action_dim": codec.action_dim,
                "material_dim": codec.material_dim,
                "condition_slot_dim": codec.condition_dim,
                "unit_dim": codec.unit_dim,
                "numeric_candidate_dim": codec.numeric_candidate_dim,
                "max_numeric_candidates": codec.max_numeric_candidates,
                "numeric_candidate_include_source": codec.numeric_candidate_include_source,
                "numeric_candidate_quantity_only": codec.numeric_candidate_quantity_only,
                "quantity_dim": codec.quantity_dim,
                "max_material_slots": codec.max_material_slots,
                "latent_norm_weight": args.latent_norm_weight,
                "graph_backbone": args.graph_backbone,
                "dit_depth": args.dit_depth,
                "dit_heads": args.dit_heads,
                "material_loss_weight": args.material_loss_weight,
                "condition_loss_weight": args.condition_loss_weight,
                "condition_value_loss_weight": args.condition_value_loss_weight,
                "structure_loss_weight": args.structure_loss_weight,
                "numeric_value_clip": args.numeric_value_clip,
                "sampling_strategy": args.sampling_strategy,
                "sample_weight_max": args.sample_weight_max,
                "skeleton_weight_max": args.skeleton_weight_max,
                "operation_weighting": args.operation_weighting,
                "operation_weight_alpha": args.operation_weight_alpha,
                "operation_weight_max": args.operation_weight_max,
                "gradient_clip_norm": args.gradient_clip_norm,
                "quantity_gate_loss_weight": args.quantity_gate_loss_weight,
                "material_none_weight": args.material_none_weight,
                "material_present_weight": args.material_present_weight,
                "condition_none_weight": args.condition_none_weight,
                "condition_present_weight": args.condition_present_weight,
                "quantity_negative_weight": args.quantity_negative_weight,
                "quantity_positive_weight": args.quantity_positive_weight,
                "quantity_gate_threshold": args.quantity_gate_threshold,
                "condition_probability_threshold": args.condition_probability_threshold,
                "use_structure_length": args.use_structure_length,
                "min_structure_steps": args.min_structure_steps,
                "argument_filler_backend": args.argument_filler_backend,
            }
        )
        if args.argument_filler_backend == "transformer":
            metrics.update(
                {
                    "argument_filler_epochs": args.argument_filler_epochs,
                    "argument_filler_batch_size": args.argument_filler_batch_size,
                    "argument_filler_learning_rate": args.argument_filler_learning_rate,
                    "argument_filler_hidden_dim": args.argument_filler_hidden_dim,
                    "argument_filler_layers": args.argument_filler_layers,
                    "argument_filler_heads": args.argument_filler_heads,
                    "argument_filler_target": args.argument_filler_target,
                    "argument_max_length": args.argument_max_length,
                    "argument_vocab_size": args.argument_vocab_size,
                    "argument_condition_on_quantities": True,
                    "argument_condition_on_quantity_units": argument_condition_on_quantity_units,
                    "argument_final_loss": argument_history[-1]["loss"] if argument_history else None,
                    "argument_examples": argument_history[-1]["examples"] if argument_history else None,
                }
            )
    elif args.decoder_backend == "graph_diffusion":
        metrics.update(
            {
                "max_steps": args.max_steps,
                "action_dim": codec.action_dim,
                "material_dim": codec.material_dim,
                "condition_slot_dim": codec.condition_dim,
                "unit_dim": codec.unit_dim,
                "quantity_dim": codec.quantity_dim,
                "numeric_candidate_dim": codec.numeric_candidate_dim,
                "max_numeric_candidates": codec.max_numeric_candidates,
                "numeric_candidate_include_source": codec.numeric_candidate_include_source,
                "numeric_candidate_quantity_only": codec.numeric_candidate_quantity_only,
                "numeric_candidate_feature_pointer": args.numeric_candidate_feature_pointer,
                "numeric_candidate_reuse_penalty": args.numeric_candidate_reuse_penalty,
                "numeric_candidate_unit_weight": args.numeric_candidate_unit_weight,
                "drop_unsupported_numeric_slots": args.drop_unsupported_numeric_slots,
                "max_material_slots": codec.max_material_slots,
                "diffusion_steps": args.diffusion_steps,
                "graph_diffusion_noise_schedule": args.graph_diffusion_noise_schedule,
                "graph_condition_encoder": args.graph_condition_encoder,
                "shared_encoder_checkpoint": args.shared_encoder_checkpoint,
                "shared_encoder_mode": args.shared_encoder_mode,
                "shared_encoder_learning_rate": args.shared_encoder_learning_rate,
                "shared_encoder_max_length": args.shared_encoder_max_length,
                "graph_diffusion_sample_steps": args.graph_diffusion_sample_steps
                or args.diffusion_steps,
                "graph_diffusion_sample_mode": args.graph_diffusion_sample_mode,
                "graph_diffusion_sampler": args.graph_diffusion_sampler,
                "graph_diffusion_sample_temperature": args.graph_diffusion_sample_temperature,
                "graph_diffusion_pad_loss_weight": args.graph_diffusion_pad_loss_weight,
                "graph_diffusion_quantity_mode": args.graph_diffusion_quantity_mode,
                "graph_diffusion_ground_numeric": graph_diffusion_ground_numeric,
                "graph_diffusion_decode_quantity_values": graph_diffusion_decode_quantity_values,
                "graph_diffusion_numeric_slot_context": graph_diffusion_numeric_slot_context,
                "graph_diffusion_timestep_sampling": args.graph_diffusion_timestep_sampling,
                "graph_diffusion_endpoint_probability": args.graph_diffusion_endpoint_probability,
                "graph_diffusion_skeleton_conditioning": args.graph_diffusion_skeleton_conditioning,
                "graph_diffusion_skeleton_loss_weight": args.graph_diffusion_skeleton_loss_weight,
                "graph_diffusion_skeleton_teacher_forcing": args.graph_diffusion_skeleton_teacher_forcing,
                "graph_diffusion_skeleton_teacher_forcing_final": (
                    args.graph_diffusion_skeleton_teacher_forcing_final
                ),
                "graph_diffusion_skeleton_corruption_probability": (
                    args.graph_diffusion_skeleton_corruption_probability
                ),
                "graph_diffusion_slot_operation_loss_weight": args.graph_diffusion_slot_operation_loss_weight,
                "sample_batch_size": args.sample_batch_size,
                "eval_skeleton_source": args.eval_skeleton_source,
                "skeleton_cache": args.skeleton_cache,
                "use_structure_length": args.use_structure_length,
                "min_structure_steps": args.min_structure_steps,
                "dit_depth": args.dit_depth,
                "dit_heads": args.dit_heads,
                "material_loss_weight": args.material_loss_weight,
                "condition_loss_weight": args.condition_loss_weight,
                "structure_loss_weight": args.structure_loss_weight,
                "sampling_strategy": args.sampling_strategy,
                "sample_weight_max": args.sample_weight_max,
                "skeleton_weight_max": args.skeleton_weight_max,
                "operation_weighting": args.operation_weighting,
                "operation_weight_alpha": args.operation_weight_alpha,
                "operation_weight_max": args.operation_weight_max,
                "gradient_clip_norm": args.gradient_clip_norm,
                "quantity_gate_loss_weight": args.quantity_gate_loss_weight,
                "quantity_value_loss_weight": args.quantity_value_loss_weight,
                "numeric_candidate_loss_weight": args.numeric_candidate_loss_weight,
                "effective_quantity_value_loss_weight": effective_quantity_value_loss_weight,
                "condition_value_loss_weight": args.condition_value_loss_weight,
                "numeric_value_clip": args.numeric_value_clip,
                "material_none_weight": args.material_none_weight,
                "material_present_weight": args.material_present_weight,
                "condition_none_weight": args.condition_none_weight,
                "condition_present_weight": args.condition_present_weight,
                "quantity_negative_weight": args.quantity_negative_weight,
                "quantity_positive_weight": args.quantity_positive_weight,
                "quantity_gate_threshold": args.quantity_gate_threshold,
                "condition_probability_threshold": args.condition_probability_threshold,
                "argument_filler_backend": args.argument_filler_backend,
                "best_eval_interval": args.best_eval_interval,
                "best_eval_limit": args.best_eval_limit,
                "best_eval_metric": graph_diffusion_best_metric_key,
                "best_eval_mode": "max",
                "restore_best": not args.no_restore_best,
                "restored_best_epoch": history[-1].get("restored_best_epoch") if history else None,
                "restored_best_metric_value": history[-1].get(
                    f"restored_best_{graph_diffusion_best_metric_key}"
                )
                if history
                else None,
            }
        )
        if args.argument_filler_backend == "transformer":
            metrics.update(
                {
                    "argument_filler_epochs": args.argument_filler_epochs,
                    "argument_filler_batch_size": args.argument_filler_batch_size,
                    "argument_filler_learning_rate": args.argument_filler_learning_rate,
                    "argument_filler_hidden_dim": args.argument_filler_hidden_dim,
                    "argument_filler_layers": args.argument_filler_layers,
                    "argument_filler_heads": args.argument_filler_heads,
                    "argument_filler_target": args.argument_filler_target,
                    "argument_max_length": args.argument_max_length,
                    "argument_vocab_size": args.argument_vocab_size,
                    "argument_condition_on_quantities": graph_diffusion_diffuse_quantities,
                    "argument_condition_on_quantity_units": argument_condition_on_quantity_units,
                    "argument_final_loss": argument_history[-1]["loss"] if argument_history else None,
                    "argument_examples": argument_history[-1]["examples"] if argument_history else None,
                }
            )
    else:
        metrics.update(
            {
                "structure_dim": featurizer.structure_dim,
                "attribute_dim": featurizer.attribute_dim,
                "diffusion_steps": args.diffusion_steps,
                "diffusion_base": args.diffusion_base,
                "dit_depth": args.dit_depth,
                "dit_heads": args.dit_heads,
            }
        )
    metrics_path = Path(args.metrics)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print(f"训练结束：训练样本={len(train_records)}，最终验证样本={len(val_records)}。")
    print(f"checkpoint 已写入：{args.checkpoint}")
    print(f"预测结果已写入：{args.predictions}")
    print(f"评估指标已写入：{args.metrics}")
    print(
        "最终验证指标："
        f"BLEU-2={metrics['bleu_2']:.4f}，"
        f"ROUGE-1={metrics['rouge_1']:.4f}，"
        f"90%LEV={metrics['levenshtein_90_rate']:.4f}，"
        f"75%LEV={metrics['levenshtein_75_rate']:.4f}，"
        f"50%LEV={metrics['levenshtein_50_rate']:.4f}，"
        f"数值归一75%LEV={metrics['number_normalized_levenshtein_75_rate']:.4f}，"
        f"exact={metrics['exact_match_rate']:.4f}，"
        f"semantic={metrics['semantic_score']:.4f}，"
        f"canonical-75%LEV={metrics['canonical_levenshtein_75_rate']:.4f}"
    )


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
