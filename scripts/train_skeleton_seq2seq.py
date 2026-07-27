"""Train a BART/T5-style seq2seq operation-skeleton predictor."""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reactgdiff.data.action_parser import parse_action_sequence
from reactgdiff.data.procedure_prompt import build_encoder_prompt
from reactgdiff.eval.lev import edit_distance
from reactgdiff.models.graph_codec import EOS_TOKEN, PAD_TOKEN, GraphTargetCodec
from reactgdiff.models.joint_diffusion import _placeholder_sort_key, _reactxt_prompt_fields, load_split_records
from reactgdiff.utils.io import write_jsonl

try:
    from transformers import (
        AutoConfig,
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        get_cosine_schedule_with_warmup,
    )
except ImportError as exc:  # pragma: no cover - exercised only in missing optional dependency envs.
    raise SystemExit(
        "transformers is required for train_skeleton_seq2seq.py. "
        "Install transformers or use scripts/train_skeleton_ar.py."
    ) from exc


ACTION_PHRASES = {
    "ADD": "add",
    "COLLECTLAYER": "collect layer",
    "CONCENTRATE": "concentrate",
    "DEGAS": "degas",
    "DRYSOLID": "dry solid",
    "DRYSOLUTION": "dry solution",
    "EXTRACT": "extract",
    "FILTER": "filter",
    "MAKESOLUTION": "make solution",
    "MICROWAVE": "microwave",
    "PARTITION": "partition",
    "PH": "adjust pH",
    "PHASESEPARATION": "separate phases",
    "QUENCH": "quench",
    "RECRYSTALLIZE": "recrystallize",
    "REFLUX": "reflux",
    "SETTEMPERATURE": "set temperature",
    "SONICATE": "sonicate",
    "STIR": "stir",
    "TRITURATE": "triturate",
    "WAIT": "wait",
    "WASH": "wash",
    "YIELD": "yield",
}


@dataclass(slots=True)
class PromptAugmentationConfig:
    enabled: bool = False
    participant_shuffle_prob: float = 0.0
    numeric_format_prob: float = 0.0
    field_mask_prob: float = 0.0


class SkeletonSeq2SeqModel(nn.Module):
    def __init__(
        self,
        base_model: nn.Module,
        *,
        hidden_size: int,
        max_steps: int,
        length_loss_weight: float,
        token_loss_weights: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.max_steps = int(max_steps)
        self.length_loss_weight = float(length_loss_weight)
        self.length_head = nn.Linear(hidden_size, self.max_steps + 1)
        self.register_buffer("token_loss_weights", token_loss_weights, persistent=False)

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        output = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=False,
            return_dict=True,
        )
        pooled = self._pool_encoder(output.encoder_last_hidden_state, attention_mask)
        length_logits = self.length_head(pooled)
        raw_token_loss = output.loss
        token_loss = (
            weighted_token_cross_entropy(output.logits, labels, self.token_loss_weights)
            if self.token_loss_weights is not None
            else raw_token_loss
        )
        length_loss = F.cross_entropy(length_logits, lengths.clamp(0, self.max_steps))
        loss = token_loss + self.length_loss_weight * length_loss
        metrics = {
            "loss": float(loss.detach()),
            "token_loss": float(token_loss.detach()),
            "raw_token_loss": float(raw_token_loss.detach()),
            "length_loss": float(length_loss.detach()),
        }
        return loss, metrics

    @torch.no_grad()
    def length_logits(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        encoder = self.base_model.get_encoder()
        output = encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        pooled = self._pool_encoder(output.last_hidden_state, attention_mask)
        return self.length_head(pooled)

    @staticmethod
    def _pool_encoder(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)


class SkeletonTextDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        records: list[dict[str, Any]],
        *,
        codec: GraphTargetCodec,
        include_numeric_evidence: bool,
        prompt_style: str,
        target_format: str,
        augmentation: PromptAugmentationConfig | None = None,
    ) -> None:
        self.records = records
        self.codec = codec
        self.include_numeric_evidence = include_numeric_evidence
        self.prompt_style = prompt_style
        self.target_format = target_format
        self.augmentation = augmentation or PromptAugmentationConfig()

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        operations = reference_operations(record, max_steps=self.codec.max_steps)
        return {
            "prompt": build_skeleton_prompt(
                record,
                include_numeric_evidence=self.include_numeric_evidence,
                prompt_style=self.prompt_style,
                augmentation=self.augmentation,
            ),
            "target": skeleton_target_text(operations, target_format=self.target_format),
            "length": min(len(operations), self.codec.max_steps),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", default="data/processed/openexp_sample/splits/train.jsonl")
    parser.add_argument("--val", default="data/processed/openexp_sample/splits/val.jsonl")
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--val-limit", type=int, default=None)
    parser.add_argument("--model-name", default="facebook/bart-base")
    parser.add_argument("--random-init", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--max-input-length", type=int, default=512)
    parser.add_argument("--max-target-length", type=int, default=96)
    parser.add_argument("--max-steps", type=int, default=32)
    parser.add_argument("--max-material-refs", type=int, default=16)
    parser.add_argument("--max-material-slots", type=int, default=4)
    parser.add_argument("--prompt-style", choices=("compact", "reactxt"), default="compact")
    parser.add_argument(
        "--target-format",
        choices=("natural_text", "special_tokens"),
        default="natural_text",
    )
    parser.add_argument("--no-numeric-evidence-input", action="store_true")
    parser.add_argument("--prompt-augmentation", action="store_true")
    parser.add_argument("--participant-shuffle-prob", type=float, default=0.0)
    parser.add_argument("--numeric-format-augment-prob", type=float, default=0.0)
    parser.add_argument("--field-mask-prob", type=float, default=0.0)
    parser.add_argument("--action-loss-weighting", choices=("none", "class_balanced"), default="none")
    parser.add_argument("--action-weight-beta", type=float, default=0.9999)
    parser.add_argument("--action-weight-min", type=float, default=0.5)
    parser.add_argument("--action-weight-max", type=float, default=4.0)
    parser.add_argument("--length-loss-weight", type=float, default=0.2)
    parser.add_argument("--length-prior-weight", type=float, default=0.0)
    parser.add_argument("--repetition-penalty-weight", type=float, default=0.1)
    parser.add_argument("--best-eval-interval", type=int, default=10)
    parser.add_argument("--no-restore-best", action="store_true")
    parser.add_argument("--beam-size", type=int, default=4)
    parser.add_argument("--num-return-sequences", type=int, default=4)
    parser.add_argument("--generation-length-penalty", type=float, default=1.0)
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--checkpoint", default="outputs/checkpoints/openexp_sample_seq2seq_skeleton.pt")
    parser.add_argument("--predictions", default="outputs/skeleton/openexp_sample_seq2seq_skeleton_val.jsonl")
    parser.add_argument("--metrics", default="outputs/metrics/openexp_sample_seq2seq_skeleton_val.json")
    parser.add_argument("--log-every", type=int, default=1)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    include_numeric_evidence = not args.no_numeric_evidence_input
    prompt_augmentation = PromptAugmentationConfig(
        enabled=bool(args.prompt_augmentation),
        participant_shuffle_prob=clamp_probability(args.participant_shuffle_prob),
        numeric_format_prob=clamp_probability(args.numeric_format_augment_prob),
        field_mask_prob=clamp_probability(args.field_mask_prob),
    )
    print(
        "Seq2seq skeleton setup: loading data "
        f"train={args.train} val={args.val} "
        f"train_limit={args.train_limit} val_limit={args.val_limit} "
        f"prompt_style={args.prompt_style}",
        flush=True,
    )
    train_records = load_split_records(args.train, limit=args.train_limit)
    val_records = load_split_records(args.val, limit=args.val_limit)
    print(
        "Seq2seq skeleton setup: data loaded "
        f"train_records={len(train_records)} val_records={len(val_records)} "
        f"device={args.device} seed={args.seed}",
        flush=True,
    )
    if prompt_augmentation.enabled:
        print(
            "Seq2seq skeleton setup: prompt augmentation enabled "
            f"participant_shuffle={prompt_augmentation.participant_shuffle_prob:g} "
            f"numeric_format={prompt_augmentation.numeric_format_prob:g} "
            f"field_mask={prompt_augmentation.field_mask_prob:g}",
            flush=True,
        )

    print("Seq2seq skeleton setup: building graph codec", flush=True)
    codec = GraphTargetCodec.fit(
        train_records,
        max_steps=args.max_steps,
        max_material_refs=args.max_material_refs,
        max_material_slots=args.max_material_slots,
    )
    action_tokens = build_action_tokens(codec) if args.target_format == "special_tokens" else {}
    if args.target_format == "natural_text" and args.action_loss_weighting != "none":
        raise ValueError(
            "--action-loss-weighting class_balanced requires --target-format special_tokens; "
            "natural_text uses standard pretrained-vocabulary cross entropy."
        )
    missing_phrases = [
        action
        for action in codec.action_vocab
        if action not in {PAD_TOKEN, EOS_TOKEN} and action not in ACTION_PHRASES
    ]
    if args.target_format == "natural_text" and missing_phrases:
        raise ValueError(f"Missing natural-language action phrases: {missing_phrases}")

    print(
        "Seq2seq skeleton setup: loading tokenizer/model "
        f"model={args.model_name} random_init={args.random_init} "
        f"local_files_only={args.local_files_only}",
        flush=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        local_files_only=args.local_files_only,
    )
    if action_tokens:
        tokenizer.add_special_tokens({"additional_special_tokens": list(action_tokens.values())})
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if action_tokens:
        validate_action_tokenization(tokenizer, action_tokens)
    token_loss_weights, action_weight_summary = build_action_loss_weights(
        tokenizer=tokenizer,
        codec=codec,
        action_tokens=action_tokens,
        records=train_records,
        weighting=args.action_loss_weighting,
        beta=args.action_weight_beta,
        min_weight=args.action_weight_min,
        max_weight=args.action_weight_max,
    )
    if action_weight_summary:
        print(
            "Seq2seq skeleton setup: action loss weights "
            f"mode={args.action_loss_weighting} beta={args.action_weight_beta:g} "
            f"min={args.action_weight_min:g} max={args.action_weight_max:g}",
            flush=True,
        )
        for item in action_weight_summary[:12]:
            print(
                f"  weight {item['action']}: count={item['count']} weight={item['weight']:.3f}",
                flush=True,
            )

    if args.random_init:
        config = AutoConfig.from_pretrained(
            args.model_name,
            local_files_only=args.local_files_only,
        )
        base_model = AutoModelForSeq2SeqLM.from_config(config)
    else:
        base_model = AutoModelForSeq2SeqLM.from_pretrained(
            args.model_name,
            local_files_only=args.local_files_only,
        )
    if len(tokenizer) != int(base_model.get_input_embeddings().num_embeddings):
        base_model.resize_token_embeddings(len(tokenizer))
    if args.freeze_encoder:
        encoder = base_model.get_encoder()
        for parameter in encoder.parameters():
            parameter.requires_grad = False

    hidden_size = infer_hidden_size(base_model)
    model = SkeletonSeq2SeqModel(
        base_model,
        hidden_size=hidden_size,
        max_steps=args.max_steps,
        length_loss_weight=args.length_loss_weight,
        token_loss_weights=token_loss_weights,
    ).to(args.device)
    if args.fp16 and args.bf16:
        raise ValueError("--fp16 and --bf16 are mutually exclusive")
    device_type = torch.device(args.device).type
    use_fp16 = bool(args.fp16 and device_type == "cuda")
    use_bf16 = bool(args.bf16 and device_type == "cuda")
    if (args.fp16 or args.bf16) and device_type != "cuda":
        raise ValueError("--fp16 and --bf16 require a CUDA device")
    use_amp = use_fp16 or use_bf16
    autocast_dtype = torch.float16 if use_fp16 else torch.bfloat16
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_fp16,
        init_scale=4096.0,
    )

    token_to_action_id = {
        int(tokenizer.convert_tokens_to_ids(token)): action_id
        for action_id, token in action_tokens.items()
    }
    action_token_ids = sorted(token_to_action_id)

    train_dataset = SkeletonTextDataset(
        train_records,
        codec=codec,
        include_numeric_evidence=include_numeric_evidence,
        prompt_style=args.prompt_style,
        target_format=args.target_format,
        augmentation=prompt_augmentation,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda rows: collate_rows(
            rows,
            tokenizer=tokenizer,
            max_input_length=args.max_input_length,
            max_target_length=args.max_target_length,
            device=args.device,
        ),
    )
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    gradient_accumulation_steps = max(int(args.gradient_accumulation_steps), 1)
    optimizer_steps_per_epoch = math.ceil(len(train_loader) / gradient_accumulation_steps)
    total_optimizer_steps = optimizer_steps_per_epoch * max(int(args.epochs), 0)
    if args.warmup_steps is not None:
        warmup_steps = max(0, min(int(args.warmup_steps), total_optimizer_steps))
    else:
        warmup_ratio = clamp_probability(args.warmup_ratio)
        warmup_steps = int(round(total_optimizer_steps * warmup_ratio))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max(total_optimizer_steps, 1),
    )

    print(
        "Seq2seq skeleton setup complete, starting training: "
        f"epochs={args.epochs} batch={args.batch_size} lr={args.learning_rate:g} "
        f"grad_accum={gradient_accumulation_steps} "
        f"target_format={args.target_format} scheduler=warmup_cosine "
        f"precision={'fp16' if use_fp16 else 'bf16' if use_bf16 else 'fp32'} "
        f"warmup_steps={warmup_steps} total_steps={total_optimizer_steps} "
        f"best_eval_interval={args.best_eval_interval} restore_best={not args.no_restore_best} "
        f"length_loss_weight={args.length_loss_weight:g} "
        f"length_prior_weight={args.length_prior_weight:g} "
        f"action_loss_weighting={args.action_loss_weighting}",
        flush=True,
    )
    history: list[dict[str, float]] = []
    validation_history: list[dict[str, float]] = []
    best_model_state: dict[str, torch.Tensor] | None = None
    best_predictions: list[dict[str, Any]] | None = None
    best_metrics: dict[str, float] | None = None
    best_epoch: int | None = None
    best_exact = -math.inf
    best_edit_similarity = -math.inf
    last_eval_predictions: list[dict[str, Any]] | None = None
    last_eval_metrics: dict[str, float] | None = None
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_token_loss = 0.0
        epoch_raw_token_loss = 0.0
        epoch_length_loss = 0.0
        seen = 0
        for batch_idx, batch in enumerate(train_loader, start=1):
            with torch.autocast(
                device_type=device_type,
                dtype=autocast_dtype,
                enabled=use_amp,
            ):
                loss, row = model(**batch)
            scaled_loss = loss / gradient_accumulation_steps
            scaler.scale(scaled_loss).backward()
            if batch_idx % gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                grad_norm = (
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip_norm)
                    if args.gradient_clip_norm > 0
                    else torch.tensor(0.0)
                )
                if not torch.isfinite(grad_norm):
                    if not use_fp16:
                        raise FloatingPointError(f"non-finite gradient norm: {grad_norm}")
                    print(
                        "FP16 gradient overflow; skipped optimizer step and reduced loss scale.",
                        flush=True,
                    )
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() >= scale_before:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            batch_size = int(batch["input_ids"].size(0))
            seen += batch_size
            epoch_loss += row["loss"] * batch_size
            epoch_token_loss += row["token_loss"] * batch_size
            epoch_raw_token_loss += row["raw_token_loss"] * batch_size
            epoch_length_loss += row["length_loss"] * batch_size
        if len(train_loader) % gradient_accumulation_steps != 0:
            scaler.unscale_(optimizer)
            grad_norm = (
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip_norm)
                if args.gradient_clip_norm > 0
                else torch.tensor(0.0)
            )
            if not torch.isfinite(grad_norm):
                if not use_fp16:
                    raise FloatingPointError(f"non-finite gradient norm: {grad_norm}")
                print(
                    "FP16 gradient overflow; skipped optimizer step and reduced loss scale.",
                    flush=True,
                )
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= scale_before:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        metrics_row = {
            "epoch": float(epoch),
            "loss": epoch_loss / max(seen, 1),
            "token_loss": epoch_token_loss / max(seen, 1),
            "raw_token_loss": epoch_raw_token_loss / max(seen, 1),
            "length_loss": epoch_length_loss / max(seen, 1),
            "records": float(seen),
            "learning_rate": float(scheduler.get_last_lr()[0]),
        }
        history.append(metrics_row)
        if args.log_every and (epoch == 1 or epoch == args.epochs or epoch % args.log_every == 0):
            print(
                f"[seq2seq skeleton {epoch:03d}/{args.epochs:03d}] "
                f"loss={metrics_row['loss']:.4f} "
                f"token={metrics_row['token_loss']:.4f} "
                f"raw_token={metrics_row['raw_token_loss']:.4f} "
                f"length={metrics_row['length_loss']:.4f} "
                f"lr={metrics_row['learning_rate']:.3e} "
                f"records={int(metrics_row['records'])}",
                flush=True,
            )
        eval_interval = max(int(args.best_eval_interval), 1)
        should_evaluate = epoch % eval_interval == 0 or epoch == args.epochs
        if should_evaluate:
            eval_predictions = predict_records(
                model,
                tokenizer,
                codec,
                val_records,
                target_format=args.target_format,
                action_token_ids=action_token_ids,
                token_to_action_id=token_to_action_id,
                include_numeric_evidence=include_numeric_evidence,
                prompt_style=args.prompt_style,
                max_input_length=args.max_input_length,
                max_new_tokens=args.max_target_length,
                beam_size=args.beam_size,
                num_return_sequences=args.num_return_sequences,
                generation_length_penalty=args.generation_length_penalty,
                length_prior_weight=args.length_prior_weight,
                repetition_penalty_weight=args.repetition_penalty_weight,
                batch_size=args.eval_batch_size,
                device=args.device,
            )
            eval_metrics = skeleton_metrics(eval_predictions, val_records)
            validation_row = {
                "epoch": float(epoch),
                **eval_metrics,
            }
            validation_history.append(validation_row)
            last_eval_predictions = eval_predictions
            last_eval_metrics = eval_metrics
            exact = float(eval_metrics["skeleton_exact_rate"])
            edit_similarity = float(eval_metrics["operation_sequence_edit_similarity"])
            improved = exact > best_exact or (
                math.isclose(exact, best_exact)
                and edit_similarity > best_edit_similarity
            )
            if improved:
                best_exact = exact
                best_edit_similarity = edit_similarity
                best_epoch = epoch
                best_metrics = dict(eval_metrics)
                best_predictions = eval_predictions
                if not args.no_restore_best:
                    best_model_state = clone_model_state_dict(model)
            print(
                f"[seq2seq skeleton val {epoch:03d}/{args.epochs:03d}] "
                f"exact={exact:.4f} edit_sim={edit_similarity:.4f} "
                f"len_err={eval_metrics['skeleton_length_error']:.2f} "
                f"best_epoch={best_epoch} best_exact={best_exact:.4f}",
                flush=True,
            )

    if not args.no_restore_best:
        if best_model_state is None or best_predictions is None or best_metrics is None:
            raise RuntimeError("Best-checkpoint restoration requested but no validation checkpoint was recorded.")
        model.load_state_dict(best_model_state)
        predictions = best_predictions
        metrics = dict(best_metrics)
        selected_epoch = best_epoch
        print(
            f"Restored best seq2seq skeleton checkpoint from epoch={best_epoch} "
            f"exact={best_exact:.4f} edit_sim={best_edit_similarity:.4f}",
            flush=True,
        )
    else:
        if last_eval_predictions is None or last_eval_metrics is None:
            raise RuntimeError("No final validation predictions were recorded.")
        predictions = last_eval_predictions
        metrics = dict(last_eval_metrics)
        selected_epoch = args.epochs

    metrics.update(
        {
            "train_records": len(train_records),
            "val_records": len(val_records),
            "epochs": args.epochs,
            "model_name": args.model_name,
            "random_init": args.random_init,
            "target_format": args.target_format,
            "include_numeric_evidence": include_numeric_evidence,
            "prompt_style": args.prompt_style,
            "prompt_augmentation": prompt_augmentation.enabled,
            "participant_shuffle_prob": prompt_augmentation.participant_shuffle_prob,
            "numeric_format_augment_prob": prompt_augmentation.numeric_format_prob,
            "field_mask_prob": prompt_augmentation.field_mask_prob,
            "max_input_length": args.max_input_length,
            "max_target_length": args.max_target_length,
            "action_loss_weighting": args.action_loss_weighting,
            "action_weight_beta": args.action_weight_beta,
            "action_weight_min": args.action_weight_min,
            "action_weight_max": args.action_weight_max,
            "length_loss_weight": args.length_loss_weight,
            "length_prior_weight": args.length_prior_weight,
            "repetition_penalty_weight": args.repetition_penalty_weight,
            "scheduler": "warmup_cosine",
            "warmup_ratio": args.warmup_ratio,
            "warmup_steps": warmup_steps,
            "total_optimizer_steps": total_optimizer_steps,
            "best_eval_interval": max(int(args.best_eval_interval), 1),
            "restore_best": not args.no_restore_best,
            "best_epoch": best_epoch,
            "best_skeleton_exact_rate": best_exact,
            "selected_epoch": selected_epoch,
            "beam_size": args.beam_size,
            "num_return_sequences": args.num_return_sequences,
            "checkpoint": args.checkpoint,
            "predictions": args.predictions,
            "final_train_loss": history[-1]["loss"] if history else None,
            "action_weight_summary": action_weight_summary,
            "validation_history": validation_history,
        }
    )

    checkpoint_path = Path(args.checkpoint)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "checkpoint_type": "seq2seq_skeleton",
            "model_name": args.model_name,
            "model_state": model.cpu().state_dict(),
            "codec": codec.to_dict(),
            "history": history,
            "validation_history": validation_history,
            "best_epoch": best_epoch,
            "best_skeleton_exact_rate": best_exact,
            "selected_epoch": selected_epoch,
            "action_tokens": action_tokens,
            "action_phrases": ACTION_PHRASES if args.target_format == "natural_text" else {},
            "action_weight_summary": action_weight_summary,
            "tokenizer_length": len(tokenizer),
            "config": {
                "max_steps": args.max_steps,
                "max_input_length": args.max_input_length,
                "max_target_length": args.max_target_length,
                "target_format": args.target_format,
                "include_numeric_evidence": include_numeric_evidence,
                "prompt_style": args.prompt_style,
                "prompt_augmentation": prompt_augmentation.enabled,
                "participant_shuffle_prob": prompt_augmentation.participant_shuffle_prob,
                "numeric_format_augment_prob": prompt_augmentation.numeric_format_prob,
                "field_mask_prob": prompt_augmentation.field_mask_prob,
                "action_loss_weighting": args.action_loss_weighting,
                "action_weight_beta": args.action_weight_beta,
                "action_weight_min": args.action_weight_min,
                "action_weight_max": args.action_weight_max,
                "length_loss_weight": args.length_loss_weight,
                "length_prior_weight": args.length_prior_weight,
                "repetition_penalty_weight": args.repetition_penalty_weight,
                "scheduler": "warmup_cosine",
                "warmup_ratio": args.warmup_ratio,
                "warmup_steps": warmup_steps,
                "total_optimizer_steps": total_optimizer_steps,
                "best_eval_interval": max(int(args.best_eval_interval), 1),
                "restore_best": not args.no_restore_best,
            },
        },
        checkpoint_path,
    )
    write_jsonl(args.predictions, predictions)
    metrics_path = Path(args.metrics)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(
        f"Seq2seq skeleton done: edit_sim={metrics['operation_sequence_edit_similarity']:.4f} "
        f"exact={metrics['skeleton_exact_rate']:.4f} len_err={metrics['skeleton_length_error']:.2f}",
        flush=True,
    )
    print(f"Wrote checkpoint to {args.checkpoint}", flush=True)
    print(f"Wrote predictions to {args.predictions}", flush=True)
    print(f"Wrote metrics to {args.metrics}", flush=True)


def clone_model_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }


def collate_rows(
    rows: list[dict[str, Any]],
    *,
    tokenizer: Any,
    max_input_length: int,
    max_target_length: int,
    device: str,
) -> dict[str, torch.Tensor]:
    prompts = [str(row["prompt"]) for row in rows]
    targets = [str(row["target"]) for row in rows]
    inputs = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=max_input_length,
        return_tensors="pt",
    )
    labels = tokenizer(
        text_target=targets,
        padding=True,
        truncation=True,
        max_length=max_target_length,
        return_tensors="pt",
    )["input_ids"]
    labels = labels.masked_fill(labels == tokenizer.pad_token_id, -100)
    return {
        "input_ids": inputs["input_ids"].to(device),
        "attention_mask": inputs["attention_mask"].to(device),
        "labels": labels.to(device),
        "lengths": torch.tensor([int(row["length"]) for row in rows], dtype=torch.long, device=device),
    }


def weighted_token_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    token_loss_weights: torch.Tensor,
) -> torch.Tensor:
    vocab_size = int(logits.size(-1))
    flat_labels = labels.reshape(-1)
    flat_loss = F.cross_entropy(
        logits.reshape(-1, vocab_size).float(),
        flat_labels,
        reduction="none",
        ignore_index=-100,
    )
    valid = flat_labels.ne(-100)
    safe_labels = flat_labels.clamp_min(0)
    weights = token_loss_weights.to(device=logits.device, dtype=flat_loss.dtype)[safe_labels]
    weights = weights * valid.to(weights.dtype)
    return (flat_loss * weights).sum() / weights.sum().clamp_min(1.0)


def build_action_loss_weights(
    *,
    tokenizer: Any,
    codec: GraphTargetCodec,
    action_tokens: dict[int, str],
    records: list[dict[str, Any]],
    weighting: str,
    beta: float,
    min_weight: float,
    max_weight: float,
) -> tuple[torch.Tensor | None, list[dict[str, float | int | str]]]:
    if weighting == "none":
        return None, []
    if weighting != "class_balanced":
        raise ValueError(f"Unsupported action-loss weighting: {weighting}")

    counts = {codec.action_vocab[action_id]: 0 for action_id in action_tokens}
    for record in records:
        for operation in reference_operations(record, max_steps=codec.max_steps):
            if operation in counts:
                counts[operation] += 1

    beta = max(0.0, min(float(beta), 0.999999))
    raw_weights: dict[int, float] = {}
    for action_id in action_tokens:
        action = codec.action_vocab[action_id]
        count = max(int(counts.get(action, 0)), 1)
        effective_count = (1.0 - beta**count) / max(1.0 - beta, 1e-12)
        raw_weights[action_id] = 1.0 / max(effective_count, 1e-12)

    count_total = sum(max(int(counts.get(codec.action_vocab[action_id], 0)), 1) for action_id in action_tokens)
    mean_raw = (
        sum(raw_weights[action_id] * max(int(counts.get(codec.action_vocab[action_id], 0)), 1) for action_id in action_tokens)
        / max(count_total, 1)
    )
    token_weights = torch.ones(len(tokenizer), dtype=torch.float32)
    summary: list[dict[str, float | int | str]] = []
    for action_id, raw_weight in raw_weights.items():
        action = codec.action_vocab[action_id]
        token = action_tokens[action_id]
        token_id = int(tokenizer.convert_tokens_to_ids(token))
        weight = raw_weight / max(mean_raw, 1e-12)
        weight = max(float(min_weight), min(float(max_weight), weight))
        token_weights[token_id] = float(weight)
        summary.append({"action": action, "count": int(counts.get(action, 0)), "weight": float(weight)})
    summary.sort(key=lambda item: (-float(item["weight"]), int(item["count"]), str(item["action"])))
    return token_weights, summary


@torch.no_grad()
def predict_records(
    model: SkeletonSeq2SeqModel,
    tokenizer: Any,
    codec: GraphTargetCodec,
    records: list[dict[str, Any]],
    *,
    target_format: str,
    action_token_ids: list[int],
    token_to_action_id: dict[int, int],
    include_numeric_evidence: bool,
    prompt_style: str,
    max_input_length: int,
    max_new_tokens: int,
    beam_size: int,
    num_return_sequences: int,
    generation_length_penalty: float,
    length_prior_weight: float,
    repetition_penalty_weight: float,
    batch_size: int,
    device: str,
) -> list[dict[str, Any]]:
    model = model.to(device)
    model.eval()
    rows: list[dict[str, Any]] = []
    return_count = max(1, min(int(num_return_sequences), max(int(beam_size), 1)))
    for batch_start in range(0, len(records), max(int(batch_size), 1)):
        batch_records = records[batch_start : batch_start + max(int(batch_size), 1)]
        prompts = [
            build_skeleton_prompt(
                record,
                include_numeric_evidence=include_numeric_evidence,
                prompt_style=prompt_style,
            )
            for record in batch_records
        ]
        inputs = tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=max_input_length,
            return_tensors="pt",
        )
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        if length_prior_weight:
            length_logits = model.length_logits(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            length_log_probs = torch.log_softmax(length_logits.float(), dim=-1).cpu()
        else:
            length_log_probs = torch.zeros(
                (len(batch_records), codec.max_steps + 1),
                dtype=torch.float32,
            )
        generation_kwargs: dict[str, Any] = {}
        if target_format == "special_tokens":
            generation_kwargs["prefix_allowed_tokens_fn"] = allowed_tokens_fn(
                action_token_ids=action_token_ids,
                eos_token_id=tokenizer.eos_token_id,
            )
        generated = model.base_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            num_beams=max(int(beam_size), 1),
            num_return_sequences=return_count,
            length_penalty=float(generation_length_penalty),
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            **generation_kwargs,
        )
        sequences = generated.sequences.detach().cpu()
        sequence_scores = getattr(generated, "sequences_scores", None)
        if sequence_scores is None:
            sequence_scores = torch.zeros(sequences.size(0), dtype=torch.float32)
        else:
            sequence_scores = sequence_scores.detach().cpu().float()
        decoded_texts = tokenizer.batch_decode(sequences, skip_special_tokens=True)
        for record_idx, record in enumerate(batch_records):
            candidates: list[dict[str, Any]] = []
            for return_idx in range(return_count):
                seq_idx = record_idx * return_count + return_idx
                token_ids = sequences[seq_idx].tolist()
                decoded_text = decoded_texts[seq_idx]
                if target_format == "special_tokens":
                    action_ids = extract_action_ids(
                        token_ids,
                        token_to_action_id=token_to_action_id,
                        codec=codec,
                    )
                    operations = [codec.action_vocab[action_id] for action_id in action_ids]
                elif target_format == "natural_text":
                    operations = parse_skeleton_target_text(
                        decoded_text,
                        max_steps=max(codec.max_steps - 1, 0),
                    )
                    action_ids = [
                        codec._id(codec.action_vocab, operation, codec.eos_id)
                        for operation in operations
                    ]
                else:
                    raise ValueError(f"Unsupported target format: {target_format}")
                length_idx = min(len(operations), codec.max_steps)
                sequence_score = float(sequence_scores[seq_idx])
                candidate_length_log_prob = float(length_log_probs[record_idx, length_idx])
                candidate_repetition_penalty = repetition_penalty(operations)
                score = sequence_score
                score += float(length_prior_weight) * candidate_length_log_prob
                score -= float(repetition_penalty_weight) * candidate_repetition_penalty
                if target_format == "natural_text" and not operations:
                    score = -math.inf
                candidates.append(
                    {
                        "score": score,
                        "sequence_score": sequence_score,
                        "length_log_prob": candidate_length_log_prob,
                        "repetition_penalty": candidate_repetition_penalty,
                        "operations": operations,
                        "ids": action_ids,
                        "raw_token_ids": token_ids,
                        "decoded_text": decoded_text,
                    }
                )
            if not candidates:
                candidates.append(
                    {
                        "score": -math.inf,
                        "sequence_score": -math.inf,
                        "length_log_prob": 0.0,
                        "repetition_penalty": 0.0,
                        "operations": [],
                        "ids": [],
                        "raw_token_ids": [],
                        "decoded_text": "",
                    }
                )
            candidates.sort(key=lambda item: float(item["score"]), reverse=True)
            best = candidates[0]
            rows.append(
                {
                    "index": record.get("index"),
                    "predicted_skeleton": best["operations"],
                    "predicted_ids": best["ids"],
                    "predicted_skeleton_text": best["decoded_text"],
                    "reference_skeleton": reference_operations(record, max_steps=codec.max_steps),
                    "seq2seq_score": best["score"],
                    "seq2seq_sequence_score": best["sequence_score"],
                    "seq2seq_length_log_prob": best["length_log_prob"],
                    "seq2seq_repetition_penalty": best["repetition_penalty"],
                }
            )
    return rows


def build_action_tokens(codec: GraphTargetCodec) -> dict[int, str]:
    tokens: dict[int, str] = {}
    for action_id, action in enumerate(codec.action_vocab):
        if action in {PAD_TOKEN, EOS_TOKEN}:
            continue
        tokens[action_id] = f"<OP_{action}>"
    return tokens


def clamp_probability(value: float) -> float:
    return max(0.0, min(float(value), 1.0))


def build_skeleton_prompt(
    record: dict[str, Any],
    *,
    include_numeric_evidence: bool,
    prompt_style: str,
    augmentation: PromptAugmentationConfig | None = None,
) -> str:
    if not _augmentation_enabled(augmentation):
        return build_encoder_prompt(
            record,
            prompt_style=prompt_style,
            include_numeric_evidence=include_numeric_evidence,
            numeric_evidence_include_source=False,
        )
    if prompt_style == "compact":
        return build_compact_skeleton_prompt(record, augmentation=augmentation)
    if prompt_style == "reactxt":
        return build_reactxt_skeleton_prompt(
            record,
            include_numeric_evidence=include_numeric_evidence,
            augmentation=augmentation,
        )
    raise ValueError(f"Unsupported prompt_style: {prompt_style}")


def build_compact_skeleton_prompt(
    record: dict[str, Any],
    *,
    augmentation: PromptAugmentationConfig | None = None,
) -> str:
    fields = _reactxt_prompt_fields(record, include_numeric_evidence=False)
    parts = ["TASK: Predict operation skeleton."]
    for label, values in zip(("REACTANT", "PRODUCT", "CATALYST", "SOLVENT"), fields[:4], strict=False):
        cleaned = [_compact_molecule_value(label, value) for value in values]
        cleaned = _augment_prompt_values(cleaned, augmentation, mask_token="<MASK_MOL>", shuffle=True)
        value_text = " | ".join(value for value in cleaned if value)
        if value_text:
            parts.append(f"{label}: {value_text}")
    temperature_text = _compact_mapping_values(record, "extracted_temperature", augmentation=augmentation)
    if temperature_text:
        parts.append(f"TEMPERATURE: {temperature_text}")
    duration_text = _compact_mapping_values(record, "extracted_duration", augmentation=augmentation)
    if duration_text:
        parts.append(f"DURATION: {duration_text}")
    return "\n".join(parts)


def _compact_molecule_value(label: str, value: Any) -> str:
    text = str(value).strip()
    prefix = f"{label}: "
    if text.startswith(prefix):
        text = text[len(prefix) :]
    text = text.replace("[START_SMILES]", "")
    text = text.replace("[END_SMILES]", "")
    return " ".join(text.split())


def _compact_mapping_values(
    record: dict[str, Any],
    mapping_name: str,
    *,
    augmentation: PromptAugmentationConfig | None = None,
) -> str:
    value_to_ref = record.get(mapping_name) or {}
    if not isinstance(value_to_ref, dict):
        return ""
    values = [
        f"{ref}:{_augment_numeric_text(str(value), augmentation)}"
        for value, ref in sorted(
            value_to_ref.items(),
            key=lambda item: _placeholder_sort_key(str(item[1])),
        )
        if str(ref).strip() or str(value).strip()
    ]
    values = _augment_prompt_values(values, augmentation, mask_token="<MASK_NUM>", shuffle=False)
    return " | ".join(values)


def build_reactxt_skeleton_prompt(
    record: dict[str, Any],
    *,
    include_numeric_evidence: bool,
    augmentation: PromptAugmentationConfig | None = None,
) -> str:
    labels = (
        "REACTANT",
        "PRODUCT",
        "CATALYST",
        "SOLVENT",
        "TEMPERATURE",
        "DURATION",
        "NUMERIC_EVIDENCE",
    )
    fields = _reactxt_prompt_fields(
        record,
        include_numeric_evidence=include_numeric_evidence,
    )
    parts = ["TASK: Predict the experimental operation skeleton."]
    for field_idx, (label, values) in enumerate(zip(labels, fields, strict=False)):
        if field_idx < 4:
            values = _augment_prompt_values(values, augmentation, mask_token="<MASK_MOL>", shuffle=True)
        else:
            values = [_augment_numeric_text(str(value), augmentation) for value in values]
            values = _augment_prompt_values(values, augmentation, mask_token="<MASK_NUM>", shuffle=False)
        value_text = " | ".join(str(value) for value in values if str(value).strip())
        parts.append(f"{label}: {value_text if value_text else '<EMPTY>'}")
    return "\n".join(parts)


NUMERIC_WITH_OPTIONAL_UNIT_RE = re.compile(
    r"(?<![A-Za-z0-9_$@#])"
    r"(?P<value>[-+]?\d+(?:[.,]\d+)?)"
    r"(?:\s*(?P<unit>ug|mg|g|kg|ul|uL|ml|mL|l|L|umol|mmol|mol|"
    r"h|hr|hrs|hour|hours|min|minute|minutes|c|C|degc|degC|"
    r"\u00b0\s*C|\u00b0C|percent|%))?"
    r"(?![A-Za-z0-9_$@#])",
    re.IGNORECASE,
)


def _augmentation_enabled(augmentation: PromptAugmentationConfig | None) -> bool:
    return bool(augmentation and augmentation.enabled)


def _augment_prompt_values(
    values: list[Any],
    augmentation: PromptAugmentationConfig | None,
    *,
    mask_token: str,
    shuffle: bool,
) -> list[str]:
    rendered = [str(value) for value in values]
    if not _augmentation_enabled(augmentation):
        return rendered
    if shuffle and len(rendered) > 1 and random.random() < augmentation.participant_shuffle_prob:
        random.shuffle(rendered)
    if augmentation.field_mask_prob > 0.0:
        rendered = [
            mask_token if random.random() < augmentation.field_mask_prob else value
            for value in rendered
        ]
    return rendered


def _augment_numeric_text(
    text: str,
    augmentation: PromptAugmentationConfig | None,
) -> str:
    if (
        not _augmentation_enabled(augmentation)
        or augmentation.numeric_format_prob <= 0.0
        or random.random() >= augmentation.numeric_format_prob
    ):
        return text

    def replace(match: re.Match[str]) -> str:
        raw_value = match.group("value")
        raw_unit = match.group("unit") or ""
        value_text = _numeric_value_variant(raw_value)
        unit_text = _unit_variant(raw_unit)
        if not raw_unit:
            return value_text
        separator = "" if unit_text.startswith("%") else " "
        return f"{value_text}{separator}{unit_text}".strip()

    return NUMERIC_WITH_OPTIONAL_UNIT_RE.sub(replace, text)


def _numeric_value_variant(raw_value: str) -> str:
    normalized = raw_value.replace(",", ".")
    try:
        value = float(normalized)
    except ValueError:
        return raw_value
    compact = f"{value:g}"
    variants = [compact]
    if "." not in raw_value and "," not in raw_value:
        variants.append(f"{value:.1f}")
    elif raw_value.endswith("0"):
        variants.append(compact)
    else:
        variants.append(f"{value:.2f}".rstrip("0").rstrip("."))
    return random.choice([variant for variant in variants if variant])


def _unit_variant(raw_unit: str) -> str:
    unit = " ".join(raw_unit.strip().split())
    if not unit:
        return ""
    key = unit.lower().replace(" ", "")
    variants_by_unit = {
        "ul": ["uL", "ul"],
        "ml": ["mL", "ml"],
        "l": ["L", "l"],
        "h": ["h", "hr", "hours"],
        "hr": ["h", "hr", "hours"],
        "hrs": ["h", "hr", "hours"],
        "hour": ["h", "hr", "hours"],
        "hours": ["h", "hr", "hours"],
        "min": ["min", "minutes"],
        "minute": ["min", "minutes"],
        "minutes": ["min", "minutes"],
        "c": ["C", "degC", "\u00b0C"],
        "degc": ["C", "degC", "\u00b0C"],
        "\u00b0c": ["C", "degC", "\u00b0C"],
        "%": ["%", "percent"],
        "percent": ["%", "percent"],
    }
    return random.choice(variants_by_unit.get(key, [unit]))


def reference_operations(record: dict[str, Any], *, max_steps: int) -> list[str]:
    return [
        step.operation_type
        for step in parse_action_sequence(str(record.get("actions") or ""))[: max(0, max_steps - 1)]
    ]


def skeleton_target_text(
    operations: list[str],
    *,
    target_format: str = "natural_text",
) -> str:
    if target_format == "natural_text":
        return "; ".join(ACTION_PHRASES[operation] for operation in operations)
    if target_format == "special_tokens":
        return "".join(f"<OP_{operation}>" for operation in operations)
    raise ValueError(f"Unsupported target format: {target_format}")


def parse_skeleton_target_text(text: str, *, max_steps: int) -> list[str]:
    normalized_text = re.sub(
        r"^\s*(?:operation\s+skeleton|operations|actions|skeleton)\s*:\s*",
        "",
        str(text),
        flags=re.IGNORECASE,
    )
    phrase_to_action = {
        normalize_action_phrase(phrase): action
        for action, phrase in ACTION_PHRASES.items()
    }
    phrase_to_action.update(
        {
            normalize_action_phrase(action): action
            for action in ACTION_PHRASES
        }
    )
    chunks = [
        normalize_action_phrase(chunk)
        for chunk in re.split(r"\s*(?:;|,|\||\n)\s*", normalized_text)
        if normalize_action_phrase(chunk)
    ]
    operations: list[str] = []
    for chunk in chunks:
        operation = phrase_to_action.get(chunk)
        if operation is None:
            return []
        operations.append(operation)
        if operation == "YIELD" or len(operations) >= max_steps:
            break
    return operations[:max_steps]


def normalize_action_phrase(text: str) -> str:
    normalized = re.sub(r"^\s*\d+\s*[.)-]\s*", "", str(text))
    normalized = normalized.strip().strip(".,:")
    return " ".join(normalized.lower().split())


def validate_action_tokenization(tokenizer: Any, action_tokens: dict[int, str]) -> None:
    for token in action_tokens.values():
        token_id = int(tokenizer.convert_tokens_to_ids(token))
        encoded = tokenizer(token, add_special_tokens=False).input_ids
        if encoded != [token_id]:
            raise ValueError(
                f"Action token must encode as a single token: {token!r} -> {encoded}, expected {[token_id]}"
            )
    sample_tokens = list(action_tokens.values())[: min(len(action_tokens), 4)]
    if len(sample_tokens) >= 2:
        expected = [int(tokenizer.convert_tokens_to_ids(token)) for token in sample_tokens]
        encoded = tokenizer("".join(sample_tokens), add_special_tokens=False).input_ids
        if encoded != expected:
            raise ValueError(
                "Concatenated action target must not insert separator tokens: "
                f"{sample_tokens!r} -> {encoded}, expected {expected}"
            )


def extract_action_ids(
    token_ids: list[int],
    *,
    token_to_action_id: dict[int, int],
    codec: GraphTargetCodec,
) -> list[int]:
    action_ids: list[int] = []
    yield_id = codec._id(codec.action_vocab, "YIELD", codec.eos_id)
    for token_id in token_ids:
        action_id = token_to_action_id.get(int(token_id))
        if action_id is None:
            continue
        action_ids.append(action_id)
        if action_id == yield_id:
            break
        if len(action_ids) >= codec.max_steps:
            break
    return action_ids[: codec.max_steps]


def allowed_tokens_fn(
    *,
    action_token_ids: list[int],
    eos_token_id: int | None,
):
    allowed_actions = [int(token_id) for token_id in action_token_ids]
    allowed_after_first = list(allowed_actions)
    if eos_token_id is not None:
        allowed_after_first.append(int(eos_token_id))

    def _allowed(_batch_id: int, input_ids: torch.Tensor) -> list[int]:
        generated_actions = sum(int(token_id) in allowed_actions for token_id in input_ids.tolist())
        return allowed_after_first if generated_actions > 0 else allowed_actions

    return _allowed


def repetition_penalty(operations: list[str]) -> float:
    penalty = 0.0
    run_length = 1
    previous = None
    for operation in operations:
        if operation == previous:
            run_length += 1
            if run_length > 3:
                penalty += float(run_length - 3)
        else:
            run_length = 1
            previous = operation

    bigram_counts: dict[tuple[str, str], int] = {}
    for left, right in zip(operations, operations[1:], strict=False):
        key = (left, right)
        bigram_counts[key] = bigram_counts.get(key, 0) + 1
        if bigram_counts[key] > 4:
            penalty += 0.5 * float(bigram_counts[key] - 4)
    return penalty


def infer_hidden_size(model: nn.Module) -> int:
    config = getattr(model, "config", None)
    for name in ("d_model", "hidden_size", "encoder_hidden_size"):
        value = getattr(config, name, None)
        if value is not None:
            return int(value)
    raise ValueError("Could not infer seq2seq hidden size from model config")


def skeleton_metrics(rows: list[dict[str, Any]], records: list[dict[str, Any]]) -> dict[str, float]:
    totals = {
        "operation_type_accuracy": 0.0,
        "operation_sequence_edit_similarity": 0.0,
        "operation_coverage": 0.0,
        "skeleton_length_error": 0.0,
        "skeleton_exact_rate": 0.0,
    }
    for row, record in zip(rows, records, strict=True):
        pred = list(row.get("predicted_skeleton") or [])
        ref = reference_operations(record, max_steps=max(len(pred), 1_000_000))
        max_len = max(len(pred), len(ref), 1)
        totals["operation_type_accuracy"] += sum(a == b for a, b in zip(pred, ref, strict=False)) / max_len
        totals["operation_sequence_edit_similarity"] += 1.0 - edit_distance(pred, ref) / max_len
        ref_counts = {op: ref.count(op) for op in set(ref)}
        pred_counts = {op: pred.count(op) for op in set(pred)}
        overlap = sum(min(pred_counts.get(op, 0), count) for op, count in ref_counts.items())
        totals["operation_coverage"] += overlap / max(len(ref), 1)
        totals["skeleton_length_error"] += abs(len(pred) - len(ref))
        totals["skeleton_exact_rate"] += float(pred == ref)
    count = len(rows)
    return {"count": float(count), **{key: value / max(count, 1) for key, value in totals.items()}}


if __name__ == "__main__":
    main()
