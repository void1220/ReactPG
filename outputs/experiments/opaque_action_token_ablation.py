"""Controlled ablation of action output representations for the small OpenExp split.

This is an experiment artifact, not production training code.  It compares:
1. the current Hugging Face default initialization for added action tokens;
2. the same one-token-per-action representation initialized from action-word embeddings;
3. natural-language action phrases using only the pretrained vocabulary.
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from reactgdiff.data.action_parser import parse_action_sequence
from reactgdiff.models.graph_codec import GraphTargetCodec
from reactgdiff.models.joint_diffusion import load_split_records
from train_skeleton_seq2seq import build_action_tokens, build_skeleton_prompt


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


def operations(record: dict[str, Any]) -> tuple[str, ...]:
    return tuple(step.operation_type for step in parse_action_sequence(str(record.get("actions") or ""))[:31])


class Rows(Dataset):
    def __init__(self, records: list[dict[str, Any]], target_style: str) -> None:
        self.records = records
        self.target_style = target_style

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        ops = operations(record)
        if self.target_style == "special":
            target = "".join(f"<OP_{op}>" for op in ops)
        else:
            target = "; ".join(ACTION_PHRASES[op] for op in ops)
        return {
            "prompt": build_skeleton_prompt(
                record,
                include_numeric_evidence=True,
                prompt_style="compact",
            ),
            "target": target,
            "operations": ops,
        }


def collate(rows: list[dict[str, Any]], tokenizer: Any, device: str) -> dict[str, Any]:
    inputs = tokenizer(
        [row["prompt"] for row in rows],
        padding=True,
        truncation=True,
        max_length=384,
        return_tensors="pt",
    )
    labels = tokenizer(
        text_target=[row["target"] for row in rows],
        padding=True,
        truncation=True,
        max_length=64,
        return_tensors="pt",
    )["input_ids"]
    labels = labels.masked_fill(labels == tokenizer.pad_token_id, -100)
    return {
        "input_ids": inputs["input_ids"].to(device),
        "attention_mask": inputs["attention_mask"].to(device),
        "labels": labels.to(device),
        "references": [row["operations"] for row in rows],
    }


def special_allowed(action_ids: list[int], eos_id: int):
    allowed = sorted(set(action_ids + [int(eos_id)]))

    def fn(_batch_id: int, _input_ids: torch.Tensor) -> list[int]:
        return allowed

    return fn


def parse_natural(text: str) -> tuple[str, ...]:
    phrase_to_action = {
        " ".join(phrase.lower().split()): action
        for action, phrase in ACTION_PHRASES.items()
    }
    chunks = [" ".join(chunk.strip().lower().split()) for chunk in text.split(";")]
    chunks = [chunk.strip(" .,:|") for chunk in chunks if chunk.strip(" .,:|")]
    parsed = []
    for chunk in chunks:
        action = phrase_to_action.get(chunk)
        if action is None:
            return ()
        parsed.append(action)
    return tuple(parsed)


def levenshtein(left: tuple[str, ...], right: tuple[str, ...]) -> int:
    previous = list(range(len(right) + 1))
    for i, lhs in enumerate(left, start=1):
        current = [i]
        for j, rhs in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + (lhs != rhs),
                )
            )
        previous = current
    return previous[-1]


@torch.inference_mode()
def evaluate(
    model: Any,
    tokenizer: Any,
    records: list[dict[str, Any]],
    *,
    target_style: str,
    action_token_ids: list[int],
    token_to_action: dict[int, str],
    device: str,
    batch_size: int,
) -> dict[str, float]:
    model.eval()
    loader = DataLoader(
        Rows(records, target_style),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda rows: collate(rows, tokenizer, device),
    )
    exact = 0
    valid = 0
    edit_sum = 0.0
    token_correct = 0
    token_total = 0
    teacher_exact = 0
    total = 0
    for batch in loader:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                return_dict=True,
            )
        predictions = output.logits.argmax(dim=-1)
        label_mask = batch["labels"].ne(-100)
        token_correct += int((predictions.eq(batch["labels"]) & label_mask).sum())
        token_total += int(label_mask.sum())
        teacher_exact += int(((predictions.eq(batch["labels"]) | ~label_mask).all(dim=1)).sum())

        generation_kwargs: dict[str, Any] = {}
        max_new_tokens = 12 if target_style == "special" else 48
        if target_style == "special":
            generation_kwargs["prefix_allowed_tokens_fn"] = special_allowed(
                action_token_ids,
                tokenizer.eos_token_id,
            )
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            generated = model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_new_tokens=max_new_tokens,
                num_beams=1,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                **generation_kwargs,
            )
        if target_style == "special":
            predicted_ops = [
                tuple(
                    token_to_action[int(token_id)]
                    for token_id in sequence.tolist()
                    if int(token_id) in token_to_action
                )
                for sequence in generated
            ]
        else:
            decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
            predicted_ops = [parse_natural(text) for text in decoded]
        for predicted, reference in zip(predicted_ops, batch["references"], strict=True):
            total += 1
            exact += int(predicted == reference)
            valid += int(bool(predicted) and all(op in ACTION_PHRASES for op in predicted))
            edit_sum += 1.0 - levenshtein(predicted, reference) / max(len(predicted), len(reference), 1)
    return {
        "n": total,
        "greedy_exact": exact / max(total, 1),
        "valid_rate": valid / max(total, 1),
        "edit_similarity": edit_sum / max(total, 1),
        "teacher_forced_token_accuracy": token_correct / max(token_total, 1),
        "teacher_forced_full_target_exact": teacher_exact / max(total, 1),
    }


def initialize_semantic_action_embeddings(
    model: Any,
    tokenizer: Any,
    action_tokens: dict[int, str],
    codec: GraphTargetCodec,
    *,
    match_existing_norm: bool = False,
) -> dict[str, float]:
    embeddings = model.get_input_embeddings().weight
    cosines = []
    with torch.no_grad():
        for action_id, special_token in action_tokens.items():
            action = codec.action_vocab[action_id]
            phrase_ids = tokenizer(
                ACTION_PHRASES[action],
                add_special_tokens=False,
            ).input_ids
            special_id = int(tokenizer.convert_tokens_to_ids(special_token))
            semantic_vector = embeddings[phrase_ids].mean(dim=0)
            if match_existing_norm:
                target_norm = embeddings[special_id].norm()
                semantic_vector = semantic_vector * (
                    target_norm / semantic_vector.norm().clamp_min(1e-12)
                )
            embeddings[special_id].copy_(semantic_vector)
            cosine = torch.nn.functional.cosine_similarity(
                embeddings[special_id].float(),
                semantic_vector.float(),
                dim=0,
            )
            cosines.append(float(cosine))
    model.tie_weights()
    return {
        "semantic_init_cosine_mean": sum(cosines) / len(cosines),
        "semantic_init_cosine_min": min(cosines),
    }


def initialization_alignment(
    model: Any,
    tokenizer: Any,
    action_tokens: dict[int, str],
    codec: GraphTargetCodec,
) -> dict[str, float]:
    embeddings = model.get_input_embeddings().weight.detach()
    cosines = []
    norms = []
    for action_id, special_token in action_tokens.items():
        action = codec.action_vocab[action_id]
        phrase_ids = tokenizer(ACTION_PHRASES[action], add_special_tokens=False).input_ids
        special_id = int(tokenizer.convert_tokens_to_ids(special_token))
        semantic_vector = embeddings[phrase_ids].mean(dim=0)
        cosines.append(
            float(
                torch.nn.functional.cosine_similarity(
                    embeddings[special_id].float(),
                    semantic_vector.float(),
                    dim=0,
                )
            )
        )
        norms.append(float(embeddings[special_id].float().norm()))
    return {
        "action_semantic_cosine_mean": sum(cosines) / len(cosines),
        "action_semantic_cosine_min": min(cosines),
        "action_embedding_norm_mean": sum(norms) / len(norms),
    }


def run_variant(
    name: str,
    train_records: list[dict[str, Any]],
    val_records: list[dict[str, Any]],
    codec: GraphTargetCodec,
    args: argparse.Namespace,
) -> dict[str, Any]:
    target_style = "natural" if name == "natural_text" else "special"
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    action_tokens = build_action_tokens(codec)
    if target_style == "special":
        tokenizer.add_special_tokens({"additional_special_tokens": list(action_tokens.values())})
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model, local_files_only=True)
    if target_style == "special":
        model.resize_token_embeddings(len(tokenizer))
    model.to(args.device)

    init_metrics: dict[str, float] = {}
    if name == "semantic_special":
        init_metrics.update(initialize_semantic_action_embeddings(model, tokenizer, action_tokens, codec))
    if name == "semantic_normmatched":
        init_metrics.update(
            initialize_semantic_action_embeddings(
                model,
                tokenizer,
                action_tokens,
                codec,
                match_existing_norm=True,
            )
        )
    if target_style == "special":
        init_metrics.update(initialization_alignment(model, tokenizer, action_tokens, codec))

    token_to_action = {
        int(tokenizer.convert_tokens_to_ids(token)): codec.action_vocab[action_id]
        for action_id, token in action_tokens.items()
    }
    action_token_ids = sorted(token_to_action)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        Rows(train_records, target_style),
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=lambda rows: collate(rows, tokenizer, args.device),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.01,
    )

    history: list[dict[str, Any]] = []
    print(json.dumps({"event": "variant_start", "variant": name, **init_metrics}), flush=True)
    epoch0 = evaluate(
        model,
        tokenizer,
        val_records,
        target_style=target_style,
        action_token_ids=action_token_ids,
        token_to_action=token_to_action,
        device=args.device,
        batch_size=args.eval_batch_size,
    )
    history.append({"epoch": 0, **epoch0})
    print(json.dumps({"event": "eval", "variant": name, "epoch": 0, **epoch0}), flush=True)

    optimizer.zero_grad(set_to_none=True)
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        seen = 0
        for batch_index, batch in enumerate(loader, start=1):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    return_dict=True,
                )
                loss = output.loss / args.gradient_accumulation
            loss.backward()
            if batch_index % args.gradient_accumulation == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            batch_n = int(batch["input_ids"].size(0))
            seen += batch_n
            loss_sum += float(output.loss.detach()) * batch_n
        if len(loader) % args.gradient_accumulation:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        metrics = evaluate(
            model,
            tokenizer,
            val_records,
            target_style=target_style,
            action_token_ids=action_token_ids,
            token_to_action=token_to_action,
            device=args.device,
            batch_size=args.eval_batch_size,
        )
        row = {
            "epoch": epoch,
            "train_loss": loss_sum / max(seen, 1),
            "elapsed_seconds": time.time() - started,
            **metrics,
        }
        history.append(row)
        print(json.dumps({"event": "eval", "variant": name, **row}), flush=True)

    result = {
        "variant": name,
        "target_style": target_style,
        "initialization": init_metrics,
        "history": history,
        "best_greedy_exact": max(row["greedy_exact"] for row in history),
        "final": history[-1],
    }
    del optimizer, model, tokenizer, loader
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/home/void/models/molt5-base")
    parser.add_argument("--train", default="outputs/prepared_splits/openexp/scale_small/train.jsonl")
    parser.add_argument("--val", default="outputs/prepared_splits/openexp/scale_small/val.jsonl")
    parser.add_argument("--train-size", type=int, default=4096)
    parser.add_argument("--val-size", type=int, default=512)
    parser.add_argument("--top-templates", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["current_special", "semantic_special", "natural_text"],
        choices=["current_special", "semantic_special", "semantic_normmatched", "natural_text"],
    )
    parser.add_argument("--output", default="outputs/experiments/opaque_action_token_ablation.json")
    args = parser.parse_args()

    all_train = load_split_records(args.train)
    all_val = load_split_records(args.val)
    codec = GraphTargetCodec.fit(
        all_train,
        max_steps=32,
        max_material_refs=16,
        max_material_slots=4,
    )
    counts = Counter(operations(record) for record in all_train)
    top_templates = {template for template, _count in counts.most_common(args.top_templates)}
    train_pool = [record for record in all_train if operations(record) in top_templates]
    val_pool = [record for record in all_val if operations(record) in top_templates]
    rng = random.Random(args.seed)
    rng.shuffle(train_pool)
    rng.shuffle(val_pool)
    train_records = train_pool[: args.train_size]
    val_records = val_pool[: args.val_size]
    if len(train_records) < args.train_size or len(val_records) < args.val_size:
        raise ValueError(
            f"Insufficient diagnostic pool: train={len(train_records)}, val={len(val_records)}"
        )

    train_template_counts = Counter(operations(record) for record in train_records)
    val_template_counts = Counter(operations(record) for record in val_records)
    majority_template, majority_count = train_template_counts.most_common(1)[0]
    majority_val_exact = sum(operations(record) == majority_template for record in val_records) / len(val_records)
    metadata = {
        "model": args.model,
        "seed": args.seed,
        "train_size": len(train_records),
        "val_size": len(val_records),
        "top_templates": args.top_templates,
        "train_unique_templates": len(train_template_counts),
        "val_unique_templates": len(val_template_counts),
        "val_templates_seen_in_diagnostic_train": sum(
            template in train_template_counts for template in val_template_counts
        )
        / max(len(val_template_counts), 1),
        "val_records_seen_template_rate": sum(
            operations(record) in train_template_counts for record in val_records
        )
        / len(val_records),
        "majority_template": list(majority_template),
        "majority_train_rate": majority_count / len(train_records),
        "majority_val_exact": majority_val_exact,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "gradient_accumulation": args.gradient_accumulation,
        "learning_rate": args.learning_rate,
        "loss": "plain pretrained-token cross entropy; no class weighting or length head",
        "decoding": "greedy; action-token vocabulary constrained only for special-token variants",
    }
    print(json.dumps({"event": "setup", **metadata}), flush=True)

    results = []
    for variant in args.variants:
        results.append(run_variant(variant, train_records, val_records, codec, args))

    payload = {"metadata": metadata, "results": results}
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"event": "done", "output": str(output_path)}), flush=True)


if __name__ == "__main__":
    main()
