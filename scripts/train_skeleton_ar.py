"""Train a small autoregressive operation-skeleton predictor."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reactgdiff.data.action_parser import parse_action_sequence
from reactgdiff.eval.lev import edit_distance
from reactgdiff.models.graph_codec import GraphTargetCodec
from reactgdiff.models.joint_diffusion import ReactGDiffFeaturizer, load_split_records
from reactgdiff.utils.io import write_jsonl


class ARSkeletonModel(nn.Module):
    def __init__(
        self,
        *,
        condition_dim: int,
        action_dim: int,
        hidden_dim: int,
        layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.condition_dim = condition_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.layers = layers
        self.dropout = dropout
        self.embedding = nn.Embedding(action_dim, hidden_dim)
        self.condition_to_hidden = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim * layers),
        )
        self.gru = nn.GRU(
            hidden_dim,
            hidden_dim,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.output = nn.Linear(hidden_dim, action_dim)

    def forward(self, condition: torch.Tensor, decoder_input: torch.Tensor) -> torch.Tensor:
        hidden = self.condition_to_hidden(condition).view(
            condition.size(0),
            self.layers,
            self.hidden_dim,
        ).transpose(0, 1).contiguous()
        embedded = self.embedding(decoder_input)
        output, _ = self.gru(embedded, hidden)
        return self.output(output)

    def config(self) -> dict[str, Any]:
        return {
            "condition_dim": self.condition_dim,
            "action_dim": self.action_dim,
            "hidden_dim": self.hidden_dim,
            "layers": self.layers,
            "dropout": self.dropout,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", default="data/processed/openexp_sample/splits/train.jsonl")
    parser.add_argument("--val", default="data/processed/openexp_sample/splits/val.jsonl")
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--val-limit", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-steps", type=int, default=32)
    parser.add_argument("--max-material-refs", type=int, default=16)
    parser.add_argument("--max-material-slots", type=int, default=4)
    parser.add_argument("--condition-encoding", choices=("reactxt_hash", "field_hash", "scalar_hash"), default="reactxt_hash")
    parser.add_argument("--field-dim", type=int, default=64)
    parser.add_argument("--ngram-min", type=int, default=2)
    parser.add_argument("--ngram-max", type=int, default=5)
    parser.add_argument("--no-numeric-evidence-input", action="store_true")
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--checkpoint", default="outputs/checkpoints/openexp_sample_ar_skeleton.pt")
    parser.add_argument("--predictions", default="outputs/skeleton/openexp_sample_ar_skeleton_val.jsonl")
    parser.add_argument("--metrics", default="outputs/metrics/openexp_sample_ar_skeleton_val.json")
    parser.add_argument("--log-every", type=int, default=1)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    print(
        "AR skeleton 准备阶段：开始读取数据 "
        f"train={args.train} val={args.val} "
        f"train_limit={args.train_limit} val_limit={args.val_limit}",
        flush=True,
    )
    train_records = load_split_records(args.train, limit=args.train_limit)
    val_records = load_split_records(args.val, limit=args.val_limit)
    print(
        "AR skeleton 准备阶段：数据读取完成 "
        f"训练集={len(train_records)} 验证集={len(val_records)} "
        f"设备={args.device} seed={args.seed}",
        flush=True,
    )
    include_numeric_evidence = not args.no_numeric_evidence_input
    print(
        "AR skeleton 准备阶段：开始构建条件 featurizer "
        f"encoding={args.condition_encoding} field_dim={args.field_dim} "
        f"numeric_evidence={'启用' if include_numeric_evidence else '关闭'}",
        flush=True,
    )
    featurizer = ReactGDiffFeaturizer.fit(
        train_records,
        condition_encoding=args.condition_encoding,
        field_dim=args.field_dim,
        ngram_min=args.ngram_min,
        ngram_max=args.ngram_max,
        include_numeric_evidence=include_numeric_evidence,
    )
    print(
        "AR skeleton 准备阶段：条件 featurizer 完成 "
        f"condition_dim={featurizer.condition_dim}",
        flush=True,
    )
    print(
        "AR skeleton 准备阶段：开始解析动作并构建骨架 codec",
        flush=True,
    )
    codec = GraphTargetCodec.fit(
        train_records,
        max_steps=args.max_steps,
        max_material_refs=args.max_material_refs,
        max_material_slots=args.max_material_slots,
    )
    print(
        "AR skeleton 准备阶段：骨架 codec 完成 "
        f"action_dim={codec.action_dim} max_steps={codec.max_steps}",
        flush=True,
    )
    model = ARSkeletonModel(
        condition_dim=featurizer.condition_dim,
        action_dim=codec.action_dim,
        hidden_dim=args.hidden_dim,
        layers=args.layers,
        dropout=args.dropout,
    ).to(args.device)

    print(
        "AR skeleton 准备阶段：开始张量化训练样本 "
        f"records={len(train_records)} condition_dim={featurizer.condition_dim}",
        flush=True,
    )
    train_conditions = torch.tensor(
        [featurizer.condition_vector(record) for record in train_records],
        dtype=torch.float32,
        device=args.device,
    )
    target_ids = torch.tensor(
        [codec.skeleton_ids_from_record(record) for record in train_records],
        dtype=torch.long,
        device=args.device,
    )
    bos_column = torch.full(
        (target_ids.size(0), 1),
        codec.eos_id,
        dtype=torch.long,
        device=args.device,
    )
    decoder_input = torch.cat((bos_column, target_ids[:, :-1]), dim=1)
    dataset = TensorDataset(train_conditions, decoder_input, target_ids)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    print(
        "AR skeleton 准备阶段完成，开始训练："
        f"epochs={args.epochs} batch={args.batch_size} lr={args.learning_rate:g} "
        f"hidden={args.hidden_dim} layers={args.layers}",
        flush=True,
    )
    history: list[dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_tokens = 0
        for condition, decoder_tokens, targets in loader:
            logits = model(condition, decoder_tokens)
            flat_logits = logits.reshape(-1, logits.size(-1))
            flat_targets = targets.reshape(-1)
            loss = F.cross_entropy(flat_logits, flat_targets, ignore_index=codec.pad_id)
            optimizer.zero_grad()
            loss.backward()
            grad_norm = (
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip_norm)
                if args.gradient_clip_norm > 0
                else torch.tensor(0.0)
            )
            optimizer.step()
            active = (flat_targets != codec.pad_id).sum().item()
            total_loss += float(loss.detach()) * active
            total_tokens += active
        row = {
            "epoch": float(epoch),
            "loss": total_loss / max(total_tokens, 1),
            "tokens": float(total_tokens),
        }
        history.append(row)
        if args.log_every and (epoch == 1 or epoch == args.epochs or epoch % args.log_every == 0):
            print(
                f"[skeleton {epoch:03d}/{args.epochs:03d}] "
                f"loss={row['loss']:.4f} tokens={int(row['tokens'])}",
                flush=True,
            )

    predictions = predict_records(
        model,
        codec,
        featurizer,
        val_records,
        device=args.device,
    )
    metrics = skeleton_metrics(predictions, val_records)
    metrics.update(
        {
            "train_records": len(train_records),
            "val_records": len(val_records),
            "epochs": args.epochs,
            "condition_dim": featurizer.condition_dim,
            "condition_encoding": args.condition_encoding,
            "include_numeric_evidence": include_numeric_evidence,
            "field_dim": args.field_dim,
            "checkpoint": args.checkpoint,
            "predictions": args.predictions,
            "final_train_loss": history[-1]["loss"] if history else None,
        }
    )

    checkpoint_path = Path(args.checkpoint)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "checkpoint_type": "ar_skeleton",
            "model_config": model.config(),
            "model_state": model.cpu().state_dict(),
            "codec": codec.to_dict(),
            "condition_featurizer": featurizer.to_dict(),
            "history": history,
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
        f"AR skeleton done: edit_sim={metrics['operation_sequence_edit_similarity']:.4f} "
        f"exact={metrics['skeleton_exact_rate']:.4f} len_err={metrics['skeleton_length_error']:.2f}"
    )
    print(f"Wrote checkpoint to {args.checkpoint}")
    print(f"Wrote predictions to {args.predictions}")
    print(f"Wrote metrics to {args.metrics}")


@torch.no_grad()
def predict_records(
    model: ARSkeletonModel,
    codec: GraphTargetCodec,
    featurizer: ReactGDiffFeaturizer,
    records: list[dict[str, Any]],
    *,
    device: str,
) -> list[dict[str, Any]]:
    model = model.to(device)
    model.eval()
    rows = []
    for record in records:
        condition = torch.tensor([featurizer.condition_vector(record)], dtype=torch.float32, device=device)
        token = torch.tensor([[codec.eos_id]], dtype=torch.long, device=device)
        hidden = model.condition_to_hidden(condition).view(1, model.layers, model.hidden_dim).transpose(0, 1).contiguous()
        ops: list[str] = []
        ids: list[int] = []
        for _ in range(codec.max_steps):
            embedded = model.embedding(token[:, -1:])
            output, hidden = model.gru(embedded, hidden)
            logits = model.output(output[:, -1])
            logits[:, codec.pad_id] = -1e9
            next_id = int(logits.argmax(dim=-1).item())
            if next_id == codec.eos_id:
                break
            ids.append(next_id)
            ops.append(codec.action_vocab[next_id])
            token = torch.cat((token, torch.tensor([[next_id]], dtype=torch.long, device=device)), dim=1)
            if ops and ops[-1] == "YIELD":
                break
        rows.append(
            {
                "index": record.get("index"),
                "predicted_skeleton": ops,
                "predicted_ids": ids,
                "reference_skeleton": [
                    step.operation_type for step in parse_action_sequence(str(record.get("actions") or ""))
                ],
            }
        )
    return rows


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
        ref = [step.operation_type for step in parse_action_sequence(str(record.get("actions") or ""))]
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
