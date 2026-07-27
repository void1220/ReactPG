"""Diagnose whether graph diffusion learns the reverse discrete graph process."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reactgdiff.eval.slots import discrete_slot_metrics
from reactgdiff.models.joint_diffusion import ReactGDiffFeaturizer, load_split_records
from reactgdiff.models.procedure_graph_diffusion import (
    ProcedureGraphDiffusion,
    corrupt_categorical,
    load_procedure_graph_diffusion_checkpoint,
    predict_procedure_graph_diffusion_records,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True, help="JSONL split used for diagnostics.")
    parser.add_argument("--limit", type=int, default=128)
    parser.add_argument("--output", default=None, help="Optional JSON metrics path.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument(
        "--timesteps",
        default="0,1,2,5,10,16,32",
        help="Comma-separated teacher-forced timesteps to evaluate.",
    )
    parser.add_argument(
        "--sample-steps",
        default="1,2,4,8,10,20,32",
        help="Comma-separated reverse-chain step counts to evaluate.",
    )
    parser.add_argument("--sample-batch-size", type=int, default=64)
    parser.add_argument(
        "--sample-mode",
        choices=("argmax", "sample", "sample_argmax_final"),
        default="argmax",
    )
    parser.add_argument(
        "--sampler",
        choices=("posterior", "single_step", "iterative"),
        default="posterior",
    )
    parser.add_argument("--quantity-gate-threshold", type=float, default=0.65)
    parser.add_argument("--condition-probability-threshold", type=float, default=0.05)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    records = load_split_records(args.data, limit=args.limit)
    model, codec, condition_featurizer, _ = load_procedure_graph_diffusion_checkpoint(
        args.checkpoint,
        device=args.device,
    )
    model = model.to(args.device).eval()
    featurizer = ReactGDiffFeaturizer.from_dict(condition_featurizer)
    condition_vectors = [featurizer.condition_vector(record) for record in records]
    timesteps = _parse_ints(args.timesteps)
    sample_steps = _parse_ints(args.sample_steps)

    report: dict[str, Any] = {
        "checkpoint": args.checkpoint,
        "data": args.data,
        "records": len(records),
        "model_diffusion_steps": model.diffusion_steps,
        "model_diffuse_quantities": bool(getattr(model, "diffuse_quantities", True)),
        "teacher_forced_denoising": teacher_forced_denoising(
            model,
            codec,
            records,
            condition_vectors,
            timesteps=timesteps,
            seed=args.seed,
            device=args.device,
        ),
        "sampling_sweep": sampling_sweep(
            model,
            codec,
            records,
            condition_vectors,
            sample_steps=sample_steps,
            sampler=args.sampler,
            sample_mode=args.sample_mode,
            sample_batch_size=args.sample_batch_size,
            seed=args.seed,
            quantity_gate_threshold=args.quantity_gate_threshold,
            condition_probability_threshold=args.condition_probability_threshold,
            device=args.device,
        ),
    }

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    print_summary(report)


@torch.no_grad()
def teacher_forced_denoising(
    model: ProcedureGraphDiffusion,
    codec,
    records: list[dict[str, Any]],
    condition_vectors: list[list[float]],
    *,
    timesteps: list[int],
    seed: int,
    device: str | torch.device,
) -> list[dict[str, float]]:
    tensors = codec.encode_records(records, condition_vectors, device=device)
    (
        condition,
        op_ids,
        material_ids,
        condition_ids,
        quantity_gate_ids,
        unit_ids,
        _quantity_values,
        _quantity_value_masks,
        _condition_values,
        _condition_value_masks,
        slot_mask,
    ) = tensors
    active_steps = slot_mask.bool()
    active_material_slots = active_steps.unsqueeze(-1).expand_as(material_ids)
    rows: list[dict[str, float]] = []
    for raw_timestep in timesteps:
        timestep_value = max(0, min(int(raw_timestep), int(model.diffusion_steps)))
        generator_seed = int(seed) + timestep_value * 9973
        torch.manual_seed(generator_seed)
        timestep = torch.full(
            (condition.size(0),),
            timestep_value,
            dtype=torch.long,
            device=condition.device,
        )
        noise_probability = model.noise_probability(timestep)
        noisy_op = corrupt_categorical(
            op_ids,
            torch.ones_like(op_ids, dtype=torch.bool),
            model.action_marginal,
            noise_probability,
        )
        noisy_material = corrupt_categorical(
            material_ids,
            active_material_slots,
            model.material_marginal,
            noise_probability,
        )
        noisy_condition = corrupt_categorical(
            condition_ids,
            active_steps,
            model.condition_marginal,
            noise_probability,
        )
        if getattr(model, "diffuse_quantities", True):
            noisy_gate = corrupt_categorical(
                quantity_gate_ids,
                active_material_slots,
                model.quantity_gate_marginal,
                noise_probability,
            )
            noisy_unit = corrupt_categorical(
                unit_ids,
                active_material_slots,
                model.unit_marginal,
                noise_probability,
            )
        else:
            noisy_gate = torch.zeros_like(quantity_gate_ids)
            noisy_unit = torch.zeros_like(unit_ids)
        output = model(
            condition,
            timestep,
            noisy_op,
            noisy_material,
            noisy_condition,
            noisy_gate,
            noisy_unit,
        )
        slot_output = output.slot_output
        rows.append(
            {
                "timestep": float(timestep_value),
                "noise_probability": float(noise_probability.mean()),
                "input_operation_accuracy": _accuracy(noisy_op, op_ids, active_steps),
                "input_material_accuracy": _accuracy(noisy_material, material_ids, active_material_slots),
                "input_condition_accuracy": _accuracy(noisy_condition, condition_ids, active_steps),
                "input_quantity_gate_accuracy": _accuracy(
                    noisy_gate,
                    quantity_gate_ids,
                    active_material_slots,
                ),
                "input_unit_accuracy": _accuracy(noisy_unit, unit_ids, active_material_slots),
                "pred_operation_accuracy": _accuracy(slot_output.op_logits.argmax(dim=-1), op_ids, active_steps),
                "pred_material_accuracy": _accuracy(
                    slot_output.material_logits.argmax(dim=-1),
                    material_ids,
                    active_material_slots,
                ),
                "pred_condition_accuracy": _accuracy(
                    slot_output.condition_logits.argmax(dim=-1),
                    condition_ids,
                    active_steps,
                ),
                "pred_quantity_gate_accuracy": _accuracy(
                    slot_output.quantity_gate_logits.argmax(dim=-1),
                    quantity_gate_ids,
                    active_material_slots,
                ),
                "pred_unit_accuracy": _accuracy(
                    slot_output.unit_logits.argmax(dim=-1),
                    unit_ids,
                    active_material_slots,
                ),
            }
        )
    return rows


@torch.no_grad()
def sampling_sweep(
    model: ProcedureGraphDiffusion,
    codec,
    records: list[dict[str, Any]],
    condition_vectors: list[list[float]],
    *,
    sample_steps: list[int],
    sampler: str,
    sample_mode: str,
    sample_batch_size: int,
    seed: int,
    quantity_gate_threshold: float,
    condition_probability_threshold: float,
    device: str | torch.device,
) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for raw_steps in sample_steps:
        effective_steps = max(1, min(int(raw_steps), int(model.diffusion_steps)))
        predictions = predict_procedure_graph_diffusion_records(
            model,
            codec,
            records,
            condition_vectors=condition_vectors,
            include_generated_graph=False,
            quantity_gate_threshold=quantity_gate_threshold,
            condition_probability_threshold=condition_probability_threshold,
            sample_steps=effective_steps,
            sample_mode=sample_mode,
            sample_batch_size=sample_batch_size,
            sampler=sampler,
            seed=seed,
            device=device,
        )
        metrics = discrete_slot_metrics(predictions, records, codec=codec)
        rows.append(
            {
                "requested_sample_steps": float(raw_steps),
                "effective_sample_steps": float(effective_steps),
                "discrete_slot_score": metrics["discrete_slot_score"],
                "operation_sequence_similarity": metrics["operation_sequence_similarity"],
                "operation_slot_accuracy": metrics["operation_slot_accuracy"],
                "material_slot_accuracy": metrics["material_slot_accuracy"],
                "condition_slot_accuracy": metrics["condition_slot_accuracy"],
                "numeric_slot_accuracy": metrics["numeric_slot_accuracy"],
                "unit_slot_accuracy": metrics["unit_slot_accuracy"],
                "length_accuracy": metrics["length_accuracy"],
                "absolute_length_error": metrics["absolute_length_error"],
                "discrete_graph_exact_rate": metrics["discrete_graph_exact_rate"],
            }
        )
    return rows


def print_summary(report: dict[str, Any]) -> None:
    print(
        "checkpoint="
        f"{report['checkpoint']} records={report['records']} "
        f"T={report['model_diffusion_steps']} "
        f"diffuse_quantities={report['model_diffuse_quantities']}"
    )
    print("teacher_forced_denoising:")
    for row in report["teacher_forced_denoising"]:
        print(
            "  "
            f"t={int(row['timestep']):>3} noise={row['noise_probability']:.3f} "
            f"in_op={row['input_operation_accuracy']:.3f} "
            f"pred_op={row['pred_operation_accuracy']:.3f} "
            f"in_mat={row['input_material_accuracy']:.3f} "
            f"pred_mat={row['pred_material_accuracy']:.3f} "
            f"in_cond={row['input_condition_accuracy']:.3f} "
            f"pred_cond={row['pred_condition_accuracy']:.3f}"
        )
    print("sampling_sweep:")
    for row in report["sampling_sweep"]:
        print(
            "  "
            f"steps={int(row['requested_sample_steps']):>3}"
            f"/{int(row['effective_sample_steps']):>3} "
            f"score={row['discrete_slot_score']:.4f} "
            f"opseq={row['operation_sequence_similarity']:.4f} "
            f"op={row['operation_slot_accuracy']:.4f} "
            f"mat={row['material_slot_accuracy']:.4f} "
            f"cond={row['condition_slot_accuracy']:.4f} "
            f"num={row['numeric_slot_accuracy']:.4f} "
            f"len_err={row['absolute_length_error']:.2f}"
        )


def _accuracy(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    mask = mask.bool()
    if not bool(mask.any()):
        return 0.0
    return float((prediction[mask] == target[mask]).float().mean())


def _parse_ints(raw: str) -> list[int]:
    values: list[int] = []
    for part in str(raw).split(","):
        part = part.strip()
        if not part:
            continue
        values.append(int(part))
    return values


if __name__ == "__main__":
    main()
