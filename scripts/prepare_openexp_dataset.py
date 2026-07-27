"""Clean and bucket OpenExp for ReactGDiff experiments."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TextIO

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reactgdiff.data.openexp_loader import iter_openexp_records
from reactgdiff.data.openexp_preprocess import SummaryBuilder, prepare_openexp_record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/raw/OpenExp.json", help="Raw OpenExp JSON path.")
    parser.add_argument("--output-dir", default="data/processed/openexp", help="Output directory.")
    parser.add_argument("--limit", type=int, default=None, help="Optional maximum records.")
    parser.add_argument("--offset", type=int, default=0, help="Optional record offset.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    splits_dir = output_dir / "splits"
    buckets_dir = output_dir / "buckets"
    output_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)
    buckets_dir.mkdir(parents=True, exist_ok=True)

    summary = SummaryBuilder()
    handles = open_output_handles(output_dir, splits_dir, buckets_dir)
    try:
        for record in iter_openexp_records(args.input, limit=args.limit, offset=args.offset):
            prepared = prepare_openexp_record(record)
            summary.add_prepared(prepared)
            row = prepared.to_dict()

            if prepared.quality["is_clean"]:
                write_row(handles["main"], row)
                write_row(handles[f"split:{prepared.split}"], row)
                write_row(handles[f"scale:{prepared.buckets['scale']}"], row)
                for bucket_name, enabled in prepared.buckets["special"].items():
                    if enabled:
                        write_row(handles[f"bucket:{bucket_name}"], row)
            else:
                write_row(handles["rejected"], row)
    finally:
        for handle in handles.values():
            handle.close()

    report_path = output_dir / "summary.json"
    report_path.write_text(
        json.dumps(summary.to_dict(), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"Wrote cleaned OpenExp datasets to {output_dir}")
    print(f"Wrote summary to {report_path}")


def open_output_handles(
    output_dir: Path,
    splits_dir: Path,
    buckets_dir: Path,
) -> dict[str, TextIO]:
    paths = {
        "main": output_dir / "main.jsonl",
        "rejected": output_dir / "rejected.jsonl",
        "split:train": splits_dir / "train.jsonl",
        "split:val": splits_dir / "val.jsonl",
        "split:test": splits_dir / "test.jsonl",
        "scale:small": buckets_dir / "scale_small.jsonl",
        "scale:medium": buckets_dir / "scale_medium.jsonl",
        "scale:large": buckets_dir / "scale_large.jsonl",
        "bucket:numeric_heavy": buckets_dir / "numeric_heavy.jsonl",
        "bucket:condition_heavy": buckets_dir / "condition_heavy.jsonl",
        "bucket:hard_numeric_condition": buckets_dir / "hard_numeric_condition.jsonl",
        "bucket:multi_reference": buckets_dir / "multi_reference.jsonl",
        "bucket:branch_workup": buckets_dir / "branch_workup.jsonl",
        "bucket:complex_overall": buckets_dir / "complex_overall.jsonl",
    }
    return {name: path.open("w", encoding="utf-8") for name, path in paths.items()}


def write_row(handle: TextIO, row: dict) -> None:
    handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
    handle.write("\n")


if __name__ == "__main__":
    main()
