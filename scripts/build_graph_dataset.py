"""Build OpenExp-derived heterogeneous process graphs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reactgdiff.data.graph_builder import build_process_graph
from reactgdiff.data.openexp_loader import iter_openexp_records
from reactgdiff.utils.io import write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/raw/OpenExp.json", help="OpenExp JSON path.")
    parser.add_argument(
        "--output",
        default="data/graphs/openexp_hetero_graphs.jsonl",
        help="Output graph JSONL path.",
    )
    parser.add_argument("--limit", type=int, default=100, help="Maximum records to convert.")
    parser.add_argument("--offset", type=int, default=0, help="Record offset before conversion.")
    args = parser.parse_args()

    records = iter_openexp_records(args.input, limit=args.limit, offset=args.offset)
    count = write_jsonl(args.output, (build_process_graph(record) for record in records))
    print(f"Wrote {count} graph records to {args.output}")


if __name__ == "__main__":
    main()
