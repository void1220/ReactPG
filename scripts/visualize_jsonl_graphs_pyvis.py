"""Generate standalone HTML views for ReactGDiff JSONL graphs.

The old project used a PyVis-style graph browser. This replacement keeps the
same purpose and filename but avoids external runtime dependencies by writing
static SVG-based HTML files.
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reactgdiff.utils.io import read_jsonl

NODE_COLORS = {
    "operation": ("#2563eb", "#eff6ff"),
    "material": ("#059669", "#ecfdf5"),
    "state": ("#7c3aed", "#f5f3ff"),
    "condition": ("#d97706", "#fffbeb"),
    "container": ("#64748b", "#f8fafc"),
    "safety_control": ("#dc2626", "#fef2f2"),
}

EDGE_COLORS = {
    "input_to": "#0f766e",
    "output_from": "#6d28d9",
    "precede": "#334155",
    "refer_to": "#0891b2",
    "has_condition": "#b45309",
    "located_in": "#475569",
    "requires_control": "#b91c1c",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="data/graphs/openexp_hetero_graphs.jsonl",
        help="Graph JSONL path.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/figures/graph_views",
        help="Directory for generated HTML pages.",
    )
    parser.add_argument("--limit", type=int, default=100, help="Maximum graphs to render.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pages: list[tuple[str, str]] = []
    for idx, graph in enumerate(read_jsonl(args.input, limit=args.limit), start=1):
        graph_id = str(graph.get("graph_id") or f"graph_{idx}")
        filename = f"{idx:03d}_{_safe_filename(graph_id)}.html"
        (output_dir / filename).write_text(render_graph_page(graph), encoding="utf-8")
        pages.append((filename, graph_id))

    (output_dir / "index.html").write_text(render_index(pages), encoding="utf-8")
    print(f"Wrote {len(pages)} graph pages to {output_dir}")


def render_index(pages: list[tuple[str, str]]) -> str:
    links = "\n".join(
        f'<li><a href="{html.escape(filename)}">{html.escape(graph_id)}</a></li>'
        for filename, graph_id in pages
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ReactGDiff OpenExp Graph Views</title>
  <style>{BASE_CSS}</style>
</head>
<body>
  <main>
    <h1>ReactGDiff OpenExp Graph Views</h1>
    <p>{len(pages)} rendered process graphs.</p>
    <ol class="index-list">
      {links}
    </ol>
  </main>
</body>
</html>
"""


def render_graph_page(graph: dict[str, Any]) -> str:
    graph_id = str(graph.get("graph_id", "graph"))
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    positions = layout_nodes(nodes)
    width = max((x for x, _ in positions.values()), default=900) + 260
    height = max((y for _, y in positions.values()), default=500) + 140

    svg_edges = "\n".join(render_edge(edge, positions) for edge in edges)
    svg_nodes = "\n".join(render_node(node, positions[node["id"]]) for node in nodes if node["id"] in positions)
    legend = render_legend()
    metadata = graph.get("metadata", {})
    actions = html.escape(str(metadata.get("actions", "")))
    source = html.escape(str(metadata.get("source", "")))
    raw_json = html.escape(json.dumps(graph, ensure_ascii=False, indent=2))

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(graph_id)}</title>
  <style>{BASE_CSS}</style>
</head>
<body>
  <main>
    <nav><a href="index.html">Index</a></nav>
    <h1>{html.escape(graph_id)}</h1>
    {legend}
    <section class="graph-shell">
      <svg viewBox="0 0 {width} {height}" role="img" aria-label="Process graph">
        <defs>
          <marker id="arrow" markerWidth="10" markerHeight="8" refX="9" refY="4" orient="auto">
            <path d="M0,0 L10,4 L0,8 Z" fill="#334155"></path>
          </marker>
        </defs>
        {svg_edges}
        {svg_nodes}
      </svg>
    </section>
    <section>
      <h2>OpenExp Actions</h2>
      <p class="mono">{actions}</p>
    </section>
    <section>
      <h2>Source Text</h2>
      <p>{source}</p>
    </section>
    <details>
      <summary>Graph JSON</summary>
      <pre>{raw_json}</pre>
    </details>
  </main>
</body>
</html>
"""


def layout_nodes(nodes: list[dict[str, Any]]) -> dict[str, tuple[int, int]]:
    columns = {
        "material": 90,
        "operation": 360,
        "condition": 630,
        "state": 900,
        "container": 630,
        "safety_control": 630,
    }
    by_type: dict[str, list[dict[str, Any]]] = {}
    for node in nodes:
        by_type.setdefault(str(node.get("type", "unknown")), []).append(node)

    positions: dict[str, tuple[int, int]] = {}
    for node_type, typed_nodes in by_type.items():
        x = columns.get(node_type, 630)
        for idx, node in enumerate(typed_nodes):
            if node_type in {"operation", "state"}:
                step = int(node.get("attrs", {}).get("step_id", idx) or idx)
                y = 90 + step * 110
            else:
                y = 90 + idx * 82
            positions[str(node["id"])] = (x, y)
    return positions


def render_edge(edge: dict[str, Any], positions: dict[str, tuple[int, int]]) -> str:
    source = str(edge.get("source"))
    target = str(edge.get("target"))
    if source not in positions or target not in positions:
        return ""
    x1, y1 = positions[source]
    x2, y2 = positions[target]
    edge_type = str(edge.get("type", "edge"))
    color = EDGE_COLORS.get(edge_type, "#475569")
    label_x = (x1 + x2) / 2
    label_y = (y1 + y2) / 2 - 8
    return f"""
      <line x1="{x1 + 62}" y1="{y1}" x2="{x2 - 62}" y2="{y2}" stroke="{color}" stroke-width="2" marker-end="url(#arrow)" opacity="0.78" />
      <text x="{label_x}" y="{label_y}" class="edge-label">{html.escape(edge_type)}</text>
    """


def render_node(node: dict[str, Any], position: tuple[int, int]) -> str:
    x, y = position
    node_type = str(node.get("type", "unknown"))
    stroke, fill = NODE_COLORS.get(node_type, ("#475569", "#f8fafc"))
    label = _shorten(str(node.get("label", node.get("id", ""))), 28)
    node_id = str(node.get("id", ""))
    return f"""
      <g class="node">
        <rect x="{x - 62}" y="{y - 28}" width="124" height="56" rx="8" fill="{fill}" stroke="{stroke}" stroke-width="2" />
        <text x="{x}" y="{y - 6}" class="node-type">{html.escape(node_type)}</text>
        <text x="{x}" y="{y + 14}" class="node-label">{html.escape(label)}</text>
        <title>{html.escape(node_id)}: {html.escape(str(node.get("label", "")))}</title>
      </g>
    """


def render_legend() -> str:
    chips = []
    for node_type, (stroke, fill) in NODE_COLORS.items():
        chips.append(
            f'<span class="chip" style="border-color:{stroke};background:{fill}">{html.escape(node_type)}</span>'
        )
    return '<div class="legend">' + "\n".join(chips) + "</div>"


def _safe_filename(value: str) -> str:
    keep = [char if char.isalnum() or char in {"-", "_"} else "_" for char in value]
    return "".join(keep).strip("_") or "graph"


def _shorten(value: str, limit: int) -> str:
    return value if len(value) <= limit else value[: limit - 1] + "..."


BASE_CSS = """
:root {
  color: #111827;
  font-family: Arial, Helvetica, sans-serif;
  line-height: 1.45;
}
body {
  margin: 0;
  background: #f8fafc;
}
main {
  max-width: 1180px;
  margin: 0 auto;
  padding: 24px;
}
a {
  color: #2563eb;
}
h1 {
  margin: 8px 0 16px;
  font-size: 28px;
}
h2 {
  margin: 24px 0 8px;
  font-size: 18px;
}
.graph-shell {
  overflow: auto;
  border: 1px solid #cbd5e1;
  background: #ffffff;
  border-radius: 8px;
}
svg {
  display: block;
  min-width: 100%;
}
.legend {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin: 0 0 14px;
}
.chip {
  border: 1px solid;
  border-radius: 999px;
  padding: 3px 9px;
  font-size: 12px;
}
.node-type {
  dominant-baseline: middle;
  fill: #334155;
  font-size: 11px;
  font-weight: 700;
  text-anchor: middle;
  text-transform: uppercase;
}
.node-label {
  dominant-baseline: middle;
  fill: #0f172a;
  font-size: 12px;
  text-anchor: middle;
}
.edge-label {
  fill: #475569;
  font-size: 10px;
  text-anchor: middle;
}
.mono,
pre {
  font-family: "SFMono-Regular", Consolas, "Liberation Mono", monospace;
}
pre {
  overflow: auto;
  padding: 12px;
  background: #0f172a;
  color: #e2e8f0;
  border-radius: 8px;
}
.index-list li {
  margin: 4px 0;
}
"""


if __name__ == "__main__":
    main()
