"""Build RGDL demo graphs and grouped HTML pages from processed OpenExp sets."""

from __future__ import annotations

import argparse
import html
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reactgdiff.compile.sequence_to_graph import compile_openexp_record_to_rgdl
from reactgdiff.utils.io import read_jsonl

DATASET_SPECS = [
    ("main", "Main Clean Set", "data/processed/openexp/main.jsonl"),
    ("scale_small", "Small-Scale Procedures", "data/processed/openexp/buckets/scale_small.jsonl"),
    ("scale_medium", "Medium-Scale Procedures", "data/processed/openexp/buckets/scale_medium.jsonl"),
    ("scale_large", "Large-Scale Procedures", "data/processed/openexp/buckets/scale_large.jsonl"),
    ("numeric_heavy", "Numeric-Heavy Cases", "data/processed/openexp/buckets/numeric_heavy.jsonl"),
    ("condition_heavy", "Condition-Heavy Cases", "data/processed/openexp/buckets/condition_heavy.jsonl"),
    (
        "hard_numeric_condition",
        "Hard Numeric / Condition Cases",
        "data/processed/openexp/buckets/hard_numeric_condition.jsonl",
    ),
    ("multi_reference", "Multi-Reference Cases", "data/processed/openexp/buckets/multi_reference.jsonl"),
    ("branch_workup", "Branch / Workup Cases", "data/processed/openexp/buckets/branch_workup.jsonl"),
]

NODE_STYLE = {
    "reaction": ("#111827", "#f9fafb"),
    "operation": ("#2563eb", "#eff6ff"),
    "material_mention": ("#0f766e", "#f0fdfa"),
    "quantity": ("#d97706", "#fffbeb"),
    "condition": ("#7c3aed", "#f5f3ff"),
    "state": ("#475569", "#f8fafc"),
}

EDGE_STYLE = {
    "next": "#1d4ed8",
    "input_to": "#0f766e",
    "output_from": "#6d28d9",
    "mentions": "#0891b2",
    "refers_to": "#059669",
    "has_quantity": "#b45309",
    "has_condition": "#7c3aed",
    "derived_from": "#64748b",
    "has_participant": "#334155",
    "initializes": "#64748b",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="outputs/figures/graph_views")
    parser.add_argument("--graph-output-dir", default="outputs/case_studies/rgdl_samples")
    parser.add_argument("--per-dataset", type=int, default=3)
    parser.add_argument("--gallery-count", type=int, default=40)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    graph_output_dir = Path(args.graph_output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    if graph_output_dir.exists():
        shutil.rmtree(graph_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    graph_output_dir.mkdir(parents=True, exist_ok=True)

    dataset_pages = []
    for dataset_id, dataset_title, path in DATASET_SPECS:
        records = list(read_jsonl(path, limit=args.per_dataset))
        graphs = [
            compile_openexp_record_to_rgdl(
                record,
                include_slots=False,
                include_surface_attrs=False,
            )
            for record in records
        ]
        write_graph_jsonl(graph_output_dir / f"{dataset_id}.rgdl.jsonl", graphs)

        dataset_dir = output_dir / dataset_id
        dataset_dir.mkdir(parents=True, exist_ok=True)
        example_links = []
        for idx, (record, graph) in enumerate(zip(records, graphs), start=1):
            filename = f"example_{idx:02d}_openexp_{record.get('index')}.html"
            (dataset_dir / filename).write_text(
                render_example_page(dataset_id, dataset_title, record, graph),
                encoding="utf-8",
            )
            example_links.append((filename, graph["graph_id"], record))
        (dataset_dir / "index.html").write_text(
            render_dataset_index(dataset_title, dataset_id, example_links),
            encoding="utf-8",
        )
        dataset_pages.append((dataset_id, dataset_title, len(graphs)))

    gallery_records = list(read_jsonl("data/processed/openexp/main.jsonl", limit=args.gallery_count))
    gallery_graphs = [
        compile_openexp_record_to_rgdl(
            record,
            include_slots=False,
            include_surface_attrs=False,
        )
        for record in gallery_records
    ]

    (output_dir / "generation_animation.html").write_text(render_generation_animation_page(), encoding="utf-8")
    (output_dir / "action_gallery.html").write_text(
        render_action_gallery_page(gallery_records, gallery_graphs),
        encoding="utf-8",
    )
    (output_dir / "index.html").write_text(render_root_index(dataset_pages), encoding="utf-8")
    (graph_output_dir / "VALIDATION.md").write_text(render_validation_note(), encoding="utf-8")
    print(f"Wrote RGDL demo pages to {output_dir}")
    print(f"Wrote RGDL sample graphs to {graph_output_dir}")


def write_graph_jsonl(path: Path, graphs: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for graph in graphs:
            handle.write(json.dumps(graph, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def render_root_index(dataset_pages: list[tuple[str, str, int]]) -> str:
    rows = "\n".join(
        f'<tr><td><a href="{html.escape(dataset_id)}/index.html">{html.escape(title)}</a></td>'
        f"<td>{count}</td></tr>"
        for dataset_id, title, count in dataset_pages
    )
    return page(
        "ReactGDiff Semantic Target Graphs",
        f"""
        <h1>ReactGDiff Semantic Target Graphs</h1>
        <p><a href="generation_animation.html">扩散到异构过程图生成动画</a></p>
        <p><a href="action_gallery.html">过程图缩略图与动作总览</a></p>
        <p>每个数据集页面展示原始 OpenExp 数据、单反应输入物料表，以及去除 surface slot 节点后的模型目标异构图。</p>
        <table>
          <thead><tr><th>Dataset</th><th>Examples</th></tr></thead>
          <tbody>{rows}</tbody>
        </table>
        """,
    )


def render_generation_animation_page() -> str:
    return r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>异构过程图生成模拟</title>
  <style>
:root {
  --ink: #111827;
  --muted: #64748b;
  --line: #cbd5e1;
  --panel: #ffffff;
  --page: #f6f8fb;
  --blue: #2563eb;
  --teal: #0f766e;
  --amber: #d97706;
  --violet: #7c3aed;
  --slate: #475569;
  color: var(--ink);
  font-family: Arial, Helvetica, sans-serif;
}
* {
  box-sizing: border-box;
}
body {
  margin: 0;
  background: var(--page);
}
main {
  width: min(1480px, calc(100vw - 32px));
  margin: 0 auto;
  padding: 20px 0 28px;
}
.topbar {
  display: grid;
  grid-template-columns: 1fr auto;
  gap: 16px;
  align-items: end;
  margin-bottom: 14px;
}
h1 {
  margin: 0 0 5px;
  font-size: 24px;
  line-height: 1.18;
  letter-spacing: 0;
}
.subtitle {
  margin: 0;
  color: var(--muted);
  font-size: 13px;
}
.navlink {
  color: #1d4ed8;
  font-size: 13px;
  text-decoration: none;
}
.stage-strip {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 8px;
  margin: 12px 0 14px;
}
.stage {
  border: 1px solid var(--line);
  border-radius: 8px;
  background: #fff;
  padding: 9px 11px;
  min-height: 58px;
}
.stage-name {
  display: block;
  font-size: 13px;
  font-weight: 700;
}
.stage-note {
  display: block;
  margin-top: 3px;
  color: var(--muted);
  font-size: 12px;
  line-height: 1.3;
}
.stage.active {
  border-color: #2563eb;
  box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.12);
}
.workspace {
  display: grid;
  grid-template-columns: minmax(360px, 0.78fr) minmax(560px, 1.22fr);
  gap: 14px;
  align-items: stretch;
}
.panel {
  min-width: 0;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: var(--panel);
  overflow: hidden;
}
.panel-head {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  align-items: center;
  min-height: 44px;
  border-bottom: 1px solid #e2e8f0;
  padding: 10px 12px;
}
.panel-title {
  margin: 0;
  font-size: 14px;
}
.metric {
  color: var(--muted);
  font-size: 12px;
  white-space: nowrap;
}
.latent-wrap {
  display: grid;
  grid-template-rows: auto minmax(0, 1fr) auto;
  height: 100%;
}
.latent-wrap .panel-head {
  justify-content: center;
  min-height: 56px;
  position: relative;
}
.latent-wrap .panel-title {
  flex: 1;
  font-size: 20px;
  text-align: center;
}
.latent-wrap .metric {
  position: absolute;
  right: 12px;
}
.latent-canvas-box {
  display: grid;
  grid-template-columns: 1fr;
  align-content: center;
  gap: 14px;
  padding: 14px;
  background:
    linear-gradient(90deg, rgba(226, 232, 240, 0.58) 1px, transparent 1px),
    linear-gradient(rgba(226, 232, 240, 0.58) 1px, transparent 1px);
  background-size: 22px 22px;
}
.latent-view {
  min-width: 0;
}
.latent-label {
  display: flex;
  justify-content: center;
  gap: 8px;
  margin: 0 0 7px;
  color: #475569;
  font-size: 13px;
  text-align: center;
}
.latent-label strong {
  color: #111827;
  font-size: 16px;
}
.latent-view canvas {
  display: block;
  width: min(100%, 500px);
  margin: 0 auto;
  aspect-ratio: 2 / 1;
  border: 1px solid #dbe3ef;
  border-radius: 8px;
  background: #0f172a;
}
#discreteCanvas {
  image-rendering: pixelated;
}
#continuousCanvas {
  image-rendering: auto;
}
.latent-footer {
  display: grid;
  gap: 8px;
  border-top: 1px solid #e2e8f0;
  padding: 11px 12px 12px;
}
.bar {
  height: 9px;
  overflow: hidden;
  border-radius: 999px;
  background: #e2e8f0;
}
.bar-fill {
  width: 0%;
  height: 100%;
  border-radius: inherit;
  background: linear-gradient(90deg, #2563eb, #0f766e, #d97706);
}
.code-row {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}
.token {
  border: 1px solid #dbe3ef;
  border-radius: 6px;
  padding: 4px 7px;
  background: #f8fafc;
  color: #334155;
  font-family: "SFMono-Regular", Consolas, "Liberation Mono", monospace;
  font-size: 12px;
}
.graph-wrap {
  display: grid;
  grid-template-rows: auto minmax(520px, 1fr) auto;
  height: 100%;
  min-height: 700px;
}
#graphSvg {
  display: block;
  width: 100%;
  height: 100%;
  min-height: 520px;
  background: #ffffff;
}
.node rect {
  transition: fill 180ms linear, stroke 180ms linear, opacity 180ms linear;
  filter: drop-shadow(0 1px 1.2px rgba(15, 23, 42, 0.12));
}
.node .type {
  fill: #0f172a;
  font-size: 12px;
  font-weight: 800;
  text-anchor: middle;
  text-transform: uppercase;
}
.node .label {
  fill: #020617;
  font-size: 15px;
  font-weight: 800;
  text-anchor: middle;
}
.node .sub {
  fill: #334155;
  font-size: 12px;
  font-weight: 700;
  text-anchor: middle;
}
.edge {
  fill: none;
  stroke-width: 2.8;
  marker-end: url(#arrow);
}
.edge-label {
  fill: #0f172a;
  font-size: 11px;
  font-weight: 800;
  text-anchor: middle;
  paint-order: stroke;
  stroke: #fff;
  stroke-width: 4.2px;
}
.attr-board {
  border-top: 1px solid #e2e8f0;
  background: #fff;
  padding: 10px 12px 12px;
}
.attr-title {
  margin: 0 0 8px;
  font-size: 14px;
  font-weight: 800;
}
.attr-grid {
  display: grid;
  grid-template-columns: 92px 1fr;
  gap: 5px 8px;
  font-size: 13px;
}
.attr-key {
  color: #334155;
  font-weight: 700;
}
.attr-value {
  color: #0f172a;
  font-family: "SFMono-Regular", Consolas, "Liberation Mono", monospace;
  word-break: break-word;
}
.controls {
  display: grid;
  grid-template-columns: auto 1fr auto;
  gap: 12px;
  align-items: center;
  margin-top: 14px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: #fff;
  padding: 11px 12px;
}
button {
  border: 1px solid #1d4ed8;
  border-radius: 8px;
  background: #1d4ed8;
  color: #fff;
  min-width: 92px;
  min-height: 36px;
  padding: 0 13px;
  font-weight: 700;
  cursor: pointer;
}
button.secondary {
  border-color: #cbd5e1;
  background: #fff;
  color: #334155;
  min-width: 70px;
}
input[type="range"] {
  width: 100%;
  accent-color: #2563eb;
}
.time-readout {
  color: #475569;
  font-family: "SFMono-Regular", Consolas, "Liberation Mono", monospace;
  font-size: 12px;
  min-width: 56px;
  text-align: right;
}
.legend {
  display: flex;
  flex-wrap: wrap;
  gap: 7px;
  margin-top: 10px;
}
.legend-chip {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  color: #0f172a;
  font-size: 13px;
  font-weight: 700;
}
.swatch {
  width: 14px;
  height: 14px;
  border-radius: 3px;
  border: 2.5px solid currentColor;
}
@media (max-width: 980px) {
  main {
    width: min(100vw - 20px, 760px);
    padding-top: 14px;
  }
  .topbar,
  .workspace,
  .controls {
    grid-template-columns: 1fr;
  }
  .stage-strip {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
  .latent-view canvas {
    width: min(100%, 520px);
  }
  .graph-wrap,
  #graphSvg {
    min-height: 620px;
  }
  .time-readout {
    text-align: left;
  }
}
  </style>
</head>
<body>
  <main>
    <header class="topbar">
      <div>
        <h1>异构过程图生成模拟</h1>
        <p class="subtitle">扩散 latent 从噪声收敛，过程图从混乱节点到骨架、属性，再到最终 RGDL 目标图。</p>
      </div>
      <a class="navlink" href="index.html">返回图示索引</a>
    </header>

    <section class="stage-strip" aria-label="generation stages">
      <div class="stage" data-stage="0"><span class="stage-name">1. 噪声采样</span><span class="stage-note">无序像素与随机连边</span></div>
      <div class="stage" data-stage="1"><span class="stage-name">2. 骨架成形</span><span class="stage-note">反应节点和操作链稳定</span></div>
      <div class="stage" data-stage="2"><span class="stage-name">3. 属性填充</span><span class="stage-note">物料索引、条件、用量显式化</span></div>
      <div class="stage" data-stage="3"><span class="stage-name">4. 联合微调</span><span class="stage-note">节点类型、边类型、属性一起收敛</span></div>
    </section>

    <section class="workspace">
      <div class="panel latent-wrap">
        <div class="panel-head">
          <h2 class="panel-title">扩散模型抽象</h2>
          <span class="metric" id="noiseMetric">噪声 1.00</span>
        </div>
        <div class="latent-canvas-box">
          <div class="latent-view">
            <p class="latent-label"><strong>离散 latent</strong><span>图骨架像素</span></p>
            <canvas id="discreteCanvas" width="420" height="210" aria-label="离散像素扩散动画"></canvas>
          </div>
          <div class="latent-view">
            <p class="latent-label"><strong>连续 latent</strong><span>属性独热编码</span></p>
            <canvas id="continuousCanvas" width="420" height="210" aria-label="连续属性扩散动画"></canvas>
          </div>
        </div>
        <div class="latent-footer">
          <div class="bar"><div class="bar-fill" id="progressFill"></div></div>
          <div class="code-row" id="latentTokens"></div>
        </div>
      </div>

      <div class="panel graph-wrap">
        <div class="panel-head">
          <h2 class="panel-title">异构过程图（样本索引 6）</h2>
          <span class="metric" id="graphMetric">节点 0 / 边 0</span>
        </div>
        <svg id="graphSvg" viewBox="0 0 920 640" role="img" aria-label="异构过程图生成动画">
          <defs>
            <marker id="arrow" markerWidth="12" markerHeight="10" refX="11" refY="5" orient="auto" markerUnits="userSpaceOnUse">
              <path d="M0,0 L12,5 L0,10 Z" fill="#0f172a"></path>
            </marker>
          </defs>
        </svg>
        <aside class="attr-board">
          <p class="attr-title">当前图属性</p>
          <div class="attr-grid" id="attrGrid"></div>
          <div class="legend">
            <span class="legend-chip"><span class="swatch" style="color:#030712;background:#e5e7eb"></span>反应</span>
            <span class="legend-chip"><span class="swatch" style="color:#1d4ed8;background:#dbeafe"></span>操作</span>
            <span class="legend-chip"><span class="swatch" style="color:#0f766e;background:#ccfbf1"></span>物料提及</span>
            <span class="legend-chip"><span class="swatch" style="color:#b45309;background:#fef3c7"></span>用量</span>
            <span class="legend-chip"><span class="swatch" style="color:#6d28d9;background:#ede9fe"></span>条件</span>
            <span class="legend-chip"><span class="swatch" style="color:#334155;background:#e2e8f0"></span>状态</span>
          </div>
        </aside>
      </div>
    </section>

    <section class="controls">
      <div>
        <button id="playBtn" type="button">暂停</button>
        <button id="resetBtn" class="secondary" type="button">重播</button>
      </div>
      <input id="timeline" type="range" min="0" max="1000" value="0" aria-label="timeline">
      <span class="time-readout" id="timeReadout">00%</span>
    </section>
  </main>

  <script>
const discreteCanvas = document.getElementById("discreteCanvas");
const discreteCtx = discreteCanvas.getContext("2d");
const continuousCanvas = document.getElementById("continuousCanvas");
const continuousCtx = continuousCanvas.getContext("2d");
const svg = document.getElementById("graphSvg");
const playBtn = document.getElementById("playBtn");
const resetBtn = document.getElementById("resetBtn");
const timeline = document.getElementById("timeline");
const timeReadout = document.getElementById("timeReadout");
const progressFill = document.getElementById("progressFill");
const noiseMetric = document.getElementById("noiseMetric");
const graphMetric = document.getElementById("graphMetric");
const attrGrid = document.getElementById("attrGrid");
const latentTokens = document.getElementById("latentTokens");
const stages = Array.from(document.querySelectorAll(".stage"));

const W = 45;
const H = 23;
const CELL_X = discreteCanvas.width / W;
const CELL_Y = discreteCanvas.height / H;
const DURATION = 18000;
let playing = true;
let manual = false;
let startTime = performance.now();
let manualProgress = 0;

const palette = {
  reaction: {stroke: "#030712", fill: "#e5e7eb"},
  operation: {stroke: "#1d4ed8", fill: "#dbeafe"},
  material_mention: {stroke: "#0f766e", fill: "#ccfbf1"},
  quantity: {stroke: "#b45309", fill: "#fef3c7"},
  condition: {stroke: "#6d28d9", fill: "#ede9fe"},
  state: {stroke: "#334155", fill: "#e2e8f0"},
  unknown: {stroke: "#64748b", fill: "#e2e8f0"}
};

const finalNodes = [
  {id:"reaction", type:"reaction", label:"样本 6", sub:"索引=6", x:456, y:54, w:132},
  {id:"op_000", type:"operation", label:"MAKESOLUTION", sub:"步骤=0", x:456, y:128, w:150},
  {id:"op_001", type:"operation", label:"ADD", sub:"步骤=1", x:456, y:218, w:122},
  {id:"op_002", type:"operation", label:"STIR", sub:"步骤=2", x:456, y:308, w:122},
  {id:"op_003", type:"operation", label:"WAIT", sub:"步骤=3", x:456, y:398, w:122},
  {id:"op_004", type:"operation", label:"YIELD", sub:"步骤=4", x:456, y:488, w:122},
  {id:"men_000_00", type:"material_mention", label:"m0", sub:"反应物", x:186, y:70, w:136},
  {id:"men_000_01", type:"material_mention", label:"m2", sub:"溶剂", x:186, y:130, w:136},
  {id:"men_000_02", type:"material_mention", label:"m3", sub:"溶剂", x:186, y:190, w:136},
  {id:"men_001_00", type:"material_mention", label:"m1", sub:"反应物", x:216, y:218, w:136},
  {id:"men_004_00", type:"material_mention", label:"m4", sub:"产物", x:216, y:488, w:136},
  {id:"qty_000_op_00", type:"quantity", label:"65 ml", sub:"体积", x:720, y:76, w:112},
  {id:"qty_001_op_00", type:"quantity", label:"20 mL", sub:"体积", x:720, y:218, w:112},
  {id:"qty_004_op_00", type:"quantity", label:"0.36 g, 53%", sub:"产率", x:720, y:488, w:126},
  {id:"cond_6", type:"condition", label:"3 h", sub:"时长", x:720, y:286, w:112},
  {id:"cond_5", type:"condition", label:"130 °C", sub:"温度", x:720, y:342, w:112},
  {id:"state_000", type:"state", label:"solution_0", sub:"溶液", x:720, y:140, w:126}
];

const finalEdges = [
  {id:"e0", source:"op_000", target:"qty_000_op_00", type:"has_quantity", stage:0.42},
  {id:"e1", source:"op_000", target:"men_000_00", type:"mentions", stage:0.38},
  {id:"e2", source:"op_000", target:"men_000_01", type:"mentions", stage:0.40},
  {id:"e3", source:"op_000", target:"men_000_02", type:"mentions", stage:0.42},
  {id:"e4", source:"op_000", target:"state_000", type:"output_from", stage:0.46},
  {id:"e5", source:"op_000", target:"op_001", type:"next", stage:0.24},
  {id:"e6", source:"op_001", target:"qty_001_op_00", type:"has_quantity", stage:0.50},
  {id:"e7", source:"op_001", target:"men_001_00", type:"mentions", stage:0.48},
  {id:"e8", source:"men_001_00", target:"qty_001_op_00", type:"has_quantity", stage:0.54},
  {id:"e9", source:"op_001", target:"op_002", type:"next", stage:0.28},
  {id:"e10", source:"state_000", target:"op_002", type:"input_to", stage:0.60},
  {id:"e11", source:"op_002", target:"cond_6", type:"has_condition", stage:0.58},
  {id:"e12", source:"op_002", target:"cond_5", type:"has_condition", stage:0.60},
  {id:"e13", source:"op_002", target:"op_003", type:"next", stage:0.32},
  {id:"e14", source:"state_000", target:"op_003", type:"input_to", stage:0.64},
  {id:"e15", source:"op_003", target:"cond_6", type:"has_condition", stage:0.62},
  {id:"e16", source:"op_003", target:"cond_5", type:"has_condition", stage:0.64},
  {id:"e17", source:"op_003", target:"op_004", type:"next", stage:0.36},
  {id:"e18", source:"op_004", target:"qty_004_op_00", type:"has_quantity", stage:0.66},
  {id:"e19", source:"op_004", target:"men_004_00", type:"mentions", stage:0.66},
  {id:"e20", source:"men_004_00", target:"qty_004_op_00", type:"has_quantity", stage:0.70},
  {id:"e21", source:"state_000", target:"op_004", type:"input_to", stage:0.70}
];

const noiseNodes = finalNodes.map((node, index) => ({
  id: node.id,
  x: 90 + seeded(index * 17 + 2) * 730,
  y: 80 + seeded(index * 23 + 9) * 470,
  dx: seeded(index * 31 + 5) * 34 - 17,
  dy: seeded(index * 41 + 7) * 34 - 17
}));

const graphLayer = document.createElementNS("http://www.w3.org/2000/svg", "g");
svg.appendChild(graphLayer);

const edgeEls = new Map();
for (const edge of finalEdges) {
  const g = document.createElementNS("http://www.w3.org/2000/svg", "g");
  const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
  const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
  path.setAttribute("class", "edge");
  text.setAttribute("class", "edge-label");
  g.appendChild(path);
  g.appendChild(text);
  graphLayer.appendChild(g);
  edgeEls.set(edge.id, {g, path, text});
}

const nodeEls = new Map();
for (const node of finalNodes) {
  const g = document.createElementNS("http://www.w3.org/2000/svg", "g");
  const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
  const type = document.createElementNS("http://www.w3.org/2000/svg", "text");
  const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
  const sub = document.createElementNS("http://www.w3.org/2000/svg", "text");
  g.setAttribute("class", "node");
  rect.setAttribute("rx", "8");
  rect.setAttribute("height", "62");
  type.setAttribute("class", "type");
  label.setAttribute("class", "label");
  sub.setAttribute("class", "sub");
  type.setAttribute("y", "-14");
  label.setAttribute("y", "6");
  sub.setAttribute("y", "25");
  g.appendChild(rect);
  g.appendChild(type);
  g.appendChild(label);
  g.appendChild(sub);
  graphLayer.appendChild(g);
  nodeEls.set(node.id, {g, rect, type, label, sub});
}

function seeded(n) {
  const x = Math.sin(n * 12.9898) * 43758.5453;
  return x - Math.floor(x);
}

function clamp(v, lo = 0, hi = 1) {
  return Math.max(lo, Math.min(hi, v));
}

function smooth(v) {
  v = clamp(v);
  return v * v * (3 - 2 * v);
}

function mix(a, b, t) {
  return a + (b - a) * t;
}

function phase(progress, start, end) {
  return smooth((progress - start) / (end - start));
}

function nodeProgress(node, progress) {
  if (node.type === "reaction" || node.type === "operation") return phase(progress, 0.10, 0.44);
  if (node.type === "material_mention") return phase(progress, 0.38, 0.64);
  if (node.type === "quantity" || node.type === "condition") return phase(progress, 0.46, 0.72);
  return phase(progress, 0.55, 0.78);
}

function edgeProgress(edge, progress) {
  return phase(progress, edge.stage, edge.stage + 0.18);
}

function currentNodePosition(node, progress) {
  const noise = noiseNodes.find(item => item.id === node.id);
  const p = nodeProgress(node, progress);
  return {
    x: mix(noise.x, node.x, p),
    y: mix(noise.y, node.y, p),
    p
  };
}

function drawLatent(progress, now) {
  drawDiscreteLatent(progress, now);
  drawContinuousLatent(progress, now);
}

function drawDiscreteLatent(progress, now) {
  discreteCtx.clearRect(0, 0, discreteCanvas.width, discreteCanvas.height);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const n0 = seeded(x * 13 + y * 97);
      const n1 = seeded(x * 83 + y * 29 + Math.floor(now / 120));
      const target = tPixelField(x, y);
      const reveal = pixelResolve(progress, x, y, target);
      const finalValue = target > 0.08 ? 0.34 + target * 0.66 : 0.08;
      const value = mix((n0 * 0.65 + n1 * 0.35), finalValue, reveal);
      const hue = mix(214, target > 0.08 ? 185 - target * 24 : 222, reveal);
      const sat = mix(26 + value * 56, target > 0.08 ? 64 : 30, reveal);
      const light = mix(20 + value * 68, target > 0.08 ? 22 + value * 48 : 14 + value * 18, reveal);
      discreteCtx.fillStyle = `hsl(${hue}, ${sat}%, ${light}%)`;
      discreteCtx.fillRect(x * CELL_X, y * CELL_Y, Math.ceil(CELL_X), Math.ceil(CELL_Y));
    }
  }
}

function drawContinuousLatent(progress, now) {
  const image = continuousCtx.createImageData(continuousCanvas.width, continuousCanvas.height);
  const data = image.data;
  let ptr = 0;
  for (let py = 0; py < continuousCanvas.height; py++) {
    for (let px = 0; px < continuousCanvas.width; px++) {
      const x = px / continuousCanvas.width;
      const y = py / continuousCanvas.height;
      const noise = 0.5
        + Math.sin(px * 0.08 + now * 0.002) * 0.19
        + Math.cos(py * 0.11 - now * 0.0017) * 0.16
        + (seeded(Math.floor(px / 7) * 19 + Math.floor(py / 7) * 31) - 0.5) * 0.34;
      const target = tContinuousField(x, y);
      const reveal = pixelResolve(progress, Math.floor(px / 8) + 71, Math.floor(py / 8) + 137, target);
      const finalValue = target > 0.04 ? target : 0.07;
      const value = clamp(mix(noise, finalValue, reveal));
      const noiseR = 22 + noise * 98;
      const noiseG = 32 + noise * 116;
      const noiseB = 56 + noise * 142;
      const hot = target > 0.44 ? 1 : 0;
      const finalR = 15 + value * 44 + hot * 176;
      const finalG = 35 + value * 104 + hot * 78;
      const finalB = 68 + value * 105 - hot * 46;
      const r = Math.round(mix(noiseR, finalR, reveal));
      const g = Math.round(mix(noiseG, finalG, reveal));
      const b = Math.round(mix(noiseB, finalB, reveal));
      data[ptr++] = r;
      data[ptr++] = g;
      data[ptr++] = b;
      data[ptr++] = 255;
    }
  }
  continuousCtx.putImageData(image, 0, 0);
}

function tPixelField(x, y) {
  const nodes = Math.max(
    rectField(x, y, 22, 3, 3.4, 1.5),
    rectField(x, y, 22, 7, 3.4, 1.5),
    rectField(x, y, 22, 11, 3.4, 1.5),
    rectField(x, y, 22, 15, 3.4, 1.5),
    rectField(x, y, 22, 19, 3.4, 1.5),
    rectField(x, y, 10, 6, 2.6, 1.2),
    rectField(x, y, 10, 9, 2.6, 1.2),
    rectField(x, y, 10, 20, 2.6, 1.2),
    rectField(x, y, 35, 6, 2.6, 1.2),
    rectField(x, y, 35, 9, 2.6, 1.2),
    rectField(x, y, 35, 11, 2.6, 1.2),
    rectField(x, y, 35, 15, 2.6, 1.2),
    rectField(x, y, 35, 19, 2.6, 1.2)
  );
  const edges = Math.max(
    lineField(x, y, 22, 4.5, 22, 17.6, 0.6),
    lineField(x, y, 13, 6, 18.6, 7, 0.45),
    lineField(x, y, 13, 9, 18.6, 7, 0.45),
    lineField(x, y, 13, 20, 18.6, 19, 0.45),
    lineField(x, y, 25.4, 7, 32, 6, 0.45),
    lineField(x, y, 25.4, 7, 32, 9, 0.45),
    lineField(x, y, 25.4, 11, 32, 11, 0.45),
    lineField(x, y, 25.4, 15, 32, 15, 0.45),
    lineField(x, y, 25.4, 19, 32, 19, 0.45)
  );
  return clamp(Math.max(nodes, edges * 0.82));
}

function tContinuousField(x, y) {
  const rows = [0.18, 0.34, 0.50, 0.66, 0.82];
  let value = 0;
  for (let row = 0; row < rows.length; row += 1) {
    const band = rectPulse(x, y, 0.50, rows[row], 0.82, 0.026) * 0.15;
    value = Math.max(value, band);
  }
  const active = [
    [0.16, 0.18, -1],
    [0.42, 0.34, 1],
    [0.66, 0.50, -1],
    [0.27, 0.66, 1],
    [0.82, 0.82, -1]
  ];
  for (const [cx, cy, direction] of active) {
    const cell = rectPulse(x, y, cx, cy, 0.052, 0.045);
    const protrude = rectPulse(x, y, cx, cy + direction * 0.074, 0.038, 0.050);
    const shoulder = rectPulse(x, y, cx + 0.040, cy, 0.020, 0.032) * 0.72;
    value = Math.max(value, cell, protrude * 0.94, shoulder);
  }
  return clamp(value);
}

function pixelResolve(progress, x, y, target) {
  const signal = target > 0.10;
  const start = signal
    ? 0.10 + (1 - target) * 0.18 + seeded(x * 47 + y * 61) * 0.34
    : 0.54 + seeded(x * 29 + y * 43) * 0.28;
  return phase(progress, start, start + 0.20);
}

function rectField(x, y, cx, cy, hw, hh) {
  const dx = Math.max(Math.abs(x - cx) - hw, 0);
  const dy = Math.max(Math.abs(y - cy) - hh, 0);
  return Math.exp(-(dx * dx + dy * dy) / 1.15);
}

function rectPulse(x, y, cx, cy, hw, hh) {
  const dx = Math.max(Math.abs(x - cx) - hw, 0);
  const dy = Math.max(Math.abs(y - cy) - hh, 0);
  return Math.exp(-(dx * dx + dy * dy) / 0.00022);
}

function lineField(px, py, ax, ay, bx, by, sigma) {
  const dx = bx - ax;
  const dy = by - ay;
  const t = clamp(((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy));
  const distance = Math.hypot(px - (ax + dx * t), py - (ay + dy * t));
  return Math.exp(-(distance * distance) / sigma);
}

function bump(x, y, cx, cy, radius) {
  const d = Math.hypot(x - cx, y - cy);
  return clamp(1 - d / radius);
}

function drawGraph(progress, now) {
  const positions = new Map();
  for (const node of finalNodes) {
    positions.set(node.id, currentNodePosition(node, progress));
  }

  let visibleNodes = 0;
  let visibleEdges = 0;

  for (const edge of finalEdges) {
    const src = positions.get(edge.source);
    const dst = positions.get(edge.target);
    const ep = edgeProgress(edge, progress);
    const el = edgeEls.get(edge.id);
    const curve = 26 * (1 - ep) * (seeded(edge.id.length * 19) > 0.5 ? 1 : -1);
    const sx = src.x;
    const sy = src.y;
    const tx = mix(src.x, dst.x, ep);
    const ty = mix(src.y, dst.y, ep);
    const mx = (sx + tx) / 2 + curve;
    const my = (sy + ty) / 2 - curve * 0.5;
    el.path.setAttribute("d", `M${sx},${sy} Q${mx},${my} ${tx},${ty}`);
    el.path.setAttribute("stroke", edgeColor(edge.type));
    el.path.setAttribute("opacity", String(0.12 + ep * 0.78));
    el.text.textContent = ep > 0.58 ? edge.type : "";
    el.text.setAttribute("x", String((sx + dst.x) / 2));
    el.text.setAttribute("y", String((sy + dst.y) / 2 - 7));
    el.g.setAttribute("opacity", String(ep > 0.02 ? 1 : 0));
    if (ep > 0.72) visibleEdges += 1;
  }

  for (const node of finalNodes) {
    const pos = positions.get(node.id);
    const el = nodeEls.get(node.id);
    const p = pos.p;
    const color = p > 0.46 ? palette[node.type] : palette.unknown;
    const w = mix(44, node.w, phase(p, 0.25, 0.75));
    const opacity = 0.42 + p * 0.58;
    el.g.setAttribute("transform", `translate(${pos.x},${pos.y})`);
    el.g.setAttribute("opacity", String(opacity));
    el.rect.setAttribute("x", String(-w / 2));
    el.rect.setAttribute("y", "-31");
    el.rect.setAttribute("width", String(w));
    el.rect.setAttribute("fill", color.fill);
    el.rect.setAttribute("stroke", color.stroke);
    el.rect.setAttribute("stroke-width", p > 0.55 ? "3.1" : "1.8");
    el.type.textContent = p > 0.58 ? nodeTypeText(node.type) : "潜变量";
    el.label.textContent = labelFor(node, p);
    el.sub.textContent = p > 0.78 ? node.sub : "";
    if (p > 0.72) visibleNodes += 1;
  }

  graphMetric.textContent = `节点 ${visibleNodes} / 边 ${visibleEdges}`;
}

function labelFor(node, p) {
  if (p < 0.35) return seeded(node.x + node.y) > 0.5 ? "?" : "...";
  if (p < 0.62) {
    if (node.type === "operation") return "op?";
    if (node.type === "reaction") return "rxn";
    if (node.type === "material_mention") return "m?";
    if (node.type === "quantity") return "q?";
    if (node.type === "condition") return "c?";
    return "state?";
  }
  return node.label;
}

function nodeTypeText(type) {
  if (type === "reaction") return "反应";
  if (type === "operation") return "操作";
  if (type === "material_mention") return "物料提及";
  if (type === "quantity") return "用量";
  if (type === "condition") return "条件";
  if (type === "state") return "状态";
  return type;
}

function edgeColor(type) {
  if (type === "next") return "#1d4ed8";
  if (type === "mentions") return "#0891b2";
  if (type === "has_quantity") return "#b45309";
  if (type === "has_condition") return "#7c3aed";
  if (type === "output_from") return "#6d28d9";
  return "#334155";
}

function updateUI(progress) {
  const pct = Math.round(progress * 100);
  timeline.value = String(Math.round(progress * 1000));
  timeReadout.textContent = String(pct).padStart(2, "0") + "%";
  progressFill.style.width = `${pct}%`;
  noiseMetric.textContent = `噪声 ${(1 - smooth(progress)).toFixed(2)}`;

  const stageIndex = progress < 0.25 ? 0 : progress < 0.50 ? 1 : progress < 0.76 ? 2 : 3;
  for (const stage of stages) {
    stage.classList.toggle("active", Number(stage.dataset.stage) === stageIndex);
  }

  const tokens = tokenState(progress);
  latentTokens.innerHTML = tokens.map(token => `<span class="token">${token}</span>`).join("");
  attrGrid.innerHTML = attrState(progress)
    .map(([key, value]) => `<span class="attr-key">${key}</span><span class="attr-value">${value}</span>`)
    .join("");
}

function tokenState(progress) {
  if (progress < 0.24) return ["扩散步 t", "噪声高", "边未定", "属性未定"];
  if (progress < 0.50) return ["反应节点", "操作链", "下一步边", "骨架生成"];
  if (progress < 0.76) return ["m0", "m1", "57.2g", "80C", "溶液"];
  return ["目标图", "类型边", "物料索引", "完成"];
}

function attrState(progress) {
  if (progress < 0.24) {
    return [["样本", "索引 6"], ["节点类型", "未知"], ["边类型", "未知"], ["属性", "待解析"]];
  }
  if (progress < 0.50) {
    return [["样本", "索引 6"], ["操作链", "MAKESOLUTION -> ADD -> STIR -> WAIT -> YIELD"], ["主边", "下一步"], ["状态", "待生成"]];
  }
  if (progress < 0.76) {
    return [["物料索引", "m0, m1, m2, m3, m4"], ["用量", "65 ml; 20 mL; 0.36 g, 53%"], ["条件", "130 °C; 3 h"], ["状态", "溶液状态"]];
  }
  return [["目标", "紧凑过程图"], ["节点", "反应, 操作, 物料提及, 用量, 条件, 状态"], ["边", "下一步, 提及, 用量, 条件, 输出, 输入"], ["表面槽位", "省略"]];
}

function frame(now) {
  let progress;
  if (playing && !manual) {
    progress = ((now - startTime) % DURATION) / DURATION;
  } else {
    progress = manualProgress;
  }
  drawLatent(progress, now);
  drawGraph(progress, now);
  updateUI(progress);
  requestAnimationFrame(frame);
}

playBtn.addEventListener("click", () => {
  playing = !playing;
  playBtn.textContent = playing ? "暂停" : "播放";
  if (playing) {
    manual = false;
    startTime = performance.now() - manualProgress * DURATION;
  }
});

resetBtn.addEventListener("click", () => {
  manual = false;
  playing = true;
  manualProgress = 0;
  startTime = performance.now();
  playBtn.textContent = "暂停";
});

timeline.addEventListener("input", () => {
  manual = true;
  playing = false;
  manualProgress = Number(timeline.value) / 1000;
  playBtn.textContent = "播放";
});

requestAnimationFrame(frame);
  </script>
</body>
</html>
"""


def render_action_gallery_page(records: list[dict[str, Any]], graphs: list[dict[str, Any]]) -> str:
    cards = "\n".join(
        render_action_gallery_card(record, graph, idx)
        for idx, (record, graph) in enumerate(zip(records, graphs), start=1)
    )
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>过程图缩略图与动作总览</title>
  <style>{ACTION_GALLERY_CSS}</style>
</head>
<body>
  <main>
    <header class="page-head">
      <div>
        <h1>过程图缩略图与动作总览</h1>
        <p>每张卡片展示一个样本的紧凑目标图缩略图、主要动作和动作序列，适合截图用于结果展示。</p>
      </div>
      <a href="index.html">返回索引</a>
    </header>
    <section class="gallery">
      {cards}
    </section>
  </main>
</body>
</html>
"""


def render_action_gallery_card(record: dict[str, Any], graph: dict[str, Any], ordinal: int) -> str:
    actions = extract_action_sequence(record.get("actions", ""))
    action_counts = sorted(
        ((action, actions.count(action)) for action in set(actions)),
        key=lambda item: (-item[1], item[0]),
    )
    main_actions = ", ".join(
        action if count == 1 else f"{action}x{count}"
        for action, count in action_counts[:3]
    )
    if not main_actions:
        main_actions = "无"
    action_chips = render_action_chips(actions)
    node_count = len(graph.get("nodes", []))
    edge_count = len(graph.get("edges", []))
    action_count = len(actions)
    return f"""
      <article class="case-card">
        <div class="card-top">
          <span class="case-id">#{html.escape(str(record.get("index", ordinal)))}</span>
        </div>
        <div class="thumb">{render_graph_thumbnail(graph, f"g{ordinal}")}</div>
        <div class="main-action"><span>主要动作</span><strong>{html.escape(main_actions)}</strong></div>
        <div class="action-row">{action_chips}</div>
        <div class="stats">
          <span>{action_count} 个动作</span>
          <span>{node_count} 个节点</span>
          <span>{edge_count} 条边</span>
        </div>
      </article>
    """


def extract_action_sequence(actions_text: str) -> list[str]:
    actions = []
    for chunk in actions_text.split(";"):
        match = re.search(r"\b([A-Z][A-Z0-9_]+)\b", chunk)
        if match:
            actions.append(match.group(1))
    return actions


def render_action_chips(actions: list[str], limit: int = 8) -> str:
    chips = [
        f'<span class="action-chip">{html.escape(action)}</span>'
        for action in actions[:limit]
    ]
    if len(actions) > limit:
        chips.append(f'<span class="action-chip more">+{len(actions) - limit}</span>')
    return "\n".join(chips)


def render_graph_thumbnail(graph: dict[str, Any], marker_suffix: str) -> str:
    positions = thumbnail_positions(graph)
    nodes_by_id = {node.get("id"): node for node in graph.get("nodes", [])}
    edges = []
    for edge in graph.get("edges", []):
        source = edge.get("source")
        target = edge.get("target")
        if source not in positions or target not in positions:
            continue
        x1, y1 = positions[source]
        x2, y2 = positions[target]
        x1, y1, x2, y2 = trim_thumbnail_edge(
            x1,
            y1,
            x2,
            y2,
            thumbnail_radius(nodes_by_id.get(source, {})) + 1.5,
            thumbnail_radius(nodes_by_id.get(target, {})) + 2.2,
        )
        color = EDGE_STYLE.get(edge.get("type"), "#94a3b8")
        edges.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{color}" stroke-width="2.15" opacity="0.74" marker-end="url(#arr-{marker_suffix})" />'
        )
    nodes = []
    for node in graph.get("nodes", []):
        node_id = node.get("id")
        if node_id not in positions:
            continue
        x, y = positions[node_id]
        stroke, fill = NODE_STYLE.get(node.get("type"), ("#64748b", "#f8fafc"))
        label = thumbnail_node_label(node)
        nodes.append(
            f'<g><circle cx="{x:.1f}" cy="{y:.1f}" r="{thumbnail_radius(node)}" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="2.15" />'
            f'<text x="{x:.1f}" y="{y + 2.6:.1f}" text-anchor="middle">{html.escape(label)}</text></g>'
        )
    return f"""
      <svg viewBox="0 0 260 126" role="img" aria-label="过程图缩略图">
        <defs>
          <marker id="arr-{marker_suffix}" markerWidth="4.8" markerHeight="4.2" refX="4.1" refY="2.1" orient="auto" markerUnits="userSpaceOnUse">
            <path d="M0,0 L4.8,2.1 L0,4.2 Z" fill="#475569"></path>
          </marker>
        </defs>
        <rect x="1" y="1" width="258" height="124" rx="8" fill="#fff" stroke="#e2e8f0" />
        {''.join(edges)}
        {''.join(nodes)}
      </svg>
    """


def trim_thumbnail_edge(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    source_pad: float,
    target_pad: float,
) -> tuple[float, float, float, float]:
    dx = x2 - x1
    dy = y2 - y1
    length = (dx * dx + dy * dy) ** 0.5
    if length <= source_pad + target_pad + 1:
        return x1, y1, x2, y2
    ux = dx / length
    uy = dy / length
    return (
        x1 + ux * source_pad,
        y1 + uy * source_pad,
        x2 - ux * target_pad,
        y2 - uy * target_pad,
    )


def thumbnail_positions(graph: dict[str, Any]) -> dict[str, tuple[float, float]]:
    operations = sorted(
        [node for node in graph.get("nodes", []) if node.get("type") == "operation"],
        key=lambda node: int(node.get("attrs", {}).get("step_id", 0)),
    )
    op_x: dict[int, float] = {}
    if operations:
        span = 205
        start = 34
        denom = max(len(operations) - 1, 1)
        for idx, node in enumerate(operations):
            step_id = int(node.get("attrs", {}).get("step_id", idx))
            op_x[step_id] = start + span * idx / denom

    step_by_node: dict[str, int] = {}
    for node in graph.get("nodes", []):
        attrs = node.get("attrs", {})
        if attrs.get("step_id") is not None:
            step_by_node[node["id"]] = int(attrs["step_id"])
    op_steps = {
        node["id"]: int(node.get("attrs", {}).get("step_id", 0))
        for node in operations
    }
    for edge in graph.get("edges", []):
        source = edge.get("source")
        target = edge.get("target")
        if source in op_steps and target not in step_by_node:
            step_by_node[target] = op_steps[source]
        if target in op_steps and source not in step_by_node:
            step_by_node[source] = op_steps[target]

    positions: dict[str, tuple[float, float]] = {}
    type_offsets: dict[tuple[int, str], int] = {}
    for node in graph.get("nodes", []):
        node_type = node.get("type")
        node_id = node.get("id")
        if node_type == "reaction":
            positions[node_id] = (16, 63)
        elif node_type == "operation":
            step_id = int(node.get("attrs", {}).get("step_id", 0))
            positions[node_id] = (op_x.get(step_id, 34), 63)
        else:
            step_id = step_by_node.get(node_id, 0)
            x = op_x.get(step_id, 34)
            key = (step_id, node_type)
            offset = type_offsets.get(key, 0)
            type_offsets[key] = offset + 1
            if node_type == "material_mention":
                positions[node_id] = (x + ((offset % 4) - 1.5) * 11, 18 + (offset // 4) * 15)
            elif node_type == "quantity":
                positions[node_id] = (x + (offset - 0.5) * 11, 100)
            elif node_type == "condition":
                positions[node_id] = (x + (offset - 0.5) * 11, 115)
            elif node_type == "state":
                positions[node_id] = (x + (offset - 0.5) * 11, 48)
            else:
                positions[node_id] = (x, 82)
    return positions


def thumbnail_node_label(node: dict[str, Any]) -> str:
    node_type = node.get("type")
    if node_type == "operation":
        return str(node.get("attrs", {}).get("step_id", ""))
    if node_type == "reaction":
        return "R"
    if node_type == "material_mention":
        return str(node.get("label", "m"))[:2]
    if node_type == "quantity":
        return "q"
    if node_type == "condition":
        return "c"
    if node_type == "state":
        return "s"
    return ""


def thumbnail_radius(node: dict[str, Any]) -> float:
    node_type = node.get("type")
    if node_type == "operation":
        return 9.2
    if node_type == "reaction":
        return 8.2
    return 6.8


def render_dataset_index(
    dataset_title: str,
    dataset_id: str,
    links: list[tuple[str, str, dict[str, Any]]],
) -> str:
    rows = "\n".join(
        f'<tr><td><a href="{html.escape(filename)}">{html.escape(graph_id)}</a></td>'
        f"<td>{record.get('index')}</td><td>{record.get('_split')}</td>"
        f"<td>{record.get('_features', {}).get('action_count')}</td>"
        f"<td>{html.escape(str(record.get('_buckets', {}).get('scale', '')))}</td></tr>"
        for filename, graph_id, record in links
    )
    return page(
        dataset_title,
        f"""
        <nav><a href="../index.html">All datasets</a></nav>
        <h1>{html.escape(dataset_title)}</h1>
        <table>
          <thead><tr><th>Graph</th><th>OpenExp index</th><th>Split</th><th>Actions</th><th>Scale</th></tr></thead>
          <tbody>{rows}</tbody>
        </table>
        """,
    )


def render_example_page(
    dataset_id: str,
    dataset_title: str,
    record: dict[str, Any],
    graph: dict[str, Any],
) -> str:
    trace = model_operation_trace(graph)
    graph_svg = render_graph_svg(graph)
    original_json = {
        key: record.get(key)
        for key in (
            "index",
            "REACTANT",
            "PRODUCT",
            "CATALYST",
            "SOLVENT",
            "actions",
            "source",
            "extracted_molecules",
            "extracted_temperature",
            "extracted_duration",
            "_features",
            "_buckets",
            "_split",
        )
    }
    graph_stats = {
        "nodes": len(graph["nodes"]),
        "edges": len(graph["edges"]),
        "input_materials": material_table_size(graph),
        "slot_nodes": count_nodes(graph, "slot"),
        "material_entity_nodes": count_nodes(graph, "material_entity"),
        "state_policy": graph["constraints"].get("state_policy"),
        "target_profile": graph["constraints"]["target_profile"],
        "includes_surface_attrs": graph["constraints"]["includes_surface_attrs"],
        "operation_count": graph["constraints"]["operation_count"],
    }
    return page(
        f"{dataset_title} / {graph['graph_id']}",
        f"""
        <nav><a href="../index.html">Dataset index</a> · <a href="../../index.html">All datasets</a></nav>
        <h1>{html.escape(dataset_title)} · {html.escape(graph['graph_id'])}</h1>
        <section class="summary-grid">
          {render_kv_card("Model Target Graph", graph_stats)}
          {render_kv_card("Bucket Labels", record.get("_buckets", {}))}
        </section>
        <section>
          <h2>Original OpenExp Data</h2>
          <pre>{html.escape(json.dumps(original_json, ensure_ascii=False, indent=2))}</pre>
        </section>
        <section>
          <h2>Model Input Materials</h2>
          {render_material_input_table(graph)}
        </section>
        <section>
          <h2>Semantic RGDL Target Graph</h2>
          <div class="graph-shell">{graph_svg}</div>
        </section>
        <section>
          <h2>Operation Trace</h2>
          <pre>{html.escape(json.dumps(trace, ensure_ascii=False, indent=2))}</pre>
        </section>
        """,
    )


def render_kv_card(title: str, data: dict[str, Any]) -> str:
    rows = "\n".join(
        f"<tr><th>{html.escape(str(key))}</th><td>{html.escape(json.dumps(value, ensure_ascii=False))}</td></tr>"
        for key, value in data.items()
    )
    return f"<section><h2>{html.escape(title)}</h2><table>{rows}</table></section>"


def render_material_input_table(graph: dict[str, Any]) -> str:
    rows = []
    input_materials = graph.get("reaction", {}).get("input_materials", {})
    for category in ("REACTANT", "CATALYST", "SOLVENT", "PRODUCT"):
        for material in input_materials.get(category, []):
            rows.append(
                "<tr>"
                f"<td>{html.escape(category)}</td>"
                f"<td>{html.escape(str(material.get('vocab_index', '')))}</td>"
                f"<td>{html.escape(str(material.get('material_id', '')))}</td>"
                f"<td>{html.escape(', '.join(material.get('names') or []))}</td>"
                f"<td>{html.escape(', '.join(material.get('sources') or []))}</td>"
                f"<td>{html.escape(str(material.get('smiles', '')))}</td>"
                "</tr>"
            )
    if not rows:
        return "<p>No input materials.</p>"
    return (
        "<table>"
        "<thead><tr><th>Category</th><th>Index</th><th>Material ID</th><th>Names</th><th>Source</th><th>SMILES</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table>"
    )


def model_operation_trace(graph: dict[str, Any]) -> list[dict[str, Any]]:
    operation_nodes = {
        node["id"]: node
        for node in graph.get("nodes", [])
        if node.get("type") == "operation"
    }
    mention_nodes = {
        node["id"]: node
        for node in graph.get("nodes", [])
        if node.get("type") == "material_mention"
    }
    mentions_by_op: dict[str, list[str]] = {}
    conditions_by_op: dict[str, list[str]] = {}
    outputs_by_op: dict[str, list[str]] = {}
    for edge in graph.get("edges", []):
        if edge.get("type") == "mentions":
            mention = mention_nodes.get(edge["target"])
            label = mention.get("label", edge["target"]) if mention else edge["target"]
            mentions_by_op.setdefault(edge["source"], []).append(label)
        elif edge.get("type") == "has_condition":
            conditions_by_op.setdefault(edge["source"], []).append(edge["target"])
        elif edge.get("type") == "output_from":
            outputs_by_op.setdefault(edge["source"], []).append(edge["target"])

    rows = []
    for operation_id, node in sorted(
        operation_nodes.items(),
        key=lambda item: int(item[1].get("attrs", {}).get("step_id", 0)),
    ):
        attrs = node.get("attrs", {})
        rows.append(
            {
                "step_id": attrs.get("step_id"),
                "operation": attrs.get("operation_type") or node.get("label"),
                "material_indices": mentions_by_op.get(operation_id, []),
                "condition_nodes": conditions_by_op.get(operation_id, []),
                "output_nodes": outputs_by_op.get(operation_id, []),
            }
        )
    return rows


def render_graph_svg(graph: dict[str, Any]) -> str:
    positions = layout_nodes(graph)
    width = max((x for x, _ in positions.values()), default=1000) + 220
    height = max((y for _, y in positions.values()), default=700) + 120
    edges = "\n".join(render_edge(edge, positions) for edge in graph["edges"])
    nodes = "\n".join(render_node(node, positions[node["id"]]) for node in graph["nodes"] if node["id"] in positions)
    legend = render_legend(graph)
    return f"""
    {legend}
    <svg viewBox="0 0 {width} {height}" role="img" aria-label="Compiled RGDL graph">
      <defs>
        <marker id="arrow" markerWidth="10" markerHeight="8" refX="9" refY="4" orient="auto">
          <path d="M0,0 L10,4 L0,8 Z" fill="#334155"></path>
        </marker>
      </defs>
      {edges}
      {nodes}
    </svg>
    """


def layout_nodes(graph: dict[str, Any]) -> dict[str, tuple[float, float]]:
    positions: dict[str, tuple[float, float]] = {}
    operation_steps = sorted(
        {
            int(node.get("attrs", {}).get("step_id", 0))
            for node in graph["nodes"]
            if node.get("type") == "operation"
        }
    )
    if not operation_steps:
        operation_steps = [0]

    step_nodes: dict[int, dict[str, list[dict[str, Any]]]] = {
        step: {"material_mention": [], "side": [], "state": [], "operation": []}
        for step in operation_steps
    }
    floating_nodes: list[dict[str, Any]] = []
    for node in graph["nodes"]:
        node_type = node["type"]
        attrs = node.get("attrs", {})
        raw_step = attrs.get("step_id")
        if node_type == "reaction":
            continue
        if raw_step is None:
            floating_nodes.append(node)
            continue
        step_id = int(raw_step)
        step_nodes.setdefault(step_id, {"material_mention": [], "side": [], "state": [], "operation": []})
        if node_type == "operation":
            step_nodes[step_id]["operation"].append(node)
        elif node_type == "material_mention":
            step_nodes[step_id]["material_mention"].append(node)
        elif node_type in {"quantity", "condition"}:
            step_nodes[step_id]["side"].append(node)
        elif node_type == "state":
            step_nodes[step_id]["state"].append(node)
        else:
            step_nodes[step_id]["side"].append(node)

    row_centers: dict[int, float] = {}
    cursor = 94.0
    for step_id in sorted(step_nodes):
        groups = step_nodes[step_id]
        max_stack = max(
            1,
            len(groups["material_mention"]),
            len(groups["side"]),
            len(groups["state"]),
        )
        row_height = max(104.0, 64.0 + max_stack * 38.0)
        row_centers[step_id] = cursor + row_height / 2.0
        cursor += row_height

    for node in graph["nodes"]:
        if node["type"] == "reaction":
            positions[node["id"]] = (560.0, 40.0)

    for step_id, groups in sorted(step_nodes.items()):
        center_y = row_centers[step_id]
        for node in groups["operation"]:
            positions[node["id"]] = (560.0, center_y)
        for x, key in ((330.0, "material_mention"), (790.0, "state"), (1010.0, "side")):
            for node, y in zip(groups[key], distribute_y(center_y, len(groups[key]))):
                positions[node["id"]] = (x, y)

    floating_counts: dict[str, int] = {}
    bottom_y = cursor + 40.0
    for node in floating_nodes:
        if node["id"] in positions:
            continue
        node_type = node["type"]
        idx = floating_counts.get(node_type, 0)
        floating_counts[node_type] = idx + 1
        if node_type == "condition":
            positions[node["id"]] = (1010.0, bottom_y + idx * 72.0)
        else:
            positions[node["id"]] = (790.0, bottom_y + idx * 72.0)
    return positions


def distribute_y(center_y: float, count: int, gap: float = 38.0) -> list[float]:
    if count <= 0:
        return []
    start = center_y - gap * (count - 1) / 2.0
    return [start + gap * idx for idx in range(count)]


def render_node(node: dict[str, Any], position: tuple[float, float]) -> str:
    x, y = position
    node_type = node["type"]
    stroke, fill = NODE_STYLE.get(node_type, ("#475569", "#f8fafc"))
    label = shorten(str(node.get("label", node["id"])), 30)
    width = 150 if node_type in {"operation", "state"} else 132
    return f"""
    <g>
      <rect x="{x - width / 2:.1f}" y="{y - 26}" width="{width}" height="52" rx="8" fill="{fill}" stroke="{stroke}" stroke-width="2.4" />
      <text x="{x}" y="{y - 6}" class="node-type">{html.escape(node_type)}</text>
      <text x="{x}" y="{y + 13}" class="node-label">{html.escape(label)}</text>
      <title>{html.escape(node['id'])}: {html.escape(str(node.get('label', '')))}</title>
    </g>
    """


def render_edge(edge: dict[str, Any], positions: dict[str, tuple[float, float]]) -> str:
    source = edge["source"]
    target = edge["target"]
    if source not in positions or target not in positions:
        return ""
    x1, y1 = positions[source]
    x2, y2 = positions[target]
    color = EDGE_STYLE.get(edge["type"], "#64748b")
    label_x = (x1 + x2) / 2
    label_y = (y1 + y2) / 2 - 4
    return f"""
    <line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="2.15" opacity="0.72" marker-end="url(#arrow)" />
    <text x="{label_x}" y="{label_y}" class="edge-label">{html.escape(edge['type'])}</text>
    """


def render_legend(graph: dict[str, Any]) -> str:
    chips = []
    node_types = sorted({node["type"] for node in graph["nodes"]})
    for node_type in node_types:
        stroke, fill = NODE_STYLE.get(node_type, ("#475569", "#f8fafc"))
        chips.append(
            f'<span class="chip" style="border-color:{stroke};background:{fill}">{html.escape(node_type)}</span>'
        )
    return '<div class="legend">' + "\n".join(chips) + "</div>"


def page(title: str, body: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>{CSS}</style>
</head>
<body>
  <main>{body}</main>
</body>
</html>
"""


def shorten(value: str, limit: int) -> str:
    return value if len(value) <= limit else value[: limit - 1] + "..."


def count_nodes(graph: dict[str, Any], node_type: str) -> int:
    return sum(1 for node in graph["nodes"] if node["type"] == node_type)


def material_table_size(graph: dict[str, Any]) -> int:
    input_materials = graph.get("reaction", {}).get("input_materials", {})
    return sum(len(items) for items in input_materials.values())


def render_validation_note() -> str:
    return """# RGDL Semantic Target Graphs

These sample graphs are compiled with:

```python
compile_openexp_record_to_rgdl(
    record,
    include_slots=False,
    include_surface_attrs=False,
)
```

They are intended as model-generation targets. Surface `slot` nodes are omitted,
quantities are attached directly to operations through `has_quantity` edges, and
all material mentions resolve to the per-reaction input material table built from
`REACTANT`, `CATALYST`, `SOLVENT`, and `PRODUCT`. Literal action materials that
can be resolved through the record's `molecules` dictionary are added to this
same table before graph construction.
Material nodes and material mentions in the target graph are labeled only by
their input-table index, for example `m0`, `m1`, and `m2`.
The compact target keeps material mentions but omits material entity nodes;
ordinary process states are omitted, while distinct states such as solution,
filtered branches, layers, and concentrated states remain explicit.
The original OpenExp text remains available on the HTML pages for comparison,
but exact character-level round-trip reconstruction is intentionally not the
objective for this target profile.
"""


ACTION_GALLERY_CSS = """
:root {
  color: #111827;
  font-family: Arial, Helvetica, sans-serif;
  line-height: 1.28;
}
* {
  box-sizing: border-box;
}
body {
  margin: 0;
  background: #f6f8fb;
}
main {
  width: min(1880px, calc(100vw - 28px));
  margin: 0 auto;
  padding: 16px 0 22px;
}
.page-head {
  display: grid;
  grid-template-columns: 1fr auto;
  align-items: end;
  gap: 16px;
  margin-bottom: 12px;
}
h1 {
  margin: 0 0 4px;
  font-size: 24px;
  letter-spacing: 0;
}
p {
  margin: 0;
  color: #64748b;
  font-size: 13px;
}
a {
  color: #1d4ed8;
  font-size: 13px;
  text-decoration: none;
}
.gallery {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(245px, 1fr));
  gap: 9px;
}
.case-card {
  min-width: 0;
  border: 1.4px solid #94a3b8;
  border-radius: 8px;
  background: #fff;
  padding: 8px;
  box-shadow: 0 6px 14px rgba(15, 23, 42, 0.10);
}
.card-top {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 8px;
  margin-bottom: 5px;
}
.case-id {
  color: #020617;
  font-size: 16px;
  font-weight: 800;
}
.thumb {
  height: 126px;
  overflow: hidden;
  border-radius: 8px;
  background: #fff;
}
.thumb svg {
  display: block;
  width: 100%;
  height: 126px;
}
.thumb text {
  fill: #020617;
  font-size: 8.3px;
  font-weight: 900;
  pointer-events: none;
}
.main-action {
  display: grid;
  grid-template-columns: 70px 1fr;
  gap: 7px;
  align-items: baseline;
  margin-top: 7px;
  min-height: 28px;
}
.main-action span {
  color: #334155;
  font-size: 11.5px;
  font-weight: 700;
}
.main-action strong {
  overflow: hidden;
  color: #020617;
  font-size: 12.8px;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.action-row {
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
  min-height: 42px;
  align-content: flex-start;
  margin-top: 6px;
}
.action-chip {
  border: 1.25px solid #64748b;
  border-radius: 999px;
  background: #ffffff;
  color: #0f172a;
  padding: 2px 6px;
  font-family: "SFMono-Regular", Consolas, "Liberation Mono", monospace;
  font-size: 11px;
  font-weight: 700;
  line-height: 1.25;
}
.action-chip.more {
  border-color: #93c5fd;
  background: #eff6ff;
  color: #1d4ed8;
}
.stats {
  display: flex;
  justify-content: space-between;
  gap: 6px;
  margin-top: 6px;
  border-top: 1px solid #cbd5e1;
  padding-top: 6px;
  color: #334155;
  font-size: 11px;
  font-weight: 700;
}
@media (max-width: 900px) {
  main {
    width: min(100vw - 18px, 720px);
  }
  .page-head {
    grid-template-columns: 1fr;
  }
  .gallery {
    grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
  }
}
"""


CSS = """
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
  max-width: 1280px;
  margin: 0 auto;
  padding: 24px;
}
h1 {
  margin: 8px 0 16px;
  font-size: 28px;
}
h2 {
  margin: 22px 0 10px;
  font-size: 18px;
}
h3 {
  margin: 14px 0 6px;
  font-size: 14px;
}
a {
  color: #1d4ed8;
}
table {
  border-collapse: collapse;
  width: 100%;
  background: #fff;
}
th,
td {
  border: 1px solid #dbe3ef;
  padding: 7px 9px;
  text-align: left;
  vertical-align: top;
}
th {
  background: #f1f5f9;
}
.summary-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
  gap: 16px;
}
.graph-shell {
  overflow: auto;
  border: 1px solid #cbd5e1;
  border-radius: 8px;
  background: #fff;
  padding: 12px;
}
svg {
  display: block;
  min-width: 1160px;
}
.legend {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin: 0 0 12px;
}
.chip {
  border: 1px solid;
  border-radius: 999px;
  padding: 3px 9px;
  font-size: 12px;
}
.node-type {
  dominant-baseline: middle;
  fill: #0f172a;
  font-size: 10.5px;
  font-weight: 800;
  text-anchor: middle;
  text-transform: uppercase;
}
.node-label {
  dominant-baseline: middle;
  fill: #020617;
  font-size: 11.5px;
  font-weight: 700;
  text-anchor: middle;
}
.edge-label {
  fill: #1f2937;
  font-size: 9.5px;
  font-weight: 700;
  text-anchor: middle;
}
pre,
.mono {
  font-family: "SFMono-Regular", Consolas, "Liberation Mono", monospace;
}
pre {
  overflow: auto;
  padding: 12px;
  border-radius: 8px;
  background: #0f172a;
  color: #e2e8f0;
}
.mono {
  padding: 10px;
  border: 1px solid #dbe3ef;
  border-radius: 8px;
  background: #fff;
}
"""


if __name__ == "__main__":
    main()
