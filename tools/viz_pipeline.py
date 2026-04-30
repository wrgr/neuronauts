#!/usr/bin/env python3
"""Pipeline inspection visualizer — generates a self-contained HTML viewer.

Usage
-----
# Path-encoder stage (shows per-window TP/FP/FN/TN in 3-D)
python tools/viz_pipeline.py path-encoder \
    --checkpoint models/path_encoder_v3_ep10.pt \
    --cache-dir data/boxes_30um \
    --n-examples 2000 \
    --output viewer.html

# Grammar stage (shows per-edge TP/FP/FN/TN)
python tools/viz_pipeline.py grammar \
    --checkpoint models/grammar_cave_real_50.pt \
    --cache-dir data/boxes_30um \
    --box-id 0216eed2e33df4df \
    --output viewer.html

# CellGNN stage
python tools/viz_pipeline.py cell-gnn \
    --checkpoint models/cell_gnn_path16_ep50.pt \
    --cache-dir data/boxes_30um \
    --box-id 0216eed2e33df4df \
    --output viewer.html
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# HTML template (Plotly CDN, no server needed)
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Neuronauts Pipeline Viewer — {stage_name}</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
  body {{ margin:0; font-family: monospace; background:#111; color:#ccc; display:flex; height:100vh; }}
  #sidebar {{ width:280px; min-width:220px; padding:12px; overflow-y:auto; background:#1a1a1a; border-right:1px solid #333; }}
  #main {{ flex:1; display:flex; flex-direction:column; }}
  #plot {{ flex:1; }}
  h2 {{ color:#7ec8e3; margin:4px 0 10px; font-size:1em; }}
  .metric-box {{ background:#222; border:1px solid #444; border-radius:4px; padding:8px; margin-bottom:8px; }}
  .metric-row {{ display:flex; justify-content:space-between; padding:2px 0; }}
  .label {{ color:#888; }}
  .val {{ color:#fff; font-weight:bold; }}
  .tp {{ color:#4caf50; }} .fp {{ color:#ff9800; }} .fn {{ color:#f44336; }} .tn {{ color:#607d8b; }}
  label {{ display:block; margin-top:8px; color:#aaa; font-size:0.85em; }}
  input[type=range] {{ width:100%; }}
  select {{ width:100%; background:#222; color:#ccc; border:1px solid #444; padding:4px; }}
  #detail {{ padding:8px 12px; background:#161616; border-top:1px solid #333; font-size:0.8em; min-height:60px; max-height:120px; overflow-y:auto; }}
  .dot-tp {{ color:#4caf50; }} .dot-fp {{ color:#ff9800; }} .dot-fn {{ color:#f44336; }} .dot-tn {{ color:#607d8b; }}
</style>
</head>
<body>
<div id="sidebar">
  <h2>&#9670; {stage_name}</h2>
  <div id="metrics" class="metric-box"></div>
  <label>Threshold (positive if score &ge; t)
    <input type="range" id="thresh" min="0" max="1" step="0.01" value="0.5"
           oninput="document.getElementById('tval').textContent=parseFloat(this.value).toFixed(2); updatePlot()"/>
    t = <span id="tval">0.50</span>
  </label>
  <label>Color by
    <select id="colorby" onchange="updatePlot()">
      <option value="outcome">Outcome (TP/FP/FN/TN)</option>
      <option value="score">Predicted score</option>
      <option value="gt">Ground truth</option>
    </select>
  </label>
  <label>Show
    <select id="showfilter" onchange="updatePlot()">
      <option value="all">All examples</option>
      <option value="errors">Errors only (FP + FN)</option>
      <option value="tp">TP only</option>
      <option value="fp">FP only</option>
      <option value="fn">FN only</option>
      <option value="tn">TN only</option>
    </select>
  </label>
  <div style="margin-top:12px; font-size:0.75em; color:#555;">
    {meta_html}
  </div>
</div>
<div id="main">
  <div id="plot"></div>
  <div id="detail">&#8592; hover a point for details</div>
</div>

<script>
const DATA = {data_json};

function classify(score, gt, thresh) {{
  const pred = score >= thresh ? 1 : 0;
  if (pred === 1 && gt === 1) return 'tp';
  if (pred === 1 && gt === 0) return 'fp';
  if (pred === 0 && gt === 1) return 'fn';
  return 'tn';
}}

const COLORS = {{tp:'#4caf50', fp:'#ff9800', fn:'#f44336', tn:'#607d8b'}};
const OUTCOME_ORDER = ['tp','fp','fn','tn'];

function updatePlot() {{
  const thresh = parseFloat(document.getElementById('thresh').value);
  const colorby = document.getElementById('colorby').value;
  const showfilter = document.getElementById('showfilter').value;

  const outcomes = DATA.scores.map((s,i) => classify(s, DATA.gt[i], thresh));

  // Metrics
  let tp=0, fp=0, fn=0, tn=0;
  outcomes.forEach(o => {{ if(o==='tp')tp++; else if(o==='fp')fp++; else if(o==='fn')fn++; else tn++; }});
  const n = outcomes.length;
  const prec = tp/(tp+fp+1e-9), rec = tp/(tp+fn+1e-9);
  const f1 = 2*prec*rec/(prec+rec+1e-9), acc = (tp+tn)/n;
  document.getElementById('metrics').innerHTML = `
    <div class="metric-row"><span class="label">n examples</span><span class="val">${{n}}</span></div>
    <div class="metric-row"><span class="label">Accuracy</span><span class="val">${{(acc*100).toFixed(1)}}%</span></div>
    <div class="metric-row"><span class="label">Precision</span><span class="val">${{(prec*100).toFixed(1)}}%</span></div>
    <div class="metric-row"><span class="label">Recall</span><span class="val">${{(rec*100).toFixed(1)}}%</span></div>
    <div class="metric-row"><span class="label">F1</span><span class="val">${{(f1*100).toFixed(1)}}%</span></div>
    <div style="margin-top:6px;">
      <span class="tp">&#9632; TP ${{tp}}</span> &nbsp;
      <span class="fp">&#9632; FP ${{fp}}</span><br/>
      <span class="fn">&#9632; FN ${{fn}}</span> &nbsp;
      <span class="tn">&#9632; TN ${{tn}}</span>
    </div>`;

  // Filter
  let mask;
  if (showfilter === 'all') mask = outcomes.map(() => true);
  else if (showfilter === 'errors') mask = outcomes.map(o => o==='fp'||o==='fn');
  else mask = outcomes.map(o => o===showfilter);

  const xs=[], ys=[], zs=[], cs=[], ts=[], hovers=[], sizes=[];
  DATA.x.forEach((x,i) => {{
    if (!mask[i]) return;
    xs.push(x); ys.push(DATA.y[i]); zs.push(DATA.z[i]);
    const o = outcomes[i];
    if (colorby === 'outcome') cs.push(COLORS[o]);
    else if (colorby === 'score') cs.push(DATA.scores[i]);
    else cs.push(DATA.gt[i]);
    ts.push(`${{o.toUpperCase()}}<br>score: ${{DATA.scores[i].toFixed(3)}}<br>gt: ${{DATA.gt[i]}}<br>${{DATA.labels[i]||''}}`);
    hovers.push(i);
    sizes.push(o==='tp'||o==='tn' ? 4 : 7);
  }});

  const trace = {{
    type: 'scatter3d', mode: 'markers',
    x: xs, y: ys, z: zs,
    text: ts,
    hoverinfo: 'text',
    customdata: hovers,
    marker: {{
      size: sizes,
      color: cs,
      colorscale: colorby==='outcome' ? null : 'RdYlGn',
      cmin: colorby==='score' ? 0 : undefined,
      cmax: colorby==='score' ? 1 : undefined,
      showscale: colorby !== 'outcome',
      opacity: 0.85,
    }}
  }};

  Plotly.react('plot', [trace], {{
    paper_bgcolor:'#111', plot_bgcolor:'#111',
    margin:{{l:0,r:0,t:0,b:0}},
    scene:{{
      xaxis:{{title:'X (µm)', color:'#666', gridcolor:'#333', backgroundcolor:'#111'}},
      yaxis:{{title:'Y (µm)', color:'#666', gridcolor:'#333', backgroundcolor:'#111'}},
      zaxis:{{title:'Z (µm)', color:'#666', gridcolor:'#333', backgroundcolor:'#111'}},
      bgcolor:'#111',
    }},
    font:{{color:'#ccc'}},
    uirevision:'stable',
  }}, {{responsive:true}});

  document.getElementById('plot').on('plotly_hover', function(ev) {{
    const pt = ev.points[0];
    const i = pt.customdata;
    const o = outcomes[i];
    document.getElementById('detail').innerHTML =
      `<span class="dot-${{o}}">&#9632; ${{o.toUpperCase()}}</span> &nbsp;|&nbsp; `+
      `score: <b>${{DATA.scores[i].toFixed(4)}}</b> &nbsp;|&nbsp; `+
      `gt: ${{DATA.gt[i]}} &nbsp;|&nbsp; `+
      `pos: (${{DATA.x[i].toFixed(1)}}, ${{DATA.y[i].toFixed(1)}}, ${{DATA.z[i].toFixed(1)}}) µm`+
      (DATA.labels[i] ? `<br/>${{DATA.labels[i]}}` : '');
  }});
}}

updatePlot();
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Stage implementations
# ---------------------------------------------------------------------------

def _nm_to_um(pts: np.ndarray) -> np.ndarray:
    return pts.astype(np.float32) * 1e-3


def _window_centroid_um(chain: np.ndarray, start: int, size: int) -> np.ndarray:
    return _nm_to_um(chain[start : start + size].mean(axis=0))


def run_path_encoder(args: argparse.Namespace) -> dict:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    import torch
    from neuronauts.dataset_builder import BoxCache
    from neuronauts.path_dataset import (
        extract_cell_chains,
        generate_path_examples,
        load_path_encoder,
    )
    from neuronauts.path_edge_encoder import pad_path_sequences

    cache = BoxCache(args.cache_dir)
    print("Extracting chains…")
    chains = extract_cell_chains(cache, role="pre", min_synapses_per_cell=5)
    print(f"  {len(chains)} cells")

    rng = np.random.default_rng(0)
    feats_list, labels = generate_path_examples(
        chains,
        window_size=args.window_size,
        hard_neg_fraction=args.hard_neg_fraction,
        max_examples=args.n_examples,
        rng=rng,
    )
    labels = labels.astype(bool)
    print(f"  {len(labels)} examples  ({labels.sum()} pos, {(~labels).sum()} neg)")

    encoder, head = load_path_encoder(args.checkpoint)
    encoder.eval()
    head.eval()

    # Build centroids for spatial display (re-generate with same seed to get positions)
    rng2 = np.random.default_rng(0)
    cell_ids = list(chains.keys())
    centroids_nm: list[np.ndarray] = []
    detail_labels: list[str] = []

    # Generate centroids alongside examples
    cell_arr = [chains[c] for c in cell_ids if len(chains[c]) >= args.window_size]
    n_pos_needed = len(labels[labels])
    n_neg_needed = int((~labels).sum())
    pos_done = neg_done = 0
    for _ in range(len(labels) * 10):
        if pos_done + neg_done >= len(labels):
            break
        c = cell_arr[rng2.integers(len(cell_arr))]
        start = rng2.integers(0, len(c) - args.window_size + 1)
        centroids_nm.append(c[start : start + args.window_size].mean(axis=0))
        if pos_done < n_pos_needed:
            detail_labels.append("positive window")
            pos_done += 1
        else:
            detail_labels.append("negative window")
            neg_done += 1

    # Pad centroids to match examples if needed
    while len(centroids_nm) < len(labels):
        centroids_nm.append(centroids_nm[-1] if centroids_nm else np.zeros(3))
        detail_labels.append("")

    # Run inference
    scores: list[float] = []
    batch_size = 256
    with torch.no_grad():
        for start in range(0, len(feats_list), batch_size):
            batch = feats_list[start : start + batch_size]
            padded, mask, has_path = pad_path_sequences(batch, max_len=args.window_size - 1)
            t = torch.tensor(padded, dtype=torch.float32)
            m = torch.tensor(mask, dtype=torch.bool)
            hp = torch.tensor(has_path, dtype=torch.bool)
            emb = encoder(t, m, hp)
            logits = head(emb).squeeze(-1)
            probs = torch.sigmoid(logits).cpu().numpy().tolist()
            scores.extend(probs)

    xyz = np.array(centroids_nm[: len(labels)], dtype=np.float32) * 1e-3  # → µm

    return {
        "x": xyz[:, 0].tolist(),
        "y": xyz[:, 1].tolist(),
        "z": xyz[:, 2].tolist(),
        "scores": scores[: len(labels)],
        "gt": labels.astype(int).tolist(),
        "labels": detail_labels[: len(labels)],
    }, {
        "stage": "Path Encoder",
        "checkpoint": args.checkpoint,
        "n_examples": len(labels),
        "window_size": args.window_size,
        "hard_neg_fraction": args.hard_neg_fraction,
    }


def run_grammar(args: argparse.Namespace) -> dict:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from neuronauts.dataset_builder import BoxCache
    from neuronauts.shared_grammar_model import load_shared_grammar_model
    import torch

    cache = BoxCache(args.cache_dir)

    # Pick a loadable box
    records = list(cache.iter_records())
    if args.box_id:
        records = [r for r in records if r.box_hash == args.box_id]
    record = synapses = None
    for r in records:
        try:
            _, synapses = cache.load(r, load_volume=False)
            record = r
            break
        except Exception:
            continue
    if record is None:
        raise ValueError(f"No loadable box found (box_id={args.box_id!r})")
    print(f"Box: {record.box_hash}")

    model = load_shared_grammar_model(args.checkpoint)
    model.eval()

    # Build edge pairs: same root_id = positive merge
    pre_ids = synapses.pre_root_id
    post_ids = synapses.post_root_id
    pre_pts = synapses.pre_pt.astype(np.float32)
    post_pts = synapses.post_pt.astype(np.float32)

    # Group pre-synapses by root_id
    from collections import defaultdict
    pre_by_root: dict[int, list[int]] = defaultdict(list)
    for i, rid in enumerate(pre_ids):
        if rid != 0:
            pre_by_root[int(rid)].append(i)

    rng = np.random.default_rng(0)
    root_list = [r for r, idx in pre_by_root.items() if len(idx) >= 3]
    print(f"  {len(root_list)} roots with ≥3 pre-synapses")

    from neuronauts.grammar import featurize_path_points
    _MIP2 = np.array([32.0, 32.0, 40.0], dtype=np.float32)
    _NM_TO_UM = np.array([1e-3, 1e-3, 1e-3], dtype=np.float32)

    pos_pairs, neg_pairs = [], []
    centroids_pos, centroids_neg = [], []

    # Positive pairs: two halves of the same cell
    for rid in rng.choice(root_list, size=min(500, len(root_list)), replace=False):
        idx = pre_by_root[rid]
        pts_nm = (pre_pts[idx] * _MIP2).astype(np.float32)
        if len(pts_nm) < 4:
            continue
        half = len(pts_nm) // 2
        left = pts_nm[:half]
        right = pts_nm[half:]
        pos_pairs.append((left, right))
        centroids_pos.append(pts_nm.mean(axis=0) * 1e-3)

    # Negative pairs: two halves from different cells
    for _ in range(len(pos_pairs)):
        ra, rb = rng.choice(root_list, size=2, replace=False)
        la = pre_pts[pre_by_root[ra]] * _MIP2
        lb = pre_pts[pre_by_root[rb]] * _MIP2
        if len(la) < 2 or len(lb) < 2:
            continue
        neg_pairs.append((la[: len(la) // 2], lb[len(lb) // 2 :]))
        centroid = (la.mean(axis=0) + lb.mean(axis=0)) / 2 * 1e-3
        centroids_neg.append(centroid)

    def score_pair(left_nm, right_nm):
        lf = featurize_path_points(left_nm * _NM_TO_UM, iso_scale=np.ones(3, dtype=np.float32))
        rf = featurize_path_points(right_nm * _NM_TO_UM, iso_scale=np.ones(3, dtype=np.float32))
        if lf is None or rf is None or len(lf) == 0 or len(rf) == 0:
            return 0.5
        lf_t = torch.tensor(lf[np.newaxis], dtype=torch.float32)
        rf_t = torch.tensor(rf[np.newaxis], dtype=torch.float32)
        lm = torch.ones(1, lf.shape[0], dtype=torch.bool)
        rm = torch.ones(1, rf.shape[0], dtype=torch.bool)
        with torch.no_grad():
            logit = model.score_merge(lf_t, lm, rf_t, rm)
            return float(torch.sigmoid(logit).squeeze())

    print("Scoring pairs…")
    scores, gts, centroids_all, detail_labels = [], [], [], []
    for left, right in pos_pairs:
        scores.append(score_pair(left, right))
        gts.append(1)
    centroids_all.extend(centroids_pos)
    detail_labels.extend([f"pos (same cell)"] * len(pos_pairs))

    for left, right in neg_pairs:
        scores.append(score_pair(left, right))
        gts.append(0)
    centroids_all.extend(centroids_neg)
    detail_labels.extend([f"neg (diff cell)"] * len(neg_pairs))

    xyz = np.array(centroids_all, dtype=np.float32)
    return {
        "x": xyz[:, 0].tolist(),
        "y": xyz[:, 1].tolist(),
        "z": xyz[:, 2].tolist(),
        "scores": scores,
        "gt": gts,
        "labels": detail_labels,
    }, {
        "stage": "Grammar (merge)",
        "checkpoint": args.checkpoint,
        "box": record.box_hash,
        "n_pos": len(pos_pairs),
        "n_neg": len(neg_pairs),
    }


def run_cell_gnn(args: argparse.Namespace) -> dict:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from neuronauts.dataset_builder import BoxCache
    from neuronauts.cell_graph import (
        load_cell_gnn,
        build_synapse_graph,
        infer_cells,
        CellGNNConfig,
    )
    import torch

    cache = BoxCache(args.cache_dir)
    records = list(cache.iter_records())
    if args.box_id:
        records = [r for r in records if r.box_hash == args.box_id]
    record = synapses = None
    for r in records:
        try:
            _, synapses = cache.load(r, load_volume=False)
            record = r
            break
        except Exception:
            continue
    if record is None:
        raise ValueError(f"No loadable box found (box_id={args.box_id!r})")
    print(f"Box: {record.box_hash}")

    model_data = load_cell_gnn(args.checkpoint)
    model = model_data["model"] if isinstance(model_data, dict) else model_data
    model.eval()

    print("Building synapse graph…")
    graph = build_synapse_graph(synapses)

    print("Running inference…")
    with torch.no_grad():
        pred_labels = infer_cells(model, graph)

    gt_labels = synapses.pre_root_id.astype(np.int64)
    _MIP2 = np.array([32.0, 32.0, 40.0], dtype=np.float32)
    pts_um = synapses.pre_pt.astype(np.float32) * _MIP2 * 1e-3

    # Map GT root_ids to consecutive ints
    unique_gt = {r: i for i, r in enumerate(np.unique(gt_labels))}
    gt_int = np.array([unique_gt[r] for r in gt_labels], dtype=np.int32)

    # Compute per-synapse "correct cell" score:
    # A synapse is correctly grouped if its predicted cluster ID matches at
    # least 50% of its GT cell peers.
    from collections import Counter
    pred_to_gt: dict[int, Counter] = defaultdict(Counter)
    for pid, gid in zip(pred_labels, gt_int):
        pred_to_gt[int(pid)][int(gid)] += 1
    pred_majority = {pid: cnt.most_common(1)[0][0] for pid, cnt in pred_to_gt.items()}

    scores, gts = [], []
    for pid, gid in zip(pred_labels, gt_int):
        majority = pred_majority[int(pid)]
        # score = fraction of cluster that is this GT cell
        cluster_size = sum(pred_to_gt[int(pid)].values())
        score = pred_to_gt[int(pid)][int(gid)] / cluster_size
        scores.append(float(score))
        gts.append(1 if int(gid) == majority else 0)

    detail_labels = [
        f"pred_cluster={int(pid)}  gt_root={int(gid)}"
        for pid, gid in zip(pred_labels, gt_labels)
    ]

    return {
        "x": pts_um[:, 0].tolist(),
        "y": pts_um[:, 1].tolist(),
        "z": pts_um[:, 2].tolist(),
        "scores": scores,
        "gt": gts,
        "labels": detail_labels,
    }, {
        "stage": "CellGNN",
        "checkpoint": args.checkpoint,
        "box": record.box_hash,
        "n_synapses": len(gt_labels),
        "n_pred_clusters": int(len(set(pred_labels))),
        "n_gt_cells": int(len(unique_gt)),
    }


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------

def _meta_html(meta: dict) -> str:
    lines = []
    for k, v in meta.items():
        lines.append(f"<b>{k}</b>: {v}")
    return "<br/>".join(lines)


def generate_html(data: dict, meta: dict, output_path: str) -> None:
    stage_name = meta.get("stage", "Pipeline")
    html = _HTML_TEMPLATE.format(
        stage_name=stage_name,
        data_json=json.dumps(data),
        meta_html=_meta_html(meta),
    )
    Path(output_path).write_text(html, encoding="utf-8")
    print(f"\nWrote {output_path}  ({Path(output_path).stat().st_size // 1024} KB)")
    print("Open in any browser — no server needed.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--checkpoint", required=True, help="Model checkpoint .pt path")
    p.add_argument("--cache-dir", default="data/boxes_30um", help="BoxCache directory")
    p.add_argument("--output", default="viewer.html", help="Output HTML path")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Neuronauts pipeline visualizer")
    sub = parser.add_subparsers(dest="stage", required=True)

    # path-encoder
    p_pe = sub.add_parser("path-encoder", help="Inspect path encoder predictions")
    _common_args(p_pe)
    p_pe.add_argument("--n-examples", type=int, default=2000)
    p_pe.add_argument("--window-size", type=int, default=12)
    p_pe.add_argument("--hard-neg-fraction", type=float, default=0.7)

    # grammar
    p_gr = sub.add_parser("grammar", help="Inspect grammar model merge predictions")
    _common_args(p_gr)
    p_gr.add_argument("--box-id", default=None)

    # cell-gnn
    p_cg = sub.add_parser("cell-gnn", help="Inspect CellGNN cell-identity predictions")
    _common_args(p_cg)
    p_cg.add_argument("--box-id", default=None)

    args = parser.parse_args(argv)

    if args.stage == "path-encoder":
        data, meta = run_path_encoder(args)
    elif args.stage == "grammar":
        data, meta = run_grammar(args)
    elif args.stage == "cell-gnn":
        data, meta = run_cell_gnn(args)
    else:
        parser.error(f"Unknown stage: {args.stage}")

    generate_html(data, meta, args.output)


if __name__ == "__main__":
    main()
