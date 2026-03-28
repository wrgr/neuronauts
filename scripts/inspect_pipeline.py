#!/usr/bin/env python3
"""Step-by-step pipeline inspector — Neuroglancer edition.

Visualises every stage of the CellGNN global topological merge pipeline
on a single cached box.  Each stage is a toggleable Neuroglancer layer
group so you can inspect the evidence graph, grammar scores, GNN
embeddings, and final cell assignments overlaid on the MICrONS EM volume.

Layer groups
------------
1. **synapses**        — pre/post points coloured by ground-truth root ID
2. **scaffold**        — points coloured by scaffold seg_id
3. **evidence_graph**  — synapse-pair edges from build_synapse_graph
                         (proximity = cyan, same-scaffold = yellow)
4. **grammar_scores**  — scaffold-group pair lines, green→red by score
5. **cell_labels**     — points coloured by CellGNN-predicted cell label
6. **assembly**        — final ConnectivityGraph edges; green = correct,
                         red = wrong cell pairing vs ground truth

Colour key (pairs):
    Green  — same cell in ground truth
    Red    — different cell in ground truth
    Cyan   — proximity edge (evidence graph)
    Yellow — scaffold edge (evidence graph)

Usage
-----
Minimal (no CellGNN, just evidence graph + grammar)::

    python scripts/inspect_pipeline.py \\
        --cache-dir data/proofread \\
        --box-idx 0

With CellGNN assembly + grammar::

    python scripts/inspect_pipeline.py \\
        --cache-dir data/proofread \\
        --grammar-path models/shared_grammar_real.pt \\
        --cell-gnn-path models/cell_gnn.pt \\
        --box-idx 0

List available boxes::

    python scripts/inspect_pipeline.py --cache-dir data/proofread --list-boxes

Requirements::

    pip install neuroglancer
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ---------------------------------------------------------------------------
# MICrONS layer URLs (same as validate_viz.py)
# ---------------------------------------------------------------------------
EM_SOURCE = "precomputed://https://bossdb-open-data.s3.amazonaws.com/iarpa_microns/minnie/minnie65/em"
SEG_SOURCE = "graphene://https://minnie65-proofreading.zetta.ai/segmentation/1.0/minnie65_8x8x40"
EM_VOX_NM = (8.0, 8.0, 40.0)
MIP2_TO_MIP0 = (4, 4, 1)


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

def _to_global_mip0(box_record, pt) -> list[float]:
    """Convert box-relative MIP-2 voxel to global MIP-0 voxel coordinates."""
    origin_nm = np.array(box_record.center_nm, dtype=np.float64) - (
        box_record.side_um * 1000.0 / 2.0
    )
    origin_mip0 = origin_nm / np.array(EM_VOX_NM, dtype=np.float64)
    global_mip0 = origin_mip0 + np.array(pt, dtype=np.float64) * np.array(MIP2_TO_MIP0, dtype=np.float64)
    return [float(global_mip0[i]) for i in range(3)]


def _hex_to_glsl(hex_color: str) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"{r/255:.3f}, {g/255:.3f}, {b/255:.3f}"


# ---------------------------------------------------------------------------
# Deterministic colour palette for integer labels
# ---------------------------------------------------------------------------

_PALETTE = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
    "#911eb4", "#42d4f4", "#f032e6", "#bfef45", "#fabed4",
    "#469990", "#dcbeff", "#9A6324", "#800000", "#aaffc3",
    "#808000", "#ffd8b1", "#000075", "#a9a9a9", "#ffffff",
]


def _label_color(label: int) -> str:
    return _PALETTE[label % len(_PALETTE)]


# ---------------------------------------------------------------------------
# Stage builders — each returns a list of (layer_name, layer) pairs
# ---------------------------------------------------------------------------

def stage_synapses(synapses, box_record, neuroglancer):
    """Stage 1: raw synapse positions coloured by ground-truth root ID."""
    layers = []

    for role, pts, root_ids, point_color in [
        ("pre", synapses.pre_pt, synapses.pre_root_id, "0.2, 0.7, 1.0"),
        ("post", synapses.post_pt, synapses.post_root_id, "1.0, 0.5, 0.2"),
    ]:
        unique_roots = sorted(set(root_ids.tolist()))
        root_to_label = {r: i for i, r in enumerate(unique_roots)}

        annotations = []
        for i in range(len(pts)):
            lbl = root_to_label[int(root_ids[i])]
            color = _label_color(lbl)
            annotations.append(neuroglancer.PointAnnotation(
                id=f"{role}_{i}",
                point=_to_global_mip0(box_record, pts[i]),
            ))

        # Single layer per role with uniform colour (root colouring via
        # separate per-root layers would create too many).  Use role colour.
        layers.append((
            f"syn_{role}",
            neuroglancer.LocalAnnotationLayer(
                annotations=annotations,
                shader=f"""
void main() {{
  setColor(vec4({point_color}, 0.85));
  setPointMarkerSize(7.0);
}}
""",
            ),
        ))

    return layers


def stage_scaffold(synapses, box_record, neuroglancer):
    """Stage 2: points coloured by scaffold seg_id."""
    layers = []
    for role, pts, seg_ids in [
        ("pre", synapses.pre_pt, getattr(synapses, "pre_seg_id", None)),
        ("post", synapses.post_pt, getattr(synapses, "post_seg_id", None)),
    ]:
        if seg_ids is None:
            continue
        unique_segs = sorted(set(int(s) for s in seg_ids if int(s) > 0))
        if not unique_segs:
            continue
        seg_to_label = {s: i for i, s in enumerate(unique_segs)}
        for seg_id in unique_segs:
            indices = [i for i, s in enumerate(seg_ids) if int(s) == seg_id]
            color = _label_color(seg_to_label[seg_id])
            annotations = [
                neuroglancer.PointAnnotation(
                    id=f"scaf_{role}_{seg_id}_{i}",
                    point=_to_global_mip0(box_record, pts[idx]),
                )
                for i, idx in enumerate(indices)
            ]
            layers.append((
                f"scaffold_{role}_seg{seg_id}",
                neuroglancer.LocalAnnotationLayer(
                    annotations=annotations,
                    shader=f"""
void main() {{
  setColor(vec4({_hex_to_glsl(color)}, 0.9));
  setPointMarkerSize(5.0);
}}
""",
                ),
            ))
    return layers


def stage_evidence_graph(synapses, box_record, neuroglancer, *, proximity_radius_nm=5000.0):
    """Stage 3: evidence graph edges from build_synapse_graph."""
    from neuronauts.cell_graph import build_synapse_graph

    layers = []
    for role, pts in [("pre", synapses.pre_pt), ("post", synapses.post_pt)]:
        graph = build_synapse_graph(
            synapses, role,
            proximity_radius_nm=proximity_radius_nm,
        )

        # Separate edges by type
        scaffold_edges = []
        proximity_edges = []
        for e in graph.edges:
            a = _to_global_mip0(box_record, pts[e.src])
            b = _to_global_mip0(box_record, pts[e.dst])
            if e.same_scaffold > 0.5:
                scaffold_edges.append((a, b, e))
            else:
                proximity_edges.append((a, b, e))

        # Proximity edges (cyan)
        if proximity_edges:
            anns = [
                neuroglancer.LineAnnotation(
                    id=f"prox_{role}_{i}",
                    point_a=a, point_b=b,
                )
                for i, (a, b, _) in enumerate(proximity_edges)
            ]
            layers.append((
                f"evidence_{role}_proximity",
                neuroglancer.LocalAnnotationLayer(
                    annotations=anns,
                    shader="""
void main() {
  setColor(vec4(0.0, 0.8, 0.9, 0.4));
  setLineWidth(1.0);
}
""",
                ),
            ))

        # Scaffold edges (yellow)
        if scaffold_edges:
            anns = [
                neuroglancer.LineAnnotation(
                    id=f"scaf_{role}_{i}",
                    point_a=a, point_b=b,
                )
                for i, (a, b, _) in enumerate(scaffold_edges)
            ]
            layers.append((
                f"evidence_{role}_scaffold",
                neuroglancer.LocalAnnotationLayer(
                    annotations=anns,
                    shader="""
void main() {
  setColor(vec4(1.0, 0.9, 0.1, 0.6));
  setLineWidth(1.5);
}
""",
                ),
            ))

        # Summary
        print(
            f"  evidence_{role}: {len(proximity_edges)} proximity + "
            f"{len(scaffold_edges)} scaffold edges  "
            f"({graph.n_synapses} nodes)"
        )

    return layers


def stage_grammar_scores(
    synapses, box_record, neuroglancer, grammar_score_fn,
    *, proximity_radius_nm=5000.0,
):
    """Stage 4: grammar pairwise scores as lines coloured green→red."""
    from neuronauts.cell_graph import extract_grammar_scores

    layers = []
    for role, pts in [("pre", synapses.pre_pt), ("post", synapses.post_pt)]:
        scores = extract_grammar_scores(
            synapses, role, grammar_score_fn,
            proximity_radius_nm=proximity_radius_nm,
        )
        if not scores:
            print(f"  grammar_{role}: no scores (no scaffold groups?)")
            continue

        vals = list(scores.values())
        vmin, vmax = min(vals), max(vals)
        vrange = max(vmax - vmin, 1e-6)

        # Green (high score = likely same cell) to red (low score)
        anns = []
        for i, ((si, sj), score) in enumerate(scores.items()):
            anns.append(neuroglancer.LineAnnotation(
                id=f"gram_{role}_{i}",
                point_a=_to_global_mip0(box_record, pts[si]),
                point_b=_to_global_mip0(box_record, pts[sj]),
            ))

        # Use a single layer with mixed colour via a medium green
        # (individual per-edge colours require separate layers; keep it simple)
        mean_score = float(np.mean(vals))
        t = max(0.0, min(1.0, (mean_score - vmin) / vrange))
        r_val = 1.0 - t
        g_val = t
        layers.append((
            f"grammar_{role}",
            neuroglancer.LocalAnnotationLayer(
                annotations=anns,
                shader=f"""
void main() {{
  setColor(vec4({r_val:.2f}, {g_val:.2f}, 0.1, 0.7));
  setLineWidth(2.0);
}}
""",
            ),
        ))
        print(
            f"  grammar_{role}: {len(scores)} scored pairs  "
            f"score range [{vmin:.2f}, {vmax:.2f}] mean={mean_score:.2f}"
        )

    return layers


def stage_cell_labels(synapses, box_record, neuroglancer, model, *, proximity_radius_nm=5000.0, threshold=0.5):
    """Stage 5: CellGNN predicted cell labels as coloured points."""
    from neuronauts.cell_graph import build_synapse_graph, infer_cells

    layers = []
    for role, pts, root_ids in [
        ("pre", synapses.pre_pt, synapses.pre_root_id),
        ("post", synapses.post_pt, synapses.post_root_id),
    ]:
        graph = build_synapse_graph(synapses, role, proximity_radius_nm=proximity_radius_nm)
        labels = infer_cells(model, graph, threshold=threshold)

        n_cells = len(set(labels.tolist()))
        # Check correctness vs ground truth
        unique_roots = sorted(set(root_ids.tolist()))
        root_to_label = {r: i for i, r in enumerate(unique_roots)}
        gt_labels = np.array([root_to_label[int(r)] for r in root_ids])

        # Per-predicted-cell layer
        for cell_id in sorted(set(labels.tolist())):
            indices = [i for i in range(len(labels)) if labels[i] == cell_id]
            color = _label_color(cell_id)
            anns = [
                neuroglancer.PointAnnotation(
                    id=f"cell_{role}_{cell_id}_{i}",
                    point=_to_global_mip0(box_record, pts[idx]),
                )
                for i, idx in enumerate(indices)
            ]
            # Check purity of this predicted cell
            gt_in_cell = [int(gt_labels[idx]) for idx in indices]
            from collections import Counter
            gt_counts = Counter(gt_in_cell)
            majority = gt_counts.most_common(1)[0][1]
            purity = majority / len(indices) if indices else 0

            layers.append((
                f"cell_{role}_{cell_id} (n={len(indices)}, pur={purity:.0%})",
                neuroglancer.LocalAnnotationLayer(
                    annotations=anns,
                    shader=f"""
void main() {{
  setColor(vec4({_hex_to_glsl(color)}, 0.9));
  setPointMarkerSize(8.0);
}}
""",
                ),
            ))

        print(
            f"  cell_{role}: {n_cells} predicted cells from {len(unique_roots)} true roots"
        )

    return layers


def stage_assembly(
    synapses, box_record, neuroglancer, model,
    *, grammar_score_fn=None, proximity_radius_nm=5000.0, threshold=0.5,
):
    """Stage 6: final ConnectivityGraph edges — green=correct, red=wrong."""
    from neuronauts.cell_graph import cell_gnn_assembly
    from neuronauts.line_graph import evaluate

    cg = cell_gnn_assembly(
        synapses, model,
        grammar_score_fn=grammar_score_fn,
        proximity_radius_nm=proximity_radius_nm,
        partition_threshold=threshold,
    )
    metrics = evaluate(cg, synapses.pre_root_id, synapses.post_root_id)

    # Build ground-truth line graph for comparison
    from neuronauts.line_graph import build_true_line_graph
    true_edges = build_true_line_graph(synapses.pre_root_id, synapses.post_root_id)
    true_set = set(true_edges)

    correct_anns = []
    wrong_anns = []
    for i, (pre_nid, post_nid, syn_idx) in enumerate(cg.edges):
        pre_neuron = cg.neurons[pre_nid]
        post_neuron = cg.neurons[post_nid]

        # Use synapse position for the edge line
        a = _to_global_mip0(box_record, synapses.pre_pt[syn_idx])
        b = _to_global_mip0(box_record, synapses.post_pt[syn_idx])

        # Check: are these two synapses in the same true line graph edge set?
        # Simplified: check if the pre-side root matches for the synapse
        pre_root = int(synapses.pre_root_id[syn_idx])
        post_root = int(synapses.post_root_id[syn_idx])

        # An edge is "correct" if the pre-side neuron groups the synapse
        # with other synapses of the same true root (and same for post)
        ann = neuroglancer.LineAnnotation(
            id=f"asm_{i}", point_a=a, point_b=b,
        )
        # Heuristic: check if all synapses in the pre neuron share root
        pre_roots_in_neuron = set(int(synapses.pre_root_id[s]) for s in pre_neuron.synapse_indices)
        post_roots_in_neuron = set(int(synapses.post_root_id[s]) for s in post_neuron.synapse_indices)

        if len(pre_roots_in_neuron) == 1 and len(post_roots_in_neuron) == 1:
            correct_anns.append(ann)
        else:
            wrong_anns.append(ann)

    layers = []
    if correct_anns:
        layers.append((
            f"assembly_correct (n={len(correct_anns)})",
            neuroglancer.LocalAnnotationLayer(
                annotations=correct_anns,
                shader="""
void main() {
  setColor(vec4(0.0, 0.85, 0.3, 0.8));
  setLineWidth(2.5);
}
""",
            ),
        ))
    if wrong_anns:
        layers.append((
            f"assembly_wrong (n={len(wrong_anns)})",
            neuroglancer.LocalAnnotationLayer(
                annotations=wrong_anns,
                shader="""
void main() {
  setColor(vec4(1.0, 0.15, 0.15, 0.8));
  setLineWidth(2.5);
}
""",
            ),
        ))

    print(
        f"  assembly: {len(cg.edges)} edges  "
        f"({len(correct_anns)} correct, {len(wrong_anns)} mixed-root)  "
        f"F1={metrics.f1:.3f} P={metrics.precision:.3f} R={metrics.recall:.3f}"
    )

    return layers, metrics


# ---------------------------------------------------------------------------
# Main viewer builder
# ---------------------------------------------------------------------------

def build_viewer(
    synapses,
    box_record,
    *,
    volume_chunk=None,
    grammar_score_fn=None,
    cell_gnn_model=None,
    proximity_radius_nm=5000.0,
    partition_threshold=0.5,
) -> str:
    """Build a Neuroglancer viewer with all pipeline stages.  Returns URL."""
    try:
        import neuroglancer
    except ImportError as exc:
        raise SystemExit("pip install neuroglancer") from exc

    neuroglancer.set_server_bind_address("127.0.0.1")
    viewer = neuroglancer.Viewer()

    all_layers: list[tuple[str, object]] = []

    # Stage 1: raw synapses
    print("Stage 1: synapses")
    all_layers.extend(stage_synapses(synapses, box_record, neuroglancer))

    # Stage 2: scaffold
    print("Stage 2: scaffold groups")
    scaffold_layers = stage_scaffold(synapses, box_record, neuroglancer)
    if scaffold_layers:
        all_layers.extend(scaffold_layers)
    else:
        print("  (no seg_ids available)")

    # Stage 3: evidence graph
    print("Stage 3: evidence graph")
    all_layers.extend(stage_evidence_graph(
        synapses, box_record, neuroglancer,
        proximity_radius_nm=proximity_radius_nm,
    ))

    # Stage 4: grammar scores
    if grammar_score_fn is not None:
        print("Stage 4: grammar scores")
        all_layers.extend(stage_grammar_scores(
            synapses, box_record, neuroglancer, grammar_score_fn,
            proximity_radius_nm=proximity_radius_nm,
        ))
    else:
        print("Stage 4: grammar scores (skipped — no grammar checkpoint)")

    # Stage 5: CellGNN cell labels
    if cell_gnn_model is not None:
        print("Stage 5: CellGNN cell labels")
        all_layers.extend(stage_cell_labels(
            synapses, box_record, neuroglancer, cell_gnn_model,
            proximity_radius_nm=proximity_radius_nm,
            threshold=partition_threshold,
        ))
    else:
        print("Stage 5: CellGNN cell labels (skipped — no CellGNN checkpoint)")

    # Stage 6: final assembly
    metrics = None
    if cell_gnn_model is not None:
        print("Stage 6: assembly")
        asm_layers, metrics = stage_assembly(
            synapses, box_record, neuroglancer, cell_gnn_model,
            grammar_score_fn=grammar_score_fn,
            proximity_radius_nm=proximity_radius_nm,
            threshold=partition_threshold,
        )
        all_layers.extend(asm_layers)
    else:
        print("Stage 6: assembly (skipped — no CellGNN checkpoint)")

    # Build viewer
    with viewer.txn() as s:
        s.dimensions = neuroglancer.CoordinateSpace(
            names=["x", "y", "z"],
            units=["vox", "vox", "vox"],
            scales=[1, 1, 1],
        )

        # EM layer
        s.layers["em"] = neuroglancer.ImageLayer(source=EM_SOURCE)

        # Local cached EM fallback
        if volume_chunk is not None and getattr(volume_chunk.data, "size", 0) > 0:
            origin_nm = np.array(box_record.center_nm, dtype=np.float64) - (
                box_record.side_um * 1000.0 / 2.0
            )
            origin_mip0 = (origin_nm / np.array(EM_VOX_NM, dtype=np.float64)).astype(np.int64)
            vol_dims = neuroglancer.CoordinateSpace(
                names=["x", "y", "z"],
                units=["vox", "vox", "vox"],
                scales=[4, 4, 1],
            )
            vol = neuroglancer.LocalVolume(
                volume=volume_chunk.data,
                dimensions=vol_dims,
                voxel_offset=origin_mip0.tolist(),
                volume_type="uint8",
            )
            s.layers["em_cached"] = neuroglancer.ImageLayer(source=vol)

        # Segmentation
        s.layers["seg"] = neuroglancer.SegmentationLayer(source=SEG_SOURCE)

        # Pipeline stage layers (later stages on top)
        for name, layer in all_layers:
            s.layers[name] = layer

        # Navigate to box centre
        centre_mip0 = [
            float(box_record.center_nm[i]) / EM_VOX_NM[i] for i in range(3)
        ]
        s.position = centre_mip0

    return viewer.get_viewer_url()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--cache-dir", default="data/proofread")
    p.add_argument("--grammar-path", default=None,
                   help="SharedGrammarModel checkpoint (enables grammar score layer).")
    p.add_argument("--cell-gnn-path", default=None,
                   help="CellGNN checkpoint (enables cell label + assembly layers).")
    p.add_argument("--box-idx", type=int, default=0)
    p.add_argument("--proximity-radius-nm", type=float, default=5000.0)
    p.add_argument("--partition-threshold", type=float, default=0.5)
    p.add_argument("--list-boxes", action="store_true")
    p.add_argument("--min-synapses", type=int, default=10)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    from neuronauts.dataset_builder import BoxCache

    cache = BoxCache(args.cache_dir)
    records = [r for r in cache.all_records() if r.n_synapses >= args.min_synapses]

    if not records:
        print(f"No boxes with >= {args.min_synapses} synapses in {args.cache_dir}")
        return 1

    if args.list_boxes:
        print(f"\n{'IDX':>4}  {'SYNAPSES':>9}  {'POS_PAIRS':>10}  BOX_HASH")
        for i, r in enumerate(records):
            print(f"{i:>4}  {r.n_synapses:>9}  {r.n_positive_pairs:>10}  {r.box_hash[:16]}")
        print(f"\n  {len(records)} boxes")
        return 0

    if args.box_idx >= len(records):
        print(f"--box-idx {args.box_idx} out of range ({len(records)} boxes). Use --list-boxes.")
        return 1

    record = records[args.box_idx]
    print(f"\nBox: {record.box_hash[:16]}  side={record.side_um}µm  "
          f"n_synapses={record.n_synapses}  pos_pairs={record.n_positive_pairs}\n")

    volume_chunk, synapses = cache.load(record)

    # Load grammar model
    grammar_score_fn = None
    if args.grammar_path and Path(args.grammar_path).exists():
        print(f"Loading grammar from {args.grammar_path} …")
        from neuronauts.run import _load_shared_merge_score_fn
        grammar_score_fn = _load_shared_merge_score_fn(args.grammar_path)

    # Load CellGNN model
    cell_gnn_model = None
    if args.cell_gnn_path and Path(args.cell_gnn_path).exists():
        print(f"Loading CellGNN from {args.cell_gnn_path} …")
        from neuronauts.cell_graph import load_cell_gnn
        cell_gnn_model = load_cell_gnn(args.cell_gnn_path)

    print()
    url = build_viewer(
        synapses,
        record,
        volume_chunk=volume_chunk,
        grammar_score_fn=grammar_score_fn,
        cell_gnn_model=cell_gnn_model,
        proximity_radius_nm=args.proximity_radius_nm,
        partition_threshold=args.partition_threshold,
    )

    print(f"\n{'='*60}")
    print(f"  Open in browser:\n\n  {url}\n")
    print("  Layer groups (toggle in sidebar):")
    print("    syn_pre / syn_post         — raw synapses by role")
    print("    scaffold_*                 — scaffold seg_id groups")
    print("    evidence_*_proximity       — spatial proximity edges (cyan)")
    print("    evidence_*_scaffold        — same-scaffold edges (yellow)")
    if grammar_score_fn:
        print("    grammar_pre / grammar_post — grammar score lines")
    if cell_gnn_model:
        print("    cell_*                     — predicted cell clusters")
        print("    assembly_correct           — correct edges (green)")
        print("    assembly_wrong             — mixed-root edges (red)")
    print(f"{'='*60}\n")
    print("Press Ctrl-C to exit.\n")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
