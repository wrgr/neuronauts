#!/usr/bin/env python3
"""Out-of-proofread-column transfer assessment for NeuronautS.

Trains on in-column bboxes (where v117→v1718 diff provides real GT) and applies
the model to a spatially distant region where v117 ≈ v1718 (no proofreading edits).
No formal ground truth is available outside the column.

Assessment strategy
-------------------
Primary — biological plausibility of assembled shapes:
  - cable_length_um distribution (expected 500–20,000 µm for cortical L2/3 neurons)
  - is_tree fraction (Kruskal guarantee; should be 1.000 by construction)
  - fully_connected fraction (1-component shapes = no stitch gap)
  - Compare to in-column shape distribution as a transfer signal

Secondary — Neuroglancer visual inspection:
  Generates a shareable URL with synapse pairs (colored by cluster), per-observation
  entropy (uncertainty), and assembled skeletons for expert assessment.

Tertiary — conservative-behavior sanity check:
  Because v117 ≡ v1718 outside the column (0 frankenmerges), each v117 root IS one
  neuron. Over-merge rate against v117 pseudo-labels confirms the model does not
  hallucinate merges in novel territory. This is a necessary but not sufficient
  condition: it cannot confirm the model finds real neurons, only that it does not
  spuriously fuse known-separate fragments.

Usage
-----
  # Default: evaluate at x=200–400k (well outside the x=750k–1550k column)
  python scripts/out_of_column_eval.py --version 1718

  # Custom out-of-column bbox:
  python scripts/out_of_column_eval.py \\
      --ooc-x0 200000 --ooc-x1 400000 \\
      --ooc-y0 400000 --ooc-y1 500000 \\
      --ooc-z0 600000 --ooc-z1 700000

  # Faster debugging (1 training region):
  python scripts/out_of_column_eval.py --quick-train --embed-epochs 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _fmt(m: dict) -> str:
    return (f"ARI={m['ari']:.4f}  P={m['merge_precision']:.3f}  "
            f"R={m['merge_recall']:.3f}  over={m['over_merge_rate']:.3f}  "
            f"fk={m['frankenmerge_split_recall']:.3f}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", type=int, default=1718)
    p.add_argument("--side", default="pre", choices=["pre", "post"])
    p.add_argument("--max-synapses", type=int, default=10_000)
    p.add_argument("--min-syn-per-fragment", type=int, default=5)
    p.add_argument("--k-spatial", type=int, default=8)
    p.add_argument("--embed-epochs", type=int, default=15)
    p.add_argument("--partition-epochs", type=int, default=100)
    p.add_argument("--cc-bias", type=float, default=-2.0)
    p.add_argument("--franken-hard-frac", type=float, default=0.30)
    p.add_argument("--seam-buffer", type=int, default=50_000)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-l2-skeletons", action="store_true")

    # Out-of-column bbox (default: x=200–400k, away from the proofread column)
    p.add_argument("--ooc-x0", type=int, default=200_000,
                   help="Out-of-column bbox x start nm")
    p.add_argument("--ooc-x1", type=int, default=400_000,
                   help="Out-of-column bbox x end nm")
    p.add_argument("--ooc-y0", type=int, default=500_000,
                   help="Out-of-column bbox y start nm")
    p.add_argument("--ooc-y1", type=int, default=570_000,
                   help="Out-of-column bbox y end nm")
    p.add_argument("--ooc-z0", type=int, default=700_000,
                   help="Out-of-column bbox z start nm")
    p.add_argument("--ooc-z1", type=int, default=800_000,
                   help="Out-of-column bbox z end nm")

    p.add_argument("--quick-train", action="store_true",
                   help="Use only one training region (faster, for debugging)")
    args = p.parse_args()

    from treestitch.assemble import assemble_partition_shapes, neuron_shape_metrics
    from treestitch.embed import FragmentEncoder, encode_fragments, train_fragment_encoder
    from treestitch.graph import build_observation_graph
    from treestitch.ngl_export import export_synapse_state
    from treestitch.partition import (
        evaluate_partition, merge_metrics,
        partition_observations_cc, partition_observations_soft,
        train_edge_partition_multi_region,
    )
    from treestitch.realworld import build_region_world
    from treestitch.risk import decision_layer, risk_summary_str

    # ------------------------------------------------------------------ Training bboxes
    # Standard in-column bboxes (same as multi_region_train.py)
    y0, y1 = 930_000, 1_000_000  # dense y-extent
    z0, z1 = 780_000, 880_000
    buf = args.seam_buffer
    train_bboxes = [
        ((750_000, y0, z0),         (950_000, y1, z1)),
        ((950_000, y0, z0),         (1_150_000 - buf, y1, z1)),
        ((1_350_000 + buf, y0, z0), (1_550_000, y1, z1)),
    ]
    if args.quick_train:
        train_bboxes = train_bboxes[:1]
        print("Quick-train mode: using only 1 training region")

    ooc_bbox = (
        (args.ooc_x0, args.ooc_y0, args.ooc_z0),
        (args.ooc_x1, args.ooc_y1, args.ooc_z1),
    )

    print("=" * 64)
    print(f"Out-of-column evaluation  (v117 → v{args.version})")
    print(f"  Train regions: {len(train_bboxes)}")
    for i, bb in enumerate(train_bboxes):
        print(f"    {chr(65+i)}: {bb}")
    print(f"  Out-of-column bbox: {ooc_bbox}")
    print(f"  NOTE: v117≈v1718 outside the proofread column.")
    print(f"        Using v117 roots as pseudo-ground-truth labels.")
    print("=" * 64)

    # ------------------------------------------------------------------ Train
    all_frags: list = []
    all_regions = []
    all_label_maps = []

    for i, bbox in enumerate(train_bboxes):
        label = chr(65 + i)
        print(f"\n[TRAIN {label}] Building world …")
        frags, region, lmap = build_region_world(
            bbox, version=args.version, side=args.side,
            max_synapses=args.max_synapses,
            min_syn_per_fragment=args.min_syn_per_fragment,
            seed=args.seed, verbose=True,
            l2_skeletons=not args.no_l2_skeletons)
        all_frags.append(frags)
        all_regions.append(region)
        all_label_maps.append(lmap)
        n_franken = sum(1 for v in lmap.values() if len(v) > 1)
        print(f"  → {len(frags)} fragments, {region.n_synapses} synapses, "
              f"{n_franken} frankenmerges")

    print(f"\nTraining FragmentEncoder ({args.embed_epochs} epochs) …")
    merged_lmap: dict = {}
    for lm in all_label_maps:
        merged_lmap.update(lm)

    encoder = FragmentEncoder(node_input_dim=4, d_model=64, output_dim=32)
    if args.embed_epochs > 0:
        train_fragment_encoder(encoder, all_frags, n_epochs=args.embed_epochs,
                               lr=1e-3, margin=1.0, device=args.device,
                               root_label_map=merged_lmap, log_every=20)

    train_graphs = []
    for i, (frags, region) in enumerate(zip(all_frags, all_regions)):
        frags_enc = encode_fragments(encoder, frags, device=args.device)
        g = build_observation_graph(region, frags_enc, side=args.side,
                                    k_spatial=args.k_spatial)
        n_franken = sum(1 for v in all_label_maps[i].values() if len(v) > 1)
        print(f"  Region {chr(65+i)}: {g.n_nodes} nodes | {g.n_edges} edges | "
              f"{int(np.unique(g.labels[g.labels != 0]).shape[0])} neurons | "
              f"{n_franken} frankenmerges")
        train_graphs.append(g)

    print(f"\nTraining EdgePartitionGNN ({args.partition_epochs} epochs) …")
    model, _ = train_edge_partition_multi_region(
        train_graphs,
        n_epochs=args.partition_epochs,
        lr=1e-3,
        franken_hard_frac=args.franken_hard_frac,
        device=args.device,
        seed=args.seed,
        log_every=50,
    )

    # ------------------------------------------------------------------ Out-of-column
    print(f"\n[OOC] Building out-of-column world …")
    print(f"  bbox: x={args.ooc_x0//1000}–{args.ooc_x1//1000}k  "
          f"y={args.ooc_y0//1000}–{args.ooc_y1//1000}k  "
          f"z={args.ooc_z0//1000}–{args.ooc_z1//1000}k nm")
    frags_ooc, region_ooc, lmap_ooc_v1718 = build_region_world(
        ooc_bbox, version=args.version, side=args.side,
        max_synapses=args.max_synapses,
        min_syn_per_fragment=args.min_syn_per_fragment,
        seed=args.seed, verbose=True,
        l2_skeletons=not args.no_l2_skeletons)

    n_ooc_frags = len(frags_ooc)
    n_ooc_syn = region_ooc.n_synapses

    # Outside the proofread column, v1718 ≈ v117.
    # Measure the divergence: if every v117 root maps to exactly one v1718 root,
    # the region is unproofread (no edits). This is our "ground truth" check.
    n_franken_v1718 = sum(1 for v in lmap_ooc_v1718.values() if len(v) > 1)
    n_multi_label = sum(1 for v in lmap_ooc_v1718.values() if len(v) > 1)

    print(f"\n  Edit signal check:")
    print(f"    {n_ooc_frags} v117 fragments in bbox")
    print(f"    Frankenmerges (v117 root → ≥2 v1718 roots): {n_franken_v1718}")
    if n_franken_v1718 == 0:
        print(f"    ✓ No frankenmerges → v117≈v1718 in this region (unproofread)")
        print(f"      Using v117 fragment ID as ground-truth label (each frag = one neuron)")
    else:
        print(f"    ⚠ {n_franken_v1718} frankenmerges found → some proofreading signal exists here")
        print(f"      Proceeding with v117 pseudo-labels AND v1718 labels for comparison")

    # Build OOC graph using v117 fragment IDs as labels (pseudo ground-truth)
    # v117 labels: each fragment is its own neuron → label = fragment index
    frags_ooc_enc = encode_fragments(encoder, frags_ooc, device=args.device)
    graph_ooc = build_observation_graph(region_ooc, frags_ooc_enc, side=args.side,
                                        k_spatial=args.k_spatial)

    # Override labels: use v117 fragment ID as pseudo ground truth
    # (each v117 root is one neuron outside the column)
    from treestitch.schemas import ObservationGraph
    frag_id_to_idx = {fid: i + 1 for i, fid in enumerate(
        np.unique(graph_ooc.fragment_id))}
    pseudo_labels = np.array([frag_id_to_idx[int(f)] for f in graph_ooc.fragment_id],
                              dtype=np.int64)

    print(f"\n  OOC graph: {graph_ooc.n_nodes} nodes | {graph_ooc.n_edges} edges | "
          f"{len(frag_id_to_idx)} pseudo-neurons (= v117 fragments)")

    # Apply trained model
    pred_ooc = partition_observations_cc(model, graph_ooc, bias=args.cc_bias,
                                         device=args.device)
    ev_ooc_pseudo = evaluate_partition(pred_ooc, pseudo_labels)
    mm_ooc_pseudo = merge_metrics(graph_ooc, pred_ooc,
                                   ignore_label=0)

    # Also evaluate with v1718 labels for direct comparison (if there's any signal)
    if n_franken_v1718 > 0:
        ev_ooc_v1718 = evaluate_partition(pred_ooc, graph_ooc.labels)
        mm_ooc_v1718 = merge_metrics(graph_ooc, pred_ooc)

    # Soft partition + risk layer
    soft_ooc = partition_observations_soft(model, graph_ooc, bias=args.cc_bias,
                                            device=args.device)
    decisions_ooc = decision_layer(
        soft_ooc,
        fragment_ids=graph_ooc.fragment_id,
        cost_merge=5.0,
        cost_split=1.0,
    )

    # Shape assembly
    shapes_ooc = assemble_partition_shapes(frags_ooc, pred_ooc, graph_ooc.fragment_id,
                                            stitch_radius_nm=5_000.0)
    mlist = [neuron_shape_metrics(s) for s in shapes_ooc.values()]
    is_tree_frac = float(np.mean([m['is_tree'] for m in mlist])) if mlist else 0.0
    cable_arr = np.array([m['cable_length_um'] for m in mlist]) if mlist else np.array([0.0])
    n_comp_arr = np.array([m['n_connected_components'] for m in mlist]) if mlist else np.array([1])

    # Neuroglancer link
    ngl_url = export_synapse_state(
        region_ooc, pred_ooc,
        soft=soft_ooc,
        shapes=shapes_ooc,
        side=args.side,
    )

    # ------------------------------------------------------------------ Summary
    print(f"\n{'='*64}")
    print(f"SUMMARY: out-of-column transfer assessment")
    print(f"{'='*64}")
    print(f"\n  OOC bbox: {n_ooc_syn} synapses, {n_ooc_frags} fragments")
    print(f"  Edit signal: {n_franken_v1718} frankenmerges "
          f"({'unproofread' if n_franken_v1718 == 0 else 'some edits'})")

    print(f"\n  [PRIMARY] Shape plausibility ({len(shapes_ooc)} assembled neurons):")
    print(f"    is_tree={is_tree_frac:.3f}")
    print(f"    cable_um: median={np.median(cable_arr):.0f}  "
          f"p5={np.percentile(cable_arr,5):.0f}  "
          f"p95={np.percentile(cable_arr,95):.0f}")
    print(f"    fully_connected={( n_comp_arr==1).mean():.1%}")

    print(f"\n  [SECONDARY] Risk / confidence distribution:")
    print(f"    {risk_summary_str(decisions_ooc)}")

    print(f"\n  [SECONDARY] Neuroglancer view (pre/post + clusters + uncertainty + skeletons):")
    print(f"    {ngl_url[:120]}...")

    print(f"\n  [TERTIARY] Conservative-behavior check vs v117 pseudo-labels:")
    print(f"    over_merge={mm_ooc_pseudo['over_merge_rate']:.3f}  "
          f"clusters={ev_ooc_pseudo['n_clusters_pred']}/{ev_ooc_pseudo['n_clusters_true']}")
    print(f"    (v117 pseudo-labels: each fragment = 1 neuron; over_merge=0 is expected "
          f"with cc_bias=-2.0)")

    if n_franken_v1718 > 0:
        print(f"\n  Using v1718 labels (some proofreading signal):")
        print(f"    {_fmt({**ev_ooc_v1718, **mm_ooc_v1718})}")

    # Key interpretation
    print(f"\n{'='*64}")
    print("INTERPRETATION")
    print(f"{'='*64}")
    over = mm_ooc_pseudo['over_merge_rate']
    n_predicted = ev_ooc_pseudo['n_clusters_pred']
    n_frags = ev_ooc_pseudo['n_clusters_true']
    cable_med = float(np.median(cable_arr))

    # Primary: shape plausibility
    if is_tree_frac >= 0.99:
        print(f"  ✓ All assembled shapes are trees (is_tree={is_tree_frac:.3f})")
    else:
        print(f"  ⚠ {(1-is_tree_frac):.0%} of shapes have cycles (is_tree={is_tree_frac:.3f})")

    if 500 <= cable_med <= 20_000:
        print(f"  ✓ Cable lengths biologically plausible: median {cable_med:.0f} µm "
              f"(expected 500–20000 µm for cortical neurons)")
    else:
        print(f"  ⚠ Cable lengths unusual: median {cable_med:.0f} µm "
              f"(expected 500–20000 µm for cortical neurons)")
        if cable_med < 500:
            print(f"    Short cables → over-fragmentation; model may not be generalizing "
                  f"well to this region")

    # Secondary: conservative-behavior check
    if over < 0.05:
        print(f"  ✓ Conservative out-of-column: over_merge={over:.3f} "
              f"({n_predicted} clusters from {n_frags} fragments)")
    elif over < 0.20:
        print(f"  ⚠ Moderate over-merge: over_merge={over:.3f} — some spurious merges")
    else:
        print(f"  ✗ High over-merge: over_merge={over:.3f} — "
              f"model hallucinates merges; increase --cc-bias magnitude")

    print(f"{'='*64}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
