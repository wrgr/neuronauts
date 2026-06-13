#!/usr/bin/env python3
"""Multi-region training benchmark for NeuronautS.

Trains the EdgePartitionGNN on N non-overlapping spatial bboxes simultaneously
(graph concatenation, edges stay intra-region), then evaluates on a held-out
test bbox.

Key question: does multi-region training fix the fk_split generalization gap?
Single-region training memorises which v117 roots are frankenmerges in one
region.  Multi-region training forces the model to learn the abstract synaptic
signature of a frankenmerge (heterogeneous synapse partners, anomalous
same-fragment edge density) because no single root is over-represented.

Training regions (non-overlapping x-strips, same y/z):
  A  x  750_000 – 950_000   (far west)
  B  x  950_000 – 1_150_000 (west, same as previous spatial-split train)
  C  x 1_350_000 – 1_550_000 (far east)

Test region (established benchmark):
     x 1_150_000 – 1_350_000

Dense mode (--dense): doubles y-extent to 100k nm for higher fragment density.

Usage
-----
  python scripts/multi_region_train.py --version 1718 --partition-epochs 150
  python scripts/multi_region_train.py --dense  # stress-test at higher density
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
    p.add_argument("--max-synapses", type=int, default=20_000,
                   help="Max synapses per bbox (applied to each region independently)")
    p.add_argument("--min-syn-per-fragment", type=int, default=5)
    p.add_argument("--k-spatial", type=int, default=8)
    p.add_argument("--embed-epochs", type=int, default=20)
    p.add_argument("--partition-epochs", type=int, default=150)
    p.add_argument("--cc-bias", type=float, default=-2.0,
                   help="Out-of-sample operating point (bias=-2.0 clears Bar2 per sweep)")
    p.add_argument("--franken-hard-frac", type=float, default=0.30)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-l2-skeletons", action="store_true")
    p.add_argument("--dense", action="store_true",
                   help="Double y-extent (930–1000k instead of 930–980k) for dense-box test")
    args = p.parse_args()

    from treestitch.assemble import assemble_partition_shapes, neuron_shape_metrics
    from treestitch.embed import FragmentEncoder, encode_fragments, train_fragment_encoder
    from treestitch.graph import build_observation_graph
    from treestitch.partition import (
        evaluate_partition, merge_metrics,
        partition_observations_cc,
        train_edge_partition_multi_region,
    )
    from treestitch.realworld import build_region_world

    y0, y1 = 930_000, 980_000
    z0, z1 = 780_000, 880_000
    if args.dense:
        y1 = 1_000_000
        print("Dense mode: y-extent = 930–1000k nm")

    # Non-overlapping training regions and the established test bbox
    train_bboxes = [
        ((750_000, y0, z0),   (950_000, y1, z1)),   # A: far west
        ((950_000, y0, z0),   (1_150_000, y1, z1)), # B: west (prev split train)
        ((1_350_000, y0, z0), (1_550_000, y1, z1)), # C: far east
    ]
    test_bbox = ((1_150_000, y0, z0), (1_350_000, y1, z1))

    print("=" * 64)
    print(f"Multi-region training  (v117 → v{args.version})")
    print(f"  Train regions: {len(train_bboxes)}")
    for i, bb in enumerate(train_bboxes):
        print(f"    {chr(65+i)}: {bb}")
    print(f"  Test  region: {test_bbox}")
    print("=" * 64)

    # ------------------------------------------------------------------ Build worlds
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

    total_frags = sum(len(f) for f in all_frags)
    total_syn = sum(r.n_synapses for r in all_regions)
    total_franken = sum(sum(1 for v in lm.values() if len(v) > 1) for lm in all_label_maps)
    print(f"\nTotal train: {total_frags} fragments, {total_syn} synapses, "
          f"{total_franken} frankenmerges across {len(train_bboxes)} regions")

    # ------------------------------------------------------------------ Encode all regions
    print(f"\nTraining FragmentEncoder on all {len(train_bboxes)} regions ({args.embed_epochs} epochs) …")
    # Merge all label maps for cross-region supervision in the encoder
    merged_lmap: dict = {}
    for lm in all_label_maps:
        merged_lmap.update(lm)

    encoder = FragmentEncoder(node_input_dim=4, d_model=64, output_dim=32)
    if args.embed_epochs > 0:
        train_fragment_encoder(encoder, all_frags, n_epochs=args.embed_epochs,
                               lr=1e-3, margin=1.0, device=args.device,
                               root_label_map=merged_lmap, log_every=20)

    # Build per-region graphs
    train_graphs = []
    for i, (frags, region) in enumerate(zip(all_frags, all_regions)):
        frags_enc = encode_fragments(encoder, frags, device=args.device)
        g = build_observation_graph(region, frags_enc, side=args.side,
                                     k_spatial=args.k_spatial)
        n_true = int(len(np.unique(g.labels[g.labels != 0])))
        n_franken = sum(1 for v in all_label_maps[i].values() if len(v) > 1)
        print(f"  Region {chr(65+i)}: {g.n_nodes} nodes | {g.n_edges} edges | "
              f"{n_true} neurons | {n_franken} frankenmerges")
        train_graphs.append(g)

    # ------------------------------------------------------------------ Multi-region train
    print(f"\nTraining EdgePartitionGNN on concatenated mega-graph ({args.partition_epochs} epochs) …")
    model, history = train_edge_partition_multi_region(
        train_graphs,
        n_epochs=args.partition_epochs,
        lr=1e-3,
        franken_hard_frac=args.franken_hard_frac,
        device=args.device,
        seed=args.seed,
        log_every=50,
    )

    # ------------------------------------------------------------------ In-sample eval
    print("\nIn-sample evaluation (per region):")
    for i, (g, frags, region, lmap) in enumerate(zip(train_graphs, all_frags, all_regions, all_label_maps)):
        pred = partition_observations_cc(model, g, bias=args.cc_bias, device=args.device)
        ev = evaluate_partition(pred, g.labels)
        mm = merge_metrics(g, pred)
        print(f"  Region {chr(65+i)}: {_fmt({**ev, **mm})}")

    # ------------------------------------------------------------------ Test
    print(f"\n[TEST] Building world (held-out bbox, model never trained on this) …")
    frags_te, region_te, lmap_te = build_region_world(
        test_bbox, version=args.version, side=args.side,
        max_synapses=args.max_synapses,
        min_syn_per_fragment=args.min_syn_per_fragment,
        seed=args.seed, verbose=True,
        l2_skeletons=not args.no_l2_skeletons)

    n_franken_te = sum(1 for v in lmap_te.values() if len(v) > 1)
    frags_te_enc = encode_fragments(encoder, frags_te, device=args.device)
    graph_te = build_observation_graph(region_te, frags_te_enc, side=args.side,
                                        k_spatial=args.k_spatial)
    n_true_te = int(len(np.unique(graph_te.labels[graph_te.labels != 0])))
    print(f"Test: {graph_te.n_nodes} nodes | {graph_te.n_edges} edges | "
          f"{n_true_te} neurons | {n_franken_te} frankenmerges")

    pred_te = partition_observations_cc(model, graph_te, bias=args.cc_bias, device=args.device)
    ev_te = evaluate_partition(pred_te, graph_te.labels)
    mm_te = merge_metrics(graph_te, pred_te)

    # Shape assembly
    shapes = assemble_partition_shapes(frags_te, pred_te, graph_te.fragment_id,
                                        stitch_radius_nm=5_000.0)
    mlist = [neuron_shape_metrics(s) for s in shapes.values()]
    is_tree_frac = float(np.mean([m['is_tree'] for m in mlist]))
    cable_arr = np.array([m['cable_length_um'] for m in mlist])

    # ------------------------------------------------------------------ Summary
    print(f"\n{'='*64}")
    print(f"SUMMARY: multi-region generalisation  ({len(train_bboxes)} train → 1 test)")
    print(f"{'='*64}")
    print(f"  Out-of-sample: {_fmt({**ev_te, **mm_te})}")
    print(f"  Shape assembly: {len(shapes)} neurons  is_tree={is_tree_frac:.3f}  "
          f"cable_median={np.median(cable_arr):.0f}um")

    bar1 = ev_te['ari'] > 0.3 and mm_te['merge_precision'] > 0.95
    bar2 = mm_te['merge_precision'] > 0.95 and mm_te['merge_recall'] > 0.70
    bar3 = mm_te['frankenmerge_split_recall'] > 0.50
    print(f"\n  Bar1 (ARI>0.3 & merge_P>0.95)     {'PASS' if bar1 else 'FAIL'}")
    print(f"  Bar2 (merge_P>0.95 & merge_R>0.70) {'PASS' if bar2 else 'FAIL'}")
    print(f"  Bar3 (fk_split>0.50)               "
          f"{'PASS' if bar3 else 'FAIL'}"
          f"  ({n_franken_te} frankenmerges in test bbox)")
    print(f"{'='*64}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
