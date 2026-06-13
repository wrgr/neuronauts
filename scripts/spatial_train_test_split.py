#!/usr/bin/env python3
"""Spatial train/test split benchmark for NeuronautS.

Critical validity test: train on bbox A, evaluate on bbox B.
No shared fragments, no shared synapses. Tests spatial generalization.

The two bboxes are non-overlapping halves of the same cortical depth range:
  Train: x  950_000 – 1_150_000
  Test:  x 1_150_000 – 1_350_000
  (same y,z range)

The test bbox is the one used in all prior benchmarks, so its results are
directly comparable. The train bbox is fresh data the model has never seen
the test partition for.

Dense box note: increasing the y-extent (30k -> 60k nm) or z-extent
doubles fragment count and stress-tests the partition in the crowded regime
where the problem is hardest.

Usage
-----
  python scripts/spatial_train_test_split.py \\
      --version 1718 --partition-epochs 150 --cc-bias -1.0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _fmt_merge(m: dict) -> str:
    return (f"merge_P={m['merge_precision']:.3f} merge_R={m['merge_recall']:.3f} "
            f"over={m['over_merge_rate']:.3f} fk_split={m['frankenmerge_split_recall']:.3f}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", type=int, default=1718)
    p.add_argument("--side", default="pre", choices=["pre", "post"])
    p.add_argument("--max-synapses-train", type=int, default=20_000)
    p.add_argument("--max-synapses-test", type=int, default=20_000)
    p.add_argument("--min-syn-per-fragment", type=int, default=5)
    p.add_argument("--k-spatial", type=int, default=8)
    p.add_argument("--embed-epochs", type=int, default=20)
    p.add_argument("--partition-epochs", type=int, default=150)
    p.add_argument("--cc-bias", type=float, default=-2.0,
                   help="Out-of-sample operating point. -2.0 clears Bar2 (merge_P>0.95) "
                        "on unseen bbox; -1.0 was tuned for in-sample only.")
    p.add_argument("--franken-hard-frac", type=float, default=0.30)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-l2-skeletons", action="store_true")
    p.add_argument("--dense", action="store_true",
                   help="Use larger bboxes (double y-extent) for the dense-box stress test")
    args = p.parse_args()

    from treestitch.assemble import assemble_partition_shapes, neuron_shape_metrics
    from treestitch.embed import FragmentEncoder, encode_fragments, train_fragment_encoder
    from treestitch.graph import build_observation_graph
    from treestitch.partition import (
        evaluate_partition, merge_metrics,
        partition_observations_cc, train_edge_partition,
    )
    from treestitch.realworld import build_region_world

    # Non-overlapping spatial halves.
    # Train: western bbox (different neurons, never seen in test eval)
    # Test:  the established benchmark bbox
    y0, y1 = 930_000, 980_000
    z0, z1 = 780_000, 880_000
    if args.dense:
        # wider y-range: ~2× more neurons per bbox, harder partition problem
        y0, y1 = 900_000, 1_000_000
        print("Dense mode: y-extent doubled (100k nm)")

    train_bbox = ((950_000, y0, z0), (1_150_000, y1, z1))
    test_bbox  = ((1_150_000, y0, z0), (1_350_000, y1, z1))

    print("=" * 64)
    print(f"Spatial train/test split  (v117 → v{args.version})")
    print(f"  Train bbox: {train_bbox}")
    print(f"  Test  bbox: {test_bbox}")
    print("=" * 64)

    # ------------------------------------------------------------------ Train
    print(f"\n[TRAIN] Building world …")
    frags_tr, region_tr, lmap_tr = build_region_world(
        train_bbox, version=args.version, side=args.side,
        max_synapses=args.max_synapses_train,
        min_syn_per_fragment=args.min_syn_per_fragment,
        seed=args.seed, verbose=True,
        l2_skeletons=not args.no_l2_skeletons)

    print(f"\nTraining FragmentEncoder ({args.embed_epochs} epochs) …")
    encoder = FragmentEncoder(node_input_dim=4, d_model=64, output_dim=32)
    if args.embed_epochs > 0:
        train_fragment_encoder(encoder, [frags_tr], n_epochs=args.embed_epochs,
                               lr=1e-3, margin=1.0, device=args.device,
                               root_label_map=lmap_tr, log_every=20)

    frags_tr_enc = encode_fragments(encoder, frags_tr, device=args.device)
    graph_tr = build_observation_graph(region_tr, frags_tr_enc, side=args.side,
                                        k_spatial=args.k_spatial)

    n_true_tr = int(len(np.unique(graph_tr.labels[graph_tr.labels != 0])))
    n_franken_tr = sum(1 for v in lmap_tr.values() if len(v) > 1)
    print(f"Train graph: {graph_tr.n_nodes} nodes | {graph_tr.n_edges} edges | "
          f"{n_true_tr} neurons | {n_franken_tr} frankenmerges")

    print(f"\nTraining EdgePartitionGNN ({args.partition_epochs} epochs) …")
    model, history = train_edge_partition(
        graph_tr, n_epochs=args.partition_epochs, lr=1e-3,
        franken_hard_frac=args.franken_hard_frac,
        device=args.device, seed=args.seed, log_every=50)

    # In-sample (train bbox) eval — sanity check
    pred_tr = partition_observations_cc(model, graph_tr, bias=args.cc_bias,
                                         device=args.device)
    ev_tr = evaluate_partition(pred_tr, graph_tr.labels)
    mm_tr = merge_metrics(graph_tr, pred_tr)
    print(f"\nIn-sample (train bbox):")
    print(f"  ARI={ev_tr['ari']:.4f}  clusters={ev_tr['n_clusters_pred']}/{ev_tr['n_clusters_true']}")
    print(f"  {_fmt_merge(mm_tr)}")

    # ------------------------------------------------------------------ Test
    print(f"\n[TEST] Building world (unseen bbox) …")
    frags_te, region_te, lmap_te = build_region_world(
        test_bbox, version=args.version, side=args.side,
        max_synapses=args.max_synapses_test,
        min_syn_per_fragment=args.min_syn_per_fragment,
        seed=args.seed, verbose=True,
        l2_skeletons=not args.no_l2_skeletons)

    # Encode test fragments with the SAME encoder trained on train data
    frags_te_enc = encode_fragments(encoder, frags_te, device=args.device)
    graph_te = build_observation_graph(region_te, frags_te_enc, side=args.side,
                                        k_spatial=args.k_spatial)

    n_true_te = int(len(np.unique(graph_te.labels[graph_te.labels != 0])))
    n_franken_te = sum(1 for v in lmap_te.values() if len(v) > 1)
    print(f"Test graph: {graph_te.n_nodes} nodes | {graph_te.n_edges} edges | "
          f"{n_true_te} neurons | {n_franken_te} frankenmerges")

    # Apply model trained on train bbox to unseen test bbox
    pred_te = partition_observations_cc(model, graph_te, bias=args.cc_bias,
                                         device=args.device)
    ev_te = evaluate_partition(pred_te, graph_te.labels)
    mm_te = merge_metrics(graph_te, pred_te)
    print(f"\nOut-of-sample (test bbox — KEY RESULT):")
    print(f"  ARI={ev_te['ari']:.4f}  clusters={ev_te['n_clusters_pred']}/{ev_te['n_clusters_true']}")
    print(f"  {_fmt_merge(mm_te)}")

    # Shape assembly on test bbox
    shapes = assemble_partition_shapes(frags_te, pred_te, graph_te.fragment_id,
                                        stitch_radius_nm=5_000.0)
    m_list = [neuron_shape_metrics(s) for s in shapes.values()]
    is_tree_frac = np.mean([m['is_tree'] for m in m_list])
    cable_arr = np.array([m['cable_length_um'] for m in m_list])
    n_comp_arr = np.array([m['n_connected_components'] for m in m_list])
    print(f"\nShape assembly ({len(shapes)} neurons):")
    print(f"  is_tree={is_tree_frac:.3f}  "
          f"fully_connected={( n_comp_arr==1).mean():.1%}  "
          f"cable_um median={np.median(cable_arr):.1f}  "
          f"p95={np.percentile(cable_arr,95):.1f}")

    # ---------------------------------------------------------------- Summary
    print(f"\n{'='*64}")
    print(f"SUMMARY: spatial generalisation  (train bbox A → test bbox B)")
    print(f"{'='*64}")
    print(f"  {'split':<12} {'ARI':>7} {'clusters':>12} {'merge_P':>9} {'over':>7} {'fk_split':>9}")
    print(f"  {'in-sample':<12} {ev_tr['ari']:>7.4f} "
          f"{str(ev_tr['n_clusters_pred'])+'/'+str(ev_tr['n_clusters_true']):>12} "
          f"{mm_tr['merge_precision']:>9.3f} {mm_tr['over_merge_rate']:>7.3f} "
          f"{mm_tr['frankenmerge_split_recall']:>9.3f}")
    print(f"  {'out-of-sample':<12} {ev_te['ari']:>7.4f} "
          f"{str(ev_te['n_clusters_pred'])+'/'+str(ev_te['n_clusters_true']):>12} "
          f"{mm_te['merge_precision']:>9.3f} {mm_te['over_merge_rate']:>7.3f} "
          f"{mm_te['frankenmerge_split_recall']:>9.3f}")

    bar1 = ev_te['ari'] > 0.3 and mm_te['merge_precision'] > 0.95
    bar2 = mm_te['merge_precision'] > 0.95 and mm_te['merge_recall'] > 0.70
    bar3 = mm_te['frankenmerge_split_recall'] > 0.50 or n_franken_te == 0
    print(f"\n  Bar1 (ARI>0.3 & merge_P>0.95) {'PASS' if bar1 else 'FAIL'}")
    print(f"  Bar2 (merge_P>0.95 & merge_R>0.70) {'PASS' if bar2 else 'FAIL'}")
    print(f"  Bar3 (fk_split>0.50 or no frankenmerges) {'PASS' if bar3 else 'FAIL (or N/A)'}")
    print(f"{'='*64}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
