#!/usr/bin/env python3
"""Spatial variance study for NeuronautS.

Trains once on the standard 3-region A/B/C protocol (saves a checkpoint),
then evaluates on multiple test bboxes — both in-column (where v1718 GT
exists) and out-of-column (where shape plausibility is the metric).

Also fits a temperature-scaling calibration on the training graph and
reports calibration quality (ECE) on each test bbox.

This answers: how stable are ARI/merge_P across different spatial locations?
A tight spread strengthens the out-of-sample claim; a wide spread flags
geographic sensitivity that the paper should acknowledge.

Test locations
--------------
In-column (v1718 GT available, y=930-1000k, z=780-880k):
  T1: x=1150-1350k  (current reference)
  T2: x=550-750k    (west of train A; reveals column extent)
  T3: x=1150-1350k, y=870-940k   (south y-shift)
  T4: x=1150-1350k, y=1000-1070k (north y-shift)

Out-of-column (shape plausibility only, no GT):
  OOC1: x=200-400k,  y=500-570k,  z=700-800k  (reference OOC; confirmed working)
  OOC2: x=1200-1400k,y=400-470k,  z=700-800k  (central x, lower y band)
  OOC3: x=600-800k,  y=600-670k,  z=700-800k  (west-central, mid y band)

Usage
-----
  # Full run (train once, eval 7 bboxes, takes ~20 min):
  python scripts/spatial_variance.py

  # Quick debug (fewer epochs, skip calibration):
  python scripts/spatial_variance.py --quick --no-calibration

  # Load existing checkpoint to skip training:
  python scripts/spatial_variance.py --checkpoint /tmp/neuronauts_variance.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _fmt_in(ev: dict, mm: dict) -> str:
    return (f"ARI={ev['ari']:.3f}  "
            f"merge_P={mm['merge_precision']:.3f}  "
            f"merge_R={mm['merge_recall']:.3f}  "
            f"over={mm['over_merge_rate']:.3f}  "
            f"fk={mm['frankenmerge_split_recall']:.3f}")


def _fmt_ooc(over: float, cable_med: float, is_tree: float) -> str:
    return (f"over={over:.3f}  cable_med={cable_med:.0f}µm  "
            f"is_tree={is_tree:.3f}")


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
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Path to a saved checkpoint; if set, skip training and load model.")
    p.add_argument("--save-checkpoint", type=str, default="/tmp/neuronauts_variance.pt",
                   help="Where to save the trained checkpoint (default: /tmp/neuronauts_variance.pt)")
    p.add_argument("--quick", action="store_true",
                   help="Fewer epochs (debug mode): embed=5, partition=30")
    p.add_argument("--no-calibration", action="store_true",
                   help="Skip temperature scaling calibration")
    p.add_argument("--dual-side", action="store_true",
                   help="Also partition the POST side at each in-column bbox and "
                        "reconstruct the connectome from both partitions (no GT root "
                        "ids), reporting directed + undirected edge F1.")
    p.add_argument("--dual-post-l2", action="store_true",
                   help="Use real L2 skeletons for the post (dendritic) side in "
                        "--dual-side mode. Default uses synapse-cloud fragments, which "
                        "is far faster because the post side has many more fragments.")
    args = p.parse_args()

    if args.quick:
        args.embed_epochs = 5
        args.partition_epochs = 30
        print("[quick mode] embed=5 epochs, partition=30 epochs")

    from treestitch.assemble import assemble_partition_shapes, neuron_shape_metrics
    from treestitch.calibration import (
        expected_calibration_error, fit_temperature,
        calibrated_obs_confidence, reliability_diagram,
    )
    from treestitch.checkpoint import load_checkpoint, save_checkpoint
    from treestitch.connectivity import (
        connectome_accuracy, dual_side_connectome_accuracy,
    )
    from treestitch.embed import FragmentEncoder, encode_fragments, train_fragment_encoder
    from treestitch.graph import build_observation_graph
    from treestitch.partition import (
        evaluate_partition, merge_metrics,
        partition_observations_cc, train_edge_partition_multi_region,
    )
    from treestitch.realworld import build_region_world, build_region_world_dual

    # ── Training bboxes (same as multi_region_train.py) ────────────────────
    y0, y1 = 930_000, 1_000_000   # dense y-extent
    z0, z1 = 780_000, 880_000
    buf = args.seam_buffer
    train_bboxes = [
        ((750_000,  y0, z0), (950_000,           y1, z1)),  # A
        ((950_000,  y0, z0), (1_150_000 - buf,   y1, z1)),  # B (seam-buffered)
        ((1_350_000 + buf, y0, z0), (1_550_000,  y1, z1)),  # C (seam-buffered)
    ]

    # ── Test bboxes ─────────────────────────────────────────────────────────
    IN_COLUMN = [
        ("T1 x=1150-1350k (reference)",
         ((1_150_000, y0, z0), (1_350_000, y1, z1))),
        ("T2 x=550-750k (west of A)",
         ((550_000, y0, z0), (750_000, y1, z1))),
        ("T3 y-shift south (y=870-940k)",
         ((1_150_000, 870_000, z0), (1_350_000, 940_000, z1))),
        ("T4 y-shift north (y=1000-1070k)",
         ((1_150_000, 1_000_000, z0), (1_350_000, 1_070_000, z1))),
    ]
    OUT_OF_COLUMN = [
        ("OOC1 x=200-400k (reference)",
         ((200_000, 500_000, 700_000), (400_000, 570_000, 800_000))),
        ("OOC2 x=1200-1400k y=400-470k",
         ((1_200_000, 400_000, 700_000), (1_400_000, 470_000, 800_000))),
        ("OOC3 x=600-800k y=600-670k",
         ((600_000, 600_000, 700_000), (800_000, 670_000, 800_000))),
    ]

    print("=" * 68)
    print(f"Spatial variance study  (v117 → v{args.version})")
    print(f"  Train: 3 regions (A/B/C, seam buffer={buf//1000}µm)")
    print(f"  In-column test locations: {len(IN_COLUMN)}")
    print(f"  OOC test locations:       {len(OUT_OF_COLUMN)}")
    print("=" * 68)

    # ── Train or load ───────────────────────────────────────────────────────
    enc_kwargs = dict(node_input_dim=4, d_model=64, output_dim=32)

    if args.checkpoint and Path(args.checkpoint).exists():
        print(f"\nLoading checkpoint from {args.checkpoint} …")
        encoder, model = load_checkpoint(args.checkpoint)
        print("  Checkpoint loaded — skipping training.")
        # We still need a training graph for calibration
        need_train_graph = not args.no_calibration
    else:
        need_train_graph = True

    if need_train_graph or not (args.checkpoint and Path(args.checkpoint).exists()):
        all_frags, all_regions, all_label_maps = [], [], []
        for i, bbox in enumerate(train_bboxes):
            label = chr(65 + i)
            print(f"\n[TRAIN {label}] Building world …")
            frags, region, lmap = build_region_world(
                bbox, version=args.version, side=args.side,
                max_synapses=args.max_synapses,
                min_syn_per_fragment=args.min_syn_per_fragment,
                seed=args.seed, verbose=True)
            all_frags.append(frags)
            all_regions.append(region)
            all_label_maps.append(lmap)
            n_fk = sum(1 for v in lmap.values() if len(v) > 1)
            print(f"  → {len(frags)} frags, {region.n_synapses} syn, {n_fk} frankenmerges")

        if not (args.checkpoint and Path(args.checkpoint).exists()):
            print(f"\nTraining FragmentEncoder ({args.embed_epochs} epochs) …")
            merged_lmap: dict = {}
            for lm in all_label_maps:
                merged_lmap.update(lm)
            encoder = FragmentEncoder(**enc_kwargs)
            if args.embed_epochs > 0:
                train_fragment_encoder(encoder, all_frags, n_epochs=args.embed_epochs,
                                       lr=1e-3, margin=1.0, device=args.device,
                                       root_label_map=merged_lmap, log_every=10)

        train_graphs = []
        for i, (frags, region) in enumerate(zip(all_frags, all_regions)):
            frags_enc = encode_fragments(encoder, frags, device=args.device)
            g = build_observation_graph(region, frags_enc, side=args.side,
                                        k_spatial=args.k_spatial)
            train_graphs.append(g)

        if not (args.checkpoint and Path(args.checkpoint).exists()):
            print(f"\nTraining EdgePartitionGNN ({args.partition_epochs} epochs) …")
            model, _ = train_edge_partition_multi_region(
                train_graphs,
                n_epochs=args.partition_epochs, lr=1e-3,
                franken_hard_frac=args.franken_hard_frac,
                device=args.device, seed=args.seed, log_every=25)

            # Derive kwargs to match what train_edge_partition_gnn actually used
            _all_etypes = np.concatenate([g.edge_type for g in train_graphs])
            _n_et = int(max(2, int(_all_etypes.max()) + 1)) if len(_all_etypes) > 0 else 2
            _ef = train_graphs[0]
            _efd = int(_ef.edge_feat.shape[1]) if _ef.edge_feat.ndim == 2 else 0
            gnn_kwargs = dict(
                input_dim=train_graphs[0].node_feat.shape[1],
                d_model=64, n_edge_types=_n_et, output_dim=32,
                dropout=0.1, edge_feat_dim=_efd,
            )
            save_checkpoint(args.save_checkpoint, encoder, model,
                            encoder_kwargs=enc_kwargs, gnn_kwargs=gnn_kwargs,
                            extra={"train_bboxes": train_bboxes,
                                   "version": args.version,
                                   "cc_bias": args.cc_bias})
            print(f"  Checkpoint saved → {args.save_checkpoint}")

        # ── Calibration (on first training graph) ─────────────────────────
        T = 1.0
        ece_train = float("nan")
        if not args.no_calibration:
            print("\nFitting temperature calibration on train graph A …")
            T = fit_temperature(model, train_graphs[0], bias=args.cc_bias,
                                device=args.device)
            diag_tr = reliability_diagram(model, train_graphs[0], T,
                                          bias=args.cc_bias, device=args.device)
            ece_train = expected_calibration_error(diag_tr)
            print(f"  T={T:.4f}  ECE(train)={ece_train:.4f}")
            if T > 1.05:
                print("  → Model is overconfident; temperature softens probabilities.")
            elif T < 0.95:
                print("  → Model is underconfident (unusual).")
            else:
                print("  → Model is well-calibrated; T≈1.0.")
    else:
        T = 1.0
        ece_train = float("nan")
        train_graphs = []   # not available when loading checkpoint

    # ── Evaluate in-column bboxes ────────────────────────────────────────
    print(f"\n{'='*68}")
    print("IN-COLUMN EVALUATION  (v1718 GT available)")
    print(f"{'='*68}")

    in_col_results = []
    for name, bbox in IN_COLUMN:
        print(f"\n[{name}]")
        try:
            frags, region, lmap = build_region_world(
                bbox, version=args.version, side=args.side,
                max_synapses=args.max_synapses,
                min_syn_per_fragment=args.min_syn_per_fragment,
                seed=args.seed, verbose=True)

            n_fk = sum(1 for v in lmap.values() if len(v) > 1)
            if n_fk == 0:
                print(f"  ⚠ 0 frankenmerges — this bbox may be outside the proofread column")

            frags_enc = encode_fragments(encoder, frags, device=args.device)
            graph = build_observation_graph(region, frags_enc, side=args.side,
                                            k_spatial=args.k_spatial)
            pred = partition_observations_cc(model, graph, bias=args.cc_bias,
                                             device=args.device)
            ev = evaluate_partition(pred, graph.labels)
            mm = merge_metrics(graph, pred)

            shapes = assemble_partition_shapes(frags, pred, graph.fragment_id,
                                               stitch_radius_nm=5_000.0)
            mlist = [neuron_shape_metrics(s) for s in shapes.values()]
            cable_med    = float(np.median([m['cable_length_um'] for m in mlist])) if mlist else 0.0
            max_path_med = float(np.median([m['max_path_length_um'] for m in mlist])) if mlist else 0.0
            tort_med     = float(np.nanmedian([m['tortuosity'] for m in mlist])) if mlist else float("nan")
            is_tree      = float(np.mean([m['is_tree'] for m in mlist])) if mlist else 0.0

            conn = connectome_accuracy(pred, region)
            has_conn = region.post_root_id is not None and int((region.post_root_id > 0).sum()) > 0

            print(f"  {_fmt_in(ev, mm)}")
            print(f"  cable_med={cable_med:.0f}µm  max_path={max_path_med:.0f}µm  "
                  f"tort={tort_med:.2f}  is_tree={is_tree:.3f}  n_neurons={len(shapes)}")
            if has_conn:
                print(f"  conn_edge_F1(dir)={conn['conn_edge_f1']:.3f}  "
                      f"conn_edge_F1(undir)={conn['conn_edge_f1_undir']:.3f}  "
                      f"syn_attr_acc={conn['synapse_attr_acc']:.3f}  "
                      f"({conn['n_true_edges']} dir / {conn['n_true_edges_undir']} undir true edges)")

            row = {
                "name": name, "ari": ev["ari"],
                "merge_p": mm["merge_precision"],
                "merge_r": mm["merge_recall"],
                "over": mm["over_merge_rate"],
                "fk": mm["frankenmerge_split_recall"],
                "cable_med": cable_med,
                "max_path_med": max_path_med,
                "tort_med": tort_med,
                "is_tree": is_tree,
                "n_franken": n_fk,
                "conn_f1": conn["conn_edge_f1"],
                "conn_f1_undir": conn["conn_edge_f1_undir"],
                "syn_attr_acc": conn["synapse_attr_acc"],
                "n_true_edges": conn["n_true_edges"],
            }

            # ── Dual-side: partition the POST side and reconstruct the connectome
            #    from BOTH partitions (no ground-truth root ids). ───────────────
            if args.dual_side:
                try:
                    # Build pre AND post worlds for the SAME synapses from a single
                    # fetch so every synapse shares a real id across both sides (a
                    # guaranteed join). The post (dendritic) side is far denser, so
                    # default to cloud fragments there (--dual-post-l2 opts into L2).
                    print("  [dual-side] building dual world (single fetch) …")
                    (frags_pre2, region_pre2, _), (frags_post, region_post, _) = \
                        build_region_world_dual(
                            bbox, version=args.version,
                            max_synapses=args.max_synapses,
                            min_syn_per_fragment=args.min_syn_per_fragment,
                            seed=args.seed, verbose=True,
                            l2_skeletons_pre=True,
                            l2_skeletons_post=args.dual_post_l2)
                    # Partition both sides with the same model.
                    fe_pre2 = encode_fragments(encoder, frags_pre2, device=args.device)
                    g_pre2 = build_observation_graph(region_pre2, fe_pre2,
                                                     side="pre", k_spatial=args.k_spatial)
                    pred_pre2 = partition_observations_cc(
                        model, g_pre2, bias=args.cc_bias, device=args.device)
                    fe_post = encode_fragments(encoder, frags_post, device=args.device)
                    g_post = build_observation_graph(region_post, fe_post,
                                                     side="post", k_spatial=args.k_spatial)
                    pred_post = partition_observations_cc(
                        model, g_post, bias=args.cc_bias, device=args.device)
                    dual = dual_side_connectome_accuracy(
                        pred_pre2, region_pre2, pred_post, region_post)
                    print(f"  [dual-side] conn_F1(dir)={dual['conn_edge_f1']:.3f}  "
                          f"conn_F1(undir)={dual['conn_edge_f1_undir']:.3f}  "
                          f"both-sides={dual['n_synapses_both_sides']} syn  "
                          f"(pre-only={dual['n_synapses_pre_only']}, "
                          f"post-only={dual['n_synapses_post_only']})")
                    row["dual_f1"] = dual["conn_edge_f1"]
                    row["dual_f1_undir"] = dual["conn_edge_f1_undir"]
                    row["dual_both"] = dual["n_synapses_both_sides"]
                except Exception as exc:
                    import traceback
                    print(f"  [dual-side] ERROR: {exc}")
                    traceback.print_exc()

            in_col_results.append(row)
        except Exception as exc:
            print(f"  ERROR: {exc}")
            in_col_results.append({"name": name, "error": str(exc)})

    # ── Evaluate OOC bboxes ───────────────────────────────────────────────
    print(f"\n{'='*68}")
    print("OUT-OF-COLUMN EVALUATION  (shape plausibility — no GT)")
    print(f"{'='*68}")

    ooc_results = []
    for name, bbox in OUT_OF_COLUMN:
        print(f"\n[{name}]")
        try:
            frags, region, lmap = build_region_world(
                bbox, version=args.version, side=args.side,
                max_synapses=args.max_synapses,
                min_syn_per_fragment=args.min_syn_per_fragment,
                seed=args.seed, verbose=True)

            n_fk = sum(1 for v in lmap.values() if len(v) > 1)
            print(f"  Edit signal: {n_fk} frankenmerges "
                  f"({'⚠ some edits' if n_fk > 0 else '✓ unproofread'})")

            frags_enc = encode_fragments(encoder, frags, device=args.device)
            graph = build_observation_graph(region, frags_enc, side=args.side,
                                            k_spatial=args.k_spatial)

            # pseudo-labels: each v117 root = 1 neuron
            frag_id_to_idx = {int(fid): i + 1
                              for i, fid in enumerate(np.unique(graph.fragment_id))}
            pseudo_labels = np.array([frag_id_to_idx[int(f)]
                                      for f in graph.fragment_id], dtype=np.int64)

            pred = partition_observations_cc(model, graph, bias=args.cc_bias,
                                             device=args.device)

            mm_pseudo = merge_metrics(graph, pred, ignore_label=0)
            over = mm_pseudo["over_merge_rate"]

            shapes = assemble_partition_shapes(frags, pred, graph.fragment_id,
                                               stitch_radius_nm=5_000.0)
            mlist = [neuron_shape_metrics(s) for s in shapes.values()]
            cable_med    = float(np.median([m['cable_length_um'] for m in mlist])) if mlist else 0.0
            max_path_med = float(np.median([m['max_path_length_um'] for m in mlist])) if mlist else 0.0
            tort_med     = float(np.nanmedian([m['tortuosity'] for m in mlist])) if mlist else float("nan")
            is_tree      = float(np.mean([m['is_tree'] for m in mlist])) if mlist else 0.0
            fully_conn   = float(np.mean([m['n_connected_components'] == 1
                                          for m in mlist])) if mlist else 0.0

            print(f"  {_fmt_ooc(over, cable_med, is_tree)}  "
                  f"max_path={max_path_med:.0f}µm  tort={tort_med:.2f}  fully_conn={fully_conn:.1%}")
            cable_ok = 500 <= cable_med <= 20_000
            print(f"  cable plausible: {'✓' if cable_ok else '⚠'}")

            ooc_results.append({
                "name": name, "over": over, "cable_med": cable_med,
                "max_path_med": max_path_med, "tort_med": tort_med,
                "is_tree": is_tree, "fully_conn": fully_conn,
                "n_neurons": len(shapes), "n_franken": n_fk,
            })
        except Exception as exc:
            print(f"  ERROR: {exc}")
            ooc_results.append({"name": name, "error": str(exc)})

    # ── Summary table ─────────────────────────────────────────────────────
    print(f"\n{'='*68}")
    print("SUMMARY: spatial variance")
    print(f"{'='*68}")

    good_in = [r for r in in_col_results if "error" not in r]
    good_ooc = [r for r in ooc_results if "error" not in r]

    if not args.no_calibration:
        print(f"\n  Calibration: T={T:.4f}  ECE(train)={ece_train:.4f}")

    print(f"\n  In-column ({len(good_in)} locations):")
    print(f"  {'Location':<38} {'ARI':>6} {'merge_P':>8} {'merge_R':>8} "
          f"{'conn_F1':>8} {'syn_acc':>8} {'cable_med':>10} {'tort':>6}")
    print(f"  {'-'*38} {'-'*6} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*10} {'-'*6}")
    for r in good_in:
        cf1 = f"{r['conn_f1']:.3f}" if r['conn_f1'] == r['conn_f1'] else "n/a"
        sa  = f"{r['syn_attr_acc']:.3f}" if r['syn_attr_acc'] == r['syn_attr_acc'] else "n/a"
        tort = f"{r['tort_med']:.2f}" if r['tort_med'] == r['tort_med'] else "n/a"
        print(f"  {r['name']:<38} {r['ari']:>6.3f} {r['merge_p']:>8.3f} "
              f"{r['merge_r']:>8.3f} {cf1:>8} {sa:>8} "
              f"{r['cable_med']:>9.0f}µ {tort:>6}")

    if len(good_in) >= 2:
        aris   = [r["ari"] for r in good_in]
        mps    = [r["merge_p"] for r in good_in]
        cf1s   = [r["conn_f1"] for r in good_in if r["conn_f1"] == r["conn_f1"]]
        print(f"\n  In-column variance:")
        print(f"    ARI:       mean={np.mean(aris):.3f}  std={np.std(aris):.3f}  "
              f"range=[{min(aris):.3f}, {max(aris):.3f}]")
        print(f"    merge_P:   mean={np.mean(mps):.3f}   std={np.std(mps):.3f}  "
              f"range=[{min(mps):.3f}, {max(mps):.3f}]")
        if cf1s:
            print(f"    conn_F1:   mean={np.mean(cf1s):.3f}  std={np.std(cf1s):.3f}  "
                  f"range=[{min(cf1s):.3f}, {max(cf1s):.3f}]")
        cf1u = [r["conn_f1_undir"] for r in good_in
                if r.get("conn_f1_undir") == r.get("conn_f1_undir")]
        if cf1u:
            print(f"    conn_F1u:  mean={np.mean(cf1u):.3f}  std={np.std(cf1u):.3f}  "
                  f"range=[{min(cf1u):.3f}, {max(cf1u):.3f}]")

    if args.dual_side:
        dual_rows = [r for r in good_in if "dual_f1" in r]
        if dual_rows:
            print(f"\n  Dual-side connectome (both partitions, NO GT root ids):")
            print(f"  {'Location':<38} {'dir_F1':>7} {'undir_F1':>9} {'both_syn':>9}")
            print(f"  {'-'*38} {'-'*7} {'-'*9} {'-'*9}")
            for r in dual_rows:
                print(f"  {r['name']:<38} {r['dual_f1']:>7.3f} "
                      f"{r['dual_f1_undir']:>9.3f} {r['dual_both']:>9d}")

    print(f"\n  OOC ({len(good_ooc)} locations):")
    print(f"  {'Location':<38} {'over':>6} {'cable_med':>10} {'max_path':>10} "
          f"{'tort':>6} {'is_tree':>8}")
    print(f"  {'-'*38} {'-'*6} {'-'*10} {'-'*10} {'-'*6} {'-'*8}")
    for r in good_ooc:
        tort = f"{r['tort_med']:.2f}" if r['tort_med'] == r['tort_med'] else "n/a"
        print(f"  {r['name']:<38} {r['over']:>6.3f} {r['cable_med']:>9.0f}µ  "
              f"{r['max_path_med']:>9.0f}µ  {tort:>6} {r['is_tree']:>8.3f}")

    if len(good_ooc) >= 2:
        cables = [r["cable_med"] for r in good_ooc]
        print(f"\n  OOC cable length variance:")
        print(f"    median: mean={np.mean(cables):.0f}µm  std={np.std(cables):.0f}µm  "
              f"range=[{min(cables):.0f}, {max(cables):.0f}]µm")

    print(f"\n{'='*68}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
