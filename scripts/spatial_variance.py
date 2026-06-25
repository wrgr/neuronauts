#!/usr/bin/env python3
"""Spatial variance study for NeuronautS.

Trains once on the standard 4-region A/B/C/D protocol (saves a checkpoint),
then evaluates on multiple test bboxes — both in-column (where v1718 GT
exists) and out-of-column (where shape plausibility is the metric).

Also fits a temperature-scaling calibration on the training graph and
reports calibration quality (ECE) on each test bbox.

This answers: how stable are ARI/merge_P across different spatial locations?
A tight spread strengthens the out-of-sample claim; a wide spread flags
geographic sensitivity that the paper should acknowledge.

Training regions (z=780-880k):
  A: x=750-950k,   y=930-1000k
  B: x=950-1150k,  y=930-1000k  (seam-buffered right edge)
  C: x=1350-1550k, y=930-1000k  (seam-buffered left edge)
  D: x=1550-1750k, y=930-1000k  (seam-buffered left edge; east expansion)
  E: x=750-950k,   y=1000-1070k (north y-band; matches T4 density regime)

Test locations
--------------
In-column (v1718 GT available, y=930-1000k, z=780-880k):
  T1: x=1150-1350k  (reference; gap between train B and C)
  T2: x=550-750k    (west of train A; reveals column extent)
  T3: x=1150-1350k, y=870-940k   (south y-shift)
  T4: x=1150-1350k, y=1000-1070k (north y-shift)
  T5: x=1750-1950k  (east of train D; east extrapolation)

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
import datetime
import json
import sys
from pathlib import Path

import gc
import math

import numpy as np


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    raise TypeError(f"Not serializable: {type(obj)}")

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _fmt_in(ev: dict, mm: dict) -> str:
    return (f"ARI={ev['ari']:.3f}  "
            f"merge_P={mm['merge_precision']:.3f}  "
            f"merge_R={mm['merge_recall']:.3f}  "
            f"over={mm['over_merge_rate']:.3f}  "
            f"under={mm['under_merge_rate']:.3f}  "
            f"fk={mm['frankenmerge_split_recall']:.3f}")


def _tile_bbox_x(bbox, n_tiles: int):
    """Split a bbox into n_tiles equal sub-bboxes along the x-axis.

    OOC regions can have 10-20k fragments — 3-5× more than typical training
    regions. Large fragment counts cause OOM during L2 skeleton fetching and
    observation-graph construction. Splitting along x keeps each tile at a
    manageable fragment count (~3-5k) while preserving the full y/z extent.
    Metrics are averaged across tiles (weighted by n_nodes for rates).
    """
    (x0, y0, z0), (x1, y1, z1) = bbox
    w = (x1 - x0) / n_tiles
    return [((x0 + i * w, y0, z0), (x0 + (i + 1) * w, y1, z1))
            for i in range(n_tiles)]


def _fmt_ooc(over: float, cable_med: float, is_tree: float) -> str:
    return (f"over={over:.3f}  cable_med={cable_med:.0f}µm  "
            f"is_tree={is_tree:.3f}")


def _subsample_observation_graph(g, n_keep: int, rng: np.random.Generator):
    """Return a copy of ObservationGraph g with n_keep randomly-sampled nodes.

    Edges whose src or dst fell outside the kept set are dropped; indices are
    remapped so the returned graph is self-consistent.
    """
    from treestitch.schemas import ObservationGraph
    n = g.n_nodes
    if n_keep >= n:
        return g
    keep = rng.choice(n, size=n_keep, replace=False)
    keep_set = set(keep.tolist())
    remap = np.full(n, -1, dtype=np.int64)
    remap[keep] = np.arange(len(keep), dtype=np.int64)
    edge_mask = np.array(
        [s in keep_set and d in keep_set
         for s, d in zip(g.edge_src.tolist(), g.edge_dst.tolist())],
        dtype=bool,
    )
    return ObservationGraph(
        node_feat=g.node_feat[keep],
        node_pos=g.node_pos[keep],
        edge_src=remap[g.edge_src[edge_mask]],
        edge_dst=remap[g.edge_dst[edge_mask]],
        edge_type=g.edge_type[edge_mask],
        edge_feat=g.edge_feat[edge_mask],
        labels=g.labels[keep],
        fragment_id=g.fragment_id[keep],
        side=g.side,
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", type=int, default=1718)
    p.add_argument("--side", default="pre", choices=["pre", "post"])
    p.add_argument("--max-synapses", type=int, default=10_000)
    p.add_argument("--eval-max-synapses", type=int, default=None,
                   help="Max synapses for EVALUATION bboxes (default: same as --max-synapses). "
                        "Set higher (e.g. 200000) for more honest full-population evaluation; "
                        "results are cached so the cost is paid only once.")
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
    p.add_argument("--save-bundle", type=str, default=None,
                   help="If set, serialize all evaluation results (metrics + skeletons + "
                        "synapse positions) to a JSON bundle at this path. Load with the "
                        "dashboard/app.py Streamlit app.")
    p.add_argument("--quick", action="store_true",
                   help="Fewer epochs (debug mode): embed=5, partition=30")
    p.add_argument("--no-calibration", action="store_true",
                   help="Skip temperature scaling calibration")
    p.add_argument("--no-eval-l2", action="store_true",
                   help="Skip L2 skeleton fetching during evaluation (use synapse-cloud "
                        "fragments instead). Required when --eval-max-synapses is large: "
                        "at 50k synapses, 1500+ fragments × L2 fetch = hours. "
                        "Morphology metrics are omitted; partition/connectome metrics are unaffected.")
    p.add_argument("--dual-side", action="store_true",
                   help="Also partition the POST side at each in-column bbox and "
                        "reconstruct the connectome from both partitions (no GT root "
                        "ids), reporting directed + undirected edge F1.")
    p.add_argument("--dual-post-l2", action="store_true",
                   help="Use real L2 skeletons for the post (dendritic) side in "
                        "--dual-side mode. Default uses synapse-cloud fragments, which "
                        "is far faster because the post side has many more fragments.")
    p.add_argument("--no-soma", action="store_true",
                   help="Skip nucleus-position download and soma detection.")
    p.add_argument("--post-tile-size", type=int, default=600,
                   help="Max nodes per spatial tile when partitioning the post "
                        "side (--dual-side). Post-side graphs are ~10x larger "
                        "than training graphs; tiling applies the GNN at the "
                        "same scale it was trained on. Default=600.")
    p.add_argument("--pre-tile-size", type=int, default=None,
                   help="Max nodes per spatial tile when partitioning the pre "
                        "side at eval. Required when --eval-max-synapses is large "
                        "(e.g. 50k → 1000–2400 fragment nodes). Same tiling + "
                        "same-fragment-edge reconciliation as --post-tile-size. "
                        "Recommended: 600 (matches typical training graph size).")
    p.add_argument("--tile-x-nm", type=float, default=0,
                   help="If >0, fetch synapses via x-axis tiling (40000 = 40µm tiles). "
                        "Matches warm_cache.py tiling so all tile cache entries are hits. "
                        "Use with --per-tile-limit=200000 for full honest coverage.")
    p.add_argument("--per-tile-limit", type=int, default=200_000,
                   help="Max synapses per tile when --tile-x-nm>0 (default 200000).")
    p.add_argument("--train-regions", default=None,
                   help="Comma-separated subset of train regions to use, e.g. 'E,D'. "
                        "Default: use all (A,B,C,D,E).")
    p.add_argument("--eval-regions", default=None,
                   help="Comma-separated subset of in-column eval regions, e.g. 'T4'. "
                        "Default: use all (T1,T2,T3,T4).")
    p.add_argument("--tile-x-nm", type=float, default=0,
                   help="x-tile width in nm for synapse fetches (0=disabled). "
                        "Set to 40000 to use the tiled synapse cache that bypasses "
                        "CAVE's ~250k per-request row cap. Required for honest "
                        "full-population coverage on dense regions.")
    p.add_argument("--per-tile-limit", type=int, default=200_000,
                   help="Max synapses per tile when --tile-x-nm > 0 (default 200k).")
    p.add_argument("--train-max-nodes", type=int, default=30_000,
                   help="Max observation-graph nodes per GNN training epoch. When the "
                        "training graph exceeds this, each epoch trains on a random "
                        "spatial subgraph of this size to keep GPU memory bounded. "
                        "0 = train on full graph (can OOM on large regions).")
    p.add_argument("--no-train-l2", action="store_true",
                   help="Use synapse point-cloud fragments for training (no L2 "
                        "skeleton fetches). Fast when the L2 cache is not yet "
                        "warm for newly-discovered fragments.")
    p.add_argument("--ooc-x-tiles", type=int, default=4,
                   help="Number of x-axis tiles to split each OOC bbox into "
                        "(default 4). OOC regions often have 10-20k fragments "
                        "vs ~3-5k per training region, causing OOM during L2 "
                        "skeleton fetch and observation-graph construction. "
                        "Tiling evaluates each sub-bbox independently and "
                        "averages shape metrics. Set to 1 to disable tiling.")
    p.add_argument("--post-cc-bias", type=float, default=None,
                   help="cc_bias override for the POST side partition (--dual-side). "
                        "Defaults to --cc-bias. Try 0.0, 1.0, 2.0 to loosen "
                        "the post-side threshold without retraining.")
    p.add_argument("--balanced-dual", action="store_true",
                   help="When --dual-side, subsample each post training graph "
                        "to match the pre-side node count before training. "
                        "Fixes the 10:1 post-to-pre imbalance that drives "
                        "over-conservative post-side partitioning.")
    p.add_argument("--post-pred", default="fragment",
                   choices=["model", "fragment"],
                   help="How to produce the post-side partition (--dual-side). "
                        "'fragment': use the raw v117 dendritic fragment id as the "
                        "cluster label — each fragment is its own cluster, no model "
                        "call needed.  Gives a meaningful AND metric at the fragment "
                        "level and avoids the scale-mismatch issue. "
                        "'model': run the GNN (requires a well-trained post-side model). "
                        "Default: fragment.")
    p.add_argument("--dual-post-model", type=str, default=None,
                   help="Path to a checkpoint trained exclusively on post-side graphs. "
                        "When provided with --post-pred=model, this model is used for "
                        "post-side partitioning instead of the shared model. "
                        "Train with: --dual-side --post-only-train --save-checkpoint PATH.")
    p.add_argument("--post-only-train", action="store_true",
                   help="Train a post-side-only model (3 post graphs, no pre) and save "
                        "to --save-checkpoint.  Use with --dual-side.  The resulting "
                        "checkpoint can then be passed as --dual-post-model in a "
                        "subsequent evaluation run.")
    args = p.parse_args()

    if args.quick:
        args.embed_epochs = 5
        args.partition_epochs = 30
        print("[quick mode] embed=5 epochs, partition=30 epochs")

    post_cc_bias = args.post_cc_bias if args.post_cc_bias is not None else args.cc_bias
    use_frag_post = (args.post_pred == "fragment")

    from neuronauts.line_graph import LineGraphSuite, evaluate_suite
    from treestitch.assemble import assemble_partition_shapes, detect_soma, neuron_shape_metrics
    from treestitch.calibration import (
        expected_calibration_error, fit_temperature,
        calibrated_obs_confidence, reliability_diagram,
    )
    from treestitch.checkpoint import load_checkpoint, save_checkpoint
    from treestitch.connectivity import (
        connectome_accuracy, dual_side_connectome_accuracy,
        _match_clusters_to_neurons,
    )
    from treestitch.embed import FragmentEncoder, encode_fragments, train_fragment_encoder
    from treestitch.graph import build_observation_graph
    from treestitch.partition import (
        evaluate_partition, merge_metrics,
        partition_observations_cc, partition_observations_tiled,
        train_edge_partition_multi_region,
        fragment_completeness, completeness_metrics,
    )
    from treestitch.realworld import build_region_world, build_region_world_dual

    # ── Training bboxes (same as multi_region_train.py) ────────────────────
    y0, y1 = 930_000, 1_000_000   # dense y-extent
    z0, z1 = 780_000, 880_000
    buf = args.seam_buffer
    _ALL_TRAIN = [
        ("A", ((750_000,  y0, z0), (950_000,             y1, z1))),
        ("B", ((950_000,  y0, z0), (1_150_000 - buf,     y1, z1))),  # seam-buffered
        ("C", ((1_350_000 + buf, y0, z0), (1_550_000,    y1, z1))),  # seam-buffered
        ("D", ((1_550_000 + buf, y0, z0), (1_750_000,    y1, z1))),  # seam-buffered
        ("E", ((750_000, 1_000_000, z0), (950_000, 1_070_000, z1))), # y-north band
    ]
    if args.train_regions:
        _sel = set(r.strip().upper() for r in args.train_regions.split(","))
        _ALL_TRAIN = [(n, b) for n, b in _ALL_TRAIN if n in _sel]
    train_labels = [n for n, _ in _ALL_TRAIN]
    train_bboxes = [b for _, b in _ALL_TRAIN]

    # ── Test bboxes ─────────────────────────────────────────────────────────
    # P1 is the proofread-dense region found by scanning nucleus edit-rate
    # (~100% of somas here have v117 != v1718). Coordinates are in TRUE nm —
    # the (4,4,40) nm frame used by fetch_region_synapses and the L2 cache's
    # rep_coord_nm. It sits south of the training y-band (930-1000k nm) so there
    # is no train/test leakage. Used by both substrates: the synapse pipeline
    # (build_region_world) and the L2 pipeline (build_region_world_l2).
    _ALL_IN_COL = [
        ("T1", "T1 x=1150-1350k (reference)",
         ((1_150_000, y0, z0), (1_350_000, y1, z1))),
        ("T2", "T2 x=550-750k (west of A)",
         ((550_000, y0, z0), (750_000, y1, z1))),
        ("T3", "T3 y-shift south (y=870-940k)",
         ((1_150_000, 870_000, z0), (1_350_000, 940_000, z1))),
        ("T4", "T4 y-shift north (y=1000-1070k)",
         ((1_150_000, 1_000_000, z0), (1_350_000, 1_070_000, z1))),
        ("P1", "P1 proofread-dense region (true-nm x=818-918k, y=685-785k, z=794-994k)",
         ((818_500, 685_000, 794_000), (918_500, 785_000, 994_000))),
    ]
    if args.eval_regions:
        _esel = set(r.strip().upper() for r in args.eval_regions.split(","))
        _ALL_IN_COL = [(k, n, b) for k, n, b in _ALL_IN_COL if k in _esel]
    IN_COLUMN = [(n, b) for _, n, b in _ALL_IN_COL]
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
    print(f"  Train: {len(train_bboxes)} regions ({','.join(train_labels)}, seam buffer={buf//1000}µm)")
    print(f"  In-column test locations: {len(IN_COLUMN)}")
    print(f"  OOC test locations:       {len(OUT_OF_COLUMN)}")
    print("=" * 68)

    # ── Nucleus positions for soma detection ────────────────────────────────
    nucleus_pos_nm: np.ndarray | None = None
    if not args.no_soma:
        try:
            from neuronauts.data.loaders import load_nucleus_positions
            print("\nLoading nucleus positions …")
            ndf = load_nucleus_positions(cache_path="/tmp/nucleus_positions.csv.gz")
            if {"x_nm", "y_nm", "z_nm"}.issubset(ndf.columns) and len(ndf) > 0:
                nucleus_pos_nm = ndf[["x_nm", "y_nm", "z_nm"]].to_numpy(dtype=float)
                print(f"  {len(nucleus_pos_nm):,} nucleus positions loaded.")
            else:
                print("  Nucleus CSV lacks xyz columns — soma detection disabled.")
        except Exception as exc:
            print(f"  Nucleus load failed ({exc}); soma detection disabled.")

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
        for i, (label, bbox) in enumerate(zip(train_labels, train_bboxes)):
            print(f"\n[TRAIN {label}] Building world …")
            frags, region, lmap = build_region_world(
                bbox, version=args.version, side=args.side,
                max_synapses=args.max_synapses,
                min_syn_per_fragment=args.min_syn_per_fragment,
                seed=args.seed, verbose=True,
                l2_skeletons=not args.no_train_l2,
                tile_x_nm=args.tile_x_nm, per_tile_limit=args.per_tile_limit)
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
            label = train_labels[i]
            frags_enc = encode_fragments(encoder, frags, device=args.device)
            g = build_observation_graph(region, frags_enc, side=args.side,
                                        k_spatial=args.k_spatial)
            train_graphs.append(g)
            if args.dual_side:
                print(f"  [post-side train {label}] Building world …")
                frags_p, region_p, _ = build_region_world(
                    train_bboxes[i], version=args.version,
                    max_synapses=args.max_synapses,
                    min_syn_per_fragment=args.min_syn_per_fragment,
                    seed=args.seed, verbose=False,
                    side="post", l2_skeletons=False,
                    tile_x_nm=args.tile_x_nm, per_tile_limit=args.per_tile_limit)
                fe_p = encode_fragments(encoder, frags_p, device=args.device)
                g_p = build_observation_graph(region_p, fe_p, side="post",
                                              k_spatial=args.k_spatial)
                if args.balanced_dual and g_p.n_nodes > g.n_nodes:
                    _rng = np.random.default_rng(args.seed + i)
                    g_p = _subsample_observation_graph(g_p, g.n_nodes, _rng)
                    print(f"    balanced: post subsampled to {g_p.n_nodes} nodes "
                          f"(matched pre-side)")
                train_graphs.append(g_p)
        if args.dual_side:
            print(f"\n  Training on {len(train_graphs)} graphs "
                  f"({len(train_bboxes)} pre + {len(train_bboxes)} post).")

        if not (args.checkpoint and Path(args.checkpoint).exists()):
            print(f"\nTraining EdgePartitionGNN ({args.partition_epochs} epochs) …")
            model, _ = train_edge_partition_multi_region(
                train_graphs,
                n_epochs=args.partition_epochs, lr=1e-3,
                franken_hard_frac=args.franken_hard_frac,
                max_train_nodes=args.train_max_nodes,
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

        # ── Optional: train a dedicated post-side model ────────────────────
        if args.post_only_train and args.dual_side and \
                not (args.dual_post_model and Path(args.dual_post_model).exists()):
            print(f"\nTraining dedicated post-side model ({args.partition_epochs} epochs) …")
            post_graphs = []
            for i, bbox in enumerate(train_bboxes):
                label = train_labels[i]
                print(f"  [post-only train {label}] Building world …")
                frags_p, region_p, _ = build_region_world(
                    bbox, version=args.version,
                    max_synapses=args.max_synapses,
                    min_syn_per_fragment=args.min_syn_per_fragment,
                    seed=args.seed, verbose=False,
                    side="post", l2_skeletons=False,
                    tile_x_nm=args.tile_x_nm, per_tile_limit=args.per_tile_limit)
                fe_p = encode_fragments(encoder, frags_p, device=args.device)
                g_p = build_observation_graph(region_p, fe_p, side="post",
                                              k_spatial=args.k_spatial)
                post_graphs.append(g_p)
                print(f"    {g_p.n_nodes} nodes, {g_p.n_edges} edges")
            post_model, _ = train_edge_partition_multi_region(
                post_graphs,
                n_epochs=args.partition_epochs, lr=1e-3,
                franken_hard_frac=args.franken_hard_frac,
                max_train_nodes=args.train_max_nodes,
                device=args.device, seed=args.seed, log_every=25)
            _pef = post_graphs[0]
            _pefd = int(_pef.edge_feat.shape[1]) if _pef.edge_feat.ndim == 2 else 0
            _pall_et = np.concatenate([g.edge_type for g in post_graphs])
            _pn_et = int(max(2, int(_pall_et.max()) + 1)) if len(_pall_et) > 0 else 2
            post_gnn_kwargs = dict(
                input_dim=post_graphs[0].node_feat.shape[1],
                d_model=64, n_edge_types=_pn_et, output_dim=32,
                dropout=0.1, edge_feat_dim=_pefd,
            )
            post_ckpt = args.dual_post_model or "/tmp/neuronauts_post_model.pt"
            save_checkpoint(post_ckpt, encoder, post_model,
                            encoder_kwargs=enc_kwargs, gnn_kwargs=post_gnn_kwargs,
                            extra={"side": "post", "cc_bias": args.cc_bias})
            print(f"  Post-side model saved → {post_ckpt}")
        elif args.dual_post_model and Path(args.dual_post_model).exists():
            print(f"\nLoading post-side model from {args.dual_post_model} …")
            _, post_model = load_checkpoint(args.dual_post_model)
        else:
            post_model = None

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

    # Release training data (fragment tensors, observation graphs) before eval.
    # Python's allocator may not return memory to the OS immediately, but
    # explicit collection prevents runaway growth across the eval regions.
    gc.collect()

    # ── Evaluate in-column bboxes ────────────────────────────────────────
    print(f"\n{'='*68}")
    print("IN-COLUMN EVALUATION  (v1718 GT available)")
    print(f"{'='*68}")

    # Separate fetch cap for evaluation so training stays fast while eval is honest.
    eval_max_syn = args.eval_max_synapses if args.eval_max_synapses is not None else args.max_synapses

    # ── Bundle accumulator (populated only when --save-bundle is set) ────────
    bbox_bundles: dict = {}

    in_col_results = []
    for name, bbox in IN_COLUMN:
        print(f"\n[{name}]")
        try:
            frags, region, lmap = build_region_world(
                bbox, version=args.version, side=args.side,
                max_synapses=eval_max_syn,
                min_syn_per_fragment=args.min_syn_per_fragment,
                seed=args.seed, verbose=True,
                l2_skeletons=not args.no_eval_l2,
                tile_x_nm=args.tile_x_nm, per_tile_limit=args.per_tile_limit)

            n_fk = sum(1 for v in lmap.values() if len(v) > 1)
            if n_fk == 0:
                print(f"  ⚠ 0 frankenmerges — this bbox may be outside the proofread column")

            frags_enc = encode_fragments(encoder, frags, device=args.device)
            graph = build_observation_graph(region, frags_enc, side=args.side,
                                            k_spatial=args.k_spatial)
            pre_tile = args.pre_tile_size
            if pre_tile and graph.n_nodes > pre_tile:
                print(f"  tiled pre-side partition "
                      f"({graph.n_nodes} nodes → tiles of {pre_tile})")
                pred = partition_observations_tiled(
                    model, graph, tile_size=pre_tile,
                    bias=args.cc_bias, device=args.device)
            else:
                pred = partition_observations_cc(model, graph, bias=args.cc_bias,
                                                 device=args.device)
            ev = evaluate_partition(pred, graph.labels)
            mm = merge_metrics(graph, pred)

            # Completeness: which v117 roots need no further merging?
            # Predicted complete = all its synapses fall in a singleton cluster
            # (model kept it isolated, no merges triggered).
            _fids = graph.fragment_id
            _frag_clusters: dict[int, set] = {}
            _cluster_frags: dict[int, set] = {}
            for _f, _c in zip(_fids.tolist(), pred.tolist()):
                if _c >= 0:
                    _frag_clusters.setdefault(int(_f), set()).add(int(_c))
                    _cluster_frags.setdefault(int(_c), set()).add(int(_f))
            pred_completeness = {
                f: (len(cs) == 1 and len(_cluster_frags[next(iter(cs))]) == 1)
                for f, cs in _frag_clusters.items()
            }
            cm = completeness_metrics(lmap, pred_completeness)

            # Synapse count distribution across predicted fragments (pre-side).
            _, _pre_counts = np.unique(pred, return_counts=True)
            syn_pre_min = int(_pre_counts.min())
            syn_pre_max = int(_pre_counts.max())
            syn_pre_med = float(np.median(_pre_counts))
            n_output_cands = int(len(_pre_counts))

            if not args.no_eval_l2:
                shapes = assemble_partition_shapes(frags, pred, graph.fragment_id,
                                                   stitch_radius_nm=5_000.0)
                mlist = [neuron_shape_metrics(s) for s in shapes.values()]
                cable_med    = float(np.median([m['cable_length_um'] for m in mlist])) if mlist else 0.0
                max_path_med = float(np.median([m['max_path_length_um'] for m in mlist])) if mlist else 0.0
                tort_med     = float(np.nanmedian([m['tortuosity'] for m in mlist])) if mlist else float("nan")
                is_tree      = float(np.mean([m['is_tree'] for m in mlist])) if mlist else 0.0
                # Soma detection
                n_with_soma = 0
                if nucleus_pos_nm is not None and shapes:
                    for s in shapes.values():
                        has_s, _ = detect_soma(s, nucleus_pos_nm)
                        if has_s:
                            n_with_soma += 1
                soma_frac = n_with_soma / len(shapes) if shapes else float("nan")
            else:
                shapes, mlist = {}, []
                cable_med = max_path_med = is_tree = 0.0
                tort_med = soma_frac = float("nan")
                n_with_soma = 0

            # ── Per-bbox bundle data (for --save-bundle) ──────────────────
            if args.save_bundle is not None:
                cluster_to_root = _match_clusters_to_neurons(pred, region.pre_root_id)
                neurons_bundle: dict = {}
                for cluster_id, frag in (shapes.items() if shapes else {}.items()):
                    nm = neuron_shape_metrics(frag)
                    has_s = False
                    if nucleus_pos_nm is not None:
                        has_s, _ = detect_soma(frag, nucleus_pos_nm)
                    edges_list = frag.edges.tolist() if frag.edges.ndim == 2 else []
                    neurons_bundle[str(cluster_id)] = {
                        "vertices_nm": frag.vertices_nm.tolist(),
                        "edges": edges_list,
                        "radius_nm": frag.radius_nm.tolist(),
                        "true_root_id": int(cluster_to_root.get(cluster_id, 0)),
                        "n_synapses": int((pred == cluster_id).sum()),
                        "metrics": {
                            "cable_length_um": nm["cable_length_um"],
                            "n_branch_points": nm["n_branch_points"],
                            "n_endpoints": nm["n_endpoints"],
                            "is_tree": nm["is_tree"],
                            "tortuosity": nm["tortuosity"],
                            "max_path_length_um": nm["max_path_length_um"],
                            "mean_caliber_um": nm.get("mean_caliber_um", float("nan")),
                        },
                        "has_soma": bool(has_s),
                    }
                bbox_bundles[name] = {
                    "bbox": [list(bbox[0]), list(bbox[1])],
                    "synapses": {
                        "positions_nm": region.pre_pt_nm.tolist(),
                        "pre_root_id": region.pre_root_id.tolist(),
                        "post_root_id": region.post_root_id.tolist()
                                        if region.post_root_id is not None
                                        else [],
                        "pred_cluster": pred.tolist(),
                        "synapse_id": region.synapse_id.tolist(),
                    },
                    "neurons": neurons_bundle,
                }

            conn = connectome_accuracy(pred, region)
            has_conn = region.post_root_id is not None and int((region.post_root_id > 0).sum()) > 0

            print(f"  {_fmt_in(ev, mm)}")
            print(f"  syn/frag(pre): min={syn_pre_min}  max={syn_pre_max}  med={syn_pre_med:.0f}  "
                  f"out_cands={n_output_cands}  "
                  f"n_merges={mm['n_merges_pred']}  n_splits={mm['n_splits_pred']}  "
                  f"n_true_merges={mm['n_true_merges']}")
            # Confusion-matrix breakdown: TP/FP/FN/TN over all evaluated edges
            print(f"  edge decisions: "
                  f"TP={mm.get('tp_merges',0)}  FP={mm.get('fp_merges',0)}  "
                  f"FN={mm.get('fn_merges',0)}  TN={mm.get('tn_splits',0)}  "
                  f"(n_eval={mm['n_edges_eval']})")
            print(f"  cable_med={cable_med:.0f}µm  max_path={max_path_med:.0f}µm  "
                  f"tort={tort_med:.2f}  is_tree={is_tree:.3f}  n_neurons={len(shapes)}  "
                  f"soma={soma_frac:.1%}" if nucleus_pos_nm is not None
                  else f"  cable_med={cable_med:.0f}µm  max_path={max_path_med:.0f}µm  "
                       f"tort={tort_med:.2f}  is_tree={is_tree:.3f}  n_neurons={len(shapes)}")
            n_complete_gt = cm["n_complete_gt"]
            n_frags_total = cm["n_fragments"]
            print(f"  completeness: P={cm['precision']:.3f}  R={cm['recall']:.3f}  "
                  f"F1={cm['f1']:.3f}  acc={cm['accuracy']:.3f}  "
                  f"(GT complete: {n_complete_gt}/{n_frags_total} "
                  f"= {n_complete_gt/n_frags_total:.0%} of fragments)")

            if has_conn:
                print(f"  conn_edge_F1(dir)={conn['conn_edge_f1']:.3f}  "
                      f"P={conn['conn_edge_precision']:.3f}  R={conn['conn_edge_recall']:.3f}  "
                      f"conn_edge_F1(undir)={conn['conn_edge_f1_undir']:.3f}  "
                      f"syn_attr_acc={conn['synapse_attr_acc']:.3f}  "
                      f"({conn['n_true_edges']} dir / {conn['n_true_edges_undir']} undir true edges)")

            # Line-graph metric suite (single-side pre-partition)
            lg: LineGraphSuite | None = None
            if has_conn:
                lg = evaluate_suite(pred, region.pre_root_id, region.post_root_id)
                print(f"  lg_pre:  F1={lg.pre_only.f1:.3f}  "
                      f"P={lg.pre_only.precision:.3f}  R={lg.pre_only.recall:.3f}  "
                      f"(penalises axonal over-fragmentation)")
                print(f"  lg_or:   F1={lg.or_metric.f1:.3f}  "
                      f"P={lg.or_metric.precision:.3f}  R={lg.or_metric.recall:.3f}  "
                      f"(OR truth; insensitive to over-fragmentation)")

            row = {
                "name": name, "ari": ev["ari"],
                "merge_p": mm["merge_precision"],
                "merge_r": mm["merge_recall"],
                "over": mm["over_merge_rate"],
                "under": mm["under_merge_rate"],
                "fk": mm["frankenmerge_split_recall"],
                "complete_p": cm["precision"],
                "complete_r": cm["recall"],
                "complete_f1": cm["f1"],
                "complete_acc": cm["accuracy"],
                "n_complete_gt": n_complete_gt,
                "complete_frac_gt": n_complete_gt / n_frags_total if n_frags_total else float("nan"),
                "syn_pre_min": syn_pre_min,
                "syn_pre_max": syn_pre_max,
                "syn_pre_med": syn_pre_med,
                "n_output_cands": n_output_cands,
                "n_merges_pred": mm["n_merges_pred"],
                "n_splits_pred": mm["n_splits_pred"],
                "n_true_merges": mm["n_true_merges"],
                "cable_med": cable_med,
                "max_path_med": max_path_med,
                "tort_med": tort_med,
                "is_tree": is_tree,
                "n_franken": n_fk,
                "conn_f1": conn["conn_edge_f1"],
                "conn_p": conn["conn_edge_precision"],
                "conn_r": conn["conn_edge_recall"],
                "conn_f1_undir": conn["conn_edge_f1_undir"],
                "conn_p_undir": conn["conn_edge_precision_undir"],
                "conn_r_undir": conn["conn_edge_recall_undir"],
                "n_true_edges": conn["n_true_edges"],
                "n_pred_edges": conn["n_pred_edges"],
                "syn_attr_acc": conn["synapse_attr_acc"],
                "soma_frac": soma_frac,
                "n_with_soma": n_with_soma,
                "lg_pre_f1": lg.pre_only.f1  if lg else float("nan"),
                "lg_pre_p":  lg.pre_only.precision if lg else float("nan"),
                "lg_pre_r":  lg.pre_only.recall    if lg else float("nan"),
                "lg_or_f1":  lg.or_metric.f1       if lg else float("nan"),
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
                            max_synapses=eval_max_syn,
                            min_syn_per_fragment=args.min_syn_per_fragment,
                            seed=args.seed, verbose=True,
                            l2_skeletons_pre=not args.no_eval_l2,
                            l2_skeletons_post=args.dual_post_l2 and not args.no_eval_l2,
                            tile_x_nm=args.tile_x_nm, per_tile_limit=args.per_tile_limit)
                    # Partition pre side.
                    fe_pre2 = encode_fragments(encoder, frags_pre2, device=args.device)
                    g_pre2 = build_observation_graph(region_pre2, fe_pre2,
                                                     side="pre", k_spatial=args.k_spatial)
                    if pre_tile and g_pre2.n_nodes > pre_tile:
                        pred_pre2 = partition_observations_tiled(
                            model, g_pre2, tile_size=pre_tile,
                            bias=args.cc_bias, device=args.device)
                    else:
                        pred_pre2 = partition_observations_cc(
                            model, g_pre2, bias=args.cc_bias, device=args.device)
                    # Partition post side.
                    fe_post = encode_fragments(encoder, frags_post, device=args.device)
                    g_post = build_observation_graph(region_post, fe_post,
                                                     side="post", k_spatial=args.k_spatial)
                    if use_frag_post:
                        # Fragment-level partition: each v117 dendritic fragment is
                        # its own cluster.  No model call; avoids the scale-mismatch
                        # problem until a post-specific model is available.
                        pred_post = g_post.fragment_id.copy()
                        print(f"  [post-side] fragment partition "
                              f"({len(np.unique(pred_post))} fragments as clusters)")
                    elif post_model is not None:
                        if g_post.n_nodes > args.post_tile_size:
                            print(f"  [post-side] tiled partition via post model "
                                  f"({g_post.n_nodes} nodes → tiles of {args.post_tile_size})")
                            pred_post = partition_observations_tiled(
                                post_model, g_post, tile_size=args.post_tile_size,
                                bias=post_cc_bias, device=args.device)
                        else:
                            pred_post = partition_observations_cc(
                                post_model, g_post, bias=post_cc_bias, device=args.device)
                    else:
                        if g_post.n_nodes > args.post_tile_size:
                            print(f"  [post-side] tiled partition "
                                  f"({g_post.n_nodes} nodes → tiles of {args.post_tile_size})")
                            pred_post = partition_observations_tiled(
                                model, g_post, tile_size=args.post_tile_size,
                                bias=post_cc_bias, device=args.device)
                        else:
                            pred_post = partition_observations_cc(
                                model, g_post, bias=post_cc_bias, device=args.device)
                    # Synapse count per predicted fragment (post-side).
                    _, _post_counts = np.unique(pred_post, return_counts=True)
                    syn_post_min = int(_post_counts.min())
                    syn_post_max = int(_post_counts.max())
                    syn_post_med = float(np.median(_post_counts))
                    print(f"  syn/frag(post): min={syn_post_min}  max={syn_post_max}  med={syn_post_med:.0f}")
                    row["syn_post_min"] = syn_post_min
                    row["syn_post_max"] = syn_post_max
                    row["syn_post_med"] = syn_post_med

                    dual = dual_side_connectome_accuracy(
                        pred_pre2, region_pre2, pred_post, region_post)
                    print(f"  [dual-side] conn_F1(dir)={dual['conn_edge_f1']:.3f}  "
                          f"P={dual['conn_edge_precision']:.3f}  R={dual['conn_edge_recall']:.3f}  "
                          f"conn_F1(undir)={dual['conn_edge_f1_undir']:.3f}  "
                          f"both-sides={dual['n_synapses_both_sides']} syn  "
                          f"(pre-only={dual['n_synapses_pre_only']}, "
                          f"post-only={dual['n_synapses_post_only']})")
                    row["dual_f1"] = dual["conn_edge_f1"]
                    row["dual_p"]  = dual["conn_edge_precision"]
                    row["dual_r"]  = dual["conn_edge_recall"]
                    row["dual_f1_undir"] = dual["conn_edge_f1_undir"]
                    row["dual_both"] = dual["n_synapses_both_sides"]

                    # ── Line-graph suite with dual partition ─────────────────
                    # Align pred_pre2 and pred_post by shared synapse_id.
                    synid_pre  = region_pre2.synapse_id
                    synid_post = region_post.synapse_id
                    shared_ids = np.intersect1d(synid_pre, synid_post)
                    if len(shared_ids) > 1:
                        pre_idx  = {int(s): i for i, s in enumerate(synid_pre)}
                        post_idx = {int(s): i for i, s in enumerate(synid_post)}
                        bp  = np.array([pre_idx[int(s)]  for s in shared_ids])
                        bpo = np.array([post_idx[int(s)] for s in shared_ids])
                        lg_dual = evaluate_suite(
                            pred_pre2[bp],
                            region_pre2.pre_root_id[bp],
                            region_pre2.post_root_id[bp],
                            pred_post=pred_post[bpo],
                        )
                        # Post-side quality (truth = same post-neuron, est = pred_post).
                        # Pass post_root_id as first arg so pre_only gives post-side F1.
                        lg_post_side = evaluate_suite(
                            pred_post,
                            region_post.post_root_id,
                            region_post.pre_root_id,
                        )
                        print(f"  [lg-and]  F1={lg_dual.and_metric.f1:.3f}  "
                              f"P={lg_dual.and_metric.precision:.3f}  "
                              f"R={lg_dual.and_metric.recall:.3f}  "
                              f"(circuit-edge; penalises both sides)")
                        print(f"  [lg-post] F1={lg_post_side.pre_only.f1:.3f}  "
                              f"P={lg_post_side.pre_only.precision:.3f}  "
                              f"R={lg_post_side.pre_only.recall:.3f}  "
                              f"(post-side partition quality)")
                        row["lg_and_f1"] = lg_dual.and_metric.f1
                        row["lg_and_p"]  = lg_dual.and_metric.precision
                        row["lg_and_r"]  = lg_dual.and_metric.recall
                        row["lg_post_f1"] = lg_post_side.pre_only.f1
                        row["lg_post_r"]  = lg_post_side.pre_only.recall
                except Exception as exc:
                    import traceback
                    print(f"  [dual-side] ERROR: {exc}")
                    traceback.print_exc()

            in_col_results.append(row)
        except Exception as exc:
            print(f"  ERROR: {exc}")
            in_col_results.append({"name": name, "error": str(exc)})
        gc.collect()

    # ── Evaluate OOC bboxes ───────────────────────────────────────────────
    print(f"\n{'='*68}")
    print("OUT-OF-COLUMN EVALUATION  (shape plausibility — no GT)")
    print(f"{'='*68}")

    def _eval_ooc_tile(tile_bbox, tile_label):
        """Evaluate one OOC sub-bbox. Returns metrics dict with shape stats."""
        frags_t, region_t, lmap_t = build_region_world(
            tile_bbox, version=args.version, side=args.side,
            max_synapses=eval_max_syn,
            min_syn_per_fragment=args.min_syn_per_fragment,
            seed=args.seed, verbose=True,
            l2_skeletons=not args.no_eval_l2,
            tile_x_nm=args.tile_x_nm, per_tile_limit=args.per_tile_limit)

        frags_enc_t = encode_fragments(encoder, frags_t, device=args.device)
        graph_t = build_observation_graph(region_t, frags_enc_t, side=args.side,
                                          k_spatial=args.k_spatial)

        pre_tile = args.pre_tile_size
        if pre_tile and graph_t.n_nodes > pre_tile:
            print(f"  [{tile_label}] tiled partition "
                  f"({graph_t.n_nodes} nodes → tiles of {pre_tile})")
            pred_t = partition_observations_tiled(
                model, graph_t, tile_size=pre_tile,
                bias=args.cc_bias, device=args.device)
        else:
            pred_t = partition_observations_cc(
                model, graph_t, bias=args.cc_bias, device=args.device)

        mm_t = merge_metrics(graph_t, pred_t, ignore_label=0)
        shapes_t = assemble_partition_shapes(
            frags_t, pred_t, graph_t.fragment_id, stitch_radius_nm=5_000.0)
        mlist_t = [neuron_shape_metrics(s) for s in shapes_t.values()]

        n_fk_t = sum(1 for v in lmap_t.values() if len(v) > 1)
        return {
            "n_nodes": graph_t.n_nodes,
            "n_franken": n_fk_t,
            "over": mm_t["over_merge_rate"],
            "cable_med": float(np.median([m['cable_length_um'] for m in mlist_t])) if mlist_t else 0.0,
            "max_path_med": float(np.median([m['max_path_length_um'] for m in mlist_t])) if mlist_t else 0.0,
            "tort_med": float(np.nanmedian([m['tortuosity'] for m in mlist_t])) if mlist_t else float("nan"),
            "is_tree": float(np.mean([m['is_tree'] for m in mlist_t])) if mlist_t else 0.0,
            "fully_conn": float(np.mean([m['n_connected_components'] == 1 for m in mlist_t])) if mlist_t else 0.0,
            "n_neurons": len(shapes_t),
        }

    ooc_results = []
    for name, bbox in OUT_OF_COLUMN:
        print(f"\n[{name}]")
        try:
            # Tile the OOC bbox along x to keep each tile at ~3-5k fragments.
            # OOC regions span novel brain areas with mostly uncached L2
            # skeletons; at 15k+ fragments the L2 fetch + observation graph
            # construction can exhaust the container's 15 GB memory limit.
            # We split into args.ooc_x_tiles sub-bboxes, evaluate each
            # independently, and report weighted-average shape metrics.
            n_ooc_tiles = max(1, args.ooc_x_tiles)
            sub_bboxes = _tile_bbox_x(bbox, n_ooc_tiles)
            tile_results = []
            n_fk_total = 0

            for ti, sub_bbox in enumerate(sub_bboxes):
                tile_label = f"x-tile {ti+1}/{n_ooc_tiles}"
                print(f"  [{tile_label}] bbox x={sub_bbox[0][0]/1e3:.0f}-{sub_bbox[1][0]/1e3:.0f}k …")
                tr = _eval_ooc_tile(sub_bbox, tile_label)
                tile_results.append(tr)
                n_fk_total += tr["n_franken"]
                gc.collect()

            # Weighted average by n_nodes for rates; median over tiles for
            # shape metrics (median of medians is a conservative estimator).
            total_nodes = sum(t["n_nodes"] for t in tile_results) or 1
            over      = sum(t["over"] * t["n_nodes"] for t in tile_results) / total_nodes
            cable_med = float(np.median([t["cable_med"] for t in tile_results]))
            max_path_med = float(np.median([t["max_path_med"] for t in tile_results]))
            tort_med  = float(np.nanmedian([t["tort_med"] for t in tile_results]))
            is_tree   = sum(t["is_tree"] * t["n_nodes"] for t in tile_results) / total_nodes
            fully_conn = sum(t["fully_conn"] * t["n_nodes"] for t in tile_results) / total_nodes
            n_neurons_total = sum(t["n_neurons"] for t in tile_results)

            print(f"  {n_ooc_tiles} x-tiles aggregated: {n_neurons_total} neurons, {n_fk_total} frankenmerges")
            print(f"  {_fmt_ooc(over, cable_med, is_tree)}  "
                  f"max_path={max_path_med:.0f}µm  tort={tort_med:.2f}  fully_conn={fully_conn:.1%}")
            cable_ok = 500 <= cable_med <= 20_000
            print(f"  cable plausible: {'✓' if cable_ok else '⚠'}")

            ooc_results.append({
                "name": name, "over": over, "cable_med": cable_med,
                "max_path_med": max_path_med, "tort_med": tort_med,
                "is_tree": is_tree, "fully_conn": fully_conn,
                "n_neurons": n_neurons_total, "n_franken": n_fk_total,
                "n_ooc_tiles": n_ooc_tiles,
            })
        except Exception as exc:
            print(f"  ERROR: {exc}")
            ooc_results.append({"name": name, "error": str(exc)})
        gc.collect()

    # ── Save bundle ───────────────────────────────────────────────────────
    if args.save_bundle is not None:
        bundle = {
            "meta": {
                "timestamp": datetime.datetime.now().isoformat(),
                "train_regions": ["A", "B", "C", "D", "E"],
                "version": args.version,
                "checkpoint": args.checkpoint or args.save_checkpoint,
                "dual_side": args.dual_side,
            },
            "calibration": {"T": T, "ece_train": ece_train},
            "in_col_results": in_col_results,
            "ooc_results": ooc_results,
            "bboxes": bbox_bundles,
        }
        with open(args.save_bundle, "w") as _bf:
            json.dump(bundle, _bf, indent=2, default=_json_default)
        print(f"\nBundle saved → {args.save_bundle}")

    # ── Summary table ─────────────────────────────────────────────────────
    print(f"\n{'='*68}")
    print("SUMMARY: spatial variance")
    print(f"{'='*68}")

    good_in = [r for r in in_col_results if "error" not in r]
    good_ooc = [r for r in ooc_results if "error" not in r]

    if not args.no_calibration:
        print(f"\n  Calibration: T={T:.4f}  ECE(train)={ece_train:.4f}")

    print(f"\n  In-column ({len(good_in)} locations):")
    print(f"  {'Location':<38} {'ARI':>6} {'merge_P':>8} {'merge_R':>8} {'under':>6} "
          f"{'conn_F1':>8} {'conn_P':>7} {'conn_R':>7} {'cable_med':>10} {'tort':>6}")
    print(f"  {'-'*38} {'-'*6} {'-'*8} {'-'*8} {'-'*6} {'-'*8} {'-'*7} {'-'*7} {'-'*10} {'-'*6}")
    for r in good_in:
        cf1 = f"{r['conn_f1']:.3f}" if r['conn_f1'] == r['conn_f1'] else "n/a"
        cp  = f"{r.get('conn_p', float('nan')):.3f}" if r.get('conn_p', float('nan')) == r.get('conn_p', float('nan')) else "n/a"
        cr  = f"{r.get('conn_r', float('nan')):.3f}" if r.get('conn_r', float('nan')) == r.get('conn_r', float('nan')) else "n/a"
        und = f"{r.get('under', float('nan')):.3f}" if r.get('under', float('nan')) == r.get('under', float('nan')) else "n/a"
        tort = f"{r['tort_med']:.2f}" if r['tort_med'] == r['tort_med'] else "n/a"
        print(f"  {r['name']:<38} {r['ari']:>6.3f} {r['merge_p']:>8.3f} "
              f"{r['merge_r']:>8.3f} {und:>6} {cf1:>8} {cp:>7} {cr:>7} "
              f"{r['cable_med']:>9.0f}µ {tort:>6}")

    # Scale metrics: merge/split decisions and output candidates.
    scale_rows = [r for r in good_in if "n_merges_pred" in r]
    if scale_rows:
        print(f"\n  Scale metrics (edge decisions + candidates):")
        print(f"  {'Location':<38} {'n_merges':>9} {'n_splits':>9} "
              f"{'n_true_mg':>10} {'out_cands':>10} {'syn/frag_med':>13}")
        print(f"  {'-'*38} {'-'*9} {'-'*9} {'-'*10} {'-'*10} {'-'*13}")
        for r in scale_rows:
            print(f"  {r['name']:<38} {r['n_merges_pred']:>9d} {r['n_splits_pred']:>9d} "
                  f"{r['n_true_merges']:>10d} {r['n_output_cands']:>10d} "
                  f"{r['syn_pre_med']:>13.0f}")

    if len(good_in) >= 2:
        aris   = [r["ari"] for r in good_in]
        mps    = [r["merge_p"] for r in good_in]
        cf1s   = [r["conn_f1"] for r in good_in if r["conn_f1"] == r["conn_f1"]]
        print(f"\n  In-column variance:")
        print(f"    ARI:       mean={np.mean(aris):.3f}  std={np.std(aris):.3f}  "
              f"range=[{min(aris):.3f}, {max(aris):.3f}]")
        print(f"    merge_P:   mean={np.mean(mps):.3f}   std={np.std(mps):.3f}  "
              f"range=[{min(mps):.3f}, {max(mps):.3f}]")
        cmplt_f1s = [r["complete_f1"] for r in good_in
                     if r.get("complete_f1") == r.get("complete_f1")]
        if cmplt_f1s:
            print(f"    cmplt_F1:  mean={np.mean(cmplt_f1s):.3f}  "
                  f"std={np.std(cmplt_f1s):.3f}  "
                  f"range=[{min(cmplt_f1s):.3f}, {max(cmplt_f1s):.3f}]")
        if cf1s:
            print(f"    conn_F1:   mean={np.mean(cf1s):.3f}  std={np.std(cf1s):.3f}  "
                  f"range=[{min(cf1s):.3f}, {max(cf1s):.3f}]")
        cf1u = [r["conn_f1_undir"] for r in good_in
                if r.get("conn_f1_undir") == r.get("conn_f1_undir")]
        if cf1u:
            print(f"    conn_F1u:  mean={np.mean(cf1u):.3f}  std={np.std(cf1u):.3f}  "
                  f"range=[{min(cf1u):.3f}, {max(cf1u):.3f}]")
        lg_pre_f1s = [r["lg_pre_f1"] for r in good_in
                      if r.get("lg_pre_f1") == r.get("lg_pre_f1")]
        if lg_pre_f1s:
            print(f"    lg_pre_F1: mean={np.mean(lg_pre_f1s):.3f}  "
                  f"std={np.std(lg_pre_f1s):.3f}  "
                  f"range=[{min(lg_pre_f1s):.3f}, {max(lg_pre_f1s):.3f}]")

    # ── Completeness summary ─────────────────────────────────────────────────
    cmp_rows = [r for r in good_in if "complete_f1" in r]
    if cmp_rows:
        print(f"\n  Completeness (predict which v117 roots need no edit):")
        print(f"  {'Location':<38} {'F1':>6} {'P':>6} {'R':>6} {'acc':>6} "
              f"{'%GT_cmplt':>10}")
        for r in cmp_rows:
            cf1  = f"{r['complete_f1']:.3f}"  if r['complete_f1']  == r['complete_f1']  else "n/a"
            cp   = f"{r['complete_p']:.3f}"   if r['complete_p']   == r['complete_p']   else "n/a"
            cr   = f"{r['complete_r']:.3f}"   if r['complete_r']   == r['complete_r']   else "n/a"
            cacc = f"{r['complete_acc']:.3f}" if r['complete_acc'] == r['complete_acc'] else "n/a"
            pct  = f"{r.get('complete_frac_gt', float('nan')):.0%}" \
                   if r.get('complete_frac_gt', float('nan')) == r.get('complete_frac_gt', float('nan')) \
                   else "n/a"
            print(f"  {r['name']:<38} {cf1:>6} {cp:>6} {cr:>6} {cacc:>6} {pct:>10}")

    # ── Soma summary ────────────────────────────────────────────────────────
    soma_rows = [r for r in good_in if r.get("soma_frac") == r.get("soma_frac")]
    if soma_rows and nucleus_pos_nm is not None:
        print(f"\n  Soma detection (nucleus bbox overlap, margin=10µm):")
        for r in soma_rows:
            sf = r["soma_frac"]
            ns = r["n_with_soma"]
            print(f"    {r['name']:<38} {sf:.1%}  ({ns} with soma)")

    # ── Line-graph metric suite summary ─────────────────────────────────────
    lg_rows = [r for r in good_in if "lg_pre_f1" in r]
    if lg_rows:
        print(f"\n  Line-graph F1 suite (single-side pre-partition):")
        print(f"  {'Location':<38} {'pre_F1':>7} {'pre_P':>7} {'pre_R':>7} {'or_F1':>7}")
        print(f"  {'-'*38} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")
        for r in lg_rows:
            print(f"  {r['name']:<38} {r['lg_pre_f1']:>7.3f} "
                  f"{r['lg_pre_p']:>7.3f} {r['lg_pre_r']:>7.3f} "
                  f"{r['lg_or_f1']:>7.3f}")

    if args.dual_side:
        dual_rows = [r for r in good_in if "dual_f1" in r]
        if dual_rows:
            print(f"\n  Dual-side connectome (both partitions, NO GT root ids):")
            print(f"  {'Location':<38} {'dir_F1':>7} {'dir_P':>6} {'dir_R':>6} {'undir_F1':>9} {'both_syn':>9}")
            print(f"  {'-'*38} {'-'*7} {'-'*6} {'-'*6} {'-'*9} {'-'*9}")
            for r in dual_rows:
                dp = f"{r.get('dual_p', float('nan')):.3f}"
                dr = f"{r.get('dual_r', float('nan')):.3f}"
                print(f"  {r['name']:<38} {r['dual_f1']:>7.3f} "
                      f"{dp:>6} {dr:>6} "
                      f"{r['dual_f1_undir']:>9.3f} {r['dual_both']:>9d}")
        and_rows = [r for r in good_in if "lg_and_f1" in r]
        if and_rows:
            print(f"\n  Line-graph AND metric (circuit-edge; penalises both sides):")
            print(f"  {'Location':<38} {'and_F1':>7} {'and_P':>7} {'and_R':>7} "
                  f"{'post_F1':>8} {'post_R':>7}")
            print(f"  {'-'*38} {'-'*7} {'-'*7} {'-'*7} {'-'*8} {'-'*7}")
            for r in and_rows:
                pf = f"{r.get('lg_post_f1', float('nan')):.3f}"
                pr = f"{r.get('lg_post_r',  float('nan')):.3f}"
                print(f"  {r['name']:<38} {r['lg_and_f1']:>7.3f} "
                      f"{r['lg_and_p']:>7.3f} {r['lg_and_r']:>7.3f} "
                      f"{pf:>8} {pr:>7}")

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
