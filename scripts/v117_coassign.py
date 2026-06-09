#!/usr/bin/env python3
"""Synapse co-assignment on real v117 MICrONS data.

This script fetches actual v117 synapses and skeletons from the CAVE database,
then runs the full co-assignment pipeline:

  1. Fetch synapses + skeletons from CAVE at materialization v117
  2. Map v117 seg_ids → v1412 for ground-truth labels
  3. Encode skeleton DNA with SkeletonGNN
  4. Build the SynapseGraph (same-seg edges + spatial k-NN)
  5. Train SynapseCoassigner (GNN encoder + edge scorer)
  6. Evaluate with K materializations: pairwise P/R/F1 and coverage@K

Usage
-----
    # Single region around a well-known neuron (default)
    python scripts/v117_coassign.py --token YOUR_CAVE_TOKEN

    # Specify a custom bounding box center (nm) and side length (µm)
    python scripts/v117_coassign.py \\
        --center-nm 661000 340000 620000 \\
        --side-um 15 \\
        --token YOUR_CAVE_TOKEN \\
        --cache-dir /tmp/v117_cache

    # Quick smoke test (fewer epochs)
    python scripts/v117_coassign.py --token YOUR_CAVE_TOKEN --epochs 10 --k 3

CAVE token
----------
Create a free account at https://cave.minnie65.brain-map.org and generate a
token. The public minnie65_public datastack works without authentication but
rate-limits unauthenticated connections heavily. Set the CAVE_TOKEN environment
variable or pass --token.
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("v117_coassign")

# ---------------------------------------------------------------------------
# Defaults — a region in mouse V1 that contains well-proofread neurons
# (L2/3 pyramidal cell soma layer, ~660,000 nm X, ~340,000 nm Y, ~620,000 nm Z)
# ---------------------------------------------------------------------------
DEFAULT_CENTER_NM = (661_000, 340_000, 620_000)
DEFAULT_SIDE_UM = 20.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Co-assignment pipeline on real v117 MICrONS data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--token",
        default=os.environ.get("CAVE_TOKEN"),
        help="CAVE auth token (or set CAVE_TOKEN env var)",
    )
    p.add_argument(
        "--center-nm",
        nargs=3,
        type=int,
        default=list(DEFAULT_CENTER_NM),
        metavar=("X", "Y", "Z"),
        help="Bounding box center in global nm",
    )
    p.add_argument(
        "--side-um",
        type=float,
        default=DEFAULT_SIDE_UM,
        help="Bounding box side length in micrometers",
    )
    p.add_argument(
        "--min-seg-synapses",
        type=int,
        default=2,
        help="Min synapses per v117 segment to include",
    )
    p.add_argument(
        "--max-segs",
        type=int,
        default=500,
        help="Max number of v117 segments to fetch skeletons for",
    )
    p.add_argument(
        "--cache-dir",
        default=None,
        help="Directory for skeleton cache (speeds up re-runs)",
    )
    p.add_argument(
        "--dna-dim",
        type=int,
        default=64,
        help="Dimension of DNA embeddings",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=60,
        help="Training epochs for SynapseCoassigner",
    )
    p.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of materializations",
    )
    p.add_argument(
        "--device",
        default="cpu",
        help="Torch device (cpu or cuda)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------
    # Imports (deferred so --help works without torch)
    # ------------------------------------------------------------------
    try:
        import torch
    except ImportError:
        sys.exit("torch is required: pip install torch --index-url https://download.pytorch.org/whl/cpu")

    from neuronauts.fetch import make_cube_bbox_nm
    from neuronauts.data.cave import encode_seg_dna, fetch_v117_region
    from neuronauts.coassign import (
        SynapseCoassigner,
        build_synapse_graph,
        coverage_at_k,
        materializations,
        pairwise_precision_recall,
        train,
    )

    rng = np.random.default_rng(args.seed)
    t0 = time.time()

    # ------------------------------------------------------------------
    # Step 1 — Fetch synapses + skeletons at v117
    # ------------------------------------------------------------------
    center_nm = tuple(args.center_nm)
    bbox_nm = make_cube_bbox_nm(center_nm, side_um=args.side_um)
    log.info("Bounding box: %s → %s (%.0f µm side)", bbox_nm[0], bbox_nm[1], args.side_um)

    region = fetch_v117_region(
        bbox_nm,
        token=args.token,
        min_seg_synapses=args.min_seg_synapses,
        max_segs=args.max_segs,
        skeleton_cache_dir=args.cache_dir,
    )

    if region.n_synapses == 0:
        sys.exit("No synapses found in the requested region. "
                 "Try a larger --side-um or a different --center-nm.")

    log.info(
        "Fetched %d synapses across %d v117 segments (%.1f s)",
        region.n_synapses, region.n_segments, time.time() - t0,
    )

    n_labeled = int((region.gt_labels > 0).sum())
    n_neurons = len(np.unique(region.gt_labels[region.gt_labels > 0]))
    log.info(
        "%d / %d synapses have v1412 ground-truth labels (%d distinct neurons)",
        n_labeled, region.n_synapses, n_neurons,
    )

    if n_labeled < 10:
        log.warning(
            "Very few labeled synapses (%d). The v117→v1412 mapping may have "
            "failed, or the region contains few proofread neurons. "
            "Evaluation metrics will not be meaningful.",
            n_labeled,
        )

    # ------------------------------------------------------------------
    # Step 2 — Encode skeleton DNA with SkeletonGNN
    # ------------------------------------------------------------------
    t1 = time.time()
    log.info("Encoding DNA for %d unique segments (dna_dim=%d) ...",
             region.n_segments, args.dna_dim)

    seg_dna = encode_seg_dna(
        region.skeletons,
        region.seg_ids,
        dna_dim=args.dna_dim,
        device=args.device,
    )
    log.info("DNA encoded in %.1f s", time.time() - t1)

    # ------------------------------------------------------------------
    # Step 3 — Build the SynapseGraph
    # ------------------------------------------------------------------
    graph = build_synapse_graph(
        region.positions_nm,
        region.seg_ids,
        region.gt_labels,
        seg_dna,
    )
    log.info(
        "SynapseGraph: %d nodes, %d edges (node_dim=%d)",
        graph.n_nodes, graph.n_edges, graph.node_dim,
    )
    same_seg_frac = float(graph.same_seg.mean())
    log.info("  %.1f%% same-segment edges, %.1f%% spatial k-NN edges",
             100 * same_seg_frac, 100 * (1 - same_seg_frac))

    # ------------------------------------------------------------------
    # Step 4 — Train SynapseCoassigner
    # ------------------------------------------------------------------
    t2 = time.time()
    model = SynapseCoassigner(node_dim=graph.node_dim)
    n_params = sum(p.numel() for p in model.parameters())
    log.info("Training SynapseCoassigner (%d parameters, %d epochs) ...",
             n_params, args.epochs)

    history = train(model, [graph], n_epochs=args.epochs, device=args.device, seed=args.seed)

    final = {k: v[-1] for k, v in history.items() if v}
    log.info(
        "Training done in %.1f s — loss=%.4f  P=%.3f  R=%.3f",
        time.time() - t2,
        final.get("loss", float("nan")),
        final.get("precision", float("nan")),
        final.get("recall", float("nan")),
    )

    # ------------------------------------------------------------------
    # Step 5 — Generate K materializations and evaluate
    # ------------------------------------------------------------------
    node_feat = torch.from_numpy(
        np.concatenate([graph.node_pos, graph.node_dna], axis=1)
    ).float()
    edge_src_t = torch.from_numpy(graph.edge_src).long()
    edge_dst_t = torch.from_numpy(graph.edge_dst).long()
    same_seg_t = torch.from_numpy(graph.same_seg).float()

    with torch.no_grad():
        probs = model.edge_probs(node_feat, edge_src_t, edge_dst_t, same_seg_t).numpy()

    mats = materializations(
        graph.n_nodes, graph.edge_src, graph.edge_dst, probs,
        K=args.k, seed=args.seed,
    )

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    total_time = time.time() - t0
    print("\n" + "=" * 60)
    print("v117 co-assignment results")
    print("=" * 60)
    print(f"  Region:      {args.side_um:.0f} µm box around {center_nm}")
    print(f"  Synapses:    {region.n_synapses} ({n_labeled} labeled)")
    print(f"  Neurons:     {n_neurons} distinct v1412 root IDs")
    print(f"  Segments:    {region.n_segments} v117 segments")
    print(f"  Edges:       {graph.n_edges}")
    print(f"  Wall time:   {total_time:.1f} s")
    print()
    print("Top-K materializations (pairwise metrics on labeled synapses):")
    for i, (labels, score) in enumerate(mats):
        r = pairwise_precision_recall(labels, graph.labels)
        print(
            f"  [{i+1}] score={score:+.1f}  "
            f"P={r['precision']:.3f}  R={r['recall']:.3f}  F1={r['f1']:.3f}"
        )

    covered = coverage_at_k(mats, graph.labels)
    print()
    print(f"  coverage@{args.k}: {covered}")

    if not covered:
        # Report the best recall achieved
        best_recall = max(
            pairwise_precision_recall(labels, graph.labels)["recall"]
            for labels, _ in mats
        )
        print(f"  (best recall in top-{args.k}: {best_recall:.3f} — "
              "threshold calibration or more training epochs may help)")

    print("=" * 60)


if __name__ == "__main__":
    main()
