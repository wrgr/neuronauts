#!/usr/bin/env python3
"""Compare partition inference methods: union-find vs correlation clustering.

Both methods consume the *same* observation graph (same fragment embeddings,
same typed edges) so the comparison isolates the inference algorithm:

  - union-find    : metric GNN (contrastive) → cosine threshold → union-find
                    (the existing partition_observations)
  - edge_cc       : edge classifier learns f(fragment→object) per edge →
                    correlation clustering (GAEC)  (partition_observations_cc)

The edge_cc path is the recommended reformulation: it directly supervises the
v117→v1412 mapping at the edge level and uses a global combinatorial clustering
that can cut a high-similarity edge when the rest of the graph disagrees —
fixing the irreversible-merge failure mode of threshold union-find.

Reports ARI, cluster counts, and the over/under-merge asymmetry for each.

Usage
-----
  # Offline, synthetic world (no network needed)
  python attic/prior_results/compare_partition_methods.py --synthetic --n-objects 20 --n-pieces 3

  # Real Minnie65 neurons (requires CAVE access)
  python attic/prior_results/compare_partition_methods.py --n-objects 20 --n-pieces 3 --cell-type 23P
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _fmt_merge(m: dict) -> str:
    return (
        f"merge_P={m['merge_precision']:.3f} merge_R={m['merge_recall']:.3f} "
        f"over={m['over_merge_rate']:.3f} under={m['under_merge_rate']:.3f}"
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--synthetic", action="store_true",
                   help="Use the offline synthetic world (no CAVE access)")
    p.add_argument("--n-objects", type=int, default=20)
    p.add_argument("--n-pieces", type=int, default=3)
    p.add_argument("--obs-per-piece", type=int, default=12)
    p.add_argument("--frankenmerge-frac", type=float, default=0.0,
                   help="(synthetic only) fraction of pieces fused across objects "
                        "into shared v117 segments — injects merge errors to correct")
    p.add_argument("--cell-type", default=None)
    p.add_argument("--max-verts", type=int, default=8000)
    p.add_argument("--endpoint-radius-nm", type=float, default=10_000.0)
    p.add_argument("--k-spatial", type=int, default=8)
    p.add_argument("--embed-epochs", type=int, default=40)
    p.add_argument("--partition-epochs", type=int, default=60)
    p.add_argument("--threshold", type=float, default=0.87,
                   help="cosine threshold for the union-find baseline")
    p.add_argument("--cc-bias", type=float, default=0.0,
                   help="log-odds bias for correlation clustering (<0 = conservative)")
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    import numpy as np

    from treestitch.embed import FragmentEncoder, encode_fragments, train_fragment_encoder
    from treestitch.graph import build_observation_graph
    from treestitch.partition import (
        PartitionGNN,
        evaluate_partition,
        merge_metrics,
        partition_observations,
        partition_observations_cc,
        train_edge_partition,
        train_partition,
    )

    # --- Load world -------------------------------------------------------
    if args.synthetic:
        from treestitch.synthetic import make_synthetic_world
        print(f"Synthetic world: {args.n_objects} objects × {args.n_pieces} pieces")
        fragments, region, label_map = make_synthetic_world(
            n_objects=args.n_objects, n_pieces=args.n_pieces,
            observations_per_piece=args.obs_per_piece,
            frankenmerge_frac=args.frankenmerge_frac, seed=args.seed, verbose=True,
        )
    else:
        from treestitch.data import load_minnie65_world
        fragments, region, label_map = load_minnie65_world(
            n_objects=args.n_objects, n_pieces=args.n_pieces,
            observations_per_piece=args.obs_per_piece, cell_type=args.cell_type,
            max_verts=args.max_verts, seed=args.seed,
        )

    # --- Shared front end: encode fragments + build one graph -------------
    print(f"\nTraining FragmentEncoder ({args.embed_epochs} epochs) …")
    encoder = FragmentEncoder(node_input_dim=4, d_model=64, output_dim=32)
    if args.embed_epochs > 0:
        train_fragment_encoder(
            encoder, [fragments], n_epochs=args.embed_epochs, lr=1e-3,
            margin=1.0, device=args.device, root_label_map=label_map, log_every=20,
        )
    frags_enc = encode_fragments(encoder, fragments, device=args.device)

    graph = build_observation_graph(
        region, frags_enc, side="pre", k_spatial=args.k_spatial,
        endpoint_radius_nm=args.endpoint_radius_nm,
    )
    n_same = int((graph.edge_type == 0).sum())
    n_sp = int((graph.edge_type == 1).sum())
    n_ep = int((graph.edge_type == 2).sum())
    n_true = int(len(np.unique(graph.labels[graph.labels != 0])))
    print(f"\nObservationGraph: {graph.n_nodes} nodes | {graph.n_edges} edges "
          f"({n_same} same-frag, {n_sp} spatial, {n_ep} endpoint-adj)")
    print(f"  true objects = {n_true}")

    # --- Method A: metric GNN + threshold union-find ----------------------
    print(f"\n{'='*64}\n[A] union-find  (metric GNN + cosine threshold {args.threshold})\n{'='*64}")
    gnn, _ = train_partition(
        graph, n_epochs=args.partition_epochs, lr=1e-3, margin=0.5,
        max_pairs=800, device=args.device, seed=args.seed, log_every=20,
    )
    pred_uf = partition_observations(gnn, graph, threshold=args.threshold, device=args.device)
    r_uf = evaluate_partition(pred_uf, graph.labels)
    m_uf = merge_metrics(graph, pred_uf)
    print(f"  ARI={r_uf['ari']:.4f}  clusters={r_uf['n_clusters_pred']}/{r_uf['n_clusters_true']}"
          f"  H={r_uf['homogeneity']:.3f} C={r_uf['completeness']:.3f}")
    print(f"  {_fmt_merge(m_uf)}")

    # --- Method B: edge classifier + correlation clustering ---------------
    print(f"\n{'='*64}\n[B] edge_cc  (learn f(117→1412) per edge + correlation clustering)\n{'='*64}")
    model, _ = train_edge_partition(
        graph, n_epochs=args.partition_epochs, lr=1e-3,
        device=args.device, seed=args.seed, log_every=20,
    )
    pred_cc = partition_observations_cc(model, graph, bias=args.cc_bias, device=args.device)
    r_cc = evaluate_partition(pred_cc, graph.labels)
    m_cc = merge_metrics(graph, pred_cc)
    print(f"  ARI={r_cc['ari']:.4f}  clusters={r_cc['n_clusters_pred']}/{r_cc['n_clusters_true']}"
          f"  H={r_cc['homogeneity']:.3f} C={r_cc['completeness']:.3f}")
    print(f"  {_fmt_merge(m_cc)}")

    # --- Summary ----------------------------------------------------------
    print(f"\n{'='*64}\nSUMMARY  ({n_true} true objects)\n{'='*64}")
    print(f"  {'method':<12} {'ARI':>7} {'clusters':>10} {'merge_P':>9} {'merge_R':>9} {'over':>7}")
    print(f"  {'union-find':<12} {r_uf['ari']:>7.4f} "
          f"{str(r_uf['n_clusters_pred'])+'/'+str(r_uf['n_clusters_true']):>10} "
          f"{m_uf['merge_precision']:>9.3f} {m_uf['merge_recall']:>9.3f} {m_uf['over_merge_rate']:>7.3f}")
    print(f"  {'edge_cc':<12} {r_cc['ari']:>7.4f} "
          f"{str(r_cc['n_clusters_pred'])+'/'+str(r_cc['n_clusters_true']):>10} "
          f"{m_cc['merge_precision']:>9.3f} {m_cc['merge_recall']:>9.3f} {m_cc['over_merge_rate']:>7.3f}")
    print(f"  ΔARI (edge_cc − union-find) = {r_cc['ari'] - r_uf['ari']:+.4f}")
    print(f"{'='*64}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
