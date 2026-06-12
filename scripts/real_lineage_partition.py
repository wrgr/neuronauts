#!/usr/bin/env python3
"""Real f(v117 → v1718) partition benchmark — no synthetic data.

Builds a fully real world from CAVE lineage (real synapses → real v117 fragments
→ real v1718 labels; see treestitch/realworld.py), then compares the two
partition inference methods on the SAME graph:

  - union-find : metric GNN + cosine threshold
  - edge_cc    : learn f(v117→v1718) per edge + correlation clustering (GAEC)

This is the honest test the synthetic benchmarks could not provide: real
"trunk + slivers" split structure and real frankenmerges, with real synapse
positions.  Reports ARI plus the over/under-merge asymmetry.

If version 1718 is unreachable for any query, fall back with --version to an
earlier AVAILABLE materialization (1621, 1507, 1300).

Usage
-----
  python scripts/real_lineage_partition.py --n-objects 15 --version 1718 \
      --max-syn-per-obj 300 --embed-epochs 40 --partition-epochs 80
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
            f"over={m['over_merge_rate']:.3f} under={m['under_merge_rate']:.3f}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-objects", type=int, default=15)
    p.add_argument("--version", type=int, default=1718,
                   help="proofread target materialization (1718/1621/1507/1300)")
    p.add_argument("--side", default="post", choices=["pre", "post"])
    p.add_argument("--max-syn-per-obj", type=int, default=300)
    p.add_argument("--min-syn-per-obj", type=int, default=20)
    p.add_argument("--endpoint-radius-nm", type=float, default=10_000.0)
    p.add_argument("--k-spatial", type=int, default=8)
    p.add_argument("--embed-epochs", type=int, default=40)
    p.add_argument("--partition-epochs", type=int, default=80)
    p.add_argument("--threshold", type=float, default=0.90)
    p.add_argument("--cc-bias", type=float, default=0.0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    from treestitch.embed import FragmentEncoder, encode_fragments, train_fragment_encoder
    from treestitch.graph import build_observation_graph
    from treestitch.partition import (
        evaluate_partition, merge_metrics, partition_observations,
        partition_observations_cc, train_edge_partition, train_partition,
    )
    from treestitch.realworld import build_lineage_world

    fragments, region, label_map = build_lineage_world(
        n_objects=args.n_objects, version=args.version, side=args.side,
        max_syn_per_obj=args.max_syn_per_obj, min_syn_per_obj=args.min_syn_per_obj,
        seed=args.seed, verbose=True)

    print(f"\nTraining FragmentEncoder ({args.embed_epochs} epochs) …")
    encoder = FragmentEncoder(node_input_dim=4, d_model=64, output_dim=32)
    if args.embed_epochs > 0:
        train_fragment_encoder(encoder, [fragments], n_epochs=args.embed_epochs, lr=1e-3,
                               margin=1.0, device=args.device, root_label_map=label_map,
                               log_every=20)
    frags_enc = encode_fragments(encoder, fragments, device=args.device)
    graph = build_observation_graph(region, frags_enc, side="pre",
                                    k_spatial=args.k_spatial,
                                    endpoint_radius_nm=args.endpoint_radius_nm)
    n_same = int((graph.edge_type == 0).sum())
    n_sp = int((graph.edge_type == 1).sum())
    n_ep = int((graph.edge_type == 2).sum())
    n_true = int(len(np.unique(graph.labels[graph.labels != 0])))
    n_frag = len(fragments)
    print(f"\nObservationGraph: {graph.n_nodes} nodes | {graph.n_edges} edges "
          f"({n_same} same-frag, {n_sp} spatial, {n_ep} endpoint-adj)")
    print(f"  true neurons={n_true}  v117 fragments={n_frag}  "
          f"(fragments/neuron={n_frag/max(n_true,1):.1f})")

    # A: union-find
    print(f"\n{'='*64}\n[A] union-find  (metric GNN + cosine threshold {args.threshold})\n{'='*64}")
    gnn, _ = train_partition(graph, n_epochs=args.partition_epochs, lr=1e-3, margin=0.5,
                             max_pairs=800, device=args.device, seed=args.seed, log_every=20)
    pred_uf = partition_observations(gnn, graph, threshold=args.threshold, device=args.device)
    r_uf = evaluate_partition(pred_uf, graph.labels)
    m_uf = merge_metrics(graph, pred_uf)
    print(f"  ARI={r_uf['ari']:.4f}  clusters={r_uf['n_clusters_pred']}/{r_uf['n_clusters_true']}  {_fmt_merge(m_uf)}")

    # B: edge_cc
    print(f"\n{'='*64}\n[B] edge_cc  (learn f(117→{args.version}) per edge + correlation clustering)\n{'='*64}")
    model, _ = train_edge_partition(graph, n_epochs=args.partition_epochs, lr=1e-3,
                                    device=args.device, seed=args.seed, log_every=20)
    pred_cc = partition_observations_cc(model, graph, bias=args.cc_bias, device=args.device)
    r_cc = evaluate_partition(pred_cc, graph.labels)
    m_cc = merge_metrics(graph, pred_cc)
    print(f"  ARI={r_cc['ari']:.4f}  clusters={r_cc['n_clusters_pred']}/{r_cc['n_clusters_true']}  {_fmt_merge(m_cc)}")

    print(f"\n{'='*64}\nSUMMARY  (real v117→v{args.version}, {n_true} neurons, {n_frag} fragments)\n{'='*64}")
    print(f"  {'method':<12} {'ARI':>7} {'clusters':>10} {'merge_P':>9} {'over':>7}")
    print(f"  {'union-find':<12} {r_uf['ari']:>7.4f} "
          f"{str(r_uf['n_clusters_pred'])+'/'+str(r_uf['n_clusters_true']):>10} "
          f"{m_uf['merge_precision']:>9.3f} {m_uf['over_merge_rate']:>7.3f}")
    print(f"  {'edge_cc':<12} {r_cc['ari']:>7.4f} "
          f"{str(r_cc['n_clusters_pred'])+'/'+str(r_cc['n_clusters_true']):>10} "
          f"{m_cc['merge_precision']:>9.3f} {m_cc['over_merge_rate']:>7.3f}")
    print(f"  ΔARI (edge_cc − union-find) = {r_cc['ari'] - r_uf['ari']:+.4f}")
    print(f"{'='*64}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
