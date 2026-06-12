#!/usr/bin/env python3
"""Real f(v117 → v1718) partition benchmark using spatial region sampling.

Builds a fully real world from a spatial bounding box (all neurons in the
region, not isolated neuron seeds) then compares the two partition methods:

  - union-find : metric GNN + cosine threshold
  - edge_cc    : learn f(v117→v1718) per edge + correlation clustering (GAEC)

Why region-based matters:
  - Neurons are spatially interleaved, so the k-NN synapse graph naturally
    contains cross-neuron edges — the training signal edge_cc needs.
  - Real frankenmerge v117 roots appear automatically; they contribute
    same-fragment edges with target=0 (cut-signals) that teach the model to
    detect and split merge errors.
  - The proofreading delta IS the training signal: merge errors = spatially
    proximate different neurons, split errors = same neuron multiple roots,
    frankenmerges = detectable from within-root synapse heterogeneity.

Compare against the neuron-seeded baseline in real_lineage_partition.py.

Usage
-----
  python scripts/real_region_partition.py \\
      --bbox-nm 1150000,930000,780000,1250000,980000,880000 \\
      --version 1718 --embed-epochs 20 --partition-epochs 40
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _parse_bbox(s: str) -> tuple:
    """Parse 'x0,y0,z0,x1,y1,z1' into ((x0,y0,z0),(x1,y1,z1))."""
    vals = [float(v) for v in s.split(",")]
    if len(vals) != 6:
        raise argparse.ArgumentTypeError(
            "bbox-nm must be 6 comma-separated floats: x0,y0,z0,x1,y1,z1")
    return (vals[0], vals[1], vals[2]), (vals[3], vals[4], vals[5])


def _fmt_merge(m: dict) -> str:
    return (f"merge_P={m['merge_precision']:.3f} merge_R={m['merge_recall']:.3f} "
            f"over={m['over_merge_rate']:.3f} under={m['under_merge_rate']:.3f}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bbox-nm", type=str,
                   default="1150000,930000,780000,1250000,980000,880000",
                   help="x0,y0,z0,x1,y1,z1 in nm")
    p.add_argument("--version", type=int, default=1718,
                   help="proofread target materialization (1718/1621/1507/1300)")
    p.add_argument("--side", default="pre", choices=["pre", "post"])
    p.add_argument("--max-synapses", type=int, default=20_000)
    p.add_argument("--min-syn-per-fragment", type=int, default=5)
    p.add_argument("--endpoint-radius-nm", type=float, default=0.0,
                   help="radius for endpoint-adj edges in nm; 0 disables them (default). "
                        "In dense regions, spatial k-NN already supplies cross-neuron signal "
                        "and endpoint edges cause edge-count OOM with hundreds of fragments.")
    p.add_argument("--max-endpoint-pairs", type=int, default=10,
                   help="cap on endpoint-adj edges per fragment pair")
    p.add_argument("--k-spatial", type=int, default=8)
    p.add_argument("--embed-epochs", type=int, default=20)
    p.add_argument("--partition-epochs", type=int, default=40)
    p.add_argument("--threshold", type=float, default=0.90)
    p.add_argument("--cc-bias", type=float, default=0.0)
    p.add_argument("--abstain-threshold", type=float, default=0.0,
                   help="uncertainty abstention: observations with "
                        "max_same_cluster_p - max_diff_cluster_p < threshold "
                        "are left unassigned instead of forced into a cluster. "
                        "0 = no abstention (default). Try 0.2-0.4 to surface "
                        "frankenmerge boundary synapses.")
    p.add_argument("--franken-hard-frac", type=float, default=0.1,
                   help="fraction of training negatives drawn from the frankenmerge "
                        "cut pool (type-0 edges crossing a neuron boundary). "
                        "Explicit oversampling of the rarest but most informative "
                        "training signal for frankenmerge detection.")
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-l2-skeletons", action="store_true",
                   help="skip L2 cache skeleton fetch; use synapse cloud only")
    args = p.parse_args()

    bbox_nm = _parse_bbox(args.bbox_nm)

    from treestitch.embed import FragmentEncoder, encode_fragments, train_fragment_encoder
    from treestitch.graph import build_observation_graph
    from treestitch.partition import (
        evaluate_partition, merge_metrics, partition_observations,
        partition_observations_cc, train_edge_partition, train_partition,
    )
    from treestitch.realworld import build_region_world

    fragments, region, label_map = build_region_world(
        bbox_nm, version=args.version, side=args.side,
        max_synapses=args.max_synapses,
        min_syn_per_fragment=args.min_syn_per_fragment,
        seed=args.seed, verbose=True,
        l2_skeletons=not args.no_l2_skeletons)

    print(f"\nTraining FragmentEncoder ({args.embed_epochs} epochs) …")
    encoder = FragmentEncoder(node_input_dim=4, d_model=64, output_dim=32)
    if args.embed_epochs > 0:
        train_fragment_encoder(encoder, [fragments], n_epochs=args.embed_epochs, lr=1e-3,
                               margin=1.0, device=args.device, root_label_map=label_map,
                               log_every=20)
    frags_enc = encode_fragments(encoder, fragments, device=args.device)
    ep_radius = args.endpoint_radius_nm if args.endpoint_radius_nm > 0 else None
    graph = build_observation_graph(region, frags_enc, side="pre",
                                    k_spatial=args.k_spatial,
                                    endpoint_radius_nm=ep_radius,
                                    max_endpoint_pairs=args.max_endpoint_pairs)

    n_same = int((graph.edge_type == 0).sum())
    n_sp = int((graph.edge_type == 1).sum())
    n_ep = int((graph.edge_type == 2).sum())
    n_true = int(len(np.unique(graph.labels[graph.labels != 0])))
    n_frag = len(fragments)
    n_franken = sum(1 for v in label_map.values() if len(v) > 1)

    # Cross-neuron edge fraction: fraction of edges connecting different v1718 neurons
    all_labels = graph.labels
    src_lab = all_labels[graph.edge_src]
    dst_lab = all_labels[graph.edge_dst]
    valid_edges = (src_lab != 0) & (dst_lab != 0)
    cross_neuron_frac = (float(((src_lab != dst_lab) & valid_edges).sum())
                         / max(int(valid_edges.sum()), 1))

    print(f"\nObservationGraph: {graph.n_nodes} nodes | {graph.n_edges} edges "
          f"({n_same} same-frag, {n_sp} spatial, {n_ep} endpoint-adj)")
    print(f"  true neurons={n_true}  v117 fragments={n_frag}  "
          f"frankenmerge fragments={n_franken}  (fragments/neuron={n_frag/max(n_true,1):.1f})")
    print(f"  cross-neuron edge fraction={cross_neuron_frac:.3f}  "
          f"(0.0 = no training signal, >0 = edge_cc can learn)")

    # A: union-find
    print(f"\n{'='*64}")
    print(f"[A] union-find  (metric GNN + cosine threshold {args.threshold})")
    print(f"{'='*64}")
    gnn, _ = train_partition(graph, n_epochs=args.partition_epochs, lr=1e-3, margin=0.5,
                             max_pairs=800, device=args.device, seed=args.seed, log_every=20)
    pred_uf = partition_observations(gnn, graph, threshold=args.threshold, device=args.device)
    r_uf = evaluate_partition(pred_uf, graph.labels)
    m_uf = merge_metrics(graph, pred_uf)
    print(f"  ARI={r_uf['ari']:.4f}  clusters={r_uf['n_clusters_pred']}/{r_uf['n_clusters_true']}  {_fmt_merge(m_uf)}")

    # B: edge_cc
    print(f"\n{'='*64}")
    print(f"[B] edge_cc  (learn f(117→{args.version}) per edge + correlation clustering)")
    print(f"{'='*64}")
    model, _ = train_edge_partition(graph, n_epochs=args.partition_epochs, lr=1e-3,
                                    franken_hard_frac=args.franken_hard_frac,
                                    device=args.device, seed=args.seed, log_every=20)
    pred_cc = partition_observations_cc(model, graph, bias=args.cc_bias,
                                        abstain_threshold=args.abstain_threshold,
                                        device=args.device)
    r_cc = evaluate_partition(pred_cc, graph.labels)
    m_cc = merge_metrics(graph, pred_cc)
    print(f"  ARI={r_cc['ari']:.4f}  clusters={r_cc['n_clusters_pred']}/{r_cc['n_clusters_true']}  {_fmt_merge(m_cc)}")

    print(f"\n{'='*64}")
    print(f"SUMMARY  (region v117→v{args.version}, {n_true} neurons, {n_frag} fragments)")
    print(f"{'='*64}")
    print(f"  {'method':<12} {'ARI':>7} {'clusters':>10} {'merge_P':>9} {'over':>7} {'fk_split':>9} {'abstain':>8}")
    print(f"  {'union-find':<12} {r_uf['ari']:>7.4f} "
          f"{str(r_uf['n_clusters_pred'])+'/'+str(r_uf['n_clusters_true']):>10} "
          f"{m_uf['merge_precision']:>9.3f} {m_uf['over_merge_rate']:>7.3f} "
          f"{m_uf['frankenmerge_split_recall']:>9.3f} {m_uf.get('abstain_rate', 0.0):>8.3f}")
    print(f"  {'edge_cc':<12} {r_cc['ari']:>7.4f} "
          f"{str(r_cc['n_clusters_pred'])+'/'+str(r_cc['n_clusters_true']):>10} "
          f"{m_cc['merge_precision']:>9.3f} {m_cc['over_merge_rate']:>7.3f} "
          f"{m_cc['frankenmerge_split_recall']:>9.3f} {m_cc.get('abstain_rate', 0.0):>8.3f}")
    print(f"  ΔARI (edge_cc − union-find) = {r_cc['ari'] - r_uf['ari']:+.4f}")
    print(f"  cross-neuron edges = {cross_neuron_frac:.3f}  "
          f"frankenmerge same-frag cut rate = {m_cc['frankenmerge_rate']:.3f}")
    print(f"  Bar1 {'PASS' if r_cc['ari'] >= r_uf['ari'] and m_cc['merge_precision'] >= m_uf['merge_precision'] else 'FAIL'}"
          f"  Bar2 {'PASS' if m_cc['merge_precision'] > 0.95 and m_cc['merge_recall'] > 0.70 else 'FAIL'}"
          f"  Bar3 {'PASS' if m_cc['frankenmerge_split_recall'] > 0.5 else 'FAIL (or N/A if no frankenmerges)'}")
    print(f"{'='*64}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
