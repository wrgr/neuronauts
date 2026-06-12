#!/usr/bin/env python3
"""Real-data validation: union-find vs correlation clustering under realistic
v117 merge errors.

Fetches real proofread v1412 neurons from the MICrONS Minnie65 dataset at their
true positions, splits each skeleton into pieces (simulating v117
over-segmentation), then injects **adjacent-neuron merge errors**: pieces of
*different* neurons whose skeletons physically come within a radius are fused
into one shared v117 segment.  This is the honest version of the synthetic
franken benchmark — the merged neurons genuinely touch, so the endpoint-distance
cue is no longer a giveaway and morphology (fragment DNA) must carry the
discrimination.

Both partition methods consume the SAME fragment embeddings and the SAME typed
observation graph; only the inference algorithm differs:

  - union-find : metric GNN + cosine threshold  (partition_observations)
  - edge_cc    : learn f(v117→v1412) per edge + correlation clustering (GAEC)

Reports ARI, cluster counts, and the over/under-merge asymmetry for each.

Usage
-----
  python scripts/real_franken_partition.py \
      --n-objects 20 --n-pieces 3 --frankenmerge-frac 0.25 \
      --franken-radius-nm 6000 --embed-epochs 40 --partition-epochs 80
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _extract_pieces(args) -> list[dict]:
    """Fetch real neurons and split each into observation-bearing pieces."""
    from neuronauts.data.loaders import load_skeleton, sample_neurons
    from treestitch.data import _split_skeleton_n_pieces

    rng = np.random.default_rng(args.seed)
    candidates = sample_neurons(args.n_objects * 6, cell_type=args.cell_type, seed=args.seed)
    print(f"Sampled {len(candidates)} candidate neurons"
          + (f" (cell_type={args.cell_type})" if args.cell_type else "") + " …")

    pieces_rec: list[dict] = []
    obj_counter = 0
    for root_id in candidates:
        if obj_counter >= args.n_objects:
            break
        skel = load_skeleton(root_id)
        if skel is None:
            continue
        verts, edges_raw, radii = skel["vertices_nm"], skel["edges"], skel["radii_nm"]
        if len(verts) < args.min_piece_verts * args.n_pieces or len(verts) > args.max_verts:
            continue
        pieces = _split_skeleton_n_pieces(verts, edges_raw, radii, args.n_pieces,
                                          min_verts=args.min_piece_verts)
        if len(pieces) < 2:
            continue
        obj_counter += 1
        for pv, pe, pr in pieces:
            anchor = rng.integers(0, len(pv), args.obs_per_piece)
            obs_pts = (pv[anchor] +
                       rng.normal(0, args.synapse_noise_nm, (args.obs_per_piece, 3)).astype(np.float32))
            pieces_rec.append({
                "obj_id": obj_counter, "verts": pv,
                "edges": pe if len(pe) else np.zeros((0, 2), dtype=np.int64),
                "radii": pr, "obs_pts": obs_pts,
            })
        print(f"  [{obj_counter:3d}] root={root_id}  V={len(verts)}  pieces={len(pieces)}")
        time.sleep(0.05)

    if obj_counter < 2:
        raise RuntimeError("Too few neurons fetched — check network/token")
    return pieces_rec


def _fmt_merge(m: dict) -> str:
    return (f"merge_P={m['merge_precision']:.3f} merge_R={m['merge_recall']:.3f} "
            f"over={m['over_merge_rate']:.3f} under={m['under_merge_rate']:.3f}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-objects", type=int, default=20)
    p.add_argument("--n-pieces", type=int, default=3)
    p.add_argument("--obs-per-piece", type=int, default=12)
    p.add_argument("--cell-type", default=None)
    p.add_argument("--frankenmerge-frac", type=float, default=0.25,
                   help="fraction of pieces fused across adjacent neurons")
    p.add_argument("--franken-radius-nm", type=float, default=6000.0,
                   help="two cross-neuron pieces fuse if skeletons come within this")
    p.add_argument("--endpoint-radius-nm", type=float, default=10_000.0)
    p.add_argument("--k-spatial", type=int, default=8)
    p.add_argument("--min-piece-verts", type=int, default=8)
    p.add_argument("--max-verts", type=int, default=8000)
    p.add_argument("--synapse-noise-nm", type=float, default=500.0)
    p.add_argument("--embed-epochs", type=int, default=40)
    p.add_argument("--partition-epochs", type=int, default=80)
    p.add_argument("--threshold", type=float, default=0.87)
    p.add_argument("--cc-bias", type=float, default=0.0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    from treestitch.embed import FragmentEncoder, encode_fragments, train_fragment_encoder
    from treestitch.graph import build_observation_graph
    from treestitch.partition import (
        evaluate_partition, merge_metrics, partition_observations,
        partition_observations_cc, train_edge_partition, train_partition,
    )
    from treestitch.worldbuild import build_world_from_pieces, frankenmerge_adjacent

    # --- Build the real-data world ---------------------------------------
    pieces_rec = _extract_pieces(args)
    rng = np.random.default_rng(args.seed)
    seg_of_piece, n_franken = frankenmerge_adjacent(
        pieces_rec, args.frankenmerge_frac, rng, radius_nm=args.franken_radius_nm)
    print(f"\nInjected {n_franken} adjacent-neuron merges "
          f"(radius {args.franken_radius_nm:.0f} nm) into {len(pieces_rec)} pieces")
    fragments, region, label_map = build_world_from_pieces(
        pieces_rec, seg_of_piece, region_id="minnie65_franken", seed=args.seed)

    # --- Shared front end -------------------------------------------------
    print(f"\nTraining FragmentEncoder ({args.embed_epochs} epochs) …")
    encoder = FragmentEncoder(node_input_dim=4, d_model=64, output_dim=32)
    if args.embed_epochs > 0:
        train_fragment_encoder(encoder, [fragments], n_epochs=args.embed_epochs, lr=1e-3,
                               margin=1.0, device=args.device, root_label_map=label_map, log_every=20)
    frags_enc = encode_fragments(encoder, fragments, device=args.device)
    graph = build_observation_graph(region, frags_enc, side="pre",
                                    k_spatial=args.k_spatial, endpoint_radius_nm=args.endpoint_radius_nm)
    n_same = int((graph.edge_type == 0).sum())
    n_sp = int((graph.edge_type == 1).sum())
    n_ep = int((graph.edge_type == 2).sum())
    n_true = int(len(np.unique(graph.labels[graph.labels != 0])))
    print(f"\nObservationGraph: {graph.n_nodes} nodes | {graph.n_edges} edges "
          f"({n_same} same-frag, {n_sp} spatial, {n_ep} endpoint-adj) | true objects={n_true}")

    # --- A: union-find ----------------------------------------------------
    print(f"\n{'='*64}\n[A] union-find  (metric GNN + cosine threshold {args.threshold})\n{'='*64}")
    gnn, _ = train_partition(graph, n_epochs=args.partition_epochs, lr=1e-3, margin=0.5,
                             max_pairs=800, device=args.device, seed=args.seed, log_every=20)
    pred_uf = partition_observations(gnn, graph, threshold=args.threshold, device=args.device)
    r_uf = evaluate_partition(pred_uf, graph.labels)
    m_uf = merge_metrics(graph, pred_uf)
    print(f"  ARI={r_uf['ari']:.4f}  clusters={r_uf['n_clusters_pred']}/{r_uf['n_clusters_true']}  {_fmt_merge(m_uf)}")

    # --- B: edge_cc -------------------------------------------------------
    print(f"\n{'='*64}\n[B] edge_cc  (learn f(117→1412) per edge + correlation clustering)\n{'='*64}")
    model, _ = train_edge_partition(graph, n_epochs=args.partition_epochs, lr=1e-3,
                                    device=args.device, seed=args.seed, log_every=20)
    pred_cc = partition_observations_cc(model, graph, bias=args.cc_bias, device=args.device)
    r_cc = evaluate_partition(pred_cc, graph.labels)
    m_cc = merge_metrics(graph, pred_cc)
    print(f"  ARI={r_cc['ari']:.4f}  clusters={r_cc['n_clusters_pred']}/{r_cc['n_clusters_true']}  {_fmt_merge(m_cc)}")

    # --- Summary ----------------------------------------------------------
    print(f"\n{'='*64}\nSUMMARY  ({n_true} true objects, {n_franken} adjacent-neuron merges)\n{'='*64}")
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
