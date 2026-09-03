#!/usr/bin/env python3
"""Half-synapse partition ablation: ARI before/after GNN training.

Evaluates how well a HalfSynapseGNN separates half-synapses by neuron identity
after training on a half-synapse graph with typed edges (same-segment + spatial).

Two modes
---------
  --synthetic   Generate a synthetic multi-neuron world (no network needed).
                Each neuron has 2–3 segments; one in three neurons has a
                frankenmerge segment shared with a neighbour.  Synapses are
                placed near segment skeletons.  Ground-truth partition =
                ``pre_root_id`` column.

  --real        (Stretch goal) Fetch real minnie65 v1412 skeletons and v117
                segment IDs from CAVE.  Requires network access and a valid
                CAVE auth token.

Usage
-----
  python scripts/half_synapse_ablation.py --synthetic --n-neurons 10 --epochs 30
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ---------------------------------------------------------------------------
# Synthetic world builder
# ---------------------------------------------------------------------------

def _make_synthetic_world(
    n_neurons: int,
    segs_per_neuron: int,
    synapses_per_seg: int,
    frankenfraction: float,
    dna_dim: int,
    rng: np.random.Generator,
    region_nm: float = 200_000.0,
    vertices_per_seg: int = 20,
) -> tuple:
    """Build a synthetic Region + Fragments for half-synapse ablation.

    Each neuron has `segs_per_neuron` segments.  With probability
    `frankenfraction`, one segment per neuron is a frankenmerge: its
    pre_seg_id is shared with the next neuron, simulating a merge error.
    Synapses are placed near the segment's skeleton vertices so spatial
    proximity and DNA are informative.

    Returns
    -------
    (region, fragments)
        region — Region with pre_seg_id, post_seg_id, pre_root_id, post_root_id
        fragments — list of Fragment with dna filled (synthetic SkeletonGNN DNA)
    """
    from neuronauts.schemas import Fragment, Region

    all_pre_pts: list[np.ndarray] = []
    all_post_pts: list[np.ndarray] = []
    all_pre_root: list[int] = []
    all_post_root: list[int] = []
    all_pre_seg: list[int] = []
    all_post_seg: list[int] = []
    all_syn_ids: list[int] = []
    fragments: list[Fragment] = []
    syn_idx = 0

    seg_id_counter = 1000  # unique seg IDs
    frankensegs: dict[int, int] = {}  # seg_id → neuron_it_belongs_to second

    # Assign a morphological type per neuron (determines DNA)
    neuron_dna_base = rng.normal(0, 1, (n_neurons, dna_dim)).astype(np.float32)
    norms = np.linalg.norm(neuron_dna_base, axis=1, keepdims=True)
    neuron_dna_base /= np.maximum(norms, 1e-8)

    for ni in range(n_neurons):
        neuron_id = ni + 1  # 1-indexed
        neuron_anchor = rng.uniform(0, region_nm, 3).astype(np.float32)

        for si in range(segs_per_neuron):
            seg_id = seg_id_counter; seg_id_counter += 1

            # Is this a frankenmerge? Share with next neuron.
            is_frank = (si == 0) and (rng.random() < frankenfraction) and (ni + 1 < n_neurons)
            frank_partner = (ni + 2) if is_frank else None  # label of second neuron

            # Build a simple branching skeleton
            offset = rng.uniform(-30_000, 30_000, 3).astype(np.float32)
            seg_anchor = neuron_anchor + offset
            verts = [seg_anchor]
            edges_list = []
            radii_list = [300.0]
            for k in range(vertices_per_seg - 1):
                step_dir = rng.normal(0, 1, 3).astype(np.float32)
                step_dir /= np.linalg.norm(step_dir)
                verts.append(verts[-1] + step_dir * 3000.0)
                edges_list.append([k, k + 1])
                radii_list.append(200.0)
            verts_nm = np.array(verts, dtype=np.float32)
            edges_arr = np.array(edges_list, dtype=np.int64)
            radii_nm = np.array(radii_list, dtype=np.float32)
            endpoints_nm = verts_nm[[0, -1]]

            # DNA: similar to neuron type + small noise
            noise = rng.normal(0, 0.1, dna_dim).astype(np.float32)
            dna = neuron_dna_base[ni] + noise
            dna /= np.linalg.norm(dna)

            frag = Fragment(
                fragment_id=seg_id,
                region_id="synthetic",
                base_root_id=seg_id,
                vertices_nm=verts_nm,
                edges=edges_arr,
                endpoints_nm=endpoints_nm,
                radius_nm=radii_nm,
                synapse_indices=np.array([], dtype=np.int64),
                dna=dna,
            ).validate()
            fragments.append(frag)

            # Place synapses near skeleton vertices
            for _ in range(synapses_per_seg):
                anchor_v = verts_nm[rng.integers(len(verts_nm))]
                pre_pt = anchor_v + rng.normal(0, 500, 3).astype(np.float32)
                post_pt = anchor_v + rng.normal(0, 2000, 3).astype(np.float32)
                all_pre_pts.append(pre_pt)
                all_post_pts.append(post_pt)
                all_pre_seg.append(seg_id)
                all_post_seg.append(seg_id)
                all_pre_root.append(neuron_id)
                all_post_root.append(neuron_id)
                if is_frank:
                    # Half of frank-synapses belong to the partner neuron
                    if rng.random() < 0.5:
                        all_pre_root[-1] = frank_partner
                        all_post_root[-1] = frank_partner
                all_syn_ids.append(syn_idx)
                syn_idx += 1

    N = syn_idx
    region = Region(
        region_id="synthetic_half_syn",
        bbox_nm=((0.0, 0.0, 0.0), (region_nm, region_nm, region_nm)),
        voxel_size_nm=(8.0, 8.0, 40.0),
        seg_version=117,
        label_version=1412,
        pre_pt_nm=np.array(all_pre_pts, dtype=np.float32),
        post_pt_nm=np.array(all_post_pts, dtype=np.float32),
        pre_root_id=np.array(all_pre_root, dtype=np.int64),
        post_root_id=np.array(all_post_root, dtype=np.int64),
        synapse_id=np.array(all_syn_ids, dtype=np.int64),
        pre_seg_id=np.array(all_pre_seg, dtype=np.int64),
        post_seg_id=np.array(all_post_seg, dtype=np.int64),
    ).validate()

    print(f"  {n_neurons} neurons × {segs_per_neuron} segs × {synapses_per_seg} syn/seg"
          f" = {N} synapses, {len(fragments)} fragments")
    frank_count = sum(1 for ni in range(n_neurons)
                      if rng.random() < frankenfraction)
    print(f"  DNA dim: {dna_dim}  |  frankenmerge fraction: {frankenfraction:.0%}")
    return region, fragments


# ---------------------------------------------------------------------------
# Run ablation
# ---------------------------------------------------------------------------

def run_half_synapse_ablation(
    region,
    fragments,
    *,
    side: str = "pre",
    n_epochs: int = 30,
    lr: float = 1e-3,
    k_spatial: int = 6,
    max_pairs: int = 500,
    threshold: float = 0.7,
    device: str = "cpu",
    seed: int = 42,
    log_every: int = 5,
) -> dict:
    from neuronauts.assemble.half_synapse_graph import build_half_synapse_graph
    from neuronauts.assemble.partition_gnn import (
        HalfSynapseGNN,
        evaluate_partition_ari,
        partition_half_synapses,
        train_partition_gnn,
    )

    print(f"\n{'='*60}")
    print(f"Building {side}-half-synapse graph …")
    graph = build_half_synapse_graph(
        region, fragments, side=side,
        k_spatial=k_spatial,
    )
    print(f"  {graph.n_nodes} nodes, {graph.n_edges} edges "
          f"({(graph.edge_type==0).sum()} same-seg, "
          f"{(graph.edge_type==1).sum()} spatial)")
    print(f"  node_dim={graph.node_dim}  (3 pos + {graph.dna_dim} DNA)")
    n_labeled = int((graph.labels != 0).sum())
    n_true_clusters = int(len(np.unique(graph.labels[graph.labels != 0])))
    print(f"  labelled nodes: {n_labeled}/{graph.n_nodes}  "
          f"({n_true_clusters} true neurons)")
    print(f"{'='*60}\n")

    # --- Baseline: untrained GNN -------------------------------------------
    print("Evaluating untrained GNN …")
    gnn_init = HalfSynapseGNN(input_dim=graph.node_dim)
    pred_init = partition_half_synapses(gnn_init, graph, threshold=threshold, device=device)
    result_before = evaluate_partition_ari(pred_init, graph.labels)
    print(f"  ARI (random init):  {result_before['ari']:.4f}")
    print(f"  Homogeneity: {result_before['homogeneity']:.3f}  "
          f"Completeness: {result_before['completeness']:.3f}")
    print(f"  Predicted clusters: {result_before['n_clusters_pred']}  "
          f"True clusters: {result_before['n_clusters_true']}")

    # --- Train ---------------------------------------------------------------
    print(f"\nTraining HalfSynapseGNN for {n_epochs} epochs …")
    gnn_trained, history = train_partition_gnn(
        graph,
        n_epochs=n_epochs,
        lr=lr,
        max_pairs=max_pairs,
        device=device,
        seed=seed,
        log_every=log_every,
    )

    # --- After training ------------------------------------------------------
    print("\nEvaluating trained GNN …")
    pred_trained = partition_half_synapses(gnn_trained, graph, threshold=threshold, device=device)
    result_after = evaluate_partition_ari(pred_trained, graph.labels)
    print(f"  ARI (trained):      {result_after['ari']:.4f}")
    print(f"  Homogeneity: {result_after['homogeneity']:.3f}  "
          f"Completeness: {result_after['completeness']:.3f}")
    print(f"  Predicted clusters: {result_after['n_clusters_pred']}  "
          f"True clusters: {result_after['n_clusters_true']}")

    delta_ari = result_after["ari"] - result_before["ari"]
    print(f"\n{'='*60}")
    print(f"  ARI improvement:  {delta_ari:+.4f}  (random → trained)")
    print(f"{'='*60}\n")

    return {
        "before": result_before,
        "after": result_after,
        "history": history,
        "delta_ari": delta_ari,
        "graph": graph,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    import argparse
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--synthetic", action="store_true",
                   help="Generate synthetic world (no network required)")
    p.add_argument("--n-neurons", type=int, default=10)
    p.add_argument("--segs-per-neuron", type=int, default=2)
    p.add_argument("--synapses-per-seg", type=int, default=8)
    p.add_argument("--frankenfraction", type=float, default=0.2,
                   help="Fraction of neurons with a frankenmerge segment")
    p.add_argument("--dna-dim", type=int, default=32,
                   help="DNA embedding dimension for synthetic fragments")
    p.add_argument("--side", choices=["pre", "post"], default="pre")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--k-spatial", type=int, default=6)
    p.add_argument("--max-pairs", type=int, default=500)
    p.add_argument("--threshold", type=float, default=0.7)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)

    if args.synthetic:
        print(f"Generating synthetic world …")
        region, fragments = _make_synthetic_world(
            n_neurons=args.n_neurons,
            segs_per_neuron=args.segs_per_neuron,
            synapses_per_seg=args.synapses_per_seg,
            frankenfraction=args.frankenfraction,
            dna_dim=args.dna_dim,
            rng=rng,
        )
    else:
        p.error("Only --synthetic mode is implemented in this script.")

    run_half_synapse_ablation(
        region, fragments,
        side=args.side,
        n_epochs=args.epochs,
        lr=args.lr,
        k_spatial=args.k_spatial,
        max_pairs=args.max_pairs,
        threshold=args.threshold,
        device=args.device,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
