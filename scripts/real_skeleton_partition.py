#!/usr/bin/env python3
"""Real-skeleton partition ablation: ARI-based evaluation on real proofread neurons.

Fetches real proofread v1412 skeletons, splits each into N pieces (simulating
the v117 pre-proofread state where a neuron is split across multiple segments),
places synapses near actual skeleton vertices, and evaluates how well the
HalfSynapseGNN can recover the original neuron partition.

Concrete evaluation scenario
-----------------------------
  1. Fetch R proofread v1412 neurons (optionally filtered by cell type).
  2. Split each skeleton into N_PIECES via recursive balanced bisection.
     Each piece = one simulated "v117 segment" with a synthetic seg_id.
  3. Assign synapses by sampling vertices from each piece and adding noise
     (~500 nm) — real synapses are ON the neuron, so this is physically correct.
  4. Set pre_seg_id  = piece segment ID  (noisy, because pieces were split from
     the same real neuron — this is the "same-segment" evidence channel).
  5. Set pre_root_id = original proofread neuron ID  (supervision only).
  6. Optionally train SkeletonGNN on the pieces so DNA captures morphological
     identity (pieces from the same neuron should have similar DNA).
  7. Build HalfSynapseGraph (same-seg edges + spatial k-NN edges, DNA features).
  8. Train HalfSynapseGNN and evaluate ARI before / after.

Why this is the honest test
-----------------------------
  - Real skeleton morphologies → real DNA variation between cell types AND individuals.
  - Synapses placed near skeleton → spatial proximity IS informative (unlike the
    uniform-random placement used in half_split_ablation.py).
  - Multiple pieces per neuron → realistic "multi-segment" scenario.
  - Evaluation via ARI (partition quality), not cosine AUC.

Usage
-----
  # Cross-type (easiest): mix of all cell types
  python scripts/real_skeleton_partition.py --n-neurons 20 --n-pieces 3

  # Within-type (hardest): only 23P pyramidal cells
  python scripts/real_skeleton_partition.py --cell-type 23P --n-neurons 30 --n-pieces 3

  # More pieces = harder test
  python scripts/real_skeleton_partition.py --n-neurons 20 --n-pieces 4 --epochs 50
"""

from __future__ import annotations

import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

TOKEN = "a08cdcba8581846f48d5742a75c53311"


# ---------------------------------------------------------------------------
# Recursive skeleton splitting → N pieces
# ---------------------------------------------------------------------------

def _build_adj(n: int, edges: np.ndarray) -> list[list[int]]:
    adj: list[list[int]] = [[] for _ in range(n)]
    for u, v in edges:
        adj[int(u)].append(int(v))
        adj[int(v)].append(int(u))
    return adj


def _subtree_sizes(root: int, adj: list[list[int]]):
    n = len(adj)
    size = np.ones(n, dtype=np.int32)
    parent = np.full(n, -1, dtype=np.int32)
    order: list[int] = []
    visited = np.zeros(n, dtype=bool)
    q = deque([root])
    visited[root] = True
    while q:
        v = q.popleft()
        order.append(v)
        for w in adj[v]:
            if not visited[w]:
                visited[w] = True
                parent[w] = v
                q.append(w)
    for v in reversed(order):
        p = parent[v]
        if p >= 0:
            size[p] += size[v]
    return size, parent, order


def _find_balance_cut(n: int, adj: list[list[int]]) -> tuple[int, int]:
    """Return (cut_child, cut_parent) for the most balanced bisection edge."""
    sizes, parent, order = _subtree_sizes(0, adj)
    best_v, best_diff = -1, n + 1
    for v in order[1:]:
        diff = abs(sizes[int(v)] - (n - sizes[int(v)]))
        if diff < best_diff:
            best_diff = diff
            best_v = int(v)
    return best_v, int(parent[best_v])


def _extract_subgraph(
    mask: np.ndarray, verts: np.ndarray, edges: np.ndarray, radii: np.ndarray
):
    old_idx = np.where(mask)[0]
    remap = np.full(len(verts), -1, dtype=np.int64)
    remap[old_idx] = np.arange(len(old_idx), dtype=np.int64)
    sub_v = verts[old_idx]
    sub_r = radii[old_idx]
    keep = mask[edges[:, 0]] & mask[edges[:, 1]]
    sub_e = remap[edges[keep]].astype(np.int64)
    return sub_v, sub_e, sub_r


def split_skeleton_n_pieces(
    verts: np.ndarray,
    edges: np.ndarray,
    radii: np.ndarray,
    n_pieces: int,
    min_verts: int = 8,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Recursively bisect a skeleton tree into (up to) n_pieces sub-trees.

    Returns a list of (verts, edges, radii) tuples, each in local vertex
    indexing.  May return fewer than n_pieces if pieces become too small.
    """
    if n_pieces <= 1 or len(verts) < min_verts * 2:
        return [(verts, edges, radii)]

    n = len(verts)
    if len(edges) == 0 or n < 4:
        return [(verts, edges, radii)]

    adj = _build_adj(n, edges)
    best_v, cut_p = _find_balance_cut(n, adj)
    if best_v < 0:
        return [(verts, edges, radii)]

    # Flood-fill the two sides
    visited_a = np.zeros(n, dtype=bool)
    q = deque([best_v])
    visited_a[best_v] = True
    while q:
        v = q.popleft()
        for w in adj[v]:
            if not visited_a[w] and w != cut_p:
                visited_a[w] = True
                q.append(w)
    visited_b = ~visited_a

    vA, eA, rA = _extract_subgraph(visited_a, verts, edges, radii)
    vB, eB, rB = _extract_subgraph(visited_b, verts, edges, radii)

    if len(vA) < min_verts or len(vB) < min_verts:
        return [(verts, edges, radii)]

    half = n_pieces // 2
    rest = n_pieces - half
    pieces_a = split_skeleton_n_pieces(vA, eA, rA, half, min_verts)
    pieces_b = split_skeleton_n_pieces(vB, eB, rB, rest, min_verts)
    return pieces_a + pieces_b


# ---------------------------------------------------------------------------
# Build world: real skeletons → multi-piece Region + Fragments
# ---------------------------------------------------------------------------

def build_real_split_world(
    root_ids: list[int],
    n_target: int,
    n_pieces: int,
    synapses_per_piece: int,
    rng: np.random.Generator,
    max_verts: int = 8000,
    min_piece_verts: int = 8,
    synapse_noise_nm: float = 500.0,
) -> tuple:
    """Fetch real proofread skeletons, split into pieces, place real-ish synapses.

    Synapses are placed at randomly sampled skeleton vertices + Gaussian noise
    (~synapse_noise_nm nm).  This gives a realistic placement: synapses are ON
    the neuron, so spatial proximity is informative (unlike uniform random).

    Returns
    -------
    (region, fragments)
        region  — Region with pre_seg_id (piece IDs) and pre_root_id (neuron IDs)
        fragments — list of Fragment (one per piece), dna=None
    """
    from neuronauts.data.loaders import load_skeleton
    from neuronauts.schemas import Fragment, Region

    print(f"\nFetching real skeletons (target {n_target} neurons, {n_pieces} pieces each) …")

    all_pre_pts: list[np.ndarray] = []
    all_pre_seg: list[int] = []
    all_pre_root: list[int] = []
    fragments: list[Fragment] = []

    neuron_counter = 0
    seg_id_counter = 1  # synthetic segment IDs
    syn_global_idx = 0

    for root_id in root_ids:
        if neuron_counter >= n_target:
            break

        skel = load_skeleton(root_id, TOKEN)
        if skel is None:
            continue
        verts = skel["vertices_nm"]
        edges = skel["edges"]
        radii = skel["radii_nm"]

        if len(verts) < min_piece_verts * n_pieces or len(verts) > max_verts:
            continue

        pieces = split_skeleton_n_pieces(verts, edges, radii, n_pieces, min_verts=min_piece_verts)
        if len(pieces) < 2:
            continue

        neuron_counter += 1
        neuron_id = neuron_counter

        piece_fragments = []
        for pv, pe, pr in pieces:
            sid = seg_id_counter
            seg_id_counter += 1

            # Leaf vertices for endpoints
            deg = np.zeros(len(pv), dtype=np.int64)
            if len(pe):
                np.add.at(deg, pe[:, 0], 1)
                np.add.at(deg, pe[:, 1], 1)
            leaf_mask = deg <= 1
            endpoints = pv[leaf_mask] if leaf_mask.any() else pv[[0]]

            # Synapses near skeleton vertices
            anchor_idxs = rng.integers(0, len(pv), synapses_per_piece)
            syn_pts = pv[anchor_idxs] + rng.normal(0, synapse_noise_nm, (synapses_per_piece, 3)).astype(np.float32)

            syn_indices = np.arange(syn_global_idx, syn_global_idx + synapses_per_piece, dtype=np.int64)
            syn_global_idx += synapses_per_piece

            all_pre_pts.append(syn_pts)
            all_pre_seg.extend([sid] * synapses_per_piece)
            all_pre_root.extend([neuron_id] * synapses_per_piece)

            frag = Fragment(
                fragment_id=sid,
                region_id="real_split",
                base_root_id=sid,
                vertices_nm=pv,
                edges=pe if len(pe) else np.zeros((0, 2), dtype=np.int64),
                endpoints_nm=endpoints,
                radius_nm=pr,
                synapse_indices=syn_indices,
                dna=None,
            ).validate()
            piece_fragments.append(frag)

        fragments.extend(piece_fragments)
        piece_sizes = [len(p[0]) for p in pieces]
        print(f"  [{neuron_counter:3d}] root={root_id}  V={len(verts)}"
              f"  pieces={'/'.join(str(s) for s in piece_sizes)}")
        time.sleep(0.05)

    if not fragments:
        raise RuntimeError("No usable skeletons fetched — check network access and token")

    N = syn_global_idx
    # Build a global coordinate frame for the synapses
    all_pts = np.concatenate(all_pre_pts, axis=0).astype(np.float32)
    post_pts = all_pts + rng.normal(0, 2000, all_pts.shape).astype(np.float32)

    # Region bbox from actual synapse positions
    mins = all_pts.min(axis=0)
    maxs = all_pts.max(axis=0)
    padding = 5000.0
    bbox = (
        tuple(float(v) for v in (mins - padding)),
        tuple(float(v) for v in (maxs + padding)),
    )

    region = Region(
        region_id="real_split_ablation",
        bbox_nm=bbox,
        voxel_size_nm=(8.0, 8.0, 40.0),
        seg_version=117,
        label_version=1412,
        pre_pt_nm=all_pts,
        post_pt_nm=post_pts,
        pre_root_id=np.array(all_pre_root, dtype=np.int64),
        post_root_id=np.zeros(N, dtype=np.int64),
        synapse_id=np.arange(N, dtype=np.int64),
        pre_seg_id=np.array(all_pre_seg, dtype=np.int64),
        post_seg_id=np.zeros(N, dtype=np.int64),
    ).validate()

    n_neurons = len(set(all_pre_root))
    print(f"\n  → {n_neurons} neurons, {len(fragments)} pieces, {N} synapses")
    return region, fragments


# ---------------------------------------------------------------------------
# DNA encoding step
# ---------------------------------------------------------------------------

def encode_dna(
    fragments,
    root_label_map,
    *,
    n_epochs_dna: int = 40,
    lr: float = 1e-3,
    d_model: int = 64,
    output_dim: int = 32,
    device: str = "cpu",
    seed: int = 42,
    log_every: int = 10,
):
    """Train SkeletonGNN on skeleton pieces and encode all fragments.

    Positive pairs: pieces from the same neuron (same label in root_label_map).
    Negative pairs: pieces from different neurons.

    Returns list of Fragments with dna filled.
    """
    from neuronauts.represent.skeleton_gnn import SkeletonGNN, encode_fragments_gnn, train_skeleton_gnn

    gnn = SkeletonGNN(
        node_input_dim=4,
        d_model=d_model,
        output_dim=output_dim,
    )

    if n_epochs_dna > 0:
        print(f"\nTraining SkeletonGNN for DNA encoding ({n_epochs_dna} epochs) …")
        history = train_skeleton_gnn(
            gnn,
            [fragments],
            n_epochs=n_epochs_dna,
            lr=lr,
            device=device,
            root_label_map=root_label_map,
            log_every=log_every,
        )
        final_pos = history["pos_cos"][-1] if history["pos_cos"] else float("nan")
        final_neg = history["neg_cos"][-1] if history["neg_cos"] else float("nan")
        print(f"  pos_cos={final_pos:.3f}  neg_cos={final_neg:.3f}")
    else:
        print("\nUsing random-init SkeletonGNN for DNA (n_epochs_dna=0) …")

    print("Encoding fragments with SkeletonGNN …")
    frags_encoded = encode_fragments_gnn(gnn, fragments, device=device)
    return frags_encoded


# ---------------------------------------------------------------------------
# Half-synapse partition evaluation
# ---------------------------------------------------------------------------

def run_partition_eval(
    region,
    fragments,
    *,
    side: str = "pre",
    n_epochs: int = 40,
    lr: float = 1e-3,
    k_spatial: int = 8,
    max_pairs: int = 800,
    threshold: float = 0.75,
    pos_scale_nm: float = 50_000.0,
    device: str = "cpu",
    seed: int = 42,
    log_every: int = 10,
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
        region, fragments, side=side, k_spatial=k_spatial, pos_scale_nm=pos_scale_nm,
    )
    dna_dim = graph.dna_dim
    n_same_seg = int((graph.edge_type == 0).sum())
    n_spatial = int((graph.edge_type == 1).sum())
    n_labeled = int((graph.labels != 0).sum())
    n_true = int(len(np.unique(graph.labels[graph.labels != 0])))
    print(f"  {graph.n_nodes} nodes | {graph.n_edges} edges "
          f"({n_same_seg} same-seg, {n_spatial} spatial)")
    print(f"  node_dim={graph.node_dim} (3 pos + {dna_dim} DNA) | "
          f"{n_labeled}/{graph.n_nodes} labelled | {n_true} true neurons")
    print(f"{'='*60}\n")

    # --- Baseline: random-init GNN ------------------------------------------
    gnn_init = HalfSynapseGNN(input_dim=graph.node_dim)
    pred_init = partition_half_synapses(gnn_init, graph, threshold=threshold, device=device)
    r_before = evaluate_partition_ari(pred_init, graph.labels)
    print(f"ARI before training:  {r_before['ari']:.4f}"
          f"  H={r_before['homogeneity']:.3f}  C={r_before['completeness']:.3f}"
          f"  clusters={r_before['n_clusters_pred']}/{r_before['n_clusters_true']}")

    # --- Train HalfSynapseGNN -----------------------------------------------
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

    # --- After training -------------------------------------------------------
    pred_trained = partition_half_synapses(gnn_trained, graph, threshold=threshold, device=device)
    r_after = evaluate_partition_ari(pred_trained, graph.labels)
    delta = r_after["ari"] - r_before["ari"]
    print(f"\nARI after training:   {r_after['ari']:.4f}"
          f"  H={r_after['homogeneity']:.3f}  C={r_after['completeness']:.3f}"
          f"  clusters={r_after['n_clusters_pred']}/{r_after['n_clusters_true']}")
    print(f"\n{'='*60}")
    print(f"  ΔARI = {delta:+.4f}  ({r_before['ari']:.4f} → {r_after['ari']:.4f})")
    print(f"{'='*60}\n")

    return {"before": r_before, "after": r_after, "history": history, "delta_ari": delta}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    import argparse
    from neuronauts.data.loaders import sample_neurons

    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--n-neurons", type=int, default=20,
                   help="Number of proofread neurons to fetch")
    p.add_argument("--cell-type", default=None,
                   help="Filter by cell type (e.g. '23P' for L2/3 pyramidal)")
    p.add_argument("--n-pieces", type=int, default=3,
                   help="Number of skeleton pieces per neuron (simulated v117 splits)")
    p.add_argument("--synapses-per-piece", type=int, default=10,
                   help="Synapses placed near each skeleton piece")
    p.add_argument("--synapse-noise-nm", type=float, default=500.0,
                   help="Gaussian noise on synapse positions (nm)")
    # DNA encoder
    p.add_argument("--epochs-dna", type=int, default=40,
                   help="SkeletonGNN training epochs (0 = random init)")
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--output-dim", type=int, default=32)
    # Partition GNN
    p.add_argument("--epochs", type=int, default=40,
                   help="HalfSynapseGNN training epochs")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--k-spatial", type=int, default=8)
    p.add_argument("--max-pairs", type=int, default=800)
    p.add_argument("--threshold", type=float, default=0.75)
    p.add_argument("--pos-scale-nm", type=float, default=50_000.0,
                   help="Divisor for position normalisation. Larger = position features less dominant.")
    p.add_argument("--side", choices=["pre", "post"], default="pre")
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-verts", type=int, default=8000)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)

    # Sample neurons
    print(f"Sampling {args.n_neurons} proofread neurons"
          + (f" (cell_type={args.cell_type})" if args.cell_type else "") + " …")
    candidates = sample_neurons(
        args.n_neurons * 6,
        cell_type=args.cell_type,
        seed=args.seed,
    )
    print(f"  {len(candidates)} candidates")

    # Build world
    region, fragments = build_real_split_world(
        candidates,
        n_target=args.n_neurons,
        n_pieces=args.n_pieces,
        synapses_per_piece=args.synapses_per_piece,
        rng=rng,
        max_verts=args.max_verts,
        synapse_noise_nm=args.synapse_noise_nm,
    )
    n_neurons_fetched = int(len(set(region.pre_root_id[region.pre_root_id > 0])))

    # Build root_label_map: piece fragment_id → {neuron_id}
    # root_label_map is used by train_skeleton_gnn for contrastive supervision
    all_pre_root = region.pre_root_id
    all_pre_seg = region.pre_seg_id
    root_label_map: dict[int, set[int]] = {}
    for seg_id, neuron_id in zip(all_pre_seg, all_pre_root):
        sid, nid = int(seg_id), int(neuron_id)
        if sid > 0 and nid > 0:
            root_label_map.setdefault(sid, set()).add(nid)

    # Encode DNA
    frags_with_dna = encode_dna(
        fragments,
        root_label_map,
        n_epochs_dna=args.epochs_dna,
        lr=args.lr,
        d_model=args.d_model,
        output_dim=args.output_dim,
        device=args.device,
        seed=args.seed,
    )

    # Partition evaluation
    print(f"\n{'='*60}")
    print(f"Real-skeleton partition ablation")
    print(f"  {n_neurons_fetched} neurons × {args.n_pieces} pieces"
          f" × {args.synapses_per_piece} syn/piece")
    print(f"  cell_type={args.cell_type or 'mixed'}"
          f"  DNA epochs={args.epochs_dna}"
          f"  Partition epochs={args.epochs}")
    print(f"{'='*60}")

    results = run_partition_eval(
        region, frags_with_dna,
        side=args.side,
        n_epochs=args.epochs,
        lr=args.lr,
        k_spatial=args.k_spatial,
        max_pairs=args.max_pairs,
        threshold=args.threshold,
        pos_scale_nm=args.pos_scale_nm,
        device=args.device,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
