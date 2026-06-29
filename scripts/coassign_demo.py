#!/usr/bin/env python3
"""Synapse co-assignment demo.

Fetches real proofread v1412 neurons, splits each skeleton into N pieces
(simulating v117 pre-proofread segments), places synapses near skeleton
vertices, and runs the co-assignment pipeline end-to-end.

Pipeline
--------
1. Fetch real skeletons and split into pieces.
2. Encode each piece's DNA with SkeletonGNN (learned, not hand-crafted).
3. Build a SynapseGraph: nodes = synapses, edges = same-seg + spatial k-NN.
4. Train SynapseCoassigner: learn P(same neuron) per edge.
5. Generate K materializations via correlation clustering.
6. Report pairwise precision/recall and coverage@K.

Usage
-----
  python scripts/coassign_demo.py --n-neurons 20 --n-pieces 3
  python scripts/coassign_demo.py --n-neurons 30 --n-pieces 3 --cell-type 23P
"""

from __future__ import annotations

import argparse
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
# Skeleton splitting (identical to real_skeleton_partition.py)
# ---------------------------------------------------------------------------

def _build_adj(n, edges):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[int(u)].append(int(v))
        adj[int(v)].append(int(u))
    return adj


def _subtree_sizes(root, adj):
    n = len(adj)
    size = np.ones(n, dtype=np.int32)
    parent = np.full(n, -1, dtype=np.int32)
    order = []
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
    return size, parent


def _split(verts, edges, radii, n_pieces, min_verts=8):
    if n_pieces <= 1 or len(verts) < min_verts * 2 or len(edges) == 0:
        return [(verts, edges, radii)]
    n = len(verts)
    adj = _build_adj(n, edges)
    sizes, parent = _subtree_sizes(0, adj)[:2]
    best_v = min(
        (v for v in range(1, n)),
        key=lambda v: abs(sizes[v] - (n - sizes[v])),
        default=-1,
    )
    if best_v < 0:
        return [(verts, edges, radii)]
    cut_p = int(parent[best_v])

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

    def subgraph(mask):
        old = np.where(mask)[0]
        remap = np.full(n, -1, dtype=np.int64)
        remap[old] = np.arange(len(old))
        sv = verts[old]
        sr = radii[old]
        keep = mask[edges[:, 0]] & mask[edges[:, 1]]
        se = remap[edges[keep]].astype(np.int64)
        return sv, se, sr

    vA, eA, rA = subgraph(visited_a)
    vB, eB, rB = subgraph(visited_b)
    if len(vA) < min_verts or len(vB) < min_verts:
        return [(verts, edges, radii)]

    half = n_pieces // 2
    return _split(vA, eA, rA, half, min_verts) + _split(vB, eB, rB, n_pieces - half, min_verts)


# ---------------------------------------------------------------------------
# Build the world: real skeletons → pieces → Region + Fragments + seg_dna
# ---------------------------------------------------------------------------

def build_world(root_ids, n_target, n_pieces, synapses_per_piece, rng,
                max_verts=8000, noise_nm=500.0):
    from neuronauts.data.loaders import load_skeleton
    from neuronauts.schemas import Fragment, Region

    print(f"\nFetching skeletons ({n_target} neurons × {n_pieces} pieces) …")

    all_pts, all_seg, all_root = [], [], []
    fragments = []
    seg_counter = 1
    syn_offset = 0
    n_neurons = 0

    for root_id in root_ids:
        if n_neurons >= n_target:
            break
        skel = load_skeleton(root_id, TOKEN)
        if skel is None:
            continue
        v, e, r = skel["vertices_nm"], skel["edges"], skel["radii_nm"]
        if not (8 * n_pieces <= len(v) <= max_verts):
            continue

        pieces = _split(v, e, r, n_pieces)
        if len(pieces) < 2:
            continue

        n_neurons += 1
        neuron_id = n_neurons
        piece_sizes = []

        for pv, pe, pr in pieces:
            sid = seg_counter
            seg_counter += 1
            anchors = rng.integers(0, len(pv), synapses_per_piece)
            pts = pv[anchors] + rng.normal(0, noise_nm, (synapses_per_piece, 3)).astype(np.float32)
            idxs = np.arange(syn_offset, syn_offset + synapses_per_piece, dtype=np.int64)
            syn_offset += synapses_per_piece

            all_pts.append(pts)
            all_seg.extend([sid] * synapses_per_piece)
            all_root.extend([neuron_id] * synapses_per_piece)

            deg = np.zeros(len(pv), dtype=np.int64)
            if len(pe):
                np.add.at(deg, pe[:, 0], 1)
                np.add.at(deg, pe[:, 1], 1)
            endpoints = pv[deg <= 1] if (deg <= 1).any() else pv[[0]]

            fragments.append(Fragment(
                fragment_id=sid,
                region_id="demo",
                base_root_id=sid,
                vertices_nm=pv,
                edges=pe if len(pe) else np.zeros((0, 2), dtype=np.int64),
                endpoints_nm=endpoints,
                radius_nm=pr,
                synapse_indices=idxs,
                dna=None,
            ).validate())
            piece_sizes.append(len(pv))

        print(f"  [{n_neurons:3d}] root={root_id}  V={len(v)}"
              f"  pieces={'/' .join(str(s) for s in piece_sizes)}")
        time.sleep(0.05)

    if not fragments:
        raise RuntimeError("No usable skeletons — check network and token")

    N = syn_offset
    pts = np.concatenate(all_pts).astype(np.float32)
    post_pts = pts + rng.normal(0, 2000, pts.shape).astype(np.float32)
    lo, hi = pts.min(0) - 5000, pts.max(0) + 5000

    region = Region(
        region_id="demo",
        bbox_nm=(tuple(float(x) for x in lo), tuple(float(x) for x in hi)),
        voxel_size_nm=(8.0, 8.0, 40.0),
        seg_version=117, label_version=1412,
        pre_pt_nm=pts, post_pt_nm=post_pts,
        pre_root_id=np.array(all_root, dtype=np.int64),
        post_root_id=np.zeros(N, dtype=np.int64),
        synapse_id=np.arange(N, dtype=np.int64),
        pre_seg_id=np.array(all_seg, dtype=np.int64),
        post_seg_id=np.zeros(N, dtype=np.int64),
    ).validate()

    print(f"\n  → {n_neurons} neurons, {len(fragments)} pieces, {N} synapses")
    return region, fragments


# ---------------------------------------------------------------------------
# Encode DNA with SkeletonGNN
# ---------------------------------------------------------------------------

def encode_dna(fragments, root_label_map, *, epochs=40, d_model=64, output_dim=32,
               device="cpu", seed=42):
    from neuronauts.represent.skeleton_gnn import SkeletonGNN, encode_fragments_gnn, train_skeleton_gnn

    gnn = SkeletonGNN(node_input_dim=4, d_model=d_model, output_dim=output_dim)
    if epochs > 0:
        print(f"\nTraining SkeletonGNN ({epochs} epochs) …")
        h = train_skeleton_gnn(gnn, [fragments], n_epochs=epochs, device=device,
                               root_label_map=root_label_map, log_every=10)
        print(f"  pos_cos={h['pos_cos'][-1]:.3f}  neg_cos={h['neg_cos'][-1]:.3f}")

    print("Encoding fragments …")
    return encode_fragments_gnn(gnn, fragments, device=device)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-neurons", type=int, default=20)
    p.add_argument("--n-pieces", type=int, default=3)
    p.add_argument("--cell-type", default=None)
    p.add_argument("--synapses-per-piece", type=int, default=10)
    p.add_argument("--epochs-dna", type=int, default=40)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--k-spatial", type=int, default=8)
    p.add_argument("--K", type=int, default=5, help="Number of materializations")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--output-dim", type=int, default=32)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)

    from neuronauts.data.loaders import sample_neurons
    candidates = sample_neurons(args.n_neurons * 6, cell_type=args.cell_type, seed=args.seed)
    print(f"Sampled {len(candidates)} candidates")

    region, fragments = build_world(
        candidates, args.n_neurons, args.n_pieces, args.synapses_per_piece, rng,
    )

    # root_label_map: piece_seg_id → {neuron_id}  (for SkeletonGNN supervision)
    root_label_map = {}
    for seg, nid in zip(region.pre_seg_id, region.pre_root_id):
        if int(seg) > 0 and int(nid) > 0:
            root_label_map.setdefault(int(seg), set()).add(int(nid))

    frags = encode_dna(fragments, root_label_map,
                       epochs=args.epochs_dna, d_model=args.d_model,
                       output_dim=args.output_dim, device=args.device, seed=args.seed)

    # Build seg_dna dict for the co-assignment graph
    seg_dna = {int(f.base_root_id): f.dna for f in frags if f.dna is not None}

    # ---- Build synapse graph ------------------------------------------------
    from neuronauts.coassign import (
        SynapseCoassigner, build_synapse_graph, coverage_at_k,
        materializations, pairwise_precision_recall, train,
    )

    graph = build_synapse_graph(
        region.pre_pt_nm, region.pre_seg_id, region.pre_root_id,
        seg_dna, k_spatial=args.k_spatial,
    )
    n_true = len(np.unique(graph.labels[graph.labels != 0]))
    n_ss   = int((graph.same_seg == 1).sum())
    n_sp   = int((graph.same_seg == 0).sum())
    print(f"\nSynapseGraph: {graph.n_nodes} nodes | {graph.n_edges} edges"
          f" ({n_ss} same-seg, {n_sp} spatial)")
    print(f"node_dim={graph.node_dim} (3 pos + {graph.dna_dim} DNA) | {n_true} true neurons")

    # ---- Before training ----------------------------------------------------
    model = SynapseCoassigner(node_dim=graph.node_dim, d_model=args.d_model)
    import torch
    with torch.no_grad():
        node_feat  = torch.from_numpy(np.concatenate([graph.node_pos, graph.node_dna], axis=1)).float()
        edge_src_t = torch.from_numpy(graph.edge_src).long()
        edge_dst_t = torch.from_numpy(graph.edge_dst).long()
        same_seg_t = torch.from_numpy(graph.same_seg).float()
        probs_init = model.edge_probs(node_feat, edge_src_t, edge_dst_t, same_seg_t).numpy()

    mats_init = materializations(graph.n_nodes, graph.edge_src, graph.edge_dst,
                                 probs_init, K=args.K, threshold=args.threshold)
    r_init = pairwise_precision_recall(mats_init[0][0], graph.labels)
    cov_init = coverage_at_k(mats_init, graph.labels)
    print(f"\n--- Before training (random init) ---")
    print(f"  top-1  P={r_init['precision']:.3f}  R={r_init['recall']:.3f}  F1={r_init['f1']:.3f}")
    print(f"  coverage@{args.K} = {cov_init}")

    # ---- Train --------------------------------------------------------------
    print(f"\nTraining SynapseCoassigner ({args.epochs} epochs) …")
    history = train(model, [graph], n_epochs=args.epochs, device=args.device,
                    seed=args.seed, log_every=10)

    # ---- After training -----------------------------------------------------
    with torch.no_grad():
        probs_trained = model.edge_probs(node_feat, edge_src_t, edge_dst_t, same_seg_t).numpy()

    mats_trained = materializations(graph.n_nodes, graph.edge_src, graph.edge_dst,
                                    probs_trained, K=args.K, threshold=args.threshold)
    r_top1 = pairwise_precision_recall(mats_trained[0][0], graph.labels)
    cov = coverage_at_k(mats_trained, graph.labels)

    print(f"\n{'='*60}")
    print(f"After training:")
    print(f"  top-1  P={r_top1['precision']:.3f}  R={r_top1['recall']:.3f}  F1={r_top1['f1']:.3f}")
    for i, (labels, score) in enumerate(mats_trained):
        ri = pairwise_precision_recall(labels, graph.labels)
        n_clusters = len(np.unique(labels))
        print(f"  mat {i+1}: P={ri['precision']:.3f}  R={ri['recall']:.3f}"
              f"  F1={ri['f1']:.3f}  clusters={n_clusters}/{n_true}  score={score:.1f}")
    print(f"  coverage@{args.K} = {cov}")
    print(f"{'='*60}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
