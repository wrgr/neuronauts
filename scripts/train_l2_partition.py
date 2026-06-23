#!/usr/bin/env python3
"""Train a partition GNN on the L2-node substrate (P1 proofread-dense region).

The shared synapse-trained checkpoint is out of distribution on the L2 substrate
(``scripts/p1_completeness_benchmark.py`` shows ARI≈0): a fragment-centroid graph
has one node per fragment and therefore *no same-fragment edges*, which is the
signal the model relies on.  This script trains a model **on the L2 substrate
itself**, where each observation is an L2 node and same-fragment edges (edge type
0) connect the L2 nodes of one v117 fragment.

Scale.  The raw P1 L2 world has ~1.38M nodes; a kNN observation graph at that
scale OOMs a 16 GB box.  We subsample each v117 fragment to ``--k-nodes`` L2 nodes
(default 50, farthest-point for spatial spread).  This preserves both the
same-fragment structure and the cross-fragment spatial-kNN edges while keeping the
graph at ~50k nodes.

Split.  A spatial train/test split along x with a buffer gap, assigned per
fragment by its node centroid, so no v117 fragment straddles the split.  Within
each split, many neurons remain spatially interleaved, so cross-neuron merge
decisions — the hard part — are present on both sides.

Trust.  A scaffold is only useful if we trust its merges.  Evaluation sweeps
``--cc-bias`` (more negative = more conservative) and reports, at each operating
point, **merge precision** (of the merges it proposes, how many are correct) and
**completeness precision** (of the fragments it calls done, how many truly are).
The high-precision end of that sweep is the trustworthy scaffold.

Usage
-----
  NEURONAUTS_L2_CACHE_DIR=$PWD/cache/l2_skeleton \
  PYTHONPATH=$PWD python3 scripts/train_l2_partition.py \
    --save-checkpoint /tmp/neuronauts_l2.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

P1_BBOX = ((818_500, 685_000, 794_000), (918_500, 785_000, 994_000))
L2_CACHE = str(_ROOT / "cache" / "l2_world" / "p1_full.npz")


def _farthest_point_subsample(pts: np.ndarray, k: int, rng) -> np.ndarray:
    """Return indices of ≤k farthest-point-sampled rows of pts (spatial spread)."""
    n = len(pts)
    if n <= k:
        return np.arange(n)
    sel = [int(rng.integers(n))]
    d = np.linalg.norm(pts - pts[sel[0]], axis=1)
    for _ in range(1, k):
        nxt = int(np.argmax(d))
        sel.append(nxt)
        d = np.minimum(d, np.linalg.norm(pts - pts[nxt], axis=1))
    return np.array(sel, dtype=np.int64)


def _build_split_world(pos, frag, label, frag_keep, *, version, k_nodes, rng):
    """Assemble (fragments, region, root_label_map) for a set of v117 fragments.

    Each kept fragment contributes ≤k_nodes farthest-point-sampled L2 nodes as
    Region observations; the fragment's DNA cloud uses the same sampled points.
    """
    from neuronauts.schemas import Region
    from treestitch.realworld import _cloud_fragment

    all_pos, all_frag, all_label = [], [], []
    fragments = []
    lmap: dict[int, set] = {}
    cursor = 0
    for fr in sorted(frag_keep):
        idx = np.where(frag == fr)[0]
        sub = _farthest_point_subsample(pos[idx], k_nodes, rng)
        gidx = idx[sub]
        p = pos[gidx]
        m = len(p)
        obs_idx = np.arange(cursor, cursor + m, dtype=np.int64)
        cursor += m
        all_pos.append(p)
        all_frag.append(np.full(m, fr, dtype=np.int64))
        all_label.append(label[gidx])
        fragments.append(_cloud_fragment(int(fr), f"minnie65_v{version}", p, obs_idx))
        lmap.setdefault(int(fr), set()).update(int(x) for x in np.unique(label[gidx]))

    pos_a = np.concatenate(all_pos).astype(np.float32)
    frag_a = np.concatenate(all_frag)
    label_a = np.concatenate(all_label)
    n = len(pos_a)
    zeros = np.zeros(n, dtype=np.int64)
    region = Region(
        region_id=f"minnie65_v{version}_l2",
        bbox_nm=(tuple(float(v) for v in pos_a.min(0) - 5000),
                 tuple(float(v) for v in pos_a.max(0) + 5000)),
        voxel_size_nm=(8.0, 8.0, 40.0),
        seg_version=117, label_version=version,
        pre_pt_nm=pos_a, post_pt_nm=pos_a.copy(),
        pre_root_id=label_a, post_root_id=zeros.copy(),
        synapse_id=np.arange(n, dtype=np.int64),
        pre_seg_id=frag_a, post_seg_id=zeros.copy(),
    ).validate()
    return fragments, region, lmap


def _reconcile_same_fragment(pred: np.ndarray, fragment_id: np.ndarray) -> np.ndarray:
    """Force every L2 node of one v117 fragment into a single cluster.

    On the L2 substrate a v117 fragment is an atomic connected supervoxel group:
    proofreading only *merges* it with other fragments, never splits it.  So
    same-fragment co-membership is ground truth, not a model decision — the model
    is responsible only for cross-fragment merges.  This unions the (possibly
    several) clusters the raw partition assigned to one fragment's nodes, exactly
    like the cross-tile reconciliation in ``partition_observations_tiled``.
    """
    pred = pred.copy()
    n_clusters = int(pred.max()) + 1 if len(pred) else 0
    parent = np.arange(n_clusters, dtype=np.int64)

    def _find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    frag_rep: dict[int, int] = {}
    for node in range(len(pred)):
        fid = int(fragment_id[node])
        c = int(pred[node])
        if fid not in frag_rep:
            frag_rep[fid] = c
        else:
            rc, rr = _find(c), _find(frag_rep[fid])
            if rc != rr:
                parent[rc] = rr

    canonical: dict[int, int] = {}
    out = np.empty(len(pred), dtype=np.int64)
    nxt = 0
    for node in range(len(pred)):
        root = _find(int(pred[node]))
        if root not in canonical:
            canonical[root] = nxt
            nxt += 1
        out[node] = canonical[root]
    return out


def _pred_completeness(fragment_id, pred) -> dict[int, bool]:
    frag_clusters: dict[int, set] = {}
    cluster_frags: dict[int, set] = {}
    for f, c in zip(fragment_id.tolist(), pred.tolist()):
        if c >= 0:
            frag_clusters.setdefault(int(f), set()).add(int(c))
            cluster_frags.setdefault(int(c), set()).add(int(f))
    return {f: (len(cs) == 1 and len(cluster_frags[next(iter(cs))]) == 1)
            for f, cs in frag_clusters.items()}


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--l2-cache", default=L2_CACHE)
    p.add_argument("--version", type=int, default=1718)
    p.add_argument("--k-nodes", type=int, default=50,
                   help="Max L2 observation nodes per v117 fragment (default 50).")
    p.add_argument("--min-l2-per-fragment", type=int, default=2)
    p.add_argument("--train-frac", type=float, default=0.60,
                   help="Fraction of x-range (from west) assigned to train.")
    p.add_argument("--buffer-frac", type=float, default=0.08,
                   help="x-range fraction dropped as a train/test buffer.")
    p.add_argument("--k-spatial", type=int, default=6)
    p.add_argument("--embed-epochs", type=int, default=15)
    p.add_argument("--partition-epochs", type=int, default=120)
    p.add_argument("--train-max-nodes", type=int, default=25_000)
    p.add_argument("--tile-size", type=int, default=4_000)
    p.add_argument("--cc-bias", default="-6,-4,-2,0,2,4",
                   help="Comma-separated cc_bias operating points for the eval sweep.")
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-checkpoint", default="/tmp/neuronauts_l2.pt")
    args = p.parse_args()

    import torch  # noqa: F401
    from treestitch.checkpoint import save_checkpoint
    from treestitch.embed import (
        FragmentEncoder, encode_fragments, train_fragment_encoder)
    from treestitch.graph import build_observation_graph
    from treestitch.partition import (
        evaluate_partition, merge_metrics,
        partition_observations_cc, partition_observations_tiled,
        train_edge_partition_multi_region,
        completeness_metrics, fragment_completeness)

    rng = np.random.default_rng(args.seed)

    # ── Load + sliver filter ──────────────────────────────────────────────
    cache = args.l2_cache if Path(args.l2_cache).exists() else L2_CACHE
    print(f"Loading raw L2 arrays from {cache} …")
    d = np.load(cache)
    pos = d["pos"].astype(np.float32)
    frag = d["frag_ids"].astype(np.int64)
    label = d["labels"].astype(np.int64)
    fu, fc = np.unique(frag, return_counts=True)
    keep = {int(f) for f, c in zip(fu, fc) if c >= args.min_l2_per_fragment}
    m = np.array([int(f) in keep for f in frag])
    pos, frag, label = pos[m], frag[m], label[m]
    print(f"  {len(keep)} fragments (≥{args.min_l2_per_fragment} nodes), {m.sum()} L2 nodes")

    # ── Spatial split by fragment centroid x, with buffer ─────────────────
    x = pos[:, 0]
    x_lo, x_hi = x.min(), x.max()
    train_cut = x_lo + (x_hi - x_lo) * args.train_frac
    test_cut = train_cut + (x_hi - x_lo) * args.buffer_frac
    cx = {fr: pos[frag == fr, 0].mean() for fr in keep}
    train_frags = {fr for fr, c in cx.items() if c < train_cut}
    test_frags = {fr for fr, c in cx.items() if c >= test_cut}
    print(f"  split @ x<{train_cut:.0f} train / x≥{test_cut:.0f} test "
          f"(buffer {train_cut:.0f}–{test_cut:.0f}); "
          f"{len(train_frags)} train frags / {len(test_frags)} test frags")

    print("\nBuilding train world …")
    tr_frags, tr_region, tr_lmap = _build_split_world(
        pos, frag, label, train_frags,
        version=args.version, k_nodes=args.k_nodes, rng=rng)
    print(f"  train: {len(tr_frags)} frags, {tr_region.n_synapses} L2 nodes, "
          f"{sum(1 for v in tr_lmap.values() if len(v) > 1)} frankenmerges")

    print("Building test world …")
    te_frags, te_region, te_lmap = _build_split_world(
        pos, frag, label, test_frags,
        version=args.version, k_nodes=args.k_nodes, rng=rng)
    n_te_complete = sum(fragment_completeness(te_lmap).values())
    print(f"  test:  {len(te_frags)} frags, {te_region.n_synapses} L2 nodes, "
          f"{sum(1 for v in te_lmap.values() if len(v) > 1)} frankenmerges, "
          f"GT complete {n_te_complete}/{len(te_lmap)} = {n_te_complete/len(te_lmap):.0%}")

    # ── Train encoder + partition GNN on the L2 substrate ─────────────────
    enc_kwargs = dict(node_input_dim=4, d_model=64, output_dim=32)
    encoder = FragmentEncoder(**enc_kwargs)
    print(f"\nTraining FragmentEncoder ({args.embed_epochs} epochs) …")
    if args.embed_epochs > 0:
        train_fragment_encoder(encoder, [tr_frags], n_epochs=args.embed_epochs,
                               lr=1e-3, margin=1.0, device=args.device,
                               root_label_map=tr_lmap, log_every=5)

    tr_enc = encode_fragments(encoder, tr_frags, device=args.device)
    tr_graph = build_observation_graph(tr_region, tr_enc, side="pre",
                                       k_spatial=args.k_spatial)
    print(f"  train graph: {tr_graph.n_nodes} nodes, {tr_graph.n_edges} edges")

    print(f"\nTraining EdgePartitionGNN ({args.partition_epochs} epochs) …")
    model, _ = train_edge_partition_multi_region(
        [tr_graph], n_epochs=args.partition_epochs, lr=1e-3,
        franken_hard_frac=0.30, max_train_nodes=args.train_max_nodes,
        device=args.device, seed=args.seed, log_every=20)

    # ── Save checkpoint ───────────────────────────────────────────────────
    _et = tr_graph.edge_type
    n_et = int(max(2, int(_et.max()) + 1)) if len(_et) else 2
    efd = int(tr_graph.edge_feat.shape[1]) if tr_graph.edge_feat.ndim == 2 else 0
    gnn_kwargs = dict(input_dim=tr_graph.node_feat.shape[1], d_model=64,
                      n_edge_types=n_et, output_dim=32, dropout=0.1,
                      edge_feat_dim=efd)
    save_checkpoint(args.save_checkpoint, encoder, model,
                    encoder_kwargs=enc_kwargs, gnn_kwargs=gnn_kwargs,
                    extra={"substrate": "l2", "bbox": P1_BBOX,
                           "version": args.version, "k_nodes": args.k_nodes})
    print(f"  checkpoint → {args.save_checkpoint}")

    # ── Eval sweep: merge precision is the trust metric ───────────────────
    te_enc = encode_fragments(encoder, te_frags, device=args.device)
    te_graph = build_observation_graph(te_region, te_enc, side="pre",
                                       k_spatial=args.k_spatial)
    print(f"\nTest graph: {te_graph.n_nodes} nodes, {te_graph.n_edges} edges")
    print("\nOperating-point sweep (held-out east split):")
    print(f"  {'cc_bias':>8} {'ARI':>6} {'merge_P':>8} {'merge_R':>8} {'over':>6} "
          f"{'cmpl_P':>7} {'cmpl_R':>7} {'cmpl_F1':>8}")
    biases = [float(b) for b in args.cc_bias.split(",")]
    for b in biases:
        if te_graph.n_nodes > args.tile_size:
            pred = partition_observations_tiled(
                model, te_graph, tile_size=args.tile_size, bias=b, device=args.device)
        else:
            pred = partition_observations_cc(model, te_graph, bias=b, device=args.device)
        pred = _reconcile_same_fragment(pred, te_graph.fragment_id)
        ev = evaluate_partition(pred, te_graph.labels)
        mm = merge_metrics(te_graph, pred)
        cm = completeness_metrics(te_lmap, _pred_completeness(te_graph.fragment_id, pred))
        print(f"  {b:>8.1f} {ev['ari']:>6.3f} {mm['merge_precision']:>8.3f} "
              f"{mm['merge_recall']:>8.3f} {mm.get('over_merge_rate', 0):>6.3f} "
              f"{cm['precision']:>7.3f} {cm['recall']:>7.3f} {cm['f1']:>8.3f}")

    print("\nDONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
