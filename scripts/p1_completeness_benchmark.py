#!/usr/bin/env python3
"""P1 proofread-dense benchmark: synapse vs L2 substrate + completeness task.

The proofread-dense **P1** region (true-nm x=818-918k, y=685-785k, z=794-994k,
in the 4,4,40 frame used by ``fetch_region_synapses`` and the L2 cache) was found
by scanning nucleus-soma edit rates: ~100% of somas here have v117 != v1718, so it
is densely populated with *real* proofreading errors — the opposite of the T-series
test boxes, whose v117 fragmentation is near-trivial (median 1, max 2).

This script builds the SAME region under two observation substrates and reports,
for each, (a) the v117→v1718 fragmentation distribution, (b) the ground-truth
completeness split (fraction of v117 roots that need no edit), and (c) the trained
GNN's completeness prediction.

Two substrates
--------------
synapse  ``build_region_world(..., l2_skeletons=False)``
    Observations are real synapses.  The ``min_syn_per_fragment=5`` sliver filter
    drops ~82% of synapses, keeping only high-degree fragments — which makes the
    region look almost edit-free (≈78% "complete", max ~5 fragments/neuron).  This
    is the same degree-bias that makes the T-series trivial.

l2       ``build_region_world_l2(..., min_l2_per_fragment=2)``
    Observations are L2 nodes (~5-10 µm chunkedgraph granularity).  Keeps every
    fragment down to 2 nodes, so the small axon/dendrite slivers that carry the
    merge signal survive (≈43% "complete", max 60 fragments/neuron, ~19% of
    neurons need ≥2 merges).  This is the honest benchmark.

The L2 walk is expensive (~52 min for 533 neurons), so its assembled arrays are
cached to ``cache/l2_world/p1_full.npz`` (git-lfs); reruns load instantly.

Completeness task
-----------------
A v117 fragment is *complete* (needs no edit) when it maps 1-to-1 onto a single
v1718 neuron (sole contributor, not a frankenmerge).  ``fragment_completeness``
gives the GT; the GNN's prediction marks a fragment complete when it lands alone
in a singleton cluster.  ``completeness_metrics`` scores P/R/F1/acc.

Note: the shared checkpoint is trained on *synapse* observation graphs.  On the L2
substrate the graph is one node per fragment with no same-fragment edges, so the
model is out of distribution (ARI≈0) — the L2 substrate needs its own training.
The GT fragmentation + completeness numbers are substrate-driven and valid
regardless of the model.

Usage
-----
  # both substrates (default):
  NEURONAUTS_L2_CACHE_DIR=$PWD/cache/l2_skeleton \
  NEURONAUTS_SYNAPSE_CACHE_DIR=$PWD/cache/synapse \
  PYTHONPATH=$PWD python3 scripts/p1_completeness_benchmark.py

  # one substrate only:
  python3 scripts/p1_completeness_benchmark.py --substrate l2
  python3 scripts/p1_completeness_benchmark.py --substrate synapse
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# P1 proofread-dense bbox, true nm (4,4,40 frame).
P1_BBOX = ((818_500, 685_000, 794_000), (918_500, 785_000, 994_000))
L2_CACHE = str(_ROOT / "cache" / "l2_world" / "p1_full.npz")


def _pred_completeness_from_partition(fragment_id, pred) -> dict[int, bool]:
    """A fragment is predicted complete iff it lands alone in a singleton cluster."""
    frag_clusters: dict[int, set] = {}
    cluster_frags: dict[int, set] = {}
    for f, c in zip(fragment_id.tolist(), pred.tolist()):
        if c >= 0:
            frag_clusters.setdefault(int(f), set()).add(int(c))
            cluster_frags.setdefault(int(c), set()).add(int(f))
    return {
        f: (len(cs) == 1 and len(cluster_frags[next(iter(cs))]) == 1)
        for f, cs in frag_clusters.items()
    }


def _report_fragmentation(lmap: dict[int, set], tag: str) -> None:
    v1718_to_n: dict[int, int] = {}
    for _frag, v1718s in lmap.items():
        for v in v1718s:
            v1718_to_n[v] = v1718_to_n.get(v, 0) + 1
    fpn = list(v1718_to_n.values())
    c = Counter(fpn)
    print(f"\n[{tag}] GT fragmentation (v117 fragments per v1718 neuron):")
    for k in sorted(c.keys())[:15]:
        star = "  <- complete (no merge)" if k == 1 else ""
        print(f"  {k:3d} frag(s): {c[k]:5d} neurons{star}")
    n_ge2 = sum(1 for n in fpn if n >= 2)
    print(f"  max={max(fpn)}  median={np.median(fpn):.1f}  mean={np.mean(fpn):.2f}  "
          f"neurons needing >=2 merges: {n_ge2}/{len(fpn)} ({n_ge2/len(fpn):.0%})")


def _build_synapse(args):
    from treestitch.realworld import build_region_world
    print("\n" + "=" * 68)
    print("SYNAPSE substrate")
    print("=" * 68)
    frags, region, lmap = build_region_world(
        P1_BBOX, version=args.version, side="pre",
        max_synapses=args.max_synapses,
        min_syn_per_fragment=args.min_syn_per_fragment,
        seed=args.seed, verbose=True,
        l2_skeletons=False,
        tile_x_nm=args.tile_x_nm, per_tile_limit=args.per_tile_limit)
    return frags, region, lmap


def _build_l2(args):
    """Build the L2-substrate world at FRAGMENT-CENTROID granularity.

    The raw L2 world has ~1.4M nodes; a kNN observation graph at that scale OOMs
    a 16 GB box.  We collapse each v117 fragment to its L2 centroid (one node per
    fragment, ~940 nodes) — the scale the GNN was trained at — for partitioning,
    while keeping the full-resolution ``root_label_map`` for the GT metrics.
    """
    import os
    from neuronauts.schemas import Region
    from treestitch.realworld import _cloud_fragment

    print("\n" + "=" * 68)
    print("L2 substrate (fragment-centroid)")
    print("=" * 68)
    cache = args.l2_cache if os.path.exists(args.l2_cache) else L2_CACHE
    if not os.path.exists(cache):
        raise FileNotFoundError(
            f"L2 world cache not found at {cache}. Build it first with "
            f"build_region_world_l2(P1_BBOX, cache_path=...).")
    print(f"  loading raw L2 arrays from {cache} …")
    d = np.load(cache)
    pos = d["pos"].astype(np.float32)
    frag = d["frag_ids"].astype(np.int64)
    label = d["labels"].astype(np.int64)

    fu, fc = np.unique(frag, return_counts=True)
    keep = {int(f) for f, c in zip(fu, fc) if c >= args.min_l2_per_fragment}
    m = np.array([int(f) in keep for f in frag])
    pos, frag, label = pos[m], frag[m], label[m]
    print(f"  {len(keep)} fragments (>= {args.min_l2_per_fragment} L2 nodes), "
          f"{m.sum()} L2 nodes")

    lmap: dict[int, set] = {}
    for f, l in zip(frag.tolist(), label.tolist()):
        lmap.setdefault(int(f), set()).add(int(l))

    frags_list = sorted(keep)
    n = len(frags_list)
    centroids = np.zeros((n, 3), dtype=np.float32)
    frag_labels = np.zeros(n, dtype=np.int64)
    cloud_frags = []
    for i, fr in enumerate(frags_list):
        idx = np.where(frag == fr)[0]
        centroids[i] = pos[idx].mean(0)
        lbls, cnts = np.unique(label[idx], return_counts=True)
        frag_labels[i] = int(lbls[cnts.argmax()])
        cloud_frags.append(_cloud_fragment(int(fr), f"minnie65_v{args.version}",
                                            pos[idx], idx))

    zeros = np.zeros(n, dtype=np.int64)
    region = Region(
        region_id=f"minnie65_v{args.version}_l2_frag",
        bbox_nm=(tuple(float(v) for v in centroids.min(0) - 5000),
                 tuple(float(v) for v in centroids.max(0) + 5000)),
        voxel_size_nm=(8.0, 8.0, 40.0),
        seg_version=117, label_version=args.version,
        pre_pt_nm=centroids, post_pt_nm=centroids.copy(),
        pre_root_id=frag_labels, post_root_id=zeros.copy(),
        synapse_id=np.arange(n, dtype=np.int64),
        pre_seg_id=np.array(frags_list, dtype=np.int64),
        post_seg_id=zeros.copy(),
    ).validate()
    return cloud_frags, region, lmap


def _eval(frags, region, lmap, args, tag):
    import torch
    from treestitch.checkpoint import load_checkpoint
    from treestitch.embed import encode_fragments
    from treestitch.graph import build_observation_graph
    from treestitch.partition import (
        partition_observations_cc, evaluate_partition, merge_metrics,
        completeness_metrics, fragment_completeness)

    _report_fragmentation(lmap, tag)

    gt = fragment_completeness(lmap)
    n_complete = sum(gt.values())
    print(f"\n[{tag}] GT completeness: {n_complete}/{len(gt)} = "
          f"{n_complete/len(gt):.1%} of fragments need no edit")

    if args.no_model:
        print(f"[{tag}] --no-model: skipping GNN prediction.")
        return

    print(f"\n[{tag}] loading checkpoint {args.checkpoint} …")
    encoder, model = load_checkpoint(args.checkpoint)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    frags_enc = encode_fragments(encoder, frags, device=device)
    graph = build_observation_graph(region, frags_enc, side="pre",
                                    k_spatial=args.k_spatial)
    print(f"[{tag}] graph: {graph.n_nodes} nodes, {graph.n_edges} edges")
    pred = partition_observations_cc(model, graph, bias=args.cc_bias, device=device)

    ev = evaluate_partition(pred, graph.labels)
    mm = merge_metrics(graph, pred)
    print(f"[{tag}] partition: ARI={ev['ari']:.3f}  "
          f"merge_P={mm['merge_precision']:.3f}  merge_R={mm['merge_recall']:.3f}  "
          f"over={mm.get('over_merge_rate', 0):.3f}")

    pred_c = _pred_completeness_from_partition(graph.fragment_id, pred)
    cm = completeness_metrics(lmap, pred_c)
    print(f"[{tag}] completeness pred: P={cm['precision']:.3f}  R={cm['recall']:.3f}  "
          f"F1={cm['f1']:.3f}  acc={cm['accuracy']:.3f}  "
          f"(pred complete: {sum(pred_c.values())}/{len(pred_c)})")


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--substrate", choices=["both", "synapse", "l2"], default="both")
    p.add_argument("--version", type=int, default=1718)
    p.add_argument("--checkpoint", default="/tmp/neuronauts_full.pt")
    p.add_argument("--no-model", action="store_true",
                   help="Report GT fragmentation + completeness only (no GNN).")
    p.add_argument("--max-synapses", type=int, default=50_000)
    p.add_argument("--min-syn-per-fragment", type=int, default=5)
    p.add_argument("--tile-x-nm", type=float, default=50_000)
    p.add_argument("--per-tile-limit", type=int, default=50_000)
    p.add_argument("--min-l2-per-fragment", type=int, default=2)
    p.add_argument("--l2-cache", default=L2_CACHE)
    p.add_argument("--k-spatial", type=int, default=6)
    p.add_argument("--cc-bias", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    print(f"P1 proofread-dense benchmark  bbox={P1_BBOX}")

    if args.substrate in ("synapse", "both"):
        frags, region, lmap = _build_synapse(args)
        _eval(frags, region, lmap, args, "synapse")

    if args.substrate in ("l2", "both"):
        frags, region, lmap = _build_l2(args)
        _eval(frags, region, lmap, args, "l2")

    print("\nDONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
