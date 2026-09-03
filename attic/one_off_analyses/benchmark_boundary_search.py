"""Benchmark boundary-edge partition search vs plain threshold inference.

For each test box reports:
  - F1 with plain infer_cells at threshold=high_sim
  - F1 with boundary_partition_search
  - Whether the oracle (ground-truth) partition is reachable within the
    search space — i.e., could the beam ever find the correct answer?

Oracle reachability definition
-------------------------------
Starting from the base partition (threshold=high_sim), the beam can only ADD
merges (accept boundary edges).  The oracle is reachable iff:

  1. The base partition has NO wrong merges (pairs merged by base that oracle
     wants separated).  Wrong base merges are uncorrectable by the beam.

  2. Every oracle merge that the base missed is connected via a path through
     the boundary-edge graph [low_sim, high_sim).

Usage
-----
    python attic/one_off_analyses/benchmark_boundary_search.py \\
        --checkpoint models/cell_gnn_5feat.pt \\
        --cache-dir data/boxes
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from neuronauts.cell_graph import (
    _UF,
    _graph_to_tensors,
    _score_partition,
    build_synapse_graph,
    boundary_partition_search,
    connectivity_graph_from_cell_labels,
    infer_cells,
    load_cell_gnn,
    partition_from_embeddings,
    select_cell_gnn_training_boxes,
)
from neuronauts.dataset_builder import BoxCache
from neuronauts.line_graph import evaluate as lg_evaluate


def _oracle_uf(root_ids: np.ndarray) -> _UF:
    """Build a UF that merges all synapses sharing a root_id."""
    n = len(root_ids)
    uf = _UF(n)
    root_to_first: dict[int, int] = {}
    for i, rid in enumerate(root_ids):
        rid = int(rid)
        if rid <= 0:
            continue
        if rid in root_to_first:
            uf.union(i, root_to_first[rid])
        else:
            root_to_first[rid] = i
    return uf


def _build_normed_embeddings(model, graph):
    """Run model, return L2-normalised embeddings [N, D]."""
    import torch
    import torch.nn.functional as F
    model.eval()
    with torch.no_grad():
        nf, es, ed, ef = _graph_to_tensors(graph)
        raw = model(nf, es, ed, ef)
        return F.normalize(raw, p=2, dim=-1).cpu().numpy()


def oracle_reachability(
    model,
    graph,
    root_ids: np.ndarray,
    *,
    low_sim: float = 0.93,
    high_sim: float = 0.99,
    max_boundary_edges: int = 12,
) -> dict:
    """Return a reachability report for one side of one box."""
    N = graph.n_synapses
    if N == 0:
        return {"reachable": True, "wrong_base_merges": 0, "missed_pairs": 0,
                "covered_pairs": 0, "n_boundary_edges": 0}

    normed = _build_normed_embeddings(model, graph)

    # --- Base partition (high-confidence only) ---
    base_labels = partition_from_embeddings(normed, threshold=high_sim)
    base_uf = _UF(N)
    label_to_rep: dict[int, int] = {}
    for i in range(N):
        lbl = int(base_labels[i])
        if lbl not in label_to_rep:
            label_to_rep[lbl] = i
        else:
            base_uf.union(i, label_to_rep[lbl])

    # --- Oracle partition ---
    oracle = _oracle_uf(root_ids)

    # --- Wrong base merges: base merges (i,j) that oracle keeps separate ---
    wrong_base = 0
    base_lbls = base_uf.labels()
    oracle_lbls = oracle.labels()
    for i in range(N):
        for j in range(i + 1, N):
            base_same = (base_lbls[i] == base_lbls[j])
            oracle_same = (oracle_lbls[i] == oracle_lbls[j])
            if base_same and not oracle_same:
                wrong_base += 1

    # --- Boundary edges ---
    seen: set[tuple[int, int]] = set()
    boundary_edges: list[tuple[int, int]] = []
    for e in graph.edges:
        key = (min(e.src, e.dst), max(e.src, e.dst))
        if key in seen:
            continue
        seen.add(key)
        sim = float(normed[e.src] @ normed[e.dst])
        if low_sim <= sim < high_sim:
            boundary_edges.append(key)

    boundary_edges = boundary_edges[:max_boundary_edges]

    # Build boundary graph adjacency for reachability
    boundary_adj: dict[int, set[int]] = {i: set() for i in range(N)}
    for i, j in boundary_edges:
        boundary_adj[i].add(j)
        boundary_adj[j].add(i)

    # --- Missed pairs: oracle merges (i,j) that base keeps separate ---
    # Check if each missed pair is reachable via boundary edges starting from
    # the same base-component.
    missed = 0
    covered = 0

    # Group indices by oracle cell
    oracle_groups: dict[int, list[int]] = {}
    for i, lbl in enumerate(oracle_lbls):
        oracle_groups.setdefault(lbl, []).append(i)

    for members in oracle_groups.values():
        if len(members) < 2:
            continue
        # For each pair in this oracle cell, check if base already merges them
        # or if they can be connected via boundary edges
        for a in range(len(members)):
            for b in range(a + 1, len(members)):
                i, j = members[a], members[b]
                if base_lbls[i] == base_lbls[j]:
                    # Already merged in base — not a missed pair
                    continue
                missed += 1
                # BFS: can we connect i and j using only boundary edges,
                # staying within their base components?
                # (The beam can only merge, so different base components
                #  need a boundary-edge path between them.)
                visited = {i}
                queue = [i]
                found = False
                while queue and not found:
                    curr = queue.pop()
                    for nbr in boundary_adj[curr]:
                        if nbr == j:
                            found = True
                            break
                        if nbr not in visited:
                            visited.add(nbr)
                            queue.append(nbr)
                if found:
                    covered += 1

    reachable = (wrong_base == 0) and (missed == 0 or covered == missed)

    return {
        "reachable": reachable,
        "wrong_base_merges": wrong_base,
        "missed_pairs": missed,
        "covered_pairs": covered,
        "n_boundary_edges": len(boundary_edges),
    }


def benchmark_box(model, synapses, *, high_sim=0.99, low_sim=0.93,
                  proximity_radius_nm=5000.0, max_boundary_edges=12):
    pre_graph = build_synapse_graph(
        synapses, "pre",
        proximity_radius_nm=proximity_radius_nm,
        partner_seg_ids=getattr(synapses, "post_seg_id", None),
    )
    post_graph = build_synapse_graph(
        synapses, "post",
        proximity_radius_nm=proximity_radius_nm,
        partner_seg_ids=getattr(synapses, "pre_seg_id", None),
    )

    # --- Plain threshold ---
    pre_plain = infer_cells(model, pre_graph, threshold=high_sim)
    post_plain = infer_cells(model, post_graph, threshold=high_sim)
    cg_plain = connectivity_graph_from_cell_labels(pre_plain, post_plain, synapses)
    m_plain = lg_evaluate(cg_plain, synapses.pre_root_id, synapses.post_root_id)

    # --- Boundary search ---
    pre_search = boundary_partition_search(
        model, pre_graph, low_sim=low_sim, high_sim=high_sim,
        max_boundary_edges=max_boundary_edges,
    )
    post_search = boundary_partition_search(
        model, post_graph, low_sim=low_sim, high_sim=high_sim,
        max_boundary_edges=max_boundary_edges,
    )
    cg_search = connectivity_graph_from_cell_labels(pre_search, post_search, synapses)
    m_search = lg_evaluate(cg_search, synapses.pre_root_id, synapses.post_root_id)

    # --- Oracle reachability (pre side only for speed; post is symmetric) ---
    reach_pre = oracle_reachability(
        model, pre_graph, synapses.pre_root_id,
        low_sim=low_sim, high_sim=high_sim,
        max_boundary_edges=max_boundary_edges,
    )
    reach_post = oracle_reachability(
        model, post_graph, synapses.post_root_id,
        low_sim=low_sim, high_sim=high_sim,
        max_boundary_edges=max_boundary_edges,
    )

    return {
        "n_synapses": len(synapses.pre_pt),
        "plain_f1": m_plain.f1,
        "plain_p": m_plain.precision,
        "plain_r": m_plain.recall,
        "search_f1": m_search.f1,
        "search_p": m_search.precision,
        "search_r": m_search.recall,
        "oracle_reachable_pre": reach_pre["reachable"],
        "oracle_reachable_post": reach_post["reachable"],
        "wrong_base_merges_pre": reach_pre["wrong_base_merges"],
        "missed_pairs_pre": reach_pre["missed_pairs"],
        "covered_pairs_pre": reach_pre["covered_pairs"],
        "n_boundary_edges_pre": reach_pre["n_boundary_edges"],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default="models/cell_gnn_5feat.pt")
    p.add_argument("--cache-dir", default="data/boxes")
    p.add_argument("--high-sim", type=float, default=0.99)
    p.add_argument("--low-sim", type=float, default=0.93)
    p.add_argument("--proximity-radius-nm", type=float, default=5000.0)
    p.add_argument("--max-boundary-edges", type=int, default=12)
    p.add_argument("--split", choices=["val", "test", "all"], default="test")
    p.add_argument("--output", default=None, help="JSON output path")
    args = p.parse_args()

    model = load_cell_gnn(args.checkpoint)
    model.eval()
    print(f"Loaded: {args.checkpoint}")

    cache = BoxCache(args.cache_dir)
    splits = select_cell_gnn_training_boxes(cache)
    if args.split == "all":
        records = splits["train"] + splits["val"] + splits["test"]
    else:
        records = splits[args.split]
    print(f"Evaluating {len(records)} {args.split} boxes\n")
    print(f"  Settings: high_sim={args.high_sim}  low_sim={args.low_sim}  "
          f"max_boundary_edges={args.max_boundary_edges}\n")

    rows = []
    for rec in records:
        try:
            _, syn = cache.load(rec)
        except Exception:
            continue
        if len(syn.pre_pt) < 5:
            continue
        key = getattr(rec, "key", None) or getattr(rec, "box_id", None) or str(rec)[:8]
        r = benchmark_box(
            model, syn,
            high_sim=args.high_sim,
            low_sim=args.low_sim,
            proximity_radius_nm=args.proximity_radius_nm,
            max_boundary_edges=args.max_boundary_edges,
        )
        r["box"] = key
        rows.append(r)
        delta = r["search_f1"] - r["plain_f1"]
        reach = "YES" if (r["oracle_reachable_pre"] and r["oracle_reachable_post"]) else "no"
        print(
            f"  {key}  "
            f"plain={r['plain_f1']:.3f}  search={r['search_f1']:.3f}  "
            f"Δ={delta:+.3f}  "
            f"boundary_edges={r['n_boundary_edges_pre']}  "
            f"missed={r['missed_pairs_pre']}  covered={r['covered_pairs_pre']}  "
            f"wrong_base={r['wrong_base_merges_pre']}  "
            f"oracle_reachable={reach}"
        )

    if not rows:
        print("No boxes evaluated.")
        return

    plain_f1s = [r["plain_f1"] for r in rows]
    search_f1s = [r["search_f1"] for r in rows]
    reachable_frac = np.mean([
        r["oracle_reachable_pre"] and r["oracle_reachable_post"] for r in rows
    ])
    wrong_base = [r["wrong_base_merges_pre"] for r in rows]

    print()
    print("=" * 70)
    print(f"Plain  F1:  mean={np.mean(plain_f1s):.4f}  median={np.median(plain_f1s):.4f}")
    print(f"Search F1:  mean={np.mean(search_f1s):.4f}  median={np.median(search_f1s):.4f}")
    print(f"ΔF1 (search - plain):  "
          f"mean={np.mean(search_f1s) - np.mean(plain_f1s):+.4f}  "
          f"median={np.median(search_f1s) - np.median(plain_f1s):+.4f}")
    print()
    print(f"Oracle reachable (both sides): {reachable_frac:.0%} of boxes")
    print(f"Wrong base merges (pre-side):  "
          f"mean={np.mean(wrong_base):.1f}  max={np.max(wrong_base)}")
    print("=" * 70)

    if args.output:
        with open(args.output, "w") as f:
            json.dump({
                "settings": vars(args),
                "summary": {
                    "plain_f1_mean": float(np.mean(plain_f1s)),
                    "plain_f1_median": float(np.median(plain_f1s)),
                    "search_f1_mean": float(np.mean(search_f1s)),
                    "search_f1_median": float(np.median(search_f1s)),
                    "oracle_reachable_frac": float(reachable_frac),
                    "wrong_base_merges_mean": float(np.mean(wrong_base)),
                },
                "boxes": rows,
            }, f, indent=2)
        print(f"\nResults → {args.output}")


if __name__ == "__main__":
    main()
