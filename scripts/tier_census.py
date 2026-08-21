#!/usr/bin/env python3
"""Tier accounting census: how much of the synapse graph is certifiable today?

Implements the tiered identity-resolution frame of
``docs/tree_assembly_algorithm.md`` §5: the connectome is a fixed set of
synapses; identity claims attach to each synapse's two sides, and a side with
no certified identity is *withheld* (anonymous node), never guessed. Tiers per
v117 root, by exact lineage soma containment (whole nucleus table):

  NAMED       — contains exactly 1 nucleus → identity = that soma's neuron
  MULTI       — contains ≥2 nuclei → catastrophic-editor caseload
  BIG-NOSOMA  — 0 nuclei, ≥ ``--big-syn`` synapses in the box → external
                neuron or must-merge candidate; deferred, never guessed
  ANON        — 0 nuclei, small → anonymous axon/dendrite fragment;
                attribution workload, ranked by synapse count

Both endpoints of the SAME synapses come from one dual fetch, so the census
reports the (post-tier × pre-tier) synapse-mass matrix and the headline
numbers of the tiered product:

  - post-side NAMED fraction  (half-edge certification — "even dendrites")
  - fully-certified edge fraction (both sides NAMED)
  - verified precision of NAMED claims against v1718 (fragment + mass level)
  - ranked ANON pre-side worklist (top fragments by synapse count)

Usage
-----
  python scripts/tier_census.py --bbox 950000 930000 780000 1150000 1000000 880000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

TIERS = ["NAMED", "MULTI", "BIG-NOSOMA", "ANON"]


def tier_of(root: int, n_syn: int, soma_counts: dict, big_syn: int) -> str:
    s = soma_counts.get(int(root), 0)
    if s == 1:
        return "NAMED"
    if s >= 2:
        return "MULTI"
    return "BIG-NOSOMA" if n_syn >= big_syn else "ANON"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bbox", type=float, nargs=6, required=True,
                   metavar=("X0", "Y0", "Z0", "X1", "Y1", "Z1"))
    p.add_argument("--version", type=int, default=1718)
    p.add_argument("--max-synapses", type=int, default=20_000)
    p.add_argument("--min-syn-per-fragment", type=int, default=2,
                   help="keep the sliver tail — it is the workload being counted")
    p.add_argument("--big-syn", type=int, default=50,
                   help="synapse count above which a 0-soma root is BIG-NOSOMA")
    p.add_argument("--tile-x-nm", type=float, default=0)
    p.add_argument("--per-tile-limit", type=int, default=200_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--nucleus-cache", default=None)
    p.add_argument("--soma-roots-cache", default=None,
                   help="npz cache for the nucleus-supervoxel→v117-root resolution")
    p.add_argument("--top-anon", type=int, default=15,
                   help="print this many top anonymous pre-side fragments")
    args = p.parse_args()

    import numpy as np

    from neuronauts.data import lineage as L
    from treestitch.realworld import build_region_world_dual, count_contained_somata

    lo3, hi3 = tuple(args.bbox[:3]), tuple(args.bbox[3:])
    (pre_w, post_w) = build_region_world_dual(
        (lo3, hi3), version=args.version, max_synapses=args.max_synapses,
        min_syn_per_fragment=args.min_syn_per_fragment, seed=args.seed,
        l2_skeletons_pre=False, l2_skeletons_post=False,
        tile_x_nm=args.tile_x_nm, per_tile_limit=args.per_tile_limit)
    frags_pre, region_pre, lmap_pre = pre_w
    frags_post, region_post, lmap_post = post_w

    soma_counts = count_contained_somata(
        version_ts=L.V117_TIMESTAMP,
        nucleus_cache_path=args.nucleus_cache,
        roots_cache_path=args.soma_roots_cache)
    print(f"nucleus containment: {len(soma_counts)} v117 roots hold ≥1 nucleus; "
          f"{sum(1 for c in soma_counts.values() if c >= 2)} hold ≥2")

    # ---- join the two sides on the shared CAVE synapse id ------------------
    pre_ids = region_pre.synapse_id
    post_ids = region_post.synapse_id
    pre_index = {int(s): i for i, s in enumerate(pre_ids)}
    joint = [(pre_index[int(s)], j) for j, s in enumerate(post_ids)
             if int(s) in pre_index]
    print(f"joined synapses: {len(joint)} "
          f"(pre side {len(pre_ids)}, post side {len(post_ids)})")

    pre_root = region_pre.pre_seg_id
    post_root = region_post.post_seg_id
    pre_true = region_pre.pre_root_id
    post_true = region_post.post_root_id

    # per-root synapse counts (side-local)
    pre_counts = {int(r): int(c) for r, c in
                  zip(*np.unique(pre_root, return_counts=True))}
    post_counts = {int(r): int(c) for r, c in
                   zip(*np.unique(post_root, return_counts=True))}

    def side_tier(root, counts):
        return tier_of(root, counts.get(int(root), 0), soma_counts, args.big_syn)

    # ---- tier matrix over joined synapses ----------------------------------
    matrix = {(a, b): 0 for a in TIERS for b in TIERS}
    for i, j in joint:
        matrix[(side_tier(post_root[j], post_counts),
                side_tier(pre_root[i], pre_counts))] += 1

    n_joint = max(len(joint), 1)
    print(f"\nSynapse-mass tier matrix (rows = POST side, cols = PRE side, "
          f"% of {n_joint} joined synapses):")
    header = "  POST \\ PRE   " + "".join(f"{t:>12s}" for t in TIERS)
    print(header)
    for a in TIERS:
        row = "".join(f"{matrix[(a, b)] / n_joint:12.1%}" for b in TIERS)
        print(f"  {a:12s}{row}")

    post_named = sum(matrix[("NAMED", b)] for b in TIERS) / n_joint
    both_named = matrix[("NAMED", "NAMED")] / n_joint
    print(f"\n  half-edge certified (post NAMED):   {post_named:6.1%}")
    print(f"  full-edge certified (both NAMED):   {both_named:6.1%}")

    # ---- verified precision of NAMED claims (against v1718) ----------------
    def named_precision(region_side, roots, true, counts, lmap, label):
        named_roots = [r for r in np.unique(roots)
                       if side_tier(r, counts) == "NAMED"]
        if not named_roots:
            print(f"  {label}: no NAMED roots")
            return
        pure = sum(1 for r in named_roots if len(lmap.get(int(r), set())) == 1)
        mass = mass_pure = 0
        for r in named_roots:
            m = roots == r
            labs = true[m]
            labs = labs[labs != 0]
            if len(labs) == 0:
                continue
            _, c = np.unique(labs, return_counts=True)
            mass += int(c.sum())
            mass_pure += int(c.max())
        print(f"  {label}: {len(named_roots)} NAMED roots — fragment purity "
              f"{pure / len(named_roots):.4f}, mass purity "
              f"{mass_pure / max(mass, 1):.4f} ({mass} synapses)")

    print("\nVerified precision of NAMED tiers vs v1718:")
    named_precision(region_post, post_root, post_true, post_counts, lmap_post,
                    "POST (dendritic scaffold)")
    named_precision(region_pre, pre_root, pre_true, pre_counts, lmap_pre,
                    "PRE  (axonal)")

    # ---- MULTI split: benign (glia/duplicate nuclei) vs catastrophic -------
    # Glia carry no synapses, so a synapse-bearing MULTI root is either a true
    # neuron-neuron merge (catastrophic) or a neuron merged with glia /
    # duplicate nucleus detections (identity-benign: every synapse still
    # belongs to the one neuron). The proofreading oracle separates them for
    # the census: a MULTI root that maps to a SINGLE v1718 neuron was left
    # whole by proofreaders → benign.
    def multi_split(roots, counts, lmap, label):
        multi = [int(r) for r in np.unique(roots)
                 if side_tier(r, counts) == "MULTI"]
        if not multi:
            print(f"  {label}: no MULTI roots")
            return
        benign = [r for r in multi if len(lmap.get(r, set())) == 1]
        mass = sum(counts.get(r, 0) for r in multi)
        mass_benign = sum(counts.get(r, 0) for r in benign)
        print(f"  {label}: {len(multi)} MULTI roots ({mass} syn) — "
              f"benign (single v1718 target): {len(benign)} roots "
              f"({mass_benign} syn, {mass_benign / max(mass, 1):.1%} of MULTI mass); "
              f"catastrophic: {len(multi) - len(benign)} roots "
              f"({mass - mass_benign} syn)")

    print("\nMULTI split (v1718 oracle; glia have no synapses):")
    multi_split(post_root, post_counts, lmap_post, "POST")
    multi_split(pre_root, pre_counts, lmap_pre, "PRE ")

    # ---- ranked anonymous pre-side worklist --------------------------------
    anon = [(pre_counts[int(r)], int(r)) for r in np.unique(pre_root)
            if side_tier(r, pre_counts) == "ANON"]
    anon.sort(reverse=True)
    anon_mass = sum(c for c, _ in anon)
    print(f"\nANON pre-side worklist: {len(anon)} fragments, "
          f"{anon_mass} synapses ({anon_mass / max(len(pre_ids), 1):.1%} of pre side)")
    for c, r in anon[:args.top_anon]:
        labs = lmap_pre.get(r, set())
        print(f"    root {r}: {c} syn, v1718 targets={len(labs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
