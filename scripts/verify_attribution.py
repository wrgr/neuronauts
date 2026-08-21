#!/usr/bin/env python3
"""First end-to-end attribution verifier: false-accept curve vs the v1718 oracle.

Implements the propose-then-verify loop of ``docs/tree_assembly_algorithm.md``
§5 on the hard tier: ANON pre-side (axonal) fragments attributed to NAMED
(single-nucleus) anchors.

Candidates are generated naively (nearest NAMED anchor by synapse-cloud gap —
the sloppy solver). Certification is a **label-free verification battery**:

  gap      — min synapse-cloud distance fragment↔anchor below τ_gap
  margin   — decoy-panel test: second-best anchor gap must exceed the best by
             a multiplicative margin (converts the scorer into a verifier)
  dna      — morphological-embedding cosine (deterministic
             ``encode_fragments_morphological``; no training, no labels)
  nucleus  — union soma count ≤ 1 (structural; ANON=0 + NAMED=1 passes)

The oracle: an accepted attribution is TRUE iff the fragment's majority v1718
root equals the anchor's v1718 root. Fragments whose true neuron has NO NAMED
anchor in the box are *unanswerable* — any acceptance on them is a false
accept, so the battery's abstention has to carry them. The report sweeps the
battery thresholds and prints, per operating point: accepted candidates,
attributed synapse mass, and the measured false-accept rate.

Usage
-----
  python scripts/verify_attribution.py --bbox 950000 930000 780000 1150000 1000000 880000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bbox", type=float, nargs=6, required=True,
                   metavar=("X0", "Y0", "Z0", "X1", "Y1", "Z1"))
    p.add_argument("--version", type=int, default=1718)
    p.add_argument("--max-synapses", type=int, default=20_000)
    p.add_argument("--min-syn-per-fragment", type=int, default=2)
    p.add_argument("--big-syn", type=int, default=50)
    p.add_argument("--candidate-radius-um", type=float, default=30.0,
                   help="max gap for candidate generation (µm)")
    p.add_argument("--panel-k", type=int, default=8,
                   help="anchors per decoy panel")
    p.add_argument("--tile-x-nm", type=float, default=0)
    p.add_argument("--per-tile-limit", type=int, default=200_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--nucleus-cache", default=None)
    p.add_argument("--soma-roots-cache", default=None)
    p.add_argument("--links", action="store_true",
                   help="verify LINKS between adjacent pre-side fragments "
                        "instead of fragment→soma-anchor attributions. "
                        "Identity flows along certified axon chains, so the "
                        "unit of certification is the link; a link is "
                        "gradeable when BOTH ends were proofreader-"
                        "adjudicated (non-self v1718 labels).")
    args = p.parse_args()

    import numpy as np
    from scipy.spatial import cKDTree

    from neuronauts.data import lineage as L
    from treestitch.embed import encode_fragments_morphological
    from treestitch.realworld import build_region_world_dual, count_contained_somata

    lo3, hi3 = tuple(args.bbox[:3]), tuple(args.bbox[3:])
    (pre_w, _post_w) = build_region_world_dual(
        (lo3, hi3), version=args.version, max_synapses=args.max_synapses,
        min_syn_per_fragment=args.min_syn_per_fragment, seed=args.seed,
        l2_skeletons_pre=False, l2_skeletons_post=False,
        tile_x_nm=args.tile_x_nm, per_tile_limit=args.per_tile_limit)
    fragments, region, lmap = pre_w

    soma_counts = count_contained_somata(
        version_ts=L.V117_TIMESTAMP,
        nucleus_cache_path=args.nucleus_cache,
        roots_cache_path=args.soma_roots_cache)

    roots = np.asarray(region.pre_seg_id, dtype=np.int64)
    pos = np.asarray(region.pre_pt_nm, dtype=np.float64)
    counts = {int(r): int(c) for r, c in zip(*np.unique(roots, return_counts=True))}

    def tier(r):
        s = soma_counts.get(int(r), 0)
        if s == 1:
            return "NAMED"
        if s >= 2:
            return "MULTI"
        return "BIG" if counts.get(int(r), 0) >= args.big_syn else "ANON"

    frag_by_root = {int(f.base_root_id): f for f in fragments}
    anchors = [int(r) for r in np.unique(roots) if tier(r) == "NAMED"]
    anon = [int(r) for r in np.unique(roots) if tier(r) == "ANON"]
    print(f"pre side: {len(anchors)} NAMED anchors, {len(anon)} ANON fragments")

    # majority v1718 label per root (oracle)
    def majority_label(r):
        m = roots == r
        labs = np.asarray(region.pre_root_id)[m]
        labs = labs[labs != 0]
        if len(labs) == 0:
            return 0
        vals, c = np.unique(labs, return_counts=True)
        return int(vals[np.argmax(c)])

    anchor_label = {r: majority_label(r) for r in anchors}
    anon_label = {r: majority_label(r) for r in anon}

    # Oracle-silence handling: a fragment whose v1718 label is ITSELF was never
    # adjudicated by proofreaders — silence is not a negative. Only
    # proofreader-touched (gradeable) fragments can score the battery.
    gradeable = {r for r in anon if anon_label[r] not in (0, r)}
    answerable = sum(1 for r in gradeable
                     if anon_label[r] in set(anchor_label.values()))
    print(f"adjudicated (gradeable) ANON fragments: {len(gradeable)}/{len(anon)}; "
          f"answerable (target among in-box NAMED anchors): {answerable}")

    # morphological DNA (deterministic, label-free)
    all_frags = [frag_by_root[r] for r in anchors + anon]
    enc = encode_fragments_morphological(all_frags)
    dna = {int(f.base_root_id): f.dna for f in enc}

    if args.links:
        return run_link_mode(args, np, cKDTree, roots, pos, region, counts,
                             anon, anon_label, dna)

    # candidate generation: nearest anchors by synapse-cloud gap
    anchor_pts = []
    anchor_idx = []
    for ai, r in enumerate(anchors):
        pts = pos[roots == r]
        anchor_pts.append(pts)
        anchor_idx.extend([ai] * len(pts))
    anchor_cloud = np.concatenate(anchor_pts, axis=0)
    anchor_of_pt = np.asarray(anchor_idx)
    tree = cKDTree(anchor_cloud)

    radius_nm = args.candidate_radius_um * 1_000.0
    cands = []  # (frag_root, anchor_root, gap, margin, cos, n_syn, is_true)
    for r in anon:
        pts = pos[roots == r]
        d, idx = tree.query(pts, k=min(60, len(anchor_cloud)))
        d, idx = np.atleast_2d(d), np.atleast_2d(idx)
        best_gap: dict[int, float] = {}
        for row_d, row_i in zip(d, idx):
            for gg, ii in zip(row_d, row_i):
                a = int(anchor_of_pt[int(ii)])
                if gg < best_gap.get(a, np.inf):
                    best_gap[a] = float(gg)
        panel = sorted(best_gap.items(), key=lambda kv: kv[1])[:args.panel_k]
        if not panel or panel[0][1] > radius_nm:
            continue
        (a0, g0) = panel[0]
        g1 = panel[1][1] if len(panel) > 1 else np.inf
        ar = anchors[a0]
        va, vb = dna[r], dna[ar]
        cos = float(np.dot(va, vb) /
                    (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-9))
        cands.append({
            "frag": r, "anchor": ar, "gap": g0,
            "margin": (g1 / max(g0, 1.0)), "cos": cos,
            "n_syn": counts.get(r, 0),
            "gradeable": r in gradeable,
            "true": anon_label[r] != 0 and anon_label[r] == anchor_label[ar],
        })

    n_c = len(cands)
    graded = [c for c in cands if c["gradeable"]]
    n_true = sum(c["true"] for c in graded)
    total_anon_syn = sum(counts.get(r, 0) for r in anon)
    print(f"\ncandidates (nearest anchor ≤ {args.candidate_radius_um:.0f} µm): "
          f"{n_c} total, {len(graded)} gradeable; naive solver precision on "
          f"gradeable = {n_true / max(len(graded), 1):.3f} (no-verifier baseline)")

    print("\nVerification battery sweep "
          "(accept iff gap ≤ G µm AND margin ≥ M AND cos ≥ C; "
          "precision/false-acc measured on GRADEABLE accepts only):")
    print(f"  {'G(µm)':>6s} {'M':>5s} {'C':>5s} {'accepted':>9s} "
          f"{'graded':>7s} {'syn mass':>9s} {'precision':>9s} {'false-acc':>9s}")
    grid_G = [30, 15, 8, 4, 2]
    grid_M = [1.0, 2.0, 4.0, 8.0]
    grid_C = [-1.0, 0.5, 0.8]
    best_zero = None
    for G in grid_G:
        for M in grid_M:
            for C in grid_C:
                acc = [c for c in cands
                       if c["gap"] <= G * 1000 and c["margin"] >= M
                       and c["cos"] >= C]
                g = [c for c in acc if c["gradeable"]]
                if not g:
                    continue
                tp = sum(c["true"] for c in g)
                fa = 1 - tp / len(g)
                mass = sum(c["n_syn"] for c in acc)
                print(f"  {G:6.0f} {M:5.1f} {C:5.1f} {len(acc):9d} "
                      f"{len(g):7d} {mass / max(total_anon_syn, 1):9.1%} "
                      f"{tp / len(g):9.3f} {fa:9.3f}")
                if fa == 0.0 and len(g) >= 10 and (best_zero is None
                                                   or mass > best_zero["mass"]):
                    best_zero = {"G": G, "M": M, "C": C,
                                 "n": len(acc), "mass": mass}
    if best_zero:
        print(f"\nbest measured-zero-false-accept point: "
              f"G={best_zero['G']}µm M={best_zero['M']} C={best_zero['C']} — "
              f"{best_zero['n']} attributions, "
              f"{best_zero['mass']} synapses "
              f"({best_zero['mass'] / max(total_anon_syn, 1):.1%} of ANON mass) "
              f"at FA=0 on this box")
    else:
        print("\nno operating point reached zero false-accepts — "
              "battery needs stronger channels (EM cut-face, trained scorer "
              "from a disjoint region)")
    return 0


def run_link_mode(args, np, cKDTree, roots, pos, region, counts,
                  anon, anon_label, dna):
    """Verify links between adjacent ANON pre-side fragments.

    Candidates: each ANON fragment's nearest other ANON fragment (by synapse-
    cloud gap) within the radius, with the decoy margin over the k-nearest
    panel. Oracle: gradeable iff BOTH ends carry non-self v1718 labels;
    TRUE iff the labels match.
    """
    anon_set = set(anon)
    a_pts, a_of_pt = [], []
    order = sorted(anon_set)
    index_of = {r: i for i, r in enumerate(order)}
    for r in order:
        pts = pos[roots == r]
        a_pts.append(pts)
        a_of_pt.extend([index_of[r]] * len(pts))
    cloud = np.concatenate(a_pts, axis=0)
    of_pt = np.asarray(a_of_pt)
    tree = cKDTree(cloud)

    radius_nm = args.candidate_radius_um * 1_000.0
    cands = []
    for r in order:
        pts = pos[roots == r]
        d, idx = tree.query(pts, k=min(40, len(cloud)))
        d, idx = np.atleast_2d(d), np.atleast_2d(idx)
        best_gap = {}
        me = index_of[r]
        for row_d, row_i in zip(d, idx):
            for gg, ii in zip(row_d, row_i):
                a = int(of_pt[int(ii)])
                if a == me:
                    continue
                if gg < best_gap.get(a, np.inf):
                    best_gap[a] = float(gg)
        panel = sorted(best_gap.items(), key=lambda kv: kv[1])[:args.panel_k]
        if not panel or panel[0][1] > radius_nm:
            continue
        (b0, g0) = panel[0]
        g1 = panel[1][1] if len(panel) > 1 else np.inf
        other = order[b0]
        if other < r:
            continue  # dedup: keep one direction per unordered pair
        va, vb = dna[r], dna[other]
        cos = float(np.dot(va, vb) /
                    (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-9))
        la, lb = anon_label[r], anon_label.get(other, 0)
        both_adj = (la not in (0, r)) and (lb not in (0, other))
        cands.append({
            "gap": g0, "margin": g1 / max(g0, 1.0), "cos": cos,
            "n_syn": counts.get(r, 0) + counts.get(other, 0),
            "gradeable": both_adj,
            "true": both_adj and la == lb,
        })

    graded = [c for c in cands if c["gradeable"]]
    n_true = sum(c["true"] for c in graded)
    total_syn = sum(counts.get(r, 0) for r in anon)
    print(f"\nLINK mode: {len(cands)} nearest-neighbour links, "
          f"{len(graded)} gradeable (both ends adjudicated); naive link "
          f"precision = {n_true / max(len(graded), 1):.3f}")

    print("\nLink battery sweep (gradeable accepts only):")
    print(f"  {'G(µm)':>6s} {'M':>5s} {'C':>5s} {'accepted':>9s} "
          f"{'graded':>7s} {'syn mass':>9s} {'precision':>9s} {'false-acc':>9s}")
    best_zero = None
    for G in [30, 15, 8, 4, 2, 1]:
        for M in [1.0, 2.0, 4.0, 8.0]:
            for C in [-1.0, 0.5, 0.8]:
                acc = [c for c in cands
                       if c["gap"] <= G * 1000 and c["margin"] >= M
                       and c["cos"] >= C]
                g = [c for c in acc if c["gradeable"]]
                if len(g) < 5:
                    continue
                tp = sum(c["true"] for c in g)
                fa = 1 - tp / len(g)
                mass = sum(c["n_syn"] for c in acc)
                print(f"  {G:6.0f} {M:5.1f} {C:5.1f} {len(acc):9d} "
                      f"{len(g):7d} {mass / max(total_syn, 1):9.1%} "
                      f"{tp / len(g):9.3f} {fa:9.3f}")
                if fa == 0.0 and len(g) >= 10 and (
                        best_zero is None or mass > best_zero["mass"]):
                    best_zero = {"G": G, "M": M, "C": C,
                                 "n": len(acc), "g": len(g), "mass": mass}
    if best_zero:
        print(f"\nbest zero-false-accept link point: G={best_zero['G']}µm "
              f"M={best_zero['M']} C={best_zero['C']} — {best_zero['n']} links "
              f"({best_zero['g']} graded) touching {best_zero['mass']} syn")
    else:
        print("\nno zero-false-accept link point with ≥10 graded accepts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
