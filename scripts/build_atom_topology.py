"""Contract every atom's L2 adjacency into the harness topology table.

Input is the raw fetch (``data/substrate/geom``): per-atom L2 node sets, real
``lvl2_graph`` adjacency, and pooled L2 attributes. Output is one NPZ holding

  * a per-atom row -- node/edge/component counts, endpoint and branch counts,
    cable length, caliber, and the atom's presynaptic/postsynaptic tallies;
  * a per-endpoint row -- position, outward tangent, the length and caliber of
    the leaf segment it terminates.

The endpoint table is the surface candidate generation runs on: a false split
appears as two endpoints facing each other. Endpoints are *not* scarce at L2
resolution, so the caliber and leaf-length columns are what a proposer filters
on; this script reports their distribution so that filter can be chosen from
data rather than guessed.

Cable length is NaN for any segment with a coordinate-less node, and the NaN
share is reported rather than silently dropped.

    uv run python scripts/build_atom_topology.py --tier 10
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neuronauts.harness.population import load_population  # noqa: E402
from neuronauts.report.provenance import write_result  # noqa: E402
from neuronauts.harness.topology import (  # noqa: E402
    L2Attributes, build_adjacency, contract, segment_lengths, segment_paths,
    segment_tip_tangents,
)

ATOM_COLS = ["n_l2", "n_edge", "n_comp", "n_iso", "n_end", "n_branch",
             "n_seg", "n_leaf_seg", "n_cycle", "cable_nm", "cable_nan_seg",
             "caliber_mean_nm", "n_pre", "n_post"]


def polarity_counts(pop):
    """Per-atom (n_pre, n_post); presynaptic = the axonal side of a contact."""
    atoms = pop.atom_id
    order = np.argsort(atoms)
    srt = atoms[order]

    def tally(side):
        v = side[side > 0]
        idx = np.searchsorted(srt, v)
        ok = (idx < len(srt)) & (srt[np.clip(idx, 0, len(srt) - 1)] == v)
        return np.bincount(order[idx[ok]], minlength=len(atoms)).astype(np.int64)

    return tally(pop.syn_atom_pre), tally(pop.syn_atom_post)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--population", default="data/substrate/c100um/population.npz")
    ap.add_argument("--geom-dir", default="data/substrate/geom")
    ap.add_argument("--tier", default="10",
                    help="shard prefix to contract, e.g. 10 -> k10_*.npz only "
                        "(that tier's own incremental shards, NOT the union "
                        "of everything with >=k synapses -- each tier's fetch "
                        "skips atoms already done in a narrower tier, so "
                        "k1_*.npz alone holds only the 1-4 synapse slice). "
                        "Pass 'all' to glob every shard (k10_+k5_+k1_), the "
                        "true complete population.")
    ap.add_argument("--span", type=int, default=5,
                    help="nodes back along a segment used for the tip tangent")
    ap.add_argument("--out", default="")
    ap.add_argument("--report", default="")
    args = ap.parse_args()

    tier_label = "all" if args.tier == "all" else int(args.tier)
    out = Path(args.out or f"data/substrate/topology/k{tier_label}.npz")
    report = Path(args.report or f"results/atom_topology_k{tier_label}.json")

    pop = load_population(args.population)
    n_pre_all, n_post_all = polarity_counts(pop)
    pop_order = np.argsort(pop.atom_id)
    pop_srt = pop.atom_id[pop_order]

    attrs = L2Attributes(Path(args.geom_dir) / "l2_attributes.npz",
                         cols=["pos_nm", "mean_dt_nm"])
    print(f"attributes: {len(attrs.l2_id):,} L2 nodes", flush=True)

    pattern = "shards/*.npz" if args.tier == "all" else f"shards/k{args.tier}_*.npz"
    shards = sorted(Path(args.geom_dir).glob(pattern))
    if not shards:
        raise SystemExit(f"no shards matching {pattern} in {args.geom_dir}")
    print(f"shards    : {len(shards)}", flush=True)

    rows: list[np.ndarray] = []
    atom_ids: list[int] = []
    ep_atom, ep_l2, ep_pos, ep_tan, ep_len, ep_cal = [], [], [], [], [], []

    t0 = time.time()
    for si, f in enumerate(shards, 1):
        with np.load(f, allow_pickle=False) as z:
            aid, npt, ept = z["atom_id"], z["node_ptr"], z["edge_ptr"]
            l2, ed = z["l2_ids"], z["edges"]

        for i in range(len(aid)):
            ids = l2[npt[i]:npt[i + 1]]
            atom = int(aid[i])
            if len(ids) == 0:
                continue
            pos = attrs.take(ids, "pos_nm")
            cal = attrs.take(ids, "mean_dt_nm")
            indptr, indices, deg = build_adjacency(ids, ed[ept[i]:ept[i + 1]])
            t = contract(indptr, indices, deg)

            flat, ptr = segment_paths(t)
            seg_len = (segment_lengths(flat, ptr, pos) if len(ptr) > 1
                       else np.zeros(0, np.float32))
            tip, tan = (segment_tip_tangents(t, flat, ptr, pos, span=args.span)
                        if len(ptr) > 1 else
                        (np.zeros(0, np.int32), np.zeros((0, 3), np.float32)))

            # the leaf segment each tip terminates, for its length and caliber
            if len(tip):
                a_at, b_at = ptr[:-1], ptr[1:] - 1
                owner = np.concatenate([
                    np.flatnonzero(t.deg[flat[a_at]] == 1),
                    np.flatnonzero(t.deg[flat[b_at]] == 1)])
                ep_atom.append(np.full(len(tip), atom, np.uint64))
                ep_l2.append(ids[tip])
                ep_pos.append(pos[tip])
                ep_tan.append(tan)
                ep_len.append(seg_len[owner])
                ep_cal.append(cal[tip])

            j = np.searchsorted(pop_srt, atom)
            has = j < len(pop_srt) and pop_srt[j] == atom
            k = pop_order[j] if has else -1

            rows.append(np.array([
                len(ids), indptr[-1] // 2, int(t.comp.max()) + 1,
                int((deg == 0).sum()), int((deg == 1).sum()),
                int((deg >= 3).sum()), len(t.seg_ends),
                int(t.seg_is_leaf.sum()), t.cycles,
                float(np.nansum(seg_len)), int(np.isnan(seg_len).sum()),
                float(np.nanmean(cal)) if len(cal) else np.nan,
                int(n_pre_all[k]) if has else 0,
                int(n_post_all[k]) if has else 0,
            ], np.float64))
            atom_ids.append(atom)

        print(f"  [{si}/{len(shards)}] {len(atom_ids):,} atoms "
              f"({time.time()-t0:.0f}s)", flush=True)

    tab = np.vstack(rows)
    atom_id = np.asarray(atom_ids, np.uint64)
    ep_atom = np.concatenate(ep_atom) if ep_atom else np.zeros(0, np.uint64)
    ep_l2 = np.concatenate(ep_l2) if len(ep_l2) else np.zeros(0, np.uint64)
    ep_pos = np.concatenate(ep_pos) if len(ep_pos) else np.zeros((0, 3), np.float32)
    ep_tan = np.concatenate(ep_tan) if len(ep_tan) else np.zeros((0, 3), np.float32)
    ep_len = np.concatenate(ep_len) if len(ep_len) else np.zeros(0, np.float32)
    ep_cal = np.concatenate(ep_cal) if len(ep_cal) else np.zeros(0, np.float32)

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out, atom_id=atom_id,
        **{c: tab[:, i].astype(np.float32) for i, c in enumerate(ATOM_COLS)},
        ep_atom=ep_atom, ep_l2_id=ep_l2, ep_pos_nm=ep_pos.astype(np.float32),
        ep_tangent=ep_tan.astype(np.float32),
        ep_seg_len_nm=ep_len.astype(np.float32),
        ep_caliber_nm=ep_cal.astype(np.float32),
        meta=np.frombuffer(json.dumps({
            "tier": tier_label, "span": args.span,
            "geom_dir": str(args.geom_dir), "population": str(args.population),
            "cols": ATOM_COLS}).encode(), np.uint8))

    col = {c: tab[:, i] for i, c in enumerate(ATOM_COLS)}
    n_seg_total = int(col["n_seg"].sum())
    nan_seg = int(col["cable_nan_seg"].sum())
    q = [10, 25, 50, 75, 90, 99]

    tier_display = "ALL (complete population)" if args.tier == "all" else f">={args.tier}"
    print(f"\n{'='*70}\nTIER {tier_display} CONTRACTED TOPOLOGY "
          f"({len(atom_id):,} atoms, {time.time()-t0:.0f}s)")
    print(f"  L2 nodes             : {int(col['n_l2'].sum()):,}")
    print(f"  L2 edges             : {int(col['n_edge'].sum()):,}")
    print(f"  components           : {int(col['n_comp'].sum()):,} "
          f"({col['n_comp'].mean():.2f}/atom)")
    print(f"  endpoints            : {int(col['n_end'].sum()):,} "
          f"({col['n_end'].mean():.0f}/atom)")
    print(f"  branch nodes         : {int(col['n_branch'].sum()):,}")
    print(f"  segments             : {n_seg_total:,} "
          f"(leaf {int(col['n_leaf_seg'].sum()):,})")
    print(f"  segments w/o length  : {nan_seg:,} "
          f"({100*nan_seg/max(n_seg_total,1):.3f}% -- missing coordinates)")
    print(f"  total cable          : {col['cable_nm'].sum()/1e9:.2f} m")
    print(f"  endpoint rows        : {len(ep_l2):,}")

    if len(ep_len):
        print(f"\n  leaf-segment length nm : " +
              "  ".join(f"p{p}={np.nanpercentile(ep_len,p):,.0f}" for p in q))
        print(f"  endpoint caliber nm    : " +
              "  ".join(f"p{p}={np.nanpercentile(ep_cal,p):,.0f}" for p in q))
        print("\n  candidate-endpoint yield under a joint filter:")
        for ln in (1000, 2000, 5000):
            for cl in (30, 50, 80):
                m = (ep_len >= ln) & (ep_cal >= cl)
                print(f"    leaf>={ln:>5}nm & caliber>={cl:>3}nm : "
                      f"{int(m.sum()):>9,} ({100*m.mean():5.2f}%)")

    write_result(report, {
        "tier": tier_label, "n_atoms": int(len(atom_id)),
        "n_l2": int(col["n_l2"].sum()), "n_edges": int(col["n_edge"].sum()),
        "n_components": int(col["n_comp"].sum()),
        "n_endpoints": int(col["n_end"].sum()),
        "n_branch": int(col["n_branch"].sum()),
        "n_segments": n_seg_total,
        "n_leaf_segments": int(col["n_leaf_seg"].sum()),
        "n_segments_without_length": nan_seg,
        "cable_m": float(col["cable_nm"].sum() / 1e9),
        "leaf_len_nm_pct": {str(p): float(np.nanpercentile(ep_len, p))
                            for p in q} if len(ep_len) else {},
        "endpoint_caliber_nm_pct": {str(p): float(np.nanpercentile(ep_cal, p))
                                    for p in q} if len(ep_cal) else {},
        "out": str(out),
    }, inputs=[args.population, str(out)],
        params={"tier": tier_label, "span": args.span,
                "geom_dir": str(args.geom_dir)},
        quick_hash=True, synthetic_fallback=False)
    print(f"\nwrote {out} and {report}")


if __name__ == "__main__":
    main()
