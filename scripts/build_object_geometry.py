"""Join the raw L2 fetch into an object point cloud: every node of every atom.

``build_atom_topology.py`` contracts the same fetch into a skeleton and keeps
only the *endpoints* -- the degree-1 nodes. That is the surface every proximity
experiment has run on. This script keeps the rest: for each atom, the position
of all of its L2 nodes, joined from

  ``geom/shards/<tier>_*.npz``   atom -> its L2 node ids (and within-atom edges)
  ``geom/l2_attributes.npz``     L2 id -> rep_coord_nm, max_dt_nm, ...

so that "how close are these two objects" can be asked of the objects rather
than of their skeleton tips.

Three integrity gates, all fatal, because a quietly incomplete point cloud would
make every distance drawn from it wrong in the safe-looking direction (too
close, never too far):

  1. every L2 id in the shards resolves in ``l2_attributes``;
  2. the tier's atom set matches the corresponding topology file exactly;
  3. an atom's endpoint L2 ids are a **subset** of its node L2 ids, and the two
     tables give the same coordinate for the same id. This is what makes the
     object measurement comparable to the endpoint one rather than merely
     different: nested point sets mean min-over-nodes <= min-over-endpoints for
     every pair, always.

    python scripts/build_object_geometry.py --tier 10
    python scripts/build_object_geometry.py --tier all
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neuronauts.harness.objgeom import ObjectGeometry  # noqa: E402

#: Which shard prefixes make up each tier. The fetch tiers are incremental --
#: each "skips atoms already done" -- so "all" is their union, not k1 alone.
#: That distinction previously cost a whole experiment (see EXP-060B's note).
TIER_SHARDS = {"10": ["k10"], "all": ["k10", "k5", "k1"]}
TIER_TOPOLOGY = {"10": "k10", "all": "kall"}


def load_l2_attributes(path: Path):
    with np.load(path, allow_pickle=False) as z:
        l2 = z["l2_id"]
        order = np.argsort(l2, kind="stable")
        return (l2[order], z["pos_nm"][order], z["max_dt_nm"][order])


def load_shards(geom_dir: Path, prefixes: list[str]):
    """Concatenate per-atom L2 node sets across every shard of a tier."""
    files = [f for p in prefixes
             for f in sorted(glob.glob(str(geom_dir / "shards" / f"{p}_*.npz")))]
    if not files:
        raise SystemExit(f"no shards under {geom_dir/'shards'} for {prefixes}")
    atom_ids, ptrs, l2_ids = [], [], []
    off = 0
    for f in files:
        with np.load(f, allow_pickle=False) as z:
            a, p, n = z["atom_id"], z["node_ptr"], z["l2_ids"]
        if len(p) != len(a) + 1 or int(p[-1]) != len(n):
            raise SystemExit(f"{f}: malformed CSR (atoms={len(a)}, ptr={len(p)}, "
                             f"nodes={len(n)}, ptr[-1]={int(p[-1])})")
        atom_ids.append(a)
        ptrs.append(p[:-1].astype(np.int64) + off)
        l2_ids.append(n)
        off += len(n)
    atom_id = np.concatenate(atom_ids)
    node_ptr = np.concatenate(ptrs + [np.array([off], np.int64)])
    return atom_id, node_ptr, np.concatenate(l2_ids), len(files)


def check_endpoint_nesting(topo: Path, atom_id, node_ptr, l2_id, pos_nm):
    """Gate 3: endpoints are a subset of nodes, at identical coordinates."""
    with np.load(topo, allow_pickle=False) as z:
        ep_atom, ep_l2, ep_pos = z["ep_atom"], z["ep_l2_id"], z["ep_pos_nm"]

    row = {int(a): k for k, a in enumerate(atom_id.tolist())}
    o = np.argsort(ep_atom, kind="stable")
    ea, el2 = ep_atom[o], ep_l2[o]
    ua, starts = np.unique(ea, return_index=True)
    ends = np.r_[starts[1:], len(ea)]

    missing_atoms, missing_ids, total = 0, 0, 0
    for a, s, e in zip(ua.tolist(), starts, ends):
        k = row.get(int(a))
        if k is None:
            missing_atoms += 1
            continue
        nodes = l2_id[int(node_ptr[k]):int(node_ptr[k + 1])]
        total += int(e - s)
        missing_ids += int((~np.isin(el2[s:e], nodes)).sum())

    srt = np.argsort(l2_id, kind="stable")
    ls, ps = l2_id[srt], pos_nm[srt]
    j = np.clip(np.searchsorted(ls, ep_l2), 0, max(len(ls) - 1, 0))
    hit = ls[j] == ep_l2
    d = np.linalg.norm(ps[j][hit] - ep_pos[hit], axis=1)
    d = d[np.isfinite(d)]
    max_off = float(d.max()) if len(d) else 0.0
    return {"endpoint_atoms_without_geometry": missing_atoms,
            "endpoint_l2_ids_not_in_node_set": missing_ids,
            "endpoint_l2_ids_checked": total,
            "max_position_disagreement_nm": max_off,
            "positions_compared": int(len(d))}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tier", choices=sorted(TIER_SHARDS), default="10")
    ap.add_argument("--geom-dir", default="data/substrate/geom")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    t0 = time.time()
    geom = Path(args.geom_dir)
    topo = Path("data/substrate/topology") / f"{TIER_TOPOLOGY[args.tier]}.npz"
    out = Path(args.out or geom / f"objgeom_{TIER_TOPOLOGY[args.tier]}.npz")

    l2, pos, mdt = load_l2_attributes(geom / "l2_attributes.npz")
    print(f"l2_attributes: {len(l2):,} nodes  ({time.time()-t0:.1f}s)")

    atom_id, node_ptr, node_l2, n_files = load_shards(geom, TIER_SHARDS[args.tier])
    print(f"tier {args.tier}: {n_files} shards, {len(atom_id):,} atoms, "
          f"{len(node_l2):,} node slots")
    if len(np.unique(atom_id)) != len(atom_id):
        raise SystemExit(f"tier {args.tier}: shards overlap -- "
                         f"{len(atom_id):,} slots, "
                         f"{len(np.unique(atom_id)):,} unique atoms")

    # Gate 1: every shard L2 id must exist in the attribute table.
    j = np.clip(np.searchsorted(l2, node_l2), 0, len(l2) - 1)
    hit = l2[j] == node_l2
    if not hit.all():
        raise SystemExit(f"{int((~hit).sum()):,} of {len(node_l2):,} L2 ids are "
                         f"absent from l2_attributes; the point cloud would have "
                         f"holes and every distance from it would read too far")
    P, R = pos[j], mdt[j]
    resolved = np.isfinite(P).all(axis=1)
    print(f"  l2 ids resolved      : {len(node_l2):,}/{len(node_l2):,} (100%)")
    print(f"  finite positions     : {int(resolved.sum()):,} "
          f"({resolved.mean():.4%})")

    # Gate 2: the atom set must match the topology file this will be compared to.
    if topo.exists():
        with np.load(topo, allow_pickle=False) as z:
            topo_atoms = z["atom_id"]
        if set(topo_atoms.tolist()) != set(atom_id.tolist()):
            raise SystemExit(
                f"atom set disagrees with {topo}: {len(topo_atoms):,} there, "
                f"{len(atom_id):,} here -- the two are not the same substrate")
        print(f"  atom set matches {topo.name}: {len(atom_id):,} atoms")

        # Gate 3: nesting.
        nest = check_endpoint_nesting(topo, atom_id, node_ptr, node_l2, P)
        print(f"  endpoint l2 ids not in their atom's node set: "
              f"{nest['endpoint_l2_ids_not_in_node_set']:,}/"
              f"{nest['endpoint_l2_ids_checked']:,}")
        print(f"  max position disagreement on shared ids: "
              f"{nest['max_position_disagreement_nm']:.3f} nm")
        if (nest["endpoint_l2_ids_not_in_node_set"]
                or nest["endpoint_atoms_without_geometry"]
                or nest["max_position_disagreement_nm"] > 1e-3):
            raise SystemExit(
                "endpoints are not a subset of nodes at identical coordinates; "
                "object and endpoint distances would not be comparable")
    else:
        print(f"  ({topo} absent -- skipping the atom-set and nesting gates)")
        nest = {}

    counts = np.diff(node_ptr)
    meta = {"tier": args.tier, "shards": TIER_SHARDS[args.tier],
            "n_shard_files": n_files, "topology": str(topo),
            "l2_attributes": str(geom / "l2_attributes.npz"),
            "n_atoms": int(len(atom_id)), "n_nodes": int(len(node_l2)),
            "n_nodes_positioned": int(resolved.sum()),
            "nodes_per_atom_median": float(np.median(counts)),
            "nodes_per_atom_p90": float(np.percentile(counts, 90)),
            "max_dt_nm_median": float(np.median(R[resolved])),
            "gates": nest, "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                      time.gmtime())}

    ObjectGeometry(atom_id=atom_id, node_ptr=node_ptr, l2_id=node_l2,
                   pos_nm=P.astype(np.float32), max_dt_nm=R.astype(np.float32),
                   resolved=resolved, meta=meta).save(out)
    print(f"  nodes/atom: median {np.median(counts):,.0f}  "
          f"p90 {np.percentile(counts,90):,.0f}  max {counts.max():,}")
    print(f"wrote {out}  ({time.time()-t0:.1f}s)")
    print(json.dumps(meta["gates"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
